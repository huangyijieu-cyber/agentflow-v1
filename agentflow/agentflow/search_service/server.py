"""Authenticated HTTP entry point for the shared search service."""
from __future__ import annotations

import argparse
import gzip
import hmac
import io
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .core import GatewayFailure, SearchService, ServiceConfig, json_dumps


LOG = logging.getLogger("search_service")


def _reject_constant(value):
    raise ValueError("Nonfinite JSON numbers are not allowed")


def _validate_json_numbers(value):
    # JSON such as 1e999 is also nonfinite, despite having no NaN literal.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON numbers are not allowed")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_numbers(item)


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, service):
        token = service.config.token
        if not token or token != token.strip() or any(c.isspace() for c in token):
            raise ValueError("SEARCH_SERVICE_TOKEN (or SEARCH_CACHE_TOKEN) is required and must contain no whitespace")
        self.service = service
        self._connections = threading.BoundedSemaphore(service.config.workers + service.config.queue_size + 16)
        super().__init__(address, ServiceRequestHandler)

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            body = b'{"error":{"code":"queue_full","message":"Search service is busy","retryable":true}}'
            try:
                request.settimeout(1.0)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: "
                                + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)
            except OSError:
                pass
            finally:
                self.service.metrics.add("connection_rejections")
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class ServiceRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AgentFlowSearchService/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(min(10.0, self.server.service.config.request_timeout))

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.server.service.metrics.add("client_disconnects")

    def log_message(self, format, *args):
        # Neither URLs containing queries, request bodies, nor auth headers are logged.
        pass

    def _authenticated(self):
        expected = ("Bearer " + self.server.service.config.token).encode("utf-8")
        supplied = self.headers.get("Authorization", "").encode("utf-8")
        return hmac.compare_digest(supplied, expected)

    def _send_json(self, status, body, *, request_id=None, retry_after=None):
        payload = json_dumps(body).encode("utf-8")
        gzip_allowed = False
        for entry in self.headers.get("Accept-Encoding", "").lower().split(","):
            parts = [part.strip() for part in entry.split(";")]
            if parts[0] in {"gzip", "*"} and "q=0" not in parts and "q=0.0" not in parts:
                gzip_allowed = True
        compressed = gzip_allowed and len(payload) >= 512
        if compressed:
            payload = gzip.compress(payload, mtime=0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        if request_id:
            self.send_header("X-Request-ID", request_id)
        if retry_after is not None:
            self.send_header("Retry-After", str(max(0, math.ceil(retry_after))))
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def _error(self, failure, request_id):
        self.server.service.metrics.add("http_errors")
        self._send_json(failure.status_code, {"error": failure.as_dict(), "request_id": request_id},
                        request_id=request_id, retry_after=failure.retry_after)

    def do_GET(self):
        request_id = uuid.uuid4().hex
        try:
            path = urlsplit(self.path).path
            if not self._authenticated():
                self._error(GatewayFailure("Bearer authentication required", code="unauthorized", status_code=401), request_id)
            elif path == "/healthz":
                self._send_json(200, {"status": "ok"}, request_id=request_id)
            elif path == "/metrics":
                self._send_json(200, self.server.service.metrics_snapshot(), request_id=request_id)
            else:
                self._error(GatewayFailure("Endpoint not found", code="not_found", status_code=404), request_id)
        except sqlite3.Error:
            LOG.exception("Cache database failure during GET; request_id=%s", request_id)
            self._error(GatewayFailure("Cache database is unavailable; inspect the service log",
                                       code="cache_database_error", status_code=503), request_id)
        except (TimeoutError, OSError):
            self.server.service.metrics.add("client_disconnects")
        except Exception:
            LOG.exception("Search service GET failed; request_id=%s", request_id)
            self._error(GatewayFailure("Internal search service error", code="internal_error", status_code=500), request_id)

    def _read_body(self):
        if self.headers.get("Transfer-Encoding"):
            raise GatewayFailure("Chunked requests are not supported", code="invalid_request", status_code=400)
        header = self.headers.get("Content-Length")
        if header is None:
            raise GatewayFailure("Content-Length is required", code="invalid_request", status_code=411)
        try:
            length = int(header)
        except ValueError:
            raise GatewayFailure("Invalid Content-Length", code="invalid_request", status_code=400) from None
        limit = self.server.service.config.max_body_bytes
        if length < 0:
            raise GatewayFailure("Invalid Content-Length", code="invalid_request", status_code=400)
        if length > limit:
            raise GatewayFailure("Request exceeds size limit", code="request_too_large", status_code=413)
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise GatewayFailure("Content-Type must be application/json", code="invalid_request", status_code=415)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise GatewayFailure("Incomplete request body", code="invalid_request", status_code=400)
        encoding = self.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
                    raw = archive.read(limit + 1)
            except (OSError, EOFError):
                raise GatewayFailure("Invalid gzip body", code="invalid_request", status_code=400) from None
            if len(raw) > limit:
                raise GatewayFailure("Expanded request exceeds size limit", code="request_too_large", status_code=413)
        elif encoding != "identity":
            raise GatewayFailure("Unsupported Content-Encoding", code="invalid_request", status_code=415)
        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            _validate_json_numbers(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise GatewayFailure("Body must contain finite JSON", code="invalid_request", status_code=400) from None
        if not isinstance(body, dict):
            raise GatewayFailure("Body must be a JSON object", code="invalid_request", status_code=400)
        return body

    def do_POST(self):
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.server.service.config.request_timeout
        try:
            if not self._authenticated():
                raise GatewayFailure("Bearer authentication required", code="unauthorized", status_code=401)
            path = urlsplit(self.path).path
            endpoints = {"/v1/search/wikipedia": "wikipedia", "/v1/search/brave": "brave", "/v1/fetch": "fetch"}
            if path not in endpoints and path != "/v1/batch":
                raise GatewayFailure("Endpoint not found", code="not_found", status_code=404)
            body = self._read_body()
            if path == "/v1/batch":
                if set(body) != {"requests"}:
                    raise GatewayFailure("Batch body requires only requests", code="invalid_request", status_code=400)
                result = self.server.service.batch(body["requests"], deadline=deadline, request_id=request_id)
            else:
                result = self.server.service.handle(endpoints[path], body, deadline=deadline, request_id=request_id)
            self._send_json(200, result, request_id=request_id)
        except GatewayFailure as exc:
            self._error(exc, request_id)
        except (TimeoutError, OSError):
            # A disconnected client cannot consume a response; don't log its body.
            self.server.service.metrics.add("client_disconnects")
        except Exception:
            self._error(GatewayFailure("Internal search service error", code="internal_error", status_code=500), request_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AgentFlow's persistent shared search/cache service")
    parser.add_argument("--host", default=os.environ.get("SEARCH_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SEARCH_SERVICE_PORT", "8091")))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args(argv)
    try:
        config = ServiceConfig.from_env()
        if args.cache_dir:
            config.cache_dir = args.cache_dir
        if args.db_path:
            config.db_path = args.db_path
        if not config.token or config.token != config.token.strip() or any(c.isspace() for c in config.token):
            parser.error("SEARCH_SERVICE_TOKEN (or SEARCH_CACHE_TOKEN) is required and must contain no whitespace")
        if not 0 <= args.port <= 65535:
            parser.error("port must be in 0..65535")
        service = SearchService(config)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    try:
        server = ServiceHTTPServer((args.host, args.port), service)
    except BaseException:
        service.close()
        raise
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Search service listening on %s:%s; state directory %s; database %s",
             args.host, server.server_port, config.cache_dir, service.cache.path)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
