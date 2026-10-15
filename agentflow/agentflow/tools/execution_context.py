"""Per-tool deadlines and safe failures, isolated across rollout threads."""
from __future__ import annotations

import contextvars
import re
import threading
import time
from contextlib import contextmanager


DEFAULT_TOOL_TIMEOUT = 300.0
DEFAULT_API_TIMEOUT = 120.0
_current = contextvars.ContextVar("agentflow_tool_execution", default=None)


class ToolExecutionError(RuntimeError):
    def __init__(self, message, *, code, phase, error_type=None, status_code=None, budget_seconds=None):
        self.message = message
        self.code, self.phase = code, phase
        self.error_type, self.status_code = error_type, status_code
        self.budget_seconds = budget_seconds
        detail = f"{code}, phase={phase}"
        if error_type:
            detail += f", type={error_type}"
        if status_code is not None:
            detail += f", HTTP={status_code}"
        if budget_seconds is not None:
            detail += f", budget={budget_seconds:.3g}s"
        super().__init__(f"{message} ({detail})")

    def copy(self):
        return type(self)(self.message, code=self.code, phase=self.phase,
                          error_type=self.error_type, status_code=self.status_code,
                          budget_seconds=self.budget_seconds)


def safe_failure(error, phase):
    """Expose type/status, never SDK exception bodies containing prompts or keys."""
    if isinstance(error, ToolExecutionError):
        return error
    names, chain = set(), []
    current = error
    while current is not None and len(chain) < 8 and all(current is not item for item in chain):
        chain.append(current)
        names.update(cls.__name__ for cls in type(current).__mro__)
        current = current.__cause__ or current.__context__
    status = (getattr(error, "upstream_status_code", None) or getattr(error, "upstream_status", None)
              or getattr(error, "status_code", None))
    if not isinstance(status, int) or isinstance(status, bool):
        status = None
    kind = type(error).__name__
    if names & {"SearchGatewayError", "GatewayFailure"}:
        code = getattr(error, "code", "search_request_error")
        code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", code) else "search_request_error"
        return ToolExecutionError("Shared search request failed", code=code, phase=phase,
                                  error_type=kind, status_code=status)
    if "ProxyError" in names:
        code = "api_proxy_error"
        if status is None:
            for item in chain:
                match = re.search(r"(?:Tunnel connection failed:\s*|HTTP(?:/\d(?:\.\d)?)?\s+)([45]\d{2})\b",
                                  str(item), re.IGNORECASE)
                if match:
                    status = int(match.group(1))
                    break
    elif names & {"SSLError", "SSLCertVerificationError"}:
        code = "api_tls_error"
    elif names & {"APITimeoutError", "Timeout", "TimeoutError", "TimeoutException",
                  "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
        code = "api_timeout"
    elif names & {"APIConnectionError", "ConnectionError", "ConnectError", "NetworkError",
                  "ReadError", "WriteError", "RemoteProtocolError", "ConnectionResetError",
                  "ConnectionRefusedError", "BrokenPipeError", "gaierror"}:
        code = "api_connection_error"
    elif status is not None or names & {"APIStatusError", "HTTPError", "RateLimitError"}:
        code = "api_http_error"
    else:
        code = "tool_processing_error"
    return ToolExecutionError("Tool operation failed", code=code, phase=phase,
                              error_type=kind, status_code=status)


class ToolExecutionContext:
    def __init__(self, timeout=DEFAULT_TOOL_TIMEOUT):
        self.timeout = timeout
        self.deadline = time.monotonic() + timeout
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._aborters = set()
        self.phase, self.last_error = "tool.execute", None

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def record_error(self, error):
        with self._lock:
            if self.last_error is None or not self.cancel_event.is_set():
                self.last_error = error

    def set_phase(self, phase):
        self.check()
        with self._lock:
            if not self.cancel_event.is_set():
                self.phase = phase
                # A previous unavailable page must not become the cause of a
                # later request's timeout, even when both share a phase name.
                self.last_error = None

    def _abort(self, callback):
        def close():
            try:
                callback()
            except Exception:
                pass
        # Transport close must not block the caller's deadline handling.
        threading.Thread(target=close, name="ToolTransportClose", daemon=True).start()

    def register_abort(self, callback):
        with self._lock:
            cancelled = self.cancel_event.is_set()
            if not cancelled:
                self._aborters.add(callback)
        if cancelled:
            self._abort(callback)
        def remove():
            with self._lock:
                self._aborters.discard(callback)
        return remove

    def cancel(self, error=None):
        with self._lock:
            if error is not None and self.last_error is None:
                self.last_error = error
            self.cancel_event.set()
            callbacks = list(self._aborters)
            self._aborters.clear()
        for callback in callbacks:
            self._abort(callback)

    def check(self):
        if self.remaining() <= 0 and not self.cancel_event.is_set():
            self.cancel(ToolExecutionError("Tool total deadline exceeded",
                                          code="tool_deadline_exceeded", phase=self.phase,
                                          budget_seconds=self.timeout))
        if self.cancel_event.is_set():
            with self._lock:
                error = self.last_error
            raise (error.copy() if error is not None else ToolExecutionError(
                "Tool execution cancelled", code="tool_cancelled", phase=self.phase))

    def diagnostic(self):
        with self._lock:
            return str(self.last_error) if self.last_error is not None else f"phase={self.phase}"


def current_context():
    return _current.get()


@contextmanager
def bind_context(context):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


@contextmanager
def ensure_tool_context(timeout=DEFAULT_TOOL_TIMEOUT):
    context = current_context()
    if context is not None:
        context.check()
        yield context
    else:
        with bind_context(ToolExecutionContext(timeout)) as context:
            yield context


def api_timeout(cap=DEFAULT_API_TIMEOUT):
    context = current_context()
    if context is None:
        return cap
    context.check()
    return min(cap, context.remaining())


def run_bounded(operation, *, phase, timeout=DEFAULT_API_TIMEOUT, abort=None, preserve_exceptions=()):
    """Bound wall time as well as SDK socket time; late work cannot start next stages."""
    with ensure_tool_context() as context:
        context.set_phase(phase)
        limit = min(timeout, context.remaining())
        copied = contextvars.copy_context()
        done, result = threading.Event(), {}
        remove_abort = context.register_abort(abort) if abort is not None else lambda: None

        def target():
            try:
                context.check()
                result["value"] = operation()
                context.check()
            except Exception as error:
                failure = safe_failure(error, phase)
                context.record_error(failure)
                # Keep shared-service error attributes for Wiki's page handling.
                known = any(cls.__name__ in {"SearchGatewayError", "GatewayFailure", "ToolExecutionError"}
                            for cls in type(error).__mro__)
                result["error"] = error if known or isinstance(error, preserve_exceptions) else failure
            finally:
                done.set()

        threading.Thread(target=lambda: copied.run(target), name="ToolIO-" + phase, daemon=True).start()
        try:
            if not done.wait(limit):
                failure = ToolExecutionError("Tool operation exceeded its time budget",
                                             code="tool_operation_timeout", phase=phase,
                                             budget_seconds=limit)
                with context._lock:
                    earlier = context.last_error
                if earlier is not None and earlier.phase == phase:
                    failure = earlier.copy()
                else:
                    context.record_error(failure)
                context.cancel()
                raise failure
            if "error" in result:
                raise result["error"]
            context.check()
            return result["value"]
        finally:
            remove_abort()


@contextmanager
def owned_api_client(factory):
    context = current_context()
    client = factory(api_timeout())
    remove = context.register_abort(client.close) if context is not None else lambda: None
    try:
        if context is not None:
            context.check()
        yield client
        if context is not None:
            context.check()
    except Exception as error:
        if context is not None:
            failure = safe_failure(error, context.phase)
            context.record_error(failure)
            if isinstance(error, ToolExecutionError):
                raise
            # This also prevents an engine's decorator from retrying a raw SDK
            # connection exception after the tool has already observed failure.
            raise failure from error
        raise
    finally:
        remove()
        try:
            client.close()
        except Exception:
            pass


@contextmanager
def tool_sdk_client(existing, factory):
    if current_context() is None:
        yield existing
    else:
        with owned_api_client(factory) as client:
            yield client


def check_api_result(result, phase):
    if isinstance(result, dict) and result.get("error"):
        kind = result["error"]
        # Some engines return an error envelope rather than raising an SDK error.
        # Its message/details may contain credentials, so retain only the type.
        kind = kind if isinstance(kind, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", kind) else "APIError"
        raise ToolExecutionError("Model API returned an error", code="api_error_response",
                                 phase=phase, error_type=kind)
    return result
