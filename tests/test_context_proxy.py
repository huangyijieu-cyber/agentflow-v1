"""Exercise the real Flask proxy routes without importing training/GPU packages.

Run with Flask installed: python3 -m unittest discover -s tests -p 'test_context_proxy.py' -v
"""
import ast
import importlib.util
import json
import logging
from pathlib import Path
import random
import sys
import time
import types
import unittest
from unittest.mock import patch

import requests
from flask import Flask, Response, abort, request, stream_with_context

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentflow"))
from agentflow.context_budget import ContextBudget
from agentflow.models.memory import Memory, MEMORY_PLACEHOLDER

spec = importlib.util.spec_from_file_location("context_endpoints", ROOT / "agentflow/verl/context_endpoints.py")
endpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endpoints)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        # Compile the actual method, replacing only its heavyweight runtime globals.
        source = ast.parse((ROOT / "agentflow/verl/daemon.py").read_text())
        daemon = next(node for node in source.body if isinstance(node, ast.ClassDef)
                      and node.name == "AgentModeDaemon")
        method = next(node for node in daemon.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_start_proxy_server")
        apps = []

        def make_app(name):
            app = Flask(name)
            app.testing = True
            apps.append(app)
            return app

        class DormantThread:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

        namespace = {
            "__name__": "context_proxy_test", "Flask": make_app,
            "register_tokenizer_routes": endpoints.register_tokenizer_routes,
            "Response": Response, "abort": abort, "request": request,
            "stream_with_context": stream_with_context, "requests": requests,
            "random": random, "time": time, "json": json,
            "threading": types.SimpleNamespace(Thread=DormantThread),
            "logger": logging.getLogger("context_proxy_test"),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "daemon_proxy", "exec"), namespace)
        self.tokenizer_calls = []

        def encode(text, **kwargs):
            self.tokenizer_calls.append((text, kwargs))
            return list(range(len(text.split())))

        instance = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(encode=encode),
            backend_llm_server_addresses=["backend:1234"],
            llm_timeout_seconds=30, proxy_port=4321,
        )
        namespace["_start_proxy_server"](instance)
        self.client = apps[0].test_client()

    def backend_response(self, payload):
        return types.SimpleNamespace(status_code=200,
                                     content=json.dumps(payload).encode(),
                                     raw=types.SimpleNamespace(headers={"Content-Type": "application/json"}))

    def test_get_models_with_empty_body_returns_backend_metadata(self):
        metadata = {"data": [{"id": "planner", "max_model_len": 10752}]}
        with patch.object(requests, "request", return_value=self.backend_response(metadata)) as backend:
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), metadata)
        self.assertEqual(backend.call_args.kwargs["url"], "http://backend:1234/v1/models")

    def test_tokenizer_routes_use_training_tokenizer_without_backend_inference(self):
        with patch.object(requests, "request", side_effect=AssertionError("No model request expected")):
            for route in ("/tokenize", "/v1/tokenize"):
                response = self.client.post(route, json={"prompt": "中文 text", "add_special_tokens": False})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["count"], 2)
        self.assertEqual(len(self.tokenizer_calls), 2)
        self.assertFalse(self.tokenizer_calls[0][1]["add_special_tokens"])

    def test_bad_tokenizer_payload_returns_400(self):
        self.assertEqual(self.client.post("/tokenize", json={"prompt": []}).status_code, 400)
        self.assertEqual(self.client.post("/tokenize", data="").status_code, 400)

    def test_normal_chat_completion_still_passes_through(self):
        answer = {"choices": [{"message": {"content": "answer"}}]}
        with patch.object(requests, "request", return_value=self.backend_response(answer)):
            response = self.client.post("/v1/chat/completions", json={"messages": [], "stream": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), answer)

    def test_full_budget_discovery_and_counting_through_proxy(self):
        def client_request(method, url, **kwargs):
            path = url.removeprefix("http://proxy")
            response = (self.client.get(path) if method == "GET"
                        else self.client.post(path, json=kwargs["json"]))
            self.assertEqual(response.status_code, 200)
            return response.get_json()
        metadata = {"data": [{"id": "planner", "max_model_len": 10752}]}
        engine = types.SimpleNamespace(model_string="planner", base_url="http://proxy/v1", system_prompt="")
        with patch.object(requests, "request", return_value=self.backend_response(metadata)), \
                patch.object(ContextBudget, "_json_request", side_effect=client_request):
            budget = ContextBudget(engine, "AGENTFLOW_AGENT")
            self.assertEqual(budget.limit, 10752)
            memory = Memory()
            long_text = "information " * 1500 + "TAIL_EVIDENCE"
            memory.add_action(1, "Web_RAG_Search_Tool", "goal", "command", [long_text])
            prompt = memory.render_prompt(MEMORY_PLACEHOLDER, engine, budget=budget)
            self.assertIn(long_text, prompt)
            self.assertTrue(budget.fits(prompt, 2048))


if __name__ == "__main__":
    unittest.main()
