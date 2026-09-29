"""Context limits and token counts for page reading and agent prompts.

vLLM exposes its actual serving limit through /v1/models and its tokenizer
through /tokenize. Older servers fall back to an explicitly configured limit
(8192 otherwise) and a conservative UTF-8 byte count, never chars/4.
"""
import os
from functools import lru_cache

import requests


class ContextBudgetError(ValueError):
    pass


class ContextBudget:
    def __init__(self, engine, env_prefix="AGENTFLOW_WEB"):
        self.engine = engine
        self.system_prompt = getattr(engine, "system_prompt", "") or ""
        self.model = getattr(engine, "model_string", "")
        self.base_url = str(getattr(engine, "base_url", "") or "").rstrip("/")
        self.headers = {}
        api_key = getattr(engine, "api_key", None)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        configured = os.getenv(f"{env_prefix}_CONTEXT_TOKENS")
        configured = int(configured) if configured else None
        if configured is not None and configured <= 0:
            raise ValueError(f"{env_prefix}_CONTEXT_TOKENS must be positive")
        discovered = self._discover_limit()
        self.limit = min(configured, discovered) if configured and discovered else (
            configured or discovered or 8192
        )
        self.margin = 256  # Chat template and serialization overhead.
        self._remote_tokenizer = bool(self.base_url)
        self.tokenizer_mode = "vllm" if self._remote_tokenizer else "utf8-upper-bound"
        if not discovered:
            print(f"[Context budget] Serving limit unavailable; using {self.limit} tokens "
                  f"({env_prefix}_CONTEXT_TOKENS can override this fallback).")

    def _json_request(self, method, url, **kwargs):
        # Session methods avoid the Wikipedia requests.get retry wrapper.
        with requests.Session() as session:
            response = session.request(method, url, headers=self.headers,
                                       timeout=(2, 5), **kwargs)
            response.raise_for_status()
            return response.json()

    def _discover_limit(self):
        if not self.base_url:
            return None
        try:
            models = self._json_request("GET", self.base_url + "/models")
            for model in models.get("data", []):
                if model.get("id") == self.model:
                    limit = model.get("max_model_len")
                    if limit and int(limit) > 0:
                        return int(limit)
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            pass
        return None

    @lru_cache(maxsize=32)
    def count(self, text):
        if self._remote_tokenizer:
            try:
                root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
                data = self._json_request("POST", root + "/tokenize", json={
                    "model": self.model, "prompt": text, "add_special_tokens": False,
                })
                count = int(data["count"])
                if count < 0 or (text and count == 0):
                    raise ValueError("Invalid token count")
                return count
            except (requests.RequestException, KeyError, ValueError, TypeError):
                self._remote_tokenizer = False
                self.tokenizer_mode = "utf8-upper-bound"
                self.count.cache_clear()
                print("[Context budget] /tokenize unavailable; using a conservative "
                      "UTF-8 byte upper bound (may create more chunks).")
        return len(text.encode("utf-8"))

    def fits(self, prompt, output_tokens):
        return (self.count(self.system_prompt + "\n" + prompt)
                + output_tokens + self.margin <= self.limit)

    def require(self, prompt, output_tokens):
        if not self.fits(prompt, output_tokens):
            raise ContextBudgetError(
                f"Prompt plus {output_tokens} output tokens exceeds the "
                f"{self.limit}-token context budget ({self.tokenizer_mode})."
            )


def get_context_budget(engine, env_prefix="AGENTFLOW_WEB"):
    # Reuse discovery and tokenizer state for this engine in subsequent rollouts.
    budgets = getattr(engine, "_agentflow_context_budgets", None)
    if budgets is None:
        budgets = {}
        engine._agentflow_context_budgets = budgets
    if env_prefix not in budgets:
        budgets[env_prefix] = ContextBudget(engine, env_prefix)
    return budgets[env_prefix]
