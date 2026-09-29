"""Context limits and token counts for page reading and agent prompts.

vLLM exposes its actual serving limit through /v1/models and its tokenizer
through /tokenize. Also support /v1/tokenize behind proxies and cached/local
model tokenizers. A UTF-8 byte bound is only a last-resort estimate, not a
measurement that can prove an agent prompt is over the model's token limit.
"""
import os
from functools import lru_cache

import requests


class ContextBudgetError(ValueError):
    pass


@lru_cache(maxsize=4)
def _load_local_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    except (ImportError, OSError, ValueError, TypeError):
        return None


class ContextBudget:
    def __init__(self, engine, env_prefix="AGENTFLOW_WEB"):
        self.engine = engine
        self.system_prompt = getattr(engine, "system_prompt", "") or ""
        self.model = getattr(engine, "model_string", "")
        self.local_tokenizer = getattr(engine, "context_tokenizer", None)
        self.tokenizer_path = os.getenv(f"{env_prefix}_TOKENIZER_PATH") or self.model
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
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.tokenizer_urls = list(dict.fromkeys([root + "/tokenize", self.base_url + "/tokenize"]))
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
            entries = models.get("data", [])
            exact = [entry for entry in entries if entry.get("id") == self.model]
            # A single served model may advertise an alias instead of its path.
            candidates = exact or (entries if len(entries) == 1 else [])
            for model in candidates:
                if model:
                    limit = model.get("max_model_len")
                    if limit and int(limit) > 0:
                        return int(limit)
        except (requests.RequestException, ValueError, TypeError, AttributeError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            print(f"[Context budget] Model metadata query failed: {type(error).__name__}"
                  f"{f' HTTP {status}' if status else ''}; model={self.model}")
        return None

    @lru_cache(maxsize=32)
    def count(self, text):
        if self._remote_tokenizer:
            for url in list(self.tokenizer_urls):
                try:
                    data = self._json_request("POST", url, json={
                        "model": self.model, "prompt": text, "add_special_tokens": False,
                    })
                    count = int(data["count"])
                    if count < 0 or (text and count == 0):
                        raise ValueError("Invalid token count")
                    self.tokenizer_urls = [url]
                    return count
                except (requests.RequestException, KeyError, ValueError, TypeError):
                    continue
            self._remote_tokenizer = False
            self.count.cache_clear()
            print("[Context budget] Tokenizer endpoints unavailable; trying the local model tokenizer.")
        if self.local_tokenizer is None:
            self.local_tokenizer = _load_local_tokenizer(self.tokenizer_path)
        if self.local_tokenizer is not None:
            self.tokenizer_mode = "local-tokenizer"
            return len(self.local_tokenizer.encode(text, add_special_tokens=False))
        if self.tokenizer_mode != "utf8-upper-bound":
            print("[Context budget] No local tokenizer available; UTF-8 bytes are an "
                  "upper-bound estimate only. Agent prompts will not be rejected solely on this estimate.")
        self.tokenizer_mode = "utf8-upper-bound"
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
