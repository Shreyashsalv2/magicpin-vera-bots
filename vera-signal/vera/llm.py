"""Groq chat client with the fallback machinery both bots rely on.

* temperature 0 + fixed seed + response cache  -> same input, same output
* per-call timeout and an overall deadline     -> never blows the 30s budget
* circuit breaker                              -> after repeated failures (bad key,
  rate limit, outage) stop calling for a while and let the template engine answer
* secondary model                              -> one retry on a smaller model
Any failure raises LLMUnavailable; callers catch it and use the deterministic path.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Optional

import httpx

from .config import settings
from .util import stable_hash


class LLMUnavailable(Exception):
    pass


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.open_until = 0.0
        self.last_error = ""

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def failure(self, reason: str, hard: bool = False) -> None:
        self.failures += 1
        self.last_error = reason
        if hard:
            # invalid key / permission: no point retrying soon
            self.open_until = time.monotonic() + 600.0
        elif self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown


class GroqClient:
    """Rotates across Groq models; each model has its own breaker because
    Groq rate limits (tokens/minute) are per model."""

    def __init__(self) -> None:
        self.breakers: dict[str, CircuitBreaker] = {}
        self.auth_breaker = CircuitBreaker(threshold=1)
        self._client: Optional[httpx.AsyncClient] = None
        self._sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        self._cache: dict[str, Any] = {}
        self.calls = 0
        self.fallbacks = 0
        self.store = None  # optional persistent cache (set by the app)

    @property
    def models(self) -> list[str]:
        return settings.models

    def _breaker(self, model: str) -> CircuitBreaker:
        if model not in self.breakers:
            self.breakers[model] = CircuitBreaker(threshold=2, cooldown=30.0)
        return self.breakers[model]

    def _ready_models(self) -> list[str]:
        return [m for m in self.models if not self._breaker(m).is_open]

    @property
    def mode(self) -> str:
        if not settings.llm_enabled:
            return "fallback (no GROQ_API_KEY)"
        if self.auth_breaker.is_open:
            return f"fallback ({self.auth_breaker.last_error})"
        ready = self._ready_models()
        if not ready:
            errs = sorted({self._breaker(m).last_error for m in self.models})
            return f"fallback (all models cooling down: {', '.join(errs)})"
        return f"groq ({len(ready)}/{len(self.models)} models ready)"

    def available(self) -> bool:
        return settings.llm_enabled and not self.auth_breaker.is_open and bool(self._ready_models())

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.groq_base_url,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=httpx.Timeout(settings.llm_call_timeout, connect=3.0),
            )
        return self._client

    async def _post(self, model: str, messages: list[dict], max_tokens: int, timeout: float) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if model.startswith("openai/gpt-oss"):
            # reasoning tokens count toward max_tokens; keep reasoning short
            body["reasoning_effort"] = "low"
            body["max_tokens"] = max_tokens + 350
        elif "qwen3" in model:
            body["reasoning_effort"] = "none"
        br = self._breaker(model)
        self.calls += 1
        resp = await asyncio.wait_for(self._http().post("/chat/completions", json=body), timeout=timeout)
        if resp.status_code in (401, 403):
            self.auth_breaker.cooldown = 600.0
            self.auth_breaker.failure(f"auth {resp.status_code}", hard=True)
            raise LLMUnavailable(f"auth {resp.status_code}")
        if resp.status_code in (400, 404) and "model" in resp.text.lower():
            br.failure(f"model unavailable {resp.status_code}", hard=True)
            raise LLMUnavailable(f"model {model} unavailable")
        if resp.status_code == 429:
            try:
                wait = float(resp.headers.get("retry-after") or 20)
            except ValueError:
                wait = 20.0
            br.failures = br.threshold
            br.last_error = "rate limited"
            br.open_until = time.monotonic() + min(max(wait, 5.0), 120.0)
            raise LLMUnavailable("rate limited")
        if resp.status_code >= 400:
            br.failure(f"http {resp.status_code}")
            raise LLMUnavailable(f"http {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def json_chat(self, system: str, user: str, *, deadline: float, max_tokens: int = 500,
                        cache_tag: str = "") -> dict:
        """Return parsed JSON from the first model that answers, or raise LLMUnavailable."""
        if not settings.llm_enabled:
            raise LLMUnavailable("no api key")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = cache_tag + ":" + stable_hash(messages, 24)
        if key in self._cache:
            return self._cache[key]
        if self.store is not None:
            try:
                hit = await self.store.get_doc("llm_cache", key)
                if hit is not None:
                    self._cache[key] = hit
                    return hit
            except Exception:
                pass
        if self.auth_breaker.is_open:
            raise LLMUnavailable(self.auth_breaker.last_error)

        last_err = "all models cooling down"
        for model in self._ready_models():
            remaining = deadline - time.monotonic()
            if remaining < 1.2:
                last_err = "deadline"
                break
            br = self._breaker(model)
            try:
                async with self._sem:
                    remaining = deadline - time.monotonic()
                    if remaining < 1.0:
                        last_err = "deadline"
                        break
                    raw = await self._post(model, messages, max_tokens, min(settings.llm_call_timeout, remaining - 0.2))
                parsed = _parse_json(raw)
                if parsed is None:
                    last_err = "bad json"
                    br.failure("bad json")
                    continue
                br.success()
                parsed["_model"] = model
                self._cache[key] = parsed
                if self.store is not None:
                    try:
                        await self.store.set_doc("llm_cache", key, parsed)
                    except Exception:
                        pass
                return parsed
            except LLMUnavailable as e:
                last_err = str(e)
                if "auth" in last_err:
                    break
            except (asyncio.TimeoutError, httpx.TimeoutException):
                last_err = "timeout"
                br.failure("timeout")
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_err = f"transport: {e.__class__.__name__}"
                br.failure(last_err)
        self.fallbacks += 1
        raise LLMUnavailable(last_err)


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                v = json.loads(m.group())
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                return None
    return None


llm = GroqClient()
