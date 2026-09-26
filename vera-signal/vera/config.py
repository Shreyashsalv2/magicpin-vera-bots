"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # identity (served on /v1/metadata)
    team_name: str = field(default_factory=lambda: _env("TEAM_NAME"))
    team_members: list[str] = field(
        default_factory=lambda: [m.strip() for m in _env("TEAM_MEMBERS").split(",") if m.strip()]
    )
    contact_email: str = field(default_factory=lambda: _env("CONTACT_EMAIL"))
    submitted_at: str = field(default_factory=lambda: _env("SUBMITTED_AT", "2026-09-26T00:00:00Z"))

    # LLM (Groq, OpenAI-compatible)
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_base_url: str = field(default_factory=lambda: _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    primary_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b"))
    secondary_model: str = field(default_factory=lambda: _env("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"))
    llm_disabled: bool = field(default_factory=lambda: _env("LLM_DISABLED", "0") in {"1", "true", "yes"})
    llm_call_timeout: float = field(default_factory=lambda: _env_float("LLM_CALL_TIMEOUT", 6.0))
    llm_max_concurrency: int = field(default_factory=lambda: int(_env_float("LLM_MAX_CONCURRENCY", 6)))

    # time budgets (judge budget is 10s for tick/reply per the API examples, 30s hard cap)
    tick_budget: float = field(default_factory=lambda: _env_float("TICK_BUDGET_SECONDS", 8.0))
    reply_budget: float = field(default_factory=lambda: _env_float("REPLY_BUDGET_SECONDS", 7.5))

    # storage
    namespace: str = field(default_factory=lambda: _env("VERA_NAMESPACE", "vera"))
    redis_url: str = field(
        default_factory=lambda: _env("KV_REST_API_URL") or _env("UPSTASH_REDIS_REST_URL")
    )
    redis_token: str = field(
        default_factory=lambda: _env("KV_REST_API_TOKEN") or _env("UPSTASH_REDIS_REST_TOKEN")
    )

    models_csv: str = field(default_factory=lambda: _env("GROQ_MODELS"))

    @property
    def models(self) -> list[str]:
        """Rotation order. Groq rate limits are per model, so spreading load across
        models multiplies the usable tokens/minute."""
        if self.models_csv:
            return [m.strip() for m in self.models_csv.split(",") if m.strip()]
        out = [self.primary_model, "qwen/qwen3.8-27b", self.secondary_model]
        return list(dict.fromkeys(out))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key) and not self.llm_disabled


settings = Settings()
