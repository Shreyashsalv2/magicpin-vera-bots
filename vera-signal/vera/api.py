"""FastAPI surface: the five judge endpoints (+ optional teardown)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .engine import BaseStrategy, Engine
from .llm import llm
from .store import build_store

log = logging.getLogger("vera")


def create_app(strategy: BaseStrategy, version: str = "1.0.0") -> FastAPI:
    app = FastAPI(title=strategy.name, version=version, docs_url=None, redoc_url=None)
    engine = Engine(build_store(), strategy)
    app.state.engine = engine

    async def _json(request: Request) -> Any:
        try:
            return await request.json()
        except Exception:
            return None

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed", "details": str(exc)[:300]})

    @app.get("/")
    async def root():
        return {"bot": strategy.name, "endpoints": ["/v1/healthz", "/v1/metadata", "/v1/context", "/v1/tick", "/v1/reply"]}

    @app.get("/v1/healthz")
    async def healthz():
        return await engine.health()

    @app.get("/v1/metadata")
    async def metadata():
        return {
            "team_name": settings.team_name or strategy.name,
            "team_members": settings.team_members or ["Solo entrant"],
            "model": f"groq/{settings.primary_model} (fallback: groq/{settings.secondary_model} -> deterministic templates)",
            "approach": strategy.approach,
            "contact_email": settings.contact_email,
            "version": version,
            "submitted_at": settings.submitted_at,
            "bot": strategy.name,
            "llm_mode": llm.mode,
        }

    @app.post("/v1/context")
    async def context(request: Request):
        body = await _json(request)
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed", "details": "JSON object expected"})
        try:
            version = int(body.get("version"))
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_version", "details": "version must be an integer"})
        status, payload = await engine.push_context(str(body.get("scope") or ""), str(body.get("context_id") or ""),
                                                    version, body.get("payload"))
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/tick")
    async def tick(request: Request):
        body = await _json(request) or {}
        try:
            return await engine.tick(body.get("now"), body.get("available_triggers") or [])
        except Exception:
            log.exception("tick failed")
            return {"actions": []}

    @app.post("/v1/reply")
    async def reply(request: Request):
        body = await _json(request) or {}
        try:
            return await engine.reply(body)
        except Exception:
            log.exception("reply failed")
            return {"action": "wait", "wait_seconds": 1800, "rationale": "Internal hiccup; backing off briefly instead of sending something wrong."}

    @app.post("/v1/teardown")
    async def teardown():
        return await engine.teardown()

    return app
