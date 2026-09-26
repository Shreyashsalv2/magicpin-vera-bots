"""State storage.

The judge requires the bot to remember every context push and every conversation
for the whole test. On a single long-lived process memory is enough; on
serverless (Vercel) several instances may serve requests, so when Upstash Redis
credentials are present every read/write goes to Redis. Both backends expose the
same async interface.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import httpx

from .config import settings

SCOPES = ("category", "merchant", "customer", "trigger")


class BaseStore:
    backend = "base"

    # contexts -------------------------------------------------------------
    async def put_context(self, scope: str, cid: str, version: int, payload: dict) -> tuple[bool, int]:
        raise NotImplementedError

    async def get_context(self, scope: str, cid: str) -> Optional[dict]:
        res = await self.get_contexts(scope, [cid])
        return res.get(cid)

    async def get_contexts(self, scope: str, cids: list[str]) -> dict[str, dict]:
        raise NotImplementedError

    async def get_versions(self, scope: str, cids: list[str]) -> dict[str, int]:
        raise NotImplementedError

    async def counts(self) -> dict[str, int]:
        raise NotImplementedError

    async def list_ids(self, scope: str) -> list[str]:
        raise NotImplementedError

    # generic json docs (conversations, merchant state, llm cache) ----------
    async def get_doc(self, kind: str, key: str) -> Optional[dict]:
        res = await self.get_docs(kind, [key])
        return res.get(key)

    async def get_docs(self, kind: str, keys: list[str]) -> dict[str, dict]:
        raise NotImplementedError

    async def set_docs(self, kind: str, docs: dict[str, dict]) -> None:
        raise NotImplementedError

    async def set_doc(self, kind: str, key: str, doc: dict) -> None:
        await self.set_docs(kind, {key: doc})

    # suppression -----------------------------------------------------------
    async def sent_keys(self, keys: list[str]) -> set[str]:
        raise NotImplementedError

    async def mark_sent(self, keys: list[str]) -> None:
        raise NotImplementedError

    async def wipe(self) -> None:
        raise NotImplementedError


class MemoryStore(BaseStore):
    backend = "memory"

    def __init__(self) -> None:
        self._ctx: dict[str, dict[str, tuple[int, dict]]] = {s: {} for s in SCOPES}
        self._docs: dict[str, dict[str, dict]] = {}
        self._sent: set[str] = set()
        self._lock = asyncio.Lock()

    async def put_context(self, scope, cid, version, payload):
        async with self._lock:
            cur = self._ctx[scope].get(cid)
            if cur and cur[0] >= version:
                return False, cur[0]
            self._ctx[scope][cid] = (version, payload)
            return True, version

    async def get_contexts(self, scope, cids):
        bucket = self._ctx.get(scope, {})
        return {c: bucket[c][1] for c in cids if c in bucket}

    async def get_versions(self, scope, cids):
        bucket = self._ctx.get(scope, {})
        return {c: bucket[c][0] for c in cids if c in bucket}

    async def counts(self):
        return {s: len(self._ctx[s]) for s in SCOPES}

    async def list_ids(self, scope):
        return sorted(self._ctx.get(scope, {}).keys())

    async def get_docs(self, kind, keys):
        bucket = self._docs.get(kind, {})
        return {k: json.loads(json.dumps(bucket[k])) for k in keys if k in bucket}

    async def set_docs(self, kind, docs):
        bucket = self._docs.setdefault(kind, {})
        for k, v in docs.items():
            bucket[k] = json.loads(json.dumps(v, default=str))

    async def sent_keys(self, keys):
        return {k for k in keys if k in self._sent}

    async def mark_sent(self, keys):
        self._sent.update(k for k in keys if k)

    async def wipe(self):
        async with self._lock:
            self._ctx = {s: {} for s in SCOPES}
            self._docs = {}
            self._sent = set()


_CAS_SCRIPT = """
local cur = redis.call('HGET', KEYS[1], ARGV[1])
if cur and tonumber(cur) >= tonumber(ARGV[2]) then
  return {0, tonumber(cur)}
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], ARGV[3])
return {1, tonumber(ARGV[2])}
"""


class UpstashStore(BaseStore):
    """Upstash Redis over its REST API (works from any serverless runtime)."""

    backend = "upstash-redis"

    def __init__(self, url: str, token: str, namespace: str) -> None:
        self.url = url.rstrip("/")
        self.ns = namespace
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, timeout=httpx.Timeout(4.0, connect=2.0)
        )

    def k(self, *parts: str) -> str:
        return ":".join((self.ns,) + parts)

    async def _pipeline(self, cmds: list[list[Any]]) -> list[Any]:
        if not cmds:
            return []
        resp = await self._client.post(f"{self.url}/pipeline", json=[[str(a) for a in c] for c in cmds])
        resp.raise_for_status()
        out = []
        for item in resp.json():
            if isinstance(item, dict) and item.get("error"):
                raise RuntimeError(item["error"])
            out.append(item.get("result") if isinstance(item, dict) else item)
        return out

    async def put_context(self, scope, cid, version, payload):
        res = await self._pipeline([[
            "EVAL", _CAS_SCRIPT, 2, self.k("ver", scope), self.k("ctx", scope),
            cid, int(version), json.dumps(payload, ensure_ascii=False),
        ]])
        ok, cur = res[0]
        return bool(int(ok)), int(cur)

    async def get_contexts(self, scope, cids):
        if not cids:
            return {}
        res = await self._pipeline([["HMGET", self.k("ctx", scope), *cids]])
        vals = res[0] or []
        return {c: json.loads(v) for c, v in zip(cids, vals) if v}

    async def get_versions(self, scope, cids):
        if not cids:
            return {}
        res = await self._pipeline([["HMGET", self.k("ver", scope), *cids]])
        return {c: int(v) for c, v in zip(cids, res[0] or []) if v}

    async def counts(self):
        res = await self._pipeline([["HLEN", self.k("ver", s)] for s in SCOPES])
        return {s: int(n or 0) for s, n in zip(SCOPES, res)}

    async def list_ids(self, scope):
        res = await self._pipeline([["HKEYS", self.k("ver", scope)]])
        return sorted(res[0] or [])

    async def get_docs(self, kind, keys):
        if not keys:
            return {}
        res = await self._pipeline([["HMGET", self.k("doc", kind), *keys]])
        return {k: json.loads(v) for k, v in zip(keys, res[0] or []) if v}

    async def set_docs(self, kind, docs):
        if not docs:
            return
        flat: list[Any] = []
        for k, v in docs.items():
            flat += [k, json.dumps(v, ensure_ascii=False, default=str)]
        await self._pipeline([["HSET", self.k("doc", kind), *flat]])

    async def sent_keys(self, keys):
        if not keys:
            return set()
        res = await self._pipeline([["SMISMEMBER", self.k("sent"), *keys]])
        return {k for k, flag in zip(keys, res[0] or []) if int(flag)}

    async def mark_sent(self, keys):
        keys = [k for k in keys if k]
        if keys:
            await self._pipeline([["SADD", self.k("sent"), *keys]])

    async def wipe(self):
        keys = [self.k("ver", s) for s in SCOPES] + [self.k("ctx", s) for s in SCOPES] + [self.k("sent")]
        kinds = ["conv", "merchant_state", "llm_cache", "sim"]
        keys += [self.k("doc", kind) for kind in kinds]
        await self._pipeline([["DEL", *keys]])


class ResilientStore(BaseStore):
    """Redis first; if Redis errors, degrade to the in-process memory store so
    endpoints keep answering (the judge disqualifies on 3 failed healthz)."""

    def __init__(self, primary: BaseStore, fallback: MemoryStore) -> None:
        self.primary = primary
        self.fallback = fallback
        self.backend = primary.backend
        self._down_until = 0.0

    async def _call(self, name: str, *args):
        if time.monotonic() >= self._down_until:
            try:
                res = await getattr(self.primary, name)(*args)
                # mirror writes into memory so a later outage still has data
                if name in {"put_context", "set_docs", "mark_sent"}:
                    await getattr(self.fallback, name)(*args)
                return res
            except Exception:
                self._down_until = time.monotonic() + 15.0
        return await getattr(self.fallback, name)(*args)

    async def put_context(self, *a):
        return await self._call("put_context", *a)

    async def get_contexts(self, *a):
        return await self._call("get_contexts", *a)

    async def get_versions(self, *a):
        return await self._call("get_versions", *a)

    async def counts(self):
        return await self._call("counts")

    async def list_ids(self, *a):
        return await self._call("list_ids", *a)

    async def get_docs(self, *a):
        return await self._call("get_docs", *a)

    async def set_docs(self, *a):
        return await self._call("set_docs", *a)

    async def sent_keys(self, *a):
        return await self._call("sent_keys", *a)

    async def mark_sent(self, *a):
        return await self._call("mark_sent", *a)

    async def wipe(self):
        await self.fallback.wipe()
        try:
            await self.primary.wipe()
        except Exception:
            pass


def build_store() -> BaseStore:
    if settings.redis_url and settings.redis_token:
        return ResilientStore(UpstashStore(settings.redis_url, settings.redis_token, settings.namespace), MemoryStore())
    return MemoryStore()
