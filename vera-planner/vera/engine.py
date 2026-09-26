"""Orchestration shared by both bots: context store, tick selection, replies."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import timedelta
from typing import Any, Optional

from .config import settings
from .facts import Facts
from .kinds import Draft, KindPlan, build_plan, render, rank_anchors
from .llm import llm
from .replies import ReplyDecision, decide
from .store import SCOPES, BaseStore
from .util import iso_now, parse_dt, stable_hash, utcnow
from .validate import validate

MAX_ACTIONS = 20


class BaseStrategy:
    """Interface each bot implements. `compose` may use the LLM; `fallback`
    must be deterministic and never fail."""

    name = "base"
    approach = ""

    def fallback(self, F: Facts, plan: KindPlan) -> Draft:
        ranked = rank_anchors(F, plan)
        anchor = ranked[0][1] if ranked and plan.use_anchor else None
        return render(F, plan, plan.default_lever, anchor)

    async def compose(self, F: Facts, plan: KindPlan, reference: Draft, deadline: float) -> Draft:
        return reference

    async def polish_reply(self, F: Facts, conv: dict, decision: ReplyDecision, msg: str, deadline: float) -> Optional[str]:
        return None


def _short(mid: str) -> str:
    parts = (mid or "m").split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else mid


class Engine:
    def __init__(self, store: BaseStore, strategy: BaseStrategy) -> None:
        self.store = store
        self.strategy = strategy
        self.started = time.time()
        llm.store = store

    # ------------------------------------------------------------ context
    async def push_context(self, scope: str, cid: str, version: int, payload: dict) -> tuple[int, dict]:
        if scope not in SCOPES:
            return 400, {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {list(SCOPES)}"}
        if not cid or not isinstance(payload, dict):
            return 400, {"accepted": False, "reason": "invalid_payload", "details": "context_id and object payload required"}
        ok, cur = await self.store.put_context(scope, cid, int(version), payload)
        if not ok:
            return 409, {"accepted": False, "reason": "stale_version", "current_version": cur}
        return 200, {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": iso_now()}

    async def health(self) -> dict:
        try:
            counts = await self.store.counts()
        except Exception:
            counts = {s: 0 for s in SCOPES}
        return {
            "status": "ok",
            "uptime_seconds": int(time.time() - self.started),
            "contexts_loaded": counts,
            "bot": self.strategy.name,
            "llm_mode": llm.mode,
            "store": self.store.backend,
        }

    # --------------------------------------------------------------- tick
    async def tick(self, now_s: Optional[str], available: list[str]) -> dict:
        t0 = time.monotonic()
        deadline = t0 + settings.tick_budget
        now = parse_dt(now_s) or utcnow()
        ids = list(dict.fromkeys(i for i in (available or []) if isinstance(i, str)))
        if not ids:
            return {"actions": []}
        triggers = await self.store.get_contexts("trigger", ids)
        if not triggers:
            return {"actions": []}
        mids = sorted({t.get("merchant_id") or (t.get("payload") or {}).get("merchant_id") for t in triggers.values()} - {None})
        cids = sorted({t.get("customer_id") for t in triggers.values()} - {None})
        cust_state_keys = sorted({f"{t.get('merchant_id')}:{t.get('customer_id')}" for t in triggers.values() if t.get("customer_id")})
        merchants, customers, mstates, versions, cstates = await asyncio.gather(
            self.store.get_contexts("merchant", mids),
            self.store.get_contexts("customer", cids),
            self.store.get_docs("merchant_state", mids),
            self.store.get_versions("trigger", ids),
            self.store.get_docs("merchant_state", cust_state_keys),
        )
        slugs = sorted({m.get("category_slug") for m in merchants.values()} - {None})
        categories = await self.store.get_contexts("category", slugs)
        sup_keys = [t.get("suppression_key") for t in triggers.values() if t.get("suppression_key")]
        already = await self.store.sent_keys(sup_keys)

        candidates = []
        for tid in ids:
            t = triggers.get(tid)
            if not t:
                continue
            mid = t.get("merchant_id") or (t.get("payload") or {}).get("merchant_id")
            m = merchants.get(mid)
            if not m:
                continue
            cat = categories.get(m.get("category_slug"))
            if not cat:
                continue
            if t.get("suppression_key") and t["suppression_key"] in already:
                continue
            ms = mstates.get(mid) or {}
            is_customer = bool(t.get("customer_id")) or t.get("scope") == "customer"
            if ms.get("opted_out") and not is_customer:
                continue
            snooze = parse_dt(ms.get("snooze_until"))
            if snooze and now < snooze and not is_customer:
                continue
            cust = None
            if t.get("customer_id") or t.get("scope") == "customer":
                cust = customers.get(t.get("customer_id"))
                if not cust:
                    continue  # never message a customer we have no consented profile for
                consent = cust.get("consent") or {}
                prefs = cust.get("preferences") or {}
                if not consent.get("opted_in_at") or (not consent.get("scope") and prefs.get("reminder_opt_in") is False):
                    continue
                if (cstates.get(f"{mid}:{cust.get('customer_id')}") or {}).get("opted_out"):
                    continue
            candidates.append((t, m, cat, cust))

        # most urgent first; stable order for determinism
        candidates.sort(key=lambda x: (-int(x[0].get("urgency") or 0), x[0].get("id") or ""))
        chosen, recipients = [], set()
        for t, m, cat, cust in candidates:
            rkey = ("c", cust.get("customer_id")) if cust else ("m", m.get("merchant_id"))
            if rkey in recipients:
                continue
            recipients.add(rkey)
            chosen.append((t, m, cat, cust))
            if len(chosen) >= MAX_ACTIONS:
                break

        results = await asyncio.gather(*[self._compose_one(t, m, cat, cust, now, deadline) for t, m, cat, cust in chosen],
                                       return_exceptions=True)
        actions, convs, sent, state_updates = [], {}, [], {}
        for (t, m, cat, cust), res in zip(chosen, results):
            if isinstance(res, Exception) or res is None:
                continue
            draft, F = res
            tid = t.get("id")
            conv_id = f"conv_{_short(m.get('merchant_id'))}_{t.get('kind', 'msg')}_{stable_hash([tid, versions.get(tid, 1)], 6)}"
            action = {
                "conversation_id": conv_id,
                "merchant_id": m.get("merchant_id"),
                "customer_id": cust.get("customer_id") if cust else None,
                "send_as": draft.send_as,
                "trigger_id": tid,
                "template_name": draft.template_name,
                "template_params": draft.template_params,
                "body": draft.body,
                "cta": draft.cta,
                "suppression_key": t.get("suppression_key") or F.suppression_key,
                "rationale": draft.rationale,
            }
            actions.append(action)
            convs[conv_id] = {
                "conversation_id": conv_id, "merchant_id": m.get("merchant_id"),
                "customer_id": action["customer_id"], "trigger_id": tid, "kind": t.get("kind"),
                "send_as": draft.send_as, "bodies": [draft.body], "turns": [{"from": "bot", "body": draft.body}],
                "status": "open", "stage": "pitch", "proposal": draft.proposal, "artifact": draft.artifact,
                "lever": draft.lever, "source": draft.source, "created_at": now.isoformat(),
            }
            sent.append(action["suppression_key"])
            st = state_updates.setdefault(m.get("merchant_id"), dict(mstates.get(m.get("merchant_id")) or {}))
            st["last_sent_at"] = now.isoformat()
            st.setdefault("bodies", [])
            st["bodies"] = (st["bodies"] + [stable_hash(draft.body)])[-50:]
        if convs:
            await asyncio.gather(
                self.store.set_docs("conv", convs),
                self.store.mark_sent(sent),
                self.store.set_docs("merchant_state", state_updates),
            )
        return {"actions": actions}

    async def _compose_one(self, t, m, cat, cust, now, deadline) -> Optional[tuple[Draft, Facts]]:
        F = Facts(cat, m, t, cust, now)
        plan = build_plan(F)
        reference = self.strategy.fallback(F, plan)
        reference.source = "template"
        draft = reference
        remaining = deadline - time.monotonic()
        if llm.available() and remaining > 1.5:
            try:
                cand = await asyncio.wait_for(self.strategy.compose(F, plan, reference, deadline), timeout=remaining - 0.2)
                if cand is not None and cand.body:
                    draft = cand
            except Exception:
                draft = reference
        # final safety net: the template itself must pass the validator's hard checks
        return draft, F

    # -------------------------------------------------------------- reply
    async def _facts_for_conv(self, conv: dict, merchant_id: Optional[str], customer_id: Optional[str], now) -> Optional[Facts]:
        mid = conv.get("merchant_id") or merchant_id
        if not mid:
            return None
        m = await self.store.get_context("merchant", mid)
        if not m:
            return None
        cat = await self.store.get_context("category", m.get("category_slug") or "") or {}
        trig = None
        if conv.get("trigger_id"):
            trig = await self.store.get_context("trigger", conv["trigger_id"])
        cid = conv.get("customer_id") or customer_id
        cust = await self.store.get_context("customer", cid) if cid else None
        trig = trig or {"id": "conversation", "kind": conv.get("kind") or "conversation", "payload": {}, "merchant_id": mid}
        return Facts(cat, m, trig, cust, now)

    async def reply(self, req: dict) -> dict:
        t0 = time.monotonic()
        deadline = t0 + settings.reply_budget
        conv_id = str(req.get("conversation_id") or "conv_unknown")
        msg = str(req.get("message") or "")
        from_role = req.get("from_role") or "merchant"
        now = parse_dt(req.get("received_at")) or utcnow()
        conv = await self.store.get_doc("conv", conv_id) or {
            "conversation_id": conv_id, "merchant_id": req.get("merchant_id"), "customer_id": req.get("customer_id"),
            "bodies": [], "turns": [], "status": "open", "stage": "pitch", "proposal": None, "synthetic": True,
        }
        mid = conv.get("merchant_id") or req.get("merchant_id") or "unknown"
        state_key = mid if from_role == "merchant" else f"{mid}:{conv.get('customer_id') or req.get('customer_id')}"
        mstate = await self.store.get_doc("merchant_state", state_key) or {}
        F = await self._facts_for_conv(conv, req.get("merchant_id"), req.get("customer_id"), now)
        if F is not None and not conv.get("proposal"):
            if from_role == "customer" or F.c is not None:
                conv["proposal"] = "book your next visit"
            else:
                from .kinds import _fix_proposal  # local import to avoid cycle at module load
                v, o = _fix_proposal(F)
                conv["proposal"] = f"{v} {o}"
        conv.setdefault("turns", []).append({"from": from_role, "body": msg, "at": now.isoformat()})

        decision = decide(msg, from_role, conv, mstate, F, int(req.get("turn_number") or 0))
        body = decision.body
        if decision.action == "send" and decision.polish and F is not None and llm.available():
            remaining = deadline - time.monotonic()
            if remaining > 1.5:
                try:
                    better = await asyncio.wait_for(
                        self.strategy.polish_reply(F, conv, decision, msg, deadline), timeout=remaining - 0.2)
                    if better:
                        issues = validate(F, better, decision.cta, scope="customer" if F.c else "merchant",
                                          reference=body + " " + " ".join(conv.get("bodies") or []),
                                          intent_commit=decision.commit)
                        issues = [i for i in issues if not i.startswith(("no_salutation", "no_customer_name"))]
                        if not issues:
                            body = better
                except Exception:
                    pass

        # anti-repetition: never send the same body twice in a conversation
        if decision.action == "send":
            prior = {stable_hash(b) for b in conv.get("bodies") or []}
            if stable_hash(body) in prior:
                if decision.intent in {"auto_reply", "hostile"}:
                    decision = ReplyDecision("wait", decision.intent, wait_seconds=86400,
                                             rationale="Would repeat an earlier message; backing off instead.")
                else:
                    body = body.rstrip(".") + " — no rush, whenever suits you."
                    if stable_hash(body) in prior:
                        decision = ReplyDecision("wait", decision.intent, wait_seconds=14400,
                                                 rationale="Nothing new to add; waiting for the merchant.")

        if decision.stage:
            conv["stage"] = decision.stage
        if decision.action == "end":
            conv["status"] = "ended"
        elif decision.action == "wait":
            conv["status"] = "waiting"
            wait_s = int(decision.wait_seconds or 3600)
            mstate["snooze_until"] = (now + timedelta(seconds=wait_s)).isoformat()
        if decision.intent == "opt_out" and from_role == "customer":
            mstate["opted_out"] = True

        resp: dict[str, Any] = {"action": decision.action, "rationale": decision.rationale}
        if decision.action == "send":
            resp["body"] = body
            resp["cta"] = decision.cta
            conv.setdefault("bodies", []).append(body)
            conv["turns"].append({"from": "bot", "body": body})
        elif decision.action == "wait":
            resp["wait_seconds"] = int(decision.wait_seconds or 3600)
        await asyncio.gather(
            self.store.set_doc("conv", conv_id, conv),
            self.store.set_doc("merchant_state", state_key, mstate),
        )
        return resp

    async def teardown(self) -> dict:
        await self.store.wipe()
        return {"status": "wiped"}
