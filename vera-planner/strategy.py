"""Bot B — vera-planner: "plan, draft three ways, judge, pick".

1. One Groq call reads ALL candidate merchant signals and writes an explicit
   plan (which signal, which hook, which lever, what Vera will do) plus three
   drafts using different compulsion levers (proof / urgency / curiosity).
2. Every draft goes through the grounding validator; survivors are scored by
   the deterministic 5-dimension rubric.
3. A second Groq call acts as a critic on the survivors (the same five
   dimensions the judge uses) and picks the winner; ties -> rubric.
   Any failure -> the deterministic version of the same process: every
   (lever x top-signal) template variant, rubric-scored, best one wins.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from vera.engine import BaseStrategy
from vera.facts import Facts
from vera.kinds import Draft, KindPlan, all_variants, rank_anchors
from vera.llm import LLMUnavailable, llm
from vera.prompts import RULES, audience_line, facts_block, voice_line
from vera.replies import ReplyDecision
from vera.util import clean_body
from vera.validate import rubric, validate

CRITIC_SYSTEM = """You are a strict reviewer scoring WhatsApp messages that magicpin's assistant Vera will send.
Score each candidate 0-10 on: specificity (verifiable facts from the context), category_fit (voice for this business type),
merchant_fit (personal to THIS merchant/customer), decision_quality (the best signal for why-now), engagement (would they reply; one low-effort CTA).
Penalise any fact not present in FACTS. Return JSON: {"scores": [{"index": 0, "total": <0-50>, "reason": "..."}], "best_index": <int>}"""


class PlannerStrategy(BaseStrategy):
    name = "vera-planner"
    approach = ("Planner + multi-draft + critic: Groq gpt-oss-120b first writes an explicit plan over all candidate "
                "merchant signals and three lever-diverse drafts (proof / urgency / curiosity); a grounding validator "
                "filters them, a deterministic 5-dimension rubric scores them, and a second Groq critic pass picks the "
                "winner. Fully deterministic fallback: rubric-selected best of all template variants.")

    def _pick(self, F: Facts, drafts: list[Draft], scope: str) -> Draft:
        scored = []
        for i, d in enumerate(drafts):
            if validate(F, d.body, d.cta, scope=scope, reference=drafts[0].body):
                continue
            r = rubric(F, d.body, d.cta, scope)
            scored.append((r["total"], -i, d))
        if not scored:
            return drafts[0]
        scored.sort(key=lambda x: (-x[0], -x[1]))
        best = scored[0][2]
        best.rationale += f"; picked by rubric ({scored[0][0]}/50) from {len(drafts)} variants"
        return best

    def fallback(self, F: Facts, plan: KindPlan) -> Draft:
        scope = "customer" if F.c else "merchant"
        return self._pick(F, all_variants(F, plan), scope)

    async def compose(self, F: Facts, plan: KindPlan, reference: Draft, deadline: float) -> Draft:
        scope = "customer" if F.c else "merchant"
        ranked = rank_anchors(F, plan)
        candidates = [{"id": a.id, "signal": a.text, "prior_score": s} for s, a in ranked[:5]] if plan.use_anchor else []
        system = (
            "You are Vera's planning brain at magicpin. First PLAN, then WRITE three different WhatsApp drafts.\n"
            + RULES + "\n" + voice_line(F) + "\n" + audience_line(F)
            + '\nReturn JSON: {"plan": {"signal_id": "<id or null>", "why_now": "...", "hook": "...", '
              '"vera_will": "...", "cta": "..."}, "drafts": [{"lever": "proof", "body": "..."}, '
              '{"lever": "urgency", "body": "..."}, {"lever": "curiosity", "body": "..."}]}'
        )
        brief = {
            "trigger_hooks_by_lever": plan.hooks,
            "candidate_merchant_signals": candidates,
            "extra_lines_you_may_use": [l for l in plan.lines if l],
            "vera_offers_to": " ".join(plan.proposal) if scope == "merchant" else None,
            "cta_type": plan.cta_type,
            "required_cta": plan.customer_cta if scope == "customer" else None,
            "reference_draft": reference.body,
        }
        user = ("FACTS (the only source of truth):\n" + facts_block(F, []) + "\n\nBRIEF:\n" + json.dumps(brief, ensure_ascii=False)
                + "\n\nPlan: choose the ONE merchant signal that makes the trigger most compelling for this merchant "
                  "(or none if the trigger alone is stronger). Then write the three drafts; each must stand alone, "
                  "use a different lever, keep numbers exactly as in FACTS and end with the single CTA.")
        try:
            res = await llm.json_chat(system, user, deadline=deadline, max_tokens=700, cache_tag="plan-msg")
        except LLMUnavailable:
            return reference
        plan_json = res.get("plan") if isinstance(res.get("plan"), dict) else {}
        drafts: list[Draft] = []
        for item in res.get("drafts") or []:
            if not isinstance(item, dict):
                continue
            body = clean_body(str(item.get("body") or "")).strip('"')
            lever = str(item.get("lever") or "proof")
            if not body or validate(F, body, plan.cta_type, scope=scope, reference=reference.body):
                continue
            drafts.append(Draft(
                body=body, cta=plan.cta_type, send_as=reference.send_as, template_name=reference.template_name,
                template_params=[reference.template_params[0], body.split(". ")[0][:160], body.split(". ")[-1][:160]],
                rationale=f"plan: {plan_json.get('why_now') or plan.why}; signal={plan_json.get('signal_id')}; lever={lever}",
                lever=lever, anchor_id=plan_json.get("signal_id"), proposal=reference.proposal,
                artifact=reference.artifact, source="groq:planner",
            ))
        if not drafts:
            return reference
        drafts.append(reference)  # the rubric-picked template competes too
        # deterministic rubric first
        scored = sorted(((rubric(F, d.body, d.cta, scope)["total"], i, d) for i, d in enumerate(drafts)),
                        key=lambda x: (-x[0], x[1]))
        winner = scored[0][2]
        winner_score = scored[0][0]
        if len(drafts) >= 2 and deadline - time.monotonic() > 2.0:
            listing = [{"index": i, "body": d.body} for i, d in enumerate(drafts)]
            try:
                crit = await llm.json_chat(
                    CRITIC_SYSTEM,
                    "FACTS:\n" + facts_block(F, []) + "\n\nCANDIDATES:\n" + json.dumps(listing, ensure_ascii=False),
                    deadline=deadline, max_tokens=250, cache_tag="plan-critic")
                idx = crit.get("best_index")
                if isinstance(idx, int) and 0 <= idx < len(drafts):
                    rub = {i: s for s, i, _ in scored}
                    # accept the critic unless it picks something the rubric rates clearly worse
                    if rub.get(idx, 0) >= winner_score - 2:
                        winner = drafts[idx]
                        winner.rationale += "; critic-selected"
            except LLMUnavailable:
                pass
        winner.rationale += f"; {len(drafts)} valid drafts"
        return winner

    async def polish_reply(self, F: Facts, conv: dict, decision: ReplyDecision, msg: str, deadline: float) -> Optional[str]:
        scope = "customer" if F.c else "merchant"
        system = (
            "You are Vera's planning brain at magicpin, handling a live WhatsApp reply. First decide in one line what the "
            "merchant needs, then write TWO candidate replies (one concise, one with slightly more substance) that keep the "
            "draft's intent, facts and single CTA.\n" + RULES + "\n" + voice_line(F)
            + ("\nThe merchant has ALREADY agreed: no qualifying questions; confirm the action and show the work."
               if decision.commit else "")
            + '\nReturn JSON: {"need": "...", "replies": ["...", "..."]}'
        )
        history = [{"from": t.get("from"), "body": t.get("body")} for t in (conv.get("turns") or [])][-6:]
        user = ("FACTS:\n" + facts_block(F) + "\n\nCONVERSATION SO FAR:\n" + json.dumps(history, ensure_ascii=False)
                + f"\n\nMERCHANT'S LATEST MESSAGE: {msg}\nDETECTED INTENT: {decision.intent}\nDRAFT REPLY:\n{decision.body}")
        try:
            res = await llm.json_chat(system, user, deadline=deadline, max_tokens=450, cache_tag="plan-reply")
        except LLMUnavailable:
            return None
        options = [clean_body(str(r)).strip('"') for r in (res.get("replies") or []) if isinstance(r, str) and r.strip()]
        valid = [o for o in options if not [i for i in validate(F, o, decision.cta, scope=scope,
                                                                 reference=decision.body, intent_commit=decision.commit)
                                            if not i.startswith(("no_salutation", "no_customer_name"))]]
        if not valid:
            return None
        return max(valid, key=lambda o: (rubric(F, o, decision.cta, scope)["total"], -len(o)))
