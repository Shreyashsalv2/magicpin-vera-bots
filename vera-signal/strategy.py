"""Bot A — vera-signal: "rank the signals, then write once".

1. A deterministic scorer ranks every merchant anchor against the trigger
   (strength x relevance) and picks the single strongest one.
2. One Groq call writes the message from ONLY that trigger + anchor + the
   concrete proposal, using the template draft as a grounded reference.
3. The grounding validator gates the output; one repair retry is allowed.
   Any failure -> the ranked template draft is sent unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from vera.engine import BaseStrategy
from vera.facts import Facts
from vera.kinds import Draft, KindPlan, rank_anchors, render
from vera.llm import LLMUnavailable, llm
from vera.prompts import RULES, audience_line, facts_block, voice_line
from vera.replies import ReplyDecision
from vera.util import clean_body
from vera.validate import rubric, validate

STYLE_EXAMPLES = """STYLE EXAMPLES (shape only — different businesses, do not reuse wording):
- "Anita, 'keratin treatment price' searches in Baner are up 18% YoY and your profile doesn't mention keratin yet. Your 4.6★ rating is above most salons nearby. Want me to add a Keratin @ ₹2,499 line to your profile today? Reply YES."
- "Dr. Rajan, your calls fell 30% this week while views held steady — people find you but don't call. Your last Google post was 26 days ago. Want me to draft 3 posts for you to approve? Reply YES."
"""


class SignalStrategy(BaseStrategy):
    name = "vera-signal"
    approach = ("Signal ranker + one-shot writer: a deterministic scorer ranks trigger-relevant merchant "
                "signals and picks the strongest; one Groq gpt-oss-120b call (temperature 0, seeded, cached) "
                "writes a category-voiced message from only that signal; a grounding validator rejects any "
                "invented number/name, extra CTA, URL or taboo word, with a ranked-template fallback.")

    def fallback(self, F: Facts, plan: KindPlan) -> Draft:
        ranked = rank_anchors(F, plan)
        anchor = ranked[0][1] if ranked and plan.use_anchor else None
        return render(F, plan, plan.default_lever, anchor)

    async def compose(self, F: Facts, plan: KindPlan, reference: Draft, deadline: float) -> Draft:
        ranked = rank_anchors(F, plan)
        anchor = ranked[0][1] if ranked and plan.use_anchor and not F.c else None
        scope = "customer" if F.c else "merchant"
        system = (
            "You are Vera, magicpin's WhatsApp assistant for Indian merchants. Write ONE outbound WhatsApp message.\n"
            + RULES + "\n" + voice_line(F) + "\n" + audience_line(F) + "\n" + STYLE_EXAMPLES
            + '\nReturn JSON: {"body": "<the message>", "rationale": "<1 sentence: why this signal, why now>"}'
        )
        brief = {
            "trigger_why_now": plan.hooks.get(plan.default_lever),
            "chosen_merchant_signal": anchor.text if anchor else None,
            "lever": plan.default_lever,
            "extra_lines_you_may_use": [l for l in plan.lines if l],
            "vera_offers_to": " ".join(plan.proposal) if scope == "merchant" else None,
            "cta_type": plan.cta_type,
            "required_cta": plan.customer_cta if scope == "customer" else None,
            "reference_draft": reference.body,
        }
        user = ("FACTS (the only source of truth):\n" + facts_block(F, [anchor] if anchor else []) + "\n\nBRIEF:\n"
                + json.dumps(brief, ensure_ascii=False)
                + "\n\nImprove on the reference draft: sharper first line, same facts, same single CTA. "
                  "Use the chosen merchant signal (it was ranked most relevant). Keep every number exactly as in FACTS.")
        try:
            res = await llm.json_chat(system, user, deadline=deadline, max_tokens=350, cache_tag="sig-msg")
        except LLMUnavailable:
            return reference
        body = clean_body(str(res.get("body") or "")).strip('"')
        issues = validate(F, body, plan.cta_type, scope=scope, reference=reference.body)
        if issues and deadline - time.monotonic() > 2.0:
            fix_user = user + "\n\nYOUR PREVIOUS DRAFT:\n" + body + "\nPROBLEMS: " + "; ".join(issues) + "\nFix every problem."
            try:
                res = await llm.json_chat(system, fix_user, deadline=deadline, max_tokens=350, cache_tag="sig-fix")
                body = clean_body(str(res.get("body") or "")).strip('"')
                issues = validate(F, body, plan.cta_type, scope=scope, reference=reference.body)
            except LLMUnavailable:
                return reference
        if issues:
            return reference
        # the rewrite must be at least as strong as the grounded template on the rubric
        if rubric(F, body, plan.cta_type, scope)["total"] < rubric(F, reference.body, plan.cta_type, scope)["total"] - 1:
            return reference
        rationale = str(res.get("rationale") or "").strip() or reference.rationale
        return Draft(
            body=body, cta=plan.cta_type, send_as=reference.send_as, template_name=reference.template_name,
            template_params=[reference.template_params[0], body.split(". ")[0][:160], body.split(". ")[-1][:160]],
            rationale=f"{rationale} [signal={anchor.id if anchor else 'trigger-only'}; lever={plan.default_lever}]",
            lever=plan.default_lever, anchor_id=anchor.id if anchor else None, proposal=reference.proposal,
            artifact=reference.artifact, source="groq:signal",
        )

    async def polish_reply(self, F: Facts, conv: dict, decision: ReplyDecision, msg: str, deadline: float) -> Optional[str]:
        system = (
            "You are Vera, magicpin's WhatsApp assistant. Rewrite the DRAFT REPLY so it answers the merchant's latest message "
            "directly and naturally, keeping the same intent, the same facts and the same single call-to-action.\n"
            + RULES + "\n" + voice_line(F)
            + ("\nThe merchant has ALREADY agreed: do not ask any qualifying question; confirm the action and show the work."
               if decision.commit else "")
            + '\nReturn JSON: {"body": "<reply>"}'
        )
        history = [{"from": t.get("from"), "body": t.get("body")} for t in (conv.get("turns") or [])][-6:]
        user = ("FACTS:\n" + facts_block(F) + "\n\nCONVERSATION SO FAR:\n" + json.dumps(history, ensure_ascii=False)
                + f"\n\nMERCHANT'S LATEST MESSAGE: {msg}\nDETECTED INTENT: {decision.intent}\nDRAFT REPLY:\n{decision.body}")
        try:
            res = await llm.json_chat(system, user, deadline=deadline, max_tokens=300, cache_tag="sig-reply")
        except LLMUnavailable:
            return None
        return clean_body(str(res.get("body") or "")).strip('"') or None
