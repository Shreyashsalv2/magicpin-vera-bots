"""Prompt building blocks shared by both strategies (identical guardrails)."""

from __future__ import annotations

import json

from .facts import Facts

VOICE_GUIDE = {
    "dentists": "clinical peer-to-peer (a colleague, not a marketer); technical terms are fine; address the owner as 'Dr. <name>'; never promise cures or outcomes",
    "salons": "warm, practical, fellow-operator; light Hindi-English code-mix is natural",
    "restaurants": "busy operator-to-operator; use words like covers, footfall, AOV, delivery; light Hinglish is natural",
    "gyms": "coach-to-owner, energetic but disciplined; no body-shaming or guaranteed results",
    "pharmacies": "trustworthy, precise neighbourhood pharmacist; exact molecule/batch names; calm, never alarmist",
}

RULES = """HARD RULES (a message that breaks any of these is discarded):
1. Use ONLY facts, numbers, names, dates, prices and sources present in FACTS. Never invent a statistic, competitor, price, date, study, or offer. If a number is not in FACTS, do not write it.
2. Exactly ONE call-to-action, placed in the last sentence. Prefer a single binary ask ("Reply YES") or one open question. Never offer multiple options unless it is a booking with named slots.
3. No URLs or links. No hashtags. No "I hope you're doing well" or self-introductions. Get to the point in the first line.
4. Lead with WHY NOW (the trigger), then one merchant-specific fact, then the low-effort next step Vera will do for them.
5. Prefer service+price ("Dental Cleaning @ ₹299") over generic discounts. Never use the category's taboo words.
6. Keep it WhatsApp-short: 2-5 sentences (bullets allowed only for a drafted plan).
7. Write your own wording — do not copy example text verbatim.
8. Never change the meaning of a fact (e.g. "free for IDA members" is a fee rule, not a free slot).
9. Customer-facing messages must NEVER mention the business's own metrics (calls, views, CTR, reviews, peers, rankings, Google profile).
10. Search trends and peer benchmarks are market-level — never present them as this business's own data.
11. Vera drafts and sets things up online; she does not physically act in the shop. Keep the proposal exactly as given.
"""


def voice_line(F: Facts) -> str:
    base = VOICE_GUIDE.get(F.slug, "respectful, specific, peer tone")
    lang = ""
    if F.c:
        mode = F.c.get("mode")
        if mode == "hinglish":
            lang = "Write in natural Hindi-English code-mix (Roman script), like 'Aapke liye 2 slots ready hain'."
        elif mode == "hindi":
            lang = "Write mostly in simple Hindi (Roman script) with a respectful 'Namaste' / 'ji' register."
        elif F.c.get("greet"):
            lang = f"Write in English; you may open with '{F.c['greet']}'."
        else:
            lang = "Write in English."
    elif F.hinglish and F.slug in {"salons", "restaurants", "pharmacies"}:
        lang = "Merchant speaks Hindi + English: English with a light Hinglish touch (one short phrase) is ideal."
    else:
        lang = "Write in English (the merchant is comfortable in English)."
    return f"VOICE: {base}. {lang}"


def facts_block(F: Facts, anchors: list | None = None) -> str:
    """Compact, fact-only context (keeps each call well inside Groq's tokens/minute).
    Customer-facing prompts never see the merchant's business metrics."""
    tp = {k: v for k, v in (F.tp or {}).items() if k not in {"placeholder", "metric_or_topic"}}
    data: dict = {
        "business": {"name": F.biz_name, "locality": F.locality, "city": F.city, "category": F.slug,
                     "active_offers": F.offers_active},
        "trigger": {"kind": F.kind, **({"payload": tp} if tp else {})},
    }
    if F.digest_item:
        d = F.digest_item
        data["digest_item"] = {k: d.get(k) for k in ("title", "source", "summary", "actionable", "trial_n",
                                                     "patient_segment", "date", "credits") if d.get(k)}
    if F.c:
        c = F.c
        data["customer"] = {k: v for k, v in {
            "name": c.get("first"), "parent": c.get("parent"), "language": c.get("lang"), "state": c.get("state"),
            "days_since_last_visit": c.get("days_since"), "visits": c.get("visits"),
            "services": (c.get("services") or [])[:4], "preferred_slots": c.get("slot_pref"),
            "senior": c.get("senior") or None,
        }.items() if v not in (None, "", [])}
    else:
        data["business"]["salutation"] = F.salutation
        chosen = anchors if anchors is not None else sorted(F.anchors, key=lambda a: -a.strength)[:4]
        data["merchant_signals"] = [{"id": a.id, "fact": a.text} for a in chosen]
        beats = F.current_beats() if F.kind in {"festival_upcoming", "category_seasonal", "seasonal_perf_dip", "curious_ask_due"} else []
        if beats:
            data["season_now"] = f"{beats[0].get('month_range')}: {beats[0].get('note')}"
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def audience_line(F: Facts) -> str:
    if F.c:
        who = F.c.get("parent") or F.c.get("first") or "the customer"
        return (f"AUDIENCE: a customer of {F.biz_name} ({who}); the message is sent FROM the business "
                f"(send_as=merchant_on_behalf). Sign as the business, not as Vera.")
    return f"AUDIENCE: the owner of {F.biz_name}; address them as '{F.salutation}'. You are Vera, magicpin's merchant assistant."
