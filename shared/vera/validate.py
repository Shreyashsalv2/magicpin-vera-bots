"""Grounding validator + heuristic rubric scorer.

`validate()` is the hard gate every LLM draft must pass (no invented numbers or
names, no URLs, no taboo words, one CTA). `rubric()` approximates the judge's
five dimensions so candidates can be compared deterministically.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .facts import Facts
from .util import extract_numbers, norm_text

URL_RE = re.compile(r"(https?://|www\.|\b[a-z0-9-]+\.(com|in|org|net|io|co)\b/?)", re.I)
PREAMBLE_RE = re.compile(r"\b(i hope (you|this)|hope you('| a)re doing|i am reaching out|i'm reaching out|my name is vera|this is vera, your)\b", re.I)
HYPE_RE = re.compile(r"(!!|\bamazing\b|\bunbelievable\b|\bhurry\b|\bact now\b|\blimited time only\b|\bbest in (the )?city\b)", re.I)
METRIC_WORDS_RE = re.compile(r"\b(calls?|views?|ctr|click[- ]through|peers?|median|dashboard|ranking|profile visits|google profile|unverified|verified|listing|yoy|searches)\b", re.I)
QUALIFYING_RE = re.compile(r"\b(would you|do you|can you tell|what if|how about)\b", re.I)

SAFE_CAPS = {
    "i", "yes", "no", "stop", "confirm", "reply", "google", "whatsapp", "gbp", "insta", "instagram", "swiggy",
    "zomato", "magicpin", "vera", "ctr", "cta", "sop", "otc", "mrp", "rx", "cde", "yoy", "ok", "ji", "dr",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december", "pm", "am", "namaste", "namaskar", "namaskaram", "vanakkam", "namaskara", "hi",
    "hello", "diwali", "holi", "ipl", "pdf", "sms", "post", "posts", "offer", "free", "bas", "aaj", "kal",
    "aapka", "aapke", "aapki", "hum", "haan", "reminder", "thank", "thanks", "current", "applied", "next",
    "suggested", "what", "fee", "format", "ages", "draft", "quick", "urgent", "heads", "new", "one", "your",
    "the", "a", "an", "and", "or", "to", "for", "hindi", "english", "india", "indian", "rvg", "iopa", "opg",
    "rct", "hiit", "pt", "sat,", "crm", "faq", "ai", "id", "ors", "gst", "ca", "fyi", "lagu", "offer:",
    "customer", "note", "audit", "checklist", "review", "sending", "done", "great", "sorry", "apologies",
    "noted", "understood", "got", "sure", "perfect", "here", "here's", "live", "step", "steps", "tier",
}


def _context_text(F: Facts) -> str:
    blob = json.dumps([F.category, F.merchant, F.trigger, F.customer or {}], ensure_ascii=False, default=str)
    return blob.lower()


def unknown_proper_nouns(F: Facts, body: str, extra_ok: str = "") -> list[str]:
    ctx = _context_text(F) + " " + extra_ok.lower()
    bad = []
    tokens = re.finditer(r"(^|[\s(\"'“‘])([A-Z][a-zA-Z'’.&-]{2,})", body)
    for m in tokens:
        word = m.group(2).strip(".'’&-")
        start = m.start(2)
        prev = body[:start].rstrip()
        sentence_start = (not prev) or not (prev[-1].isalnum() or prev[-1] in ",;&")
        lw = word.lower()
        if lw.startswith(("i'", "i’")):
            continue
        if lw in SAFE_CAPS or lw.rstrip("s") in SAFE_CAPS:
            continue
        if sentence_start:
            continue
        if lw in ctx or lw.rstrip("s") in ctx:
            continue
        bad.append(word)
    return bad


def count_ctas(body: str) -> int:
    b = body.lower()
    replies = len(re.findall(r"\breply\b", b))
    return replies


def validate(F: Facts, body: str, cta_type: str, *, scope: str = "merchant",
             reference: str = "", intent_commit: bool = False) -> list[str]:
    """Return a list of problems; empty list = OK to send."""
    issues: list[str] = []
    if not body or len(body.strip()) < 25:
        return ["empty_or_too_short"]
    if len(body) > 900:
        issues.append("too_long")
    if URL_RE.search(body):
        issues.append("contains_url")
    low = body.lower()
    for t in F.taboos:
        if t and t in low:
            issues.append(f"taboo:{t}")
    if PREAMBLE_RE.search(body):
        issues.append("preamble")
    if HYPE_RE.search(body) and F.slug in {"dentists", "pharmacies"}:
        issues.append("hype_tone")
    ref_nums = set(extract_numbers(reference))
    bank = F.nums
    if scope == "customer":
        # customer messages may only use customer/trigger/offer numbers — never merchant metrics
        from .facts import NumberBank
        bank = NumberBank()
        bank.absorb([F.customer or {}, F.tp, F.offers_active, F.biz_name, F.locality])
        if F.c and F.c.get("days_since") is not None:
            ds = F.c["days_since"]
            bank.add(ds, round(ds / 7), round(ds / 30.4))
        if METRIC_WORDS_RE.search(body) and not METRIC_WORDS_RE.search(reference):
            issues.append("merchant_metrics_leaked_to_customer")
    for n in extract_numbers(body):
        if n in ref_nums:
            continue
        if not bank.allows(n):
            issues.append(f"ungrounded_number:{n:g}")
    bad_names = unknown_proper_nouns(F, body, reference)
    if bad_names:
        issues.append("ungrounded_names:" + ",".join(sorted(set(bad_names))[:5]))
    replies = count_ctas(body)
    if cta_type == "multi_choice_slot":
        if replies > 2:
            issues.append("multiple_ctas")
    elif replies > 1:
        issues.append("multiple_ctas")
    if body.count("?") > max(1, reference.count("?")) if reference else body.count("?") > 2:
        issues.append("too_many_questions")
    if reference and cta_type != "open_ended" and "reply" in reference.lower() and "reply" not in low:
        issues.append("missing_reply_cta")
    if reference and scope == "merchant" and len(body) < 0.45 * len(reference) and "\n" in reference:
        issues.append("dropped_draft_content")
    if scope == "merchant":
        sal = F.salutation.lower().replace("dr. ", "")
        if sal and sal.split()[0] not in low and F.biz_name.lower() not in low:
            issues.append("no_salutation")
    elif F.c and F.c.get("first"):
        name = (F.c.get("parent") or F.c.get("first") or "").lower().split()[0]
        honor = (F.c.get("name") or "").lower().replace("mr. ", "").split()[:1]
        if name and name not in low and not (honor and honor[0] in low):
            issues.append("no_customer_name")
    if intent_commit and QUALIFYING_RE.search(body):
        issues.append("qualifying_after_commit")
    if intent_commit and not re.search(r"\b(done|sending|draft|here|confirm|proceed|next|live)\b", low):
        issues.append("commit_without_action_words")
    return issues


# ---------------------------------------------------------------------------
# heuristic rubric (0-10 per dimension, mirrors the judge's five dimensions)

LEVER_WORDS = {
    "effort": ["i'll", "i will", "draft", "ready", "5 minutes", "2 minutes", "few minutes", "just reply", "done for you", "kar doon"],
    "loss": ["missing", "down", "before", "lapsed", "gap", "losing", "costs you", "behind"],
    "curiosity": ["want to see", "spotted", "one thing", "worth a look", "?"],
    "social": ["peers", "similar", "median", "other "],
    "reassure": ["no commitment", "no judgment", "no auto-charge"],
}


def rubric(F: Facts, body: str, cta_type: str, scope: str = "merchant") -> dict:
    low = body.lower()
    nums = [n for n in extract_numbers(body) if n > 12 or "%" in body or "₹" in body]
    grounded = [n for n in extract_numbers(body) if F.nums.allows(n)]
    distinct = len(set(grounded))
    spec = {0: 2, 1: 5, 2: 7, 3: 8}.get(distinct, 9)
    d = F.digest_item or {}
    if d.get("source") and d["source"].lower().split(",")[0][:12] in low:
        spec = min(10, spec + 1)
    if "₹" in body:
        spec = min(10, spec + 1) if distinct >= 2 else spec

    cat = 8
    if any(t in low for t in F.taboos if t):
        cat = 2
    if HYPE_RE.search(body):
        cat -= 3
    if any(v.lower() in low for v in F.vocab):
        cat += 1
    if scope == "merchant" and F.slug == "dentists" and "dr." not in low:
        cat -= 2
    cat = max(0, min(10, cat))

    mf = 4
    if scope == "merchant":
        if F.salutation.lower() in low or (F.owner and F.owner.lower() in low):
            mf += 2
        if F.biz_name.lower() in low or (F.locality and F.locality.lower() in low):
            mf += 1
    elif F.c:
        if F.c.get("first") and F.c["first"].lower().split()[0] in low:
            mf += 2
        if F.biz_name.lower().split()[0] in low:
            mf += 1
        if F.c.get("mode") == "hinglish" and re.search(r"\b(aap|hai|hain|karein|ya|ke liye)\b", low):
            mf += 1
    anchor_hit = any(a.text[:25].lower() in low for a in F.anchors)
    merchant_nums = set()
    for k in ("views", "calls", "directions"):
        v = F.perf.get(k)
        if isinstance(v, (int, float)):
            merchant_nums.add(float(v))
    for v in F.agg.values():
        if isinstance(v, (int, float)):
            merchant_nums.add(float(v))
    if anchor_hit or merchant_nums & set(extract_numbers(body)):
        mf += 2
    if any(o.lower() in low for o in F.offers_active):
        mf += 1
    mf = min(10, mf)

    dq = 4
    trig_tokens = []
    for v in F.tp.values():
        if isinstance(v, str) and len(v) > 3 and "_" not in v:
            trig_tokens.append(v.lower())
        elif isinstance(v, list):
            trig_tokens += [str(x).lower() for x in v if isinstance(x, str)]
    if d.get("title"):
        trig_tokens += [w for w in norm_text(d["title"]).split() if len(w) > 5][:4]
    hits = sum(1 for t in trig_tokens if t and t in low)
    dq += min(4, hits * 2)
    if re.search(r"\b(today|tonight|this week|days (away|left|from)|just|new|now|due)\b", low):
        dq += 2
    dq = min(10, dq)

    eng = 3
    tail = body.strip()[-160:].lower()
    if cta_type == "open_ended" and body.strip().endswith(("?", "time.", "time")) or "reply" in tail or tail.endswith("?"):
        eng += 3
    if count_ctas(body) <= 1 or cta_type == "multi_choice_slot":
        eng += 1
    levers = sum(1 for words in LEVER_WORDS.values() if any(w in low for w in words))
    eng += min(3, levers)
    if len(body) > 650:
        eng -= 1
    eng = max(0, min(10, eng))

    total = spec + cat + mf + dq + eng
    return {"specificity": spec, "category_fit": cat, "merchant_fit": mf, "decision_quality": dq,
            "engagement": eng, "total": total}
