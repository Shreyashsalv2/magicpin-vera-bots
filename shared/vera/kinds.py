"""Deterministic, grounded message plans for every trigger kind.

A KindPlan holds the trigger-specific pieces (hooks per lever, body lines,
the concrete thing Vera offers to do, CTA type). `render()` assembles a plan +
a merchant anchor + a lever into a final Draft. The same library is the
no-LLM fallback for both bots and the "reference draft" the LLM improves on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .facts import Anchor, Facts
from .util import (
    clean_body,
    ctr_pct,
    days_between,
    fmt_day,
    fmt_day_full,
    fmt_int,
    fmt_money,
    humanize,
    parse_dt,
    pct,
    stable_hash,
)

LEVERS = ("proof", "urgency", "curiosity")

# which merchant anchors make each trigger more personal (ranked preference)
ANCHOR_PREFS: dict[str, list[str]] = {
    "research_digest": ["clinical_cohort", "retention", "content", "visibility_gap"],
    "regulation_change": ["clinical_cohort", "base"],
    "cde_opportunity": ["clinical_cohort", "strength", "review_pos"],
    "supply_alert": ["clinical_cohort", "base"],
    "category_seasonal": ["base", "momentum_up", "offer", "offer_gap"],
    "perf_dip": ["visibility_gap", "offer_gap", "content", "review_neg", "subscription"],
    "perf_spike": ["offer", "offer_gap", "content", "strength"],
    "milestone_reached": ["review_pos", "strength", "base"],
    "dormant_with_vera": ["momentum_down", "visibility_gap", "offer_gap", "retention", "subscription"],
    "review_theme_emerged": ["review_pos", "visibility_gap", "base"],
    "competitor_opened": ["review_pos", "offer", "offer_gap", "strength", "visibility_gap"],
    "festival_upcoming": ["offer", "offer_gap", "base", "strength"],
    "ipl_match_today": ["base", "offer", "review_neg"],
    "active_planning_intent": ["history", "base", "offer"],
    "seasonal_perf_dip": ["base", "retention", "strength"],
    "renewal_due": ["momentum_down", "visibility_gap", "strength", "base"],
    "winback_eligible": ["retention", "visibility_gap", "momentum_down"],
    "gbp_unverified": ["visibility_gap", "base", "offer_gap"],
    "curious_ask_due": ["strength", "review_pos", "base"],
    "weather_heatwave": ["offer", "offer_gap", "base"],
    "local_news_event": ["offer", "base"],
    "category_trend_movement": ["offer_gap", "offer", "content"],
}
DEFAULT_PREFS = ["visibility_gap", "momentum_down", "offer_gap", "review_neg", "retention", "content", "strength", "base"]


@dataclass
class KindPlan:
    kind: str
    scope: str  # "merchant" | "customer"
    hooks: dict[str, str]
    lines: list[str] = field(default_factory=list)
    proposal: tuple[str, str] = ("draft", "a quick next step")
    cta_type: str = "binary_yes_no"
    default_lever: str = "proof"
    use_anchor: bool = True
    anchor_prefs: list[str] = field(default_factory=list)
    why: str = ""
    # customer-facing plans carry their full CTA text (booking flows differ)
    customer_cta: Optional[str] = None
    signoff: Optional[str] = None
    artifact: Optional[str] = None  # deliverable text used when merchant says yes
    exclude_anchors: list[str] = field(default_factory=list)  # anchor id prefixes already covered by the hook


@dataclass
class Draft:
    body: str
    cta: str
    send_as: str
    template_name: str
    template_params: list[str]
    rationale: str
    lever: str
    anchor_id: Optional[str]
    proposal: str
    artifact: Optional[str] = None
    source: str = "template"

    def as_dict(self) -> dict:
        return {
            "body": self.body, "cta": self.cta, "send_as": self.send_as,
            "template_name": self.template_name, "template_params": self.template_params,
            "rationale": self.rationale, "lever": self.lever, "anchor_id": self.anchor_id,
            "proposal": self.proposal, "source": self.source,
        }


# ---------------------------------------------------------------------------
# helpers


def _sentence(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if not s.lower().startswith("magicpin"):
        s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + "."


_ABBREV = re.compile(r"(\b(?:Dr|Mr|Mrs|Ms|No|p|pp|vs|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|[A-Z])\.|\d\.)$")


def _sentences(text: str) -> list[str]:
    out, buf = [], ""
    for chunk in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        buf = f"{buf} {chunk}".strip() if buf else chunk
        if _ABBREV.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _first_sentence(text: str) -> str:
    parts = _sentences(text)
    return parts[0] if parts else ""


def _payload_text(tp: dict) -> str:
    bits = []
    for k, v in tp.items():
        if k in {"placeholder", "metric_or_topic", "category", "merchant_id", "customer_id"} or v in (None, "", [], {}):
            continue
        if isinstance(v, float) and -1 < v < 1:
            v = pct(v)
        elif isinstance(v, list):
            v = ", ".join(humanize(x) if not isinstance(x, dict) else humanize(x.get("label") or x.get("title") or "") for x in v[:4])
        elif isinstance(v, dict):
            continue
        bits.append(f"{humanize(k)}: {humanize(v) if isinstance(v, str) and '_' in v else v}")
    return "; ".join(bits[:4])


def _metric_word(metric: str) -> str:
    return {"calls": "calls", "views": "profile views", "ctr": "CTR", "directions": "direction requests",
            "leads": "leads", "review_count": "reviews"}.get(metric, humanize(metric))


def _worst_delta(F: Facts) -> Optional[tuple[str, float]]:
    best = None
    for k, v in F.d7.items():
        if isinstance(v, (int, float)) and v < 0 and (best is None or v < best[1]):
            best = (k.replace("_pct", ""), v)
    return best


def _best_delta(F: Facts) -> Optional[tuple[str, float]]:
    best = None
    for k, v in F.d7.items():
        if isinstance(v, (int, float)) and v > 0 and (best is None or v > best[1]):
            best = (k.replace("_pct", ""), v)
    return best


def _fix_proposal(F: Facts) -> tuple[str, str]:
    """Most useful concrete action given the merchant's gaps."""
    if not F.offers_active:
        title, _ = F.offer_or_catalog()
        if title:
            return ("set up", f"a \"{title}\" offer on your profile")
    if isinstance(F.signals.get("stale_posts"), int) or F.signals.get("no_recent_post"):
        return ("draft", "3 fresh Google posts for you to approve")
    if F.verified is False:
        return ("start", "your Google verification (a phone call is the fastest route)")
    if F.offers_active:
        return ("draft", f"a Google post pushing your \"{F.offers_active[0]}\" offer")
    return ("draft", "a 3-step plan for this week")


def _find_digest(F: Facts, *needles: str) -> Optional[dict]:
    for d in F.digest:
        hay = f"{d.get('id', '')} {d.get('title', '')}".lower()
        if all(n in hay for n in needles):
            return d
    return None


# ---------------------------------------------------------------------------
# merchant-facing plans


def plan_research(F: Facts) -> KindPlan:
    d = F.digest_item or (F.digest[0] if F.digest else None)
    if not d:
        return plan_generic(F)
    src = d.get("source") or "this week's digest"
    title = d.get("title", "")
    seg = humanize(d.get("patient_segment")) if d.get("patient_segment") else ""
    n = d.get("trial_n")
    trial = f" ({fmt_int(n)}-patient trial)" if n else ""
    first = _first_sentence(d.get("summary", ""))
    cohort = F.anchor("high_risk")
    seg_line = f"your {seg} {F.people}" if seg else f"your {F.people}"
    if F.slug == "dentists":
        proposal = ("pull", "the abstract and draft a patient-ed WhatsApp you can share")
    else:
        proposal = ("draft", f"a short {F.words['person']}-facing WhatsApp note on this")
    who = seg or F.people
    artifact = (f"Abstract ({src}): {d.get('summary', title)}\n\n"
                f"{F.words['person'].title()} WhatsApp draft: \"New research ({src}) on {title[0].lower() + title[1:]}. "
                f"If this sounds like you, ask us at your next visit whether it applies — just reply here to book a quick check.\"")
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{src} has one worth 2 minutes for {seg_line}: {title}{trial}.",
            "urgency": f"new this week in {src} — {title}{trial}.",
            "curiosity": f"one item from {src} could change how you plan care for {seg_line}.",
        },
        lines=[_sentence(first)] + ([f"Suggested move: {d['actionable']}."] if d.get("actionable") and not cohort else []),
        proposal=proposal, default_lever="proof",
        anchor_prefs=ANCHOR_PREFS["research_digest"],
        why=f"research digest item {d.get('id')} ({src})",
        signoff=f"— {src}" if F.slug == "dentists" else None,
        artifact=artifact,
    )


def plan_compliance(F: Facts) -> KindPlan:
    d = F.digest_item or next((x for x in F.digest if x.get("kind") == "compliance"), None)
    if not d:
        return plan_generic(F)
    deadline = parse_dt(F.tp.get("deadline_iso")) or parse_dt(d.get("effective_date"))
    if not deadline:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", d.get("title", ""))
        deadline = parse_dt(m.group(1)) if m else None
    left = days_between(F.now, deadline) if deadline else None
    if left is not None and left > 0:
        F.nums.add(left)
    when = f"effective {fmt_day(deadline)} {deadline.year}" if deadline else ""
    left_txt = f" — {left} days from today" if left and left > 0 else ""
    first = _first_sentence(d.get("summary", ""))
    rest = d.get("summary", "")[len(first):].strip()
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "urgency": f"compliance heads-up: {d.get('title')} ({d.get('source')}){left_txt}.",
            "proof": f"{d.get('source')} — {d.get('title')}{left_txt}.",
            "curiosity": f"one compliance change {when} will need a quick check of your setup{left_txt}.",
        },
        lines=[_sentence(first), _sentence(rest) if rest else "", f"What to do: {d['actionable']}." if d.get("actionable") else ""],
        proposal=("draft", "a 1-page audit checklist your team can run this week"),
        default_lever="urgency", anchor_prefs=ANCHOR_PREFS["regulation_change"], use_anchor=False,
        why=f"regulation change {d.get('id')} with deadline {deadline.date() if deadline else 'n/a'}",
        artifact="Audit checklist: 1) list every unit/process affected, 2) note current spec vs new limit, 3) fix or replace non-compliant items, 4) record the change in your SOP file with date + signature.",
    )


def plan_cde(F: Facts) -> KindPlan:
    d = F.digest_item or next((x for x in F.digest if x.get("kind") == "cde"), None)
    if not d:
        return plan_generic(F)
    dt = parse_dt(d.get("date"))
    when = ""
    if dt:
        hour = dt.hour
        tod = f", {hour % 12 or 12}{'pm' if hour >= 12 else 'am'}" if hour else ""
        when = f"{fmt_day_full(dt)}{tod}"
    credits = F.tp.get("credits") or d.get("credits")
    cred = f", {credits} CDE credits" if credits else ""
    fee = d.get("actionable") if d.get("actionable") and "₹" in d.get("actionable", "") else humanize(F.tp.get("fee") or "")
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{d.get('title')} ({d.get('source')}) — {when}{cred}.",
            "urgency": f"registration is open for {d.get('title')}, {when}{cred}.",
            "curiosity": f"a session on {d.get('title').split(':')[-1].strip()} is coming up {when}{cred}.",
        },
        lines=[_sentence(d.get("summary", "")) if len(d.get("summary", "")) <= 220 else _sentence(_first_sentence(d.get("summary", ""))),
               _sentence(f"Fee: {fee}") if fee else ""],
        proposal=("block", "the slot in your calendar and share the registration details"),
        default_lever="proof", anchor_prefs=ANCHOR_PREFS["cde_opportunity"], use_anchor=False,
        why=f"CDE opportunity {d.get('id')} on {when}",
    )


def plan_supply_alert(F: Facts) -> KindPlan:
    d = F.digest_item
    mol = F.tp.get("molecule") or (d.get("title") if d else "an item you stock")
    batches = F.tp.get("affected_batches") or []
    mfr = F.tp.get("manufacturer")
    bt = f" ({', '.join(batches)})" if batches else ""
    by = f" by {mfr}" if mfr else ""
    summary = d.get("summary", "") if d else ""
    risk = ""
    if summary:
        sents = re.split(r"(?<=[.!?])\s+", summary)
        risk = " ".join(s for s in sents if "risk" in s.lower() or "potency" in s.lower())[:220]
    cohort = F.anchor("chronic_rx")
    who = f"your {fmt_int(F.agg.get('chronic_rx_count'))} chronic-Rx customers" if cohort else "your repeat-Rx customers"
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "urgency": f"urgent — voluntary recall on {len(batches) or 'some'} {mol} batches{bt}{by}.",
            "proof": f"{(d or {}).get('source', 'CDSCO')} alert: {mol} batches{bt}{by} recalled.",
            "curiosity": f"a {mol} recall{by} may touch some of {who}.",
        },
        lines=[_sentence(risk) if risk else "", f"Some of {who} may have been dispensed these batches."],
        proposal=("pull", f"the list of customers who got these batches in the last 90 days and draft their replacement note"),
        default_lever="urgency", use_anchor=False, anchor_prefs=ANCHOR_PREFS["supply_alert"],
        why=f"supply alert on {mol} batches {', '.join(batches)}",
        artifact=f"Customer note: \"Namaste, {F.biz_name} here. A batch of {mol} you received is part of a voluntary recall (sub-potency, not a safety issue). Please bring or keep the strip aside — we'll replace it free. Reply YES and we'll deliver the replacement.\"",
    )


def plan_category_seasonal(F: Facts) -> KindPlan:
    trends = F.tp.get("trends") or []
    parsed = []
    for t in trends:
        m = re.match(r"(.+?)_demand_([+-]?\d+)", str(t))
        if m:
            name = humanize(m.group(1)).replace("cold cough", "cold & cough")
            name = name if name.isupper() else name
            parsed.append(f"{name} {int(m.group(2)):+d}%")
    d = F.digest_item
    season = humanize(F.tp.get("season") or "")
    shift = ", ".join(parsed[:4])
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"the {season or 'seasonal'} demand shift is here: {shift}." if shift else f"{d.get('title') if d else 'Seasonal demand is shifting'}.",
            "urgency": f"shelves need a {season or 'seasonal'} reset this week — {shift}." if shift else "your shelves need a seasonal reset this week.",
            "curiosity": f"your {season or 'seasonal'} bestsellers are about to change — {shift}." if shift else "your seasonal bestsellers are about to change.",
        },
        lines=[f"Suggested: {d['actionable']}." if d and d.get("actionable") else ""],
        proposal=("draft", "a counter-display list and a Google post for this week"),
        default_lever="proof", anchor_prefs=ANCHOR_PREFS["category_seasonal"],
        why=f"category seasonal shift ({season})",
    )


def plan_perf_dip(F: Facts) -> KindPlan:
    metric = F.tp.get("metric")
    delta = F.tp.get("delta_pct")
    if not isinstance(delta, (int, float)):
        wd = _worst_delta(F)
        if wd:
            metric, delta = wd
    window = F.tp.get("window") or "7d"
    base = F.tp.get("vs_baseline")
    base_txt = f" (your usual is ~{base})" if base else ""
    if isinstance(delta, (int, float)):
        mw = _metric_word(metric or "calls")
        head = f"your {mw} are down {pct(delta)} over the last {window.replace('7d', '7 days')}{base_txt}"
    else:
        head = "your listing has slowed down this week"
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "urgency": f"{head} — worth fixing before it compounds.",
            "proof": f"{head}.",
            "curiosity": f"{head}, and I can see a likely reason.",
        },
        lines=[],
        proposal=_fix_proposal(F), default_lever="urgency",
        anchor_prefs=ANCHOR_PREFS["perf_dip"], why=f"performance dip in {metric or 'engagement'}",
        exclude_anchors=[f"d7_{metric}"] if metric else [],
    )


def plan_perf_spike(F: Facts) -> KindPlan:
    metric = F.tp.get("metric")
    delta = F.tp.get("delta_pct")
    if not isinstance(delta, (int, float)):
        bd = _best_delta(F)
        if bd:
            metric, delta = bd
    driver = humanize(F.tp.get("likely_driver") or "")
    base = F.tp.get("vs_baseline")
    base_txt = f" (vs your usual ~{base})" if base else ""
    mw = _metric_word(metric or "views")
    head = f"your {mw} are up {pct(delta)} this week{base_txt}" if isinstance(delta, (int, float)) else "your profile is picking up this week"
    drv = f" — looks driven by your {driver}" if driver else ""
    if driver:
        proposal = ("draft", f"a follow-up to the {driver} while interest is warm")
    else:
        proposal = _fix_proposal(F)
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{head}{drv}.",
            "urgency": f"{head}{drv} — a good window to follow up.",
            "curiosity": f"{head}{drv}. Want to keep it going?",
        },
        proposal=proposal, default_lever="proof", anchor_prefs=ANCHOR_PREFS["perf_spike"],
        why=f"performance spike in {metric or 'engagement'}", exclude_anchors=[f"d7_{metric}"] if metric else [],
    )


def plan_milestone(F: Facts) -> KindPlan:
    metric = F.tp.get("metric")
    now_v, goal = F.tp.get("value_now"), F.tp.get("milestone_value")
    if isinstance(now_v, (int, float)) and isinstance(goal, (int, float)):
        gap = int(goal - now_v)
        F.nums.add(gap)
        mw = _metric_word(metric or "reviews")
        if gap > 0:
            head = f"you're at {fmt_int(now_v)} {mw} — just {gap} away from {fmt_int(goal)}"
        else:
            head = f"you just crossed {fmt_int(goal)} {mw}"
        proposal = ("draft", "a 2-line review request you can send to this week's happy customers") if "review" in (metric or "") else ("draft", "a thank-you post to mark it")
    else:
        v = F.perf.get("views")
        head = f"{fmt_int(v)} people viewed your profile in the last 30 days" if v else "your profile hit a new high this month"
        proposal = ("draft", "a thank-you post and a review request to lock in the momentum")
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={"proof": f"{head}.", "urgency": f"{head} — easy win this week.", "curiosity": f"{head}. Want to get there faster?"},
        proposal=proposal, default_lever="proof", anchor_prefs=ANCHOR_PREFS["milestone_reached"],
        why=f"milestone {metric or 'profile'}",
        artifact=f"Review request: \"Thank you for visiting {F.biz_name}! If we made your day a little better, a 1-line Google review would mean a lot to our team 🙏\"",
    )


def plan_dormant(F: Facts) -> KindPlan:
    days = F.tp.get("days_since_last_merchant_message")
    topic = humanize(F.tp.get("last_topic") or "")
    head = f"it's been {days} days since we last spoke" if days else "it's been a while since we caught up"
    topic_txt = f" (last time it was about {topic})" if topic else ""
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "curiosity": f"{head}{topic_txt} — I spotted something on your profile.",
            "proof": f"{head}{topic_txt}. A quick update from your dashboard:",
            "urgency": f"{head}{topic_txt}, and one number needs attention.",
        },
        proposal=_fix_proposal(F), default_lever="curiosity", anchor_prefs=ANCHOR_PREFS["dormant_with_vera"],
        why=f"dormant merchant re-engagement ({days or 'n/a'} days)",
    )


def plan_review_theme(F: Facts) -> KindPlan:
    theme = F.tp.get("theme")
    occ = F.tp.get("occurrences_30d")
    quote = F.tp.get("common_quote")
    trend = F.tp.get("trend")
    if not theme:
        neg = next((r for r in F.reviews if r.get("sentiment") == "neg"), None)
        if neg:
            theme, occ, quote = neg.get("theme"), neg.get("occurrences_30d"), neg.get("common_quote")
    if not theme:
        return plan_generic(F)
    th = humanize(theme)
    trend_txt = f", and it's {trend}" if trend else ""
    q = f" One reads: \"{quote}\"." if quote else ""
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{occ} reviews in the last 30 days mention {th}{trend_txt}.{q}" if occ else f"a pattern in your reviews: {th}{trend_txt}.{q}",
            "urgency": f"{th} is showing up in your reviews ({occ} in 30 days{trend_txt}) — new customers read these first.{q}" if occ else f"{th} keeps showing up in your reviews.{q}",
            "curiosity": f"your reviews are telling you something — {occ} mention {th} this month.{q}" if occ else f"your reviews are telling you something about {th}.{q}",
        },
        proposal=("draft", "polite public replies to those reviews plus a fix note you can pin"),
        default_lever="proof", anchor_prefs=ANCHOR_PREFS["review_theme_emerged"],
        why=f"review theme '{theme}' emerging", exclude_anchors=[f"review_neg_{theme}"],
        artifact=f"Reply draft: \"Thank you for the honest feedback — you're right about the {th}. We've made changes this week and would love to host you again. — {F.salutation}\"",
    )


def plan_competitor(F: Facts) -> KindPlan:
    name = F.tp.get("competitor_name")
    dist = F.tp.get("distance_km")
    offer = F.tp.get("their_offer")
    opened = parse_dt(F.tp.get("opened_date"))
    where = f" {dist} km from you" if dist else (f" near {F.locality}" if F.locality else " nearby")
    when = f" on {fmt_day(opened)}" if opened else ""
    who = name or f"a new {F.words['biz']}"
    offer_txt = f", leading with \"{offer}\"" if offer else ""
    yours = f" Your \"{F.offers_active[0]}\" is live" if F.offers_active else ""
    line = ""
    if F.offers_active and offer:
        line = f"{yours.strip()} — worth positioning on what they can't copy, not on price."
    elif not F.offers_active:
        line = "You don't have an offer live right now, so their listing will look sharper in search."
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "curiosity": f"{who} opened{where}{when}{offer_txt}.",
            "proof": f"new listing alert — {who} opened{where}{when}{offer_txt}.",
            "urgency": f"{who} opened{where}{when}{offer_txt}, and they're competing for the same searches.",
        },
        lines=[line],
        proposal=("draft", "a counter-post that leads with your strongest reviews"),
        default_lever="curiosity", anchor_prefs=ANCHOR_PREFS["competitor_opened"],
        why=f"competitor opened ({name or 'unnamed'})", exclude_anchors=["offer_live", "offer_gap"],
    )


def plan_festival(F: Facts) -> KindPlan:
    fest = F.tp.get("festival")
    date = parse_dt(F.tp.get("date"))
    days = F.tp.get("days_until") if isinstance(F.tp.get("days_until"), int) else (days_between(F.now, date) if date else None)
    if isinstance(days, int) and days > 0:
        F.nums.add(days)
    from .facts import month_in_range
    beat = None
    if date:
        beat = next((b for b in F.seasonal_beats if month_in_range(b.get("month_range", ""), date.month)), None)
    if beat is None:
        beats = F.current_beats(4)
        beat = next((b for b in beats if "festiv" in (b.get("note") or "").lower() or "wedding" in (b.get("note") or "").lower()), beats[0] if beats else None)
    offer, live = F.offer_or_catalog()
    if fest:
        when = f"{fest} is {days} days away ({fmt_day(date)})" if isinstance(days, int) and days > 0 and date else f"{fest} is coming up"
    else:
        when = f"peak season is coming ({beat.get('month_range')})" if beat else "the festive window is coming"
    beat_line = f"For {F.words['label']}, {beat['month_range']} is {beat['note']}." if beat else ""
    pkg = f"a {fest or 'festive'} package built around your \"{offer}\"" if offer and live else f"a {fest or 'festive'} package (a \"{offer}\" style offer works well)" if offer else f"a {fest or 'festive'} campaign"
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{when}.",
            "urgency": f"{when} — the early bookings go to whoever posts first.",
            "curiosity": f"{when}. Have you planned your offer yet?",
        },
        lines=[beat_line],
        proposal=("draft", pkg), default_lever="proof", anchor_prefs=ANCHOR_PREFS["festival_upcoming"],
        why=f"festival upcoming ({fest or 'seasonal'})", exclude_anchors=["offer_live", "offer_gap"],
    )


def plan_ipl(F: Facts) -> KindPlan:
    match, venue = F.tp.get("match"), F.tp.get("venue")
    t = parse_dt(F.tp.get("match_time_iso"))
    time_txt = ""
    if t:
        m = re.search(r"T(\d{2}):(\d{2})", F.tp.get("match_time_iso", ""))
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            time_txt = f"{h % 12 or 12}{':' + str(mi).zfill(2) if mi else ''}{'pm' if h >= 12 else 'am'}"
    weeknight = F.tp.get("is_weeknight")
    d = _find_digest(F, "ipl")
    offer, live = F.offer_or_catalog(["pizza", "combo", "match"])
    head = f"{match} at {venue} tonight{', ' + time_txt if time_txt else ''}"
    lines = []
    if d and weeknight is False:
        sat = next((s for s in re.split(r"(?<=[.!?])\s+", d.get("summary", "")) if "saturday" in s.lower()), "")
        lines.append(_sentence(f"Heads-up from {d.get('source')}: {sat}" if sat else d.get("title")))
        lines.append(f"So skip a dine-in match promo tonight and push your \"{offer}\" as a delivery special instead." if offer and live else "So lean on delivery tonight rather than a dine-in promo.")
        proposal = ("draft", "a delivery banner and an Insta story for tonight")
    else:
        wk = next((s for s in re.split(r"(?<=[.!?])\s+", (d or {}).get("summary", "")) if "weeknight" in s.lower()), "")
        if wk:
            lines.append(_sentence(f"{d.get('source')}: {wk}"))
        lines.append(f"Your \"{offer}\" is the obvious match-night hook." if offer and live else "A match-night combo is the obvious hook.")
        proposal = ("draft", "a match-night post and WhatsApp blast for 6pm")
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{head}.",
            "urgency": f"{head} — you have a few hours to set this up.",
            "curiosity": f"{head}. One thing most restaurants get wrong on match days:",
        },
        lines=lines, proposal=proposal, default_lever="urgency", anchor_prefs=ANCHOR_PREFS["ipl_match_today"],
        use_anchor=False, why=f"IPL match today ({match}, weeknight={weeknight})",
    )


def _price_from_offer(title: str) -> Optional[int]:
    m = re.search(r"₹\s?([\d,]+)", title or "")
    return int(m.group(1).replace(",", "")) if m else None


def plan_planning(F: Facts) -> KindPlan:
    topic = humanize(F.tp.get("intent_topic") or "the plan")
    last_msg = F.tp.get("merchant_last_message")
    t = topic.lower()
    lines: list[str] = []
    if any(w in t for w in ("thali", "corporate", "bulk", "catering")):
        base_offer = next((o for o in F.offers_active if _price_from_offer(o)), None)
        base = _price_from_offer(base_offer) if base_offer else None
        if base:
            tiers = [(10, int(round(base * 0.9 / 5) * 5)), (25, int(round(base * 0.85 / 5) * 5)), (50, int(round(base * 0.8 / 5) * 5))]
            for q, p in tiers:
                F.nums.add(q, p, base - p)
            lines.append(f"{F.biz_name} Corporate Thali — for offices around {F.locality}:")
            lines.append(f"• {tiers[0][0]}+ thalis/day @ {fmt_money(tiers[0][1])} each (your regular is {fmt_money(base)})")
            lines.append(f"• {tiers[1][0]}+ @ {fmt_money(tiers[1][1])} each + free delivery")
            lines.append(f"• {tiers[2][0]}+ @ {fmt_money(tiers[2][1])} each + monthly billing")
            lines.append("• Order on WhatsApp by 5pm the day before; delivered for lunch")
        else:
            lines.append(f"Draft {topic}: 3 volume tiers, next-day ordering on WhatsApp by 5pm, and delivery in a fixed lunch window.")
        proposal = ("draft", f"a 3-line WhatsApp pitch for office admins near {F.locality}")
    elif any(w in t for w in ("kid", "yoga", "camp", "program", "class", "batch")):
        hist = " ".join(h.get("body", "") for h in F.history if h.get("from") == "vera")
        weeks = re.search(r"(\d+)-week", hist)
        per = re.search(r"(\d+) classes/week", hist)
        age = re.search(r"age (\d+)-(\d+)", hist)
        price = re.search(r"₹\s?([\d,]+)", hist)
        name = topic.title()
        lines.append(f"{name} — draft outline:")
        lines.append(f"• Format: {weeks.group(1) + '-week program' if weeks else '4-week program'}, {per.group(1) + ' classes/week' if per else '3 classes/week'}")
        if age:
            lines.append(f"• Ages {age.group(1)}-{age.group(2)}, small batches so every child gets attention")
        if price:
            lines.append(f"• Fee: ₹{price.group(1)} for the full program (sibling discount optional)")
        lines.append("• Weekend morning slots so parents can drop in")
        proposal = ("draft", "the GBP post and a parent-facing WhatsApp invite")
    else:
        offer, _ = F.offer_or_catalog()
        lines.append(f"Draft plan for {topic}: 1) define the package, 2) price it against your \"{offer}\" anchor, 3) launch with a Google post + WhatsApp broadcast." if offer else f"Draft plan for {topic}: 1) define the package, 2) set a clear price, 3) launch with a Google post + WhatsApp broadcast.")
        proposal = ("draft", "the launch post for you to approve")
    echo = f" (you said: \"{last_msg}\")" if last_msg else ""
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"here's a first cut of the {topic}{echo} — edit anything:",
            "urgency": f"picking up your {topic} idea — here's a ready draft:",
            "curiosity": f"I sketched the {topic} you asked about:",
        },
        lines=lines, proposal=proposal, default_lever="proof", use_anchor=False,
        anchor_prefs=ANCHOR_PREFS["active_planning_intent"], why=f"merchant planning intent: {topic}",
    )


def plan_seasonal_dip(F: Facts) -> KindPlan:
    metric = F.tp.get("metric") or "views"
    delta = F.tp.get("delta_pct")
    head = f"your {_metric_word(metric)} are down {pct(delta)} this week" if isinstance(delta, (int, float)) else "your numbers dipped this week"
    raw_note = str(F.tp.get("season_note") or "")
    months = re.findall(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", raw_note.lower())
    window = f"{months[0].title()}-{months[-1].title()}" if len(months) >= 2 else ""
    label = humanize(re.sub(r"_?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(_|$)", " ", raw_note.lower())).strip()
    label = label.replace("post resolution", "post-resolution")
    beat = None
    if window:
        beat = next((b for b in F.seasonal_beats if (b.get("month_range") or "").lower() == window.lower()), None)
    if beat is None and not raw_note:
        beat = next((b for b in F.current_beats(0)), None)
    if beat:
        reframe = f"that's the normal {beat['month_range']} pattern for {F.words['label']} ({beat['note']}), not a problem with your listing"
    else:
        what = " ".join(x for x in (label, f"({window})" if window else "") if x)
        reframe = f"that's the expected seasonal lull{(' — ' + what) if what else ''}, not a problem with your listing"
    d = next((x for x in F.digest if x.get("kind") == "seasonal"), None)
    lines = []
    if d:
        sent = next((x for x in _sentences(d.get("summary", "")) if window and window.split("-")[0].lower() in x.lower()), "")
        if sent:
            lines.append(_sentence(f"{d.get('source')}: {sent}"))
        if d.get("actionable"):
            lines.append(f"Suggested: {d['actionable']}.")
    members = F.agg.get("total_active_members")
    if members:
        lines.append(f"The bigger lever right now is keeping your {fmt_int(members)} active members showing up.")
    proposal = (("draft", f"a 4-week attendance challenge for your {fmt_int(members)} members") if members
                else ("draft", "a retention push for your current members"))
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"{head} — {reframe}.",
            "urgency": f"{head}. Before you spend on ads: {reframe}.",
            "curiosity": f"{head}, but I don't think you should worry — {reframe}.",
        },
        lines=lines, proposal=proposal, default_lever="proof", use_anchor=False,
        anchor_prefs=ANCHOR_PREFS["seasonal_perf_dip"], why=f"expected seasonal dip reframe ({window or 'seasonal'})",
    )


def plan_renewal(F: Facts) -> KindPlan:
    days = F.tp.get("days_remaining", F.sub.get("days_remaining"))
    plan = F.tp.get("plan") or F.sub.get("plan") or "magicpin"
    amt = F.tp.get("renewal_amount")
    amt_txt = f" ({fmt_money(amt)})" if amt else ""
    p = F.perf
    recap = ""
    if p.get("views"):
        recap = f"Last 30 days on your profile: {fmt_int(p.get('views'))} views, {fmt_int(p.get('calls', 0))} calls, {fmt_int(p.get('directions', 0))} direction requests."
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "urgency": f"your {plan} plan renews in {days} days{amt_txt}.",
            "proof": f"quick renewal note — {days} days left on your {plan} plan{amt_txt}.",
            "curiosity": f"{days} days left on your {plan} plan{amt_txt} — here's what it's been doing for you.",
        },
        lines=[recap], proposal=("share", "the renewal details here so there's no gap in your profile upkeep"),
        default_lever="urgency", anchor_prefs=ANCHOR_PREFS["renewal_due"], why=f"renewal due in {days} days",
        exclude_anchors=["sub_"],
    )


def plan_winback(F: Facts) -> KindPlan:
    days = F.tp.get("days_since_expiry", F.sub.get("days_since_expiry"))
    dip = F.tp.get("perf_dip_pct")
    lapsed = F.tp.get("lapsed_customers_added_since_expiry")
    bits = []
    if isinstance(dip, (int, float)):
        bits.append(f"calls are down {pct(dip)}")
    if lapsed:
        bits.append(f"{lapsed} more {F.people} have gone quiet")
    since = f"since your plan lapsed {days} days ago" if days else "since your plan lapsed"
    head = f"{since}, {' and '.join(bits)}" if bits else f"{since}, your profile has been running without upkeep"
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={"proof": f"{head}.", "urgency": f"{head} — and that gap widens every week.", "curiosity": f"{head}. There's a quick way back."},
        proposal=("reactivate", f"your profile with a win-back offer for those {lapsed} {F.people}" if lapsed else "your profile with a win-back offer"),
        default_lever="proof", anchor_prefs=ANCHOR_PREFS["winback_eligible"], why=f"winback eligible ({days} days since expiry)",
        exclude_anchors=["sub_"],
    )


def plan_unverified(F: Facts) -> KindPlan:
    up = F.tp.get("estimated_uplift_pct")
    path = humanize(F.tp.get("verification_path") or "").replace(" or ", " or a ")
    up_txt = f" — verified listings get an estimated {pct(up)} more visibility" if isinstance(up, (int, float)) else ""
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "proof": f"your Google profile for {F.biz_name} is still unverified{up_txt}.",
            "urgency": f"every week unverified costs you searches{up_txt}.",
            "curiosity": f"one setting is holding your Google listing back{up_txt}.",
        },
        lines=[f"Verification is via {path}." if path else ""],
        proposal=("start", "the verification with you now — it takes a few minutes"),
        default_lever="proof", anchor_prefs=ANCHOR_PREFS["gbp_unverified"], why="GBP unverified",
        exclude_anchors=["unverified"],
    )


def plan_curious(F: Facts) -> KindPlan:
    service_word = {"dentists": "treatment", "salons": "service", "restaurants": "dish", "gyms": "class", "pharmacies": "product"}.get(F.slug, "service")
    pool = [t for t in F.trend_signals if t.get("query") and not re.search(r"\b(offer|cost|near me|delivery|monitor|program)\b", t["query"])]
    trend = max(pool, key=lambda t: t.get("delta_yoy") or 0, default=None)
    guess = ""
    if trend:
        q = re.sub(r"\b(price|near me|delhi|mumbai|bangalore|classes)\b", "", trend["query"]).strip()
        guess = f" My guess is {q} — searches for it are up {pct(trend.get('delta_yoy'))} YoY."
    return KindPlan(
        kind=F.kind, scope="merchant",
        hooks={
            "curiosity": f"quick one — what's been the most asked-for {service_word} at {F.biz_name} this week?{guess}",
            "proof": f"quick one — which {service_word} are people asking about most this week?{guess}",
            "urgency": f"one question before the weekend — what's the most asked-for {service_word} this week?{guess}",
        },
        lines=["Tell me and I'll turn it into a Google post plus a ready WhatsApp reply for price questions — 5 minutes of your time."],
        proposal=("turn", "your answer into a post"), cta_type="open_ended", default_lever="curiosity",
        use_anchor=False, anchor_prefs=ANCHOR_PREFS["curious_ask_due"], why="weekly curious-ask cadence",
    )


def plan_weather(F: Facts) -> KindPlan:
    temp = F.tp.get("temperature_c") or F.tp.get("temp_c") or F.tp.get("temperature")
    city = F.tp.get("city") or F.city
    head = f"{temp}°C in {city} today" if temp else f"heatwave conditions in {city} today"
    tip = {
        "restaurants": "Expect orders to shift to delivery and cold drinks — worth pushing both this afternoon.",
        "pharmacies": "ORS, sunscreen and electrolyte demand usually jumps on days like this — keep them at the counter.",
        "gyms": "Members skip the hot afternoon slots — nudge them to early-morning and evening classes.",
        "salons": "Walk-ins drop in the afternoon heat — morning and evening slots are the ones to promote.",
        "dentists": "Patients reschedule afternoon visits in this heat — offering morning or evening slots helps.",
    }.get(F.slug, "Customer timing shifts on days like this.")
    return KindPlan(kind=F.kind, scope="merchant",
                    hooks={"proof": f"{head}.", "urgency": f"{head} — plan today around it.", "curiosity": f"{head}. Quick idea:"},
                    lines=[tip], proposal=("draft", "a same-day post for it"), default_lever="urgency",
                    anchor_prefs=ANCHOR_PREFS["weather_heatwave"], why="weather trigger")


def plan_trend(F: Facts) -> KindPlan:
    q = F.tp.get("query")
    delta = F.tp.get("delta_yoy") or F.tp.get("delta_pct")
    if not q:
        t = max(F.trend_signals, key=lambda x: x.get("delta_yoy") or 0, default=None)
        if t:
            q, delta = t.get("query"), t.get("delta_yoy")
    if not q:
        return plan_generic(F)
    head = f"searches for \"{q}\" are up {pct(delta)} YoY" if isinstance(delta, (int, float)) else f"searches for \"{q}\" are climbing"
    return KindPlan(kind=F.kind, scope="merchant",
                    hooks={"proof": f"{head}.", "urgency": f"{head} — the listings that mention it now catch that demand.",
                           "curiosity": f"{head}. Is it on your profile yet?"},
                    proposal=("draft", f"a post and offer line that puts \"{q}\" on your profile"), default_lever="proof",
                    anchor_prefs=ANCHOR_PREFS["category_trend_movement"], why=f"trend movement: {q}")


def plan_generic(F: Facts) -> KindPlan:
    ptxt = _payload_text(F.tp)
    label = humanize(F.kind)
    head = f"{label} — {ptxt}" if ptxt else f"a {label} update for {F.biz_name}"
    return KindPlan(kind=F.kind, scope="merchant",
                    hooks={"proof": f"{head}.", "urgency": f"{head}. Worth acting on this week.", "curiosity": f"{head}. Here's what I'd do."},
                    proposal=_fix_proposal(F), default_lever="proof", anchor_prefs=DEFAULT_PREFS,
                    why=f"{label} trigger")


# ---------------------------------------------------------------------------
# customer-facing plans (sent as merchant_on_behalf)


def _cust_open(F: Facts, emoji: str = "") -> str:
    c = F.c or {}
    name = c.get("parent") or c.get("first") or ""
    mode = c.get("mode")
    shop = F.biz_name
    if F.slug == "dentists" and F.owner:
        shop = f"{F.salutation}'s clinic"
    owner = F.owner if F.slug != "dentists" else ""
    who = f"{owner} from {shop}" if owner and c.get("state") in {"lapsed_hard", "lapsed_soft", "new"} else shop
    if mode == "hindi":
        honor = f"{name} ji" if name and not name.lower().startswith("mr") else (name.replace("Mr. ", "") + " ji" if name else "")
        return f"Namaste{' ' + honor if honor else ''} — {shop}{', ' + F.locality if F.locality else ''} se{(' ' + emoji) if emoji else '.'}"
    greet = c.get("greet") or "Hi"
    return f"{greet}{' ' + name if name else ''}, {who} here{(' ' + emoji) if emoji else '.'}"


def _slot_labels(F: Facts, key: str = "available_slots") -> list[tuple[str, Optional[object]]]:
    out = []
    for s in F.tp.get(key) or F.tp.get("next_session_options") or []:
        if isinstance(s, dict) and s.get("label"):
            out.append((s["label"], parse_dt(s.get("iso"))))
    return out


def _cust_cta_slots(F: Facts, slots: list[tuple[str, object]]) -> tuple[str, str]:
    mode = (F.c or {}).get("mode")
    if len(slots) >= 2:
        a, b = slots[0][0], slots[1][0]
        if mode in {"hinglish", "hindi"}:
            return f"Aapke liye 2 slots rakhe hain: {a} ya {b}. Reply 1 ya 2 karein, ya apna convenient time bata dijiye.", "multi_choice_slot"
        return f"We've kept 2 slots for you: {a} or {b}. Reply 1 or 2, or tell us a time that suits you.", "multi_choice_slot"
    if len(slots) == 1:
        a = slots[0][0]
        if mode in {"hinglish", "hindi"}:
            return f"Next slot: {a}. Reply YES to confirm karein.", "binary_yes_no"
        return f"Next slot: {a}. Reply YES to confirm it.", "binary_yes_no"
    pref = (F.c or {}).get("slot_pref")
    pref_txt = f" ({pref})" if pref else ""
    if mode in {"hinglish", "hindi"}:
        return f"Reply YES karein aur hum is hafte ka slot{pref_txt} share kar denge.", "binary_yes_no"
    return f"Reply YES and we'll share this week's open slots{pref_txt}.", "binary_yes_no"


def _cust_offer_line(F: Facts, prefer: Optional[list[str]] = None) -> str:
    if not F.offers_active:
        return ""
    offer = F.offers_active[0]
    if prefer:
        for p in prefer:
            for o in F.offers_active:
                if p.lower() in o.lower():
                    offer = o
                    break
    mode = (F.c or {}).get("mode")
    if mode in {"hinglish", "hindi"}:
        return f"Offer abhi live hai: {offer}."
    return f"Current offer: {offer}."


VISIT_WORD = {"dentists": "check-up", "gyms": "session", "salons": "appointment", "pharmacies": "refill", "restaurants": "visit"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def plan_recall(F: Facts) -> KindPlan:
    c = F.c or {}
    svc = humanize(F.tp.get("service_due") or "") or f"regular {VISIT_WORD.get(F.slug, 'visit')}"
    last = parse_dt(F.tp.get("last_service_date")) or c.get("last_visit")
    months = None
    if last:
        dd = days_between(last, F.now)
        if dd and dd > 0:
            months = max(1, round(dd / 30.4))
            F.nums.add(months, dd)
    slots = _slot_labels(F)
    cta, cta_type = _cust_cta_slots(F, slots)
    mode = c.get("mode")
    pref = c.get("slot_pref")
    if slots and pref and "evening" in pref and all(dt and (dt.hour >= 17) for _, dt in slots):
        tag = " (both evening slots, as you prefer)" if mode == "english" else " (dono evening slots, aapki preference ke hisaab se)"
        cta = cta.replace(".", tag + ".", 1)
    if mode in {"hinglish", "hindi"}:
        line = (f"Aapki last visit ko {_plural(months, 'mahina').replace('mahinas', 'mahine')} ho gaye — aapka {svc} due hai."
                if months else f"Aapka {svc} due hai.")
    else:
        line = f"It's been {_plural(months, 'month')} since your last visit — your {svc} is due." if months else f"Your {svc} is due."
    offer = _cust_offer_line(F, ["clean", "check", "session", "month"])
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": line}, lines=[offer],
                    customer_cta=cta, cta_type=cta_type, proposal=("book", "your slot"),
                    why=f"recall due for {c.get('first') or 'customer'} ({svc})")


def plan_wedding(F: Facts) -> KindPlan:
    c = F.c or {}
    wd = parse_dt(F.tp.get("wedding_date") or (c.get("prefs") or {}).get("wedding_date"))
    days = F.tp.get("days_to_wedding") if isinstance(F.tp.get("days_to_wedding"), int) else (days_between(F.now, wd) if wd else None)
    if isinstance(days, int) and days > 0:
        F.nums.add(days)
    trial = parse_dt(F.tp.get("trial_completed"))
    step = humanize(F.tp.get("next_step_window_open") or "skin prep program").replace("30day", "30-day")
    count = f"{days} days to your wedding" if isinstance(days, int) and days > 0 else "Your wedding is coming up"
    trial_txt = f" since your bridal trial on {fmt_day(trial)}" if trial else ""
    pref = c.get("slot_pref")
    pref_txt = f"a {pref.title()} slot" if pref else "a slot"
    return KindPlan(kind=F.kind, scope="customer",
                    hooks={"proof": f"{count} 💍 — it's been a few weeks{trial_txt}, so this is the right window to start the {step} before the main bridal rush fills our calendar."},
                    customer_cta=f"Shall I block {pref_txt} for your first session next week? Reply YES and I'll confirm.",
                    cta_type="binary_yes_no", proposal=("block", "your first session"),
                    why=f"bridal follow-up ({days} days to wedding)")


def plan_lapsed_customer(F: Facts) -> KindPlan:
    c = F.c or {}
    days = F.tp.get("days_since_last_visit") or c.get("days_since")
    focus = humanize(F.tp.get("previous_focus") or (c.get("prefs") or {}).get("training_focus") or "")
    months_member = F.tp.get("previous_membership_months")
    mode = c.get("mode")
    hi = mode in {"hinglish", "hindi"}
    if isinstance(days, int) and days > 0:
        weeks, months = round(days / 7), max(1, round(days / 30.4))
        F.nums.add(weeks, months)
        gap = (f"about {_plural(weeks, 'week')}" if days < 90 else f"about {_plural(months, 'month')}")
        if hi:
            gap = f"lagbhag {weeks} hafte" if days < 90 else f"lagbhag {months} mahine"
    else:
        gap = "kaafi samay" if hi else "a while"
    if F.slug == "gyms":
        body = f"It's been {gap} since your last session — happens to everyone, no judgment."
        if focus:
            body += f" Your {focus} goal is still very doable, and we'd love to help you pick it back up."
    elif hi:
        body = f"Aapko dekhe {gap} ho gaye — umeed hai sab theek hai."
    else:
        body = f"It's been {gap} since we last saw you — hope all is well."
    if months_member:
        body += f" Thank you for your {months_member} months with us."
    offer = _cust_offer_line(F)
    pref = c.get("slot_pref")
    pref_txt = f" at your usual {pref} time" if pref else ""
    ctas = {
        "gyms": ("Want us to hold a spot for you{p} this week? Reply YES — no commitment.",
                 "Is hafte aapke liye ek spot{p} hold kar dein? Reply YES — koi commitment nahi."),
        "dentists": ("Want us to keep a check-up slot for you{p} this week? Reply YES and we'll share the timing.",
                     "Is hafte ek check-up slot{p} rakh dein? Reply YES karein, timing share kar denge."),
        "salons": ("Want us to keep an appointment slot for you{p} this week? Reply YES and we'll confirm.",
                   "Is hafte aapke liye ek slot{p} rakh dein? Reply YES karein, confirm kar denge."),
        "restaurants": ("Want us to reserve a table for you this weekend? Reply YES and we'll hold it.",
                        "Is weekend aapke liye table reserve kar dein? Reply YES karein."),
        "pharmacies": ("Reply YES and we'll check whether any of your regular medicines are due for a refill.",
                       "Reply YES karein — hum check kar lenge ki aapki koi regular medicine refill due to nahi."),
    }
    en, hn = ctas.get(F.slug, ("Reply YES and we'll help with your next visit.", "Reply YES karein, agli visit hum arrange kar denge."))
    cta = (hn if hi else en).replace("{p}", pref_txt)
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": body}, lines=[offer], customer_cta=cta,
                    cta_type="binary_yes_no", proposal=("hold", "your next visit"), why=f"customer lapsed ({days} days)")


def plan_trial_followup(F: Facts) -> KindPlan:
    c = F.c or {}
    trial = parse_dt(F.tp.get("trial_date"))
    slots = _slot_labels(F, "next_session_options")
    child = c.get("first") if c.get("parent") else None
    who = f"{child}'s" if child else "your"
    what = "kids yoga trial" if child and F.slug == "gyms" else "trial session"
    line = f"Thank you for coming in for {who} {what}{' on ' + fmt_day(trial) if trial else ''}! 🙏"
    if slots:
        cta = f"The next session is {slots[0][0]} — shall we save {child + chr(39) + 's' if child else 'your'} spot? Reply YES."
    else:
        cta = "Shall we save a spot in the next session? Reply YES and we'll confirm the time."
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": line}, lines=[_cust_offer_line(F)], customer_cta=cta,
                    cta_type="binary_yes_no", proposal=("book", "the next session"), why="trial follow-up")


def plan_refill(F: Facts) -> KindPlan:
    c = F.c or {}
    mols = F.tp.get("molecule_list") or []
    if not mols and F.slug != "pharmacies":
        return plan_recall(F)
    runs_out = parse_dt(F.tp.get("stock_runs_out_iso"))
    mode = c.get("mode")
    hi = mode in {"hindi", "hinglish"}
    senior = c.get("senior")
    name = c.get("name") or ""
    subject = name.replace("Mr. ", "") + " ji" if senior and name else (c.get("first") or "")
    delivery = F.tp.get("delivery_address_saved")
    offers = [o for o in F.offers_active if ("senior" in o.lower() and senior) or "deliver" in o.lower()]
    if mols:
        mol_txt = ", ".join(mols)
        if hi:
            line = (f"{subject + ' ki' if subject else 'Aapki'} {len(mols)} monthly medicines ({mol_txt}) "
                    f"{fmt_day(runs_out) + ' ko' if runs_out else 'jald'} khatam hongi. Same dose, same brand pack ready hai.")
        else:
            line = (f"{subject + chr(39) + 's' if subject else 'Your'} {len(mols)} monthly medicines ({mol_txt}) run out "
                    f"{('on ' + fmt_day(runs_out)) if runs_out else 'soon'}. Same dose, same pack is ready.")
    else:
        line = "Aapki regular medicines ka refill due hai." if hi else "Your regular medicines are due for a refill."
    off = ""
    if offers:
        off = ("Aapke liye lagu: " if hi else "Applied for you: ") + " + ".join(offers) + "."
    dl = ""
    if delivery:
        dl = " Saved address par home delivery ho jayegi." if hi else " We'll deliver to your saved address."
    cta = ("Reply CONFIRM karein to dispatch kar denge, ya dosage mein koi change ho to bata dijiye." if hi
           else "Reply CONFIRM to dispatch, or tell us if the dosage has changed.")
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": line}, lines=[(off + dl).strip()], customer_cta=cta,
                    cta_type="binary_confirm_cancel", proposal=("dispatch", "your refill"), why="chronic refill due")


def plan_appointment(F: Facts) -> KindPlan:
    c = F.c or {}
    when = F.tp.get("appointment_label") or F.tp.get("time_label") or F.tp.get("slot_label")
    dt = parse_dt(F.tp.get("appointment_iso") or F.tp.get("iso"))
    if not when and dt:
        when = f"{dt.hour % 12 or 12}{'pm' if dt.hour >= 12 else 'am'}"
    svc = humanize(F.tp.get("service") or "")
    hi = c.get("mode") in {"hinglish", "hindi"}
    at = f" at {when}" if when else ""
    what = f" for your {svc}" if svc else ""
    where = f" at our {F.locality} {F.words['biz']}" if F.locality else ""
    if hi:
        line = f"Yaad dila dein — kal{at} aapka appointment hai{what}{(' (' + F.locality + ')') if F.locality else ''}."
        cta = "Reply YES karke confirm karein, ya reschedule karna ho to bata dijiye."
    else:
        line = f"Quick reminder: your appointment{what} is tomorrow{at}{where}."
        cta = "Reply YES to confirm, or tell us if you'd like to reschedule."
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": line}, customer_cta=cta,
                    cta_type="binary_confirm_cancel", proposal=("confirm", "your appointment"), why="appointment tomorrow")


def plan_customer_generic(F: Facts) -> KindPlan:
    c = F.c or {}
    ptxt = _payload_text(F.tp)
    line = f"A quick update from us — {ptxt}." if ptxt else "A quick update from us."
    offer = _cust_offer_line(F)
    cta = "Reply YES and we'll take care of it." if c.get("mode") == "english" else "Reply YES karein, baaki hum sambhal lenge."
    return KindPlan(kind=F.kind, scope="customer", hooks={"proof": line}, lines=[offer], customer_cta=cta,
                    cta_type="binary_yes_no", proposal=("help", "with the next step"), why=f"{humanize(F.kind)} (customer)")


MERCHANT_PLANS = {
    "research_digest": plan_research,
    "category_research_digest_release": plan_research,
    "regulation_change": plan_compliance,
    "compliance_alert": plan_compliance,
    "cde_opportunity": plan_cde,
    "supply_alert": plan_supply_alert,
    "category_seasonal": plan_category_seasonal,
    "perf_dip": plan_perf_dip,
    "perf_spike": plan_perf_spike,
    "milestone_reached": plan_milestone,
    "dormant_with_vera": plan_dormant,
    "review_theme_emerged": plan_review_theme,
    "competitor_opened": plan_competitor,
    "festival_upcoming": plan_festival,
    "ipl_match_today": plan_ipl,
    "active_planning_intent": plan_planning,
    "seasonal_perf_dip": plan_seasonal_dip,
    "renewal_due": plan_renewal,
    "winback_eligible": plan_winback,
    "gbp_unverified": plan_unverified,
    "curious_ask_due": plan_curious,
    "scheduled_recurring": plan_curious,
    "weather_heatwave": plan_weather,
    "category_trend_movement": plan_trend,
}
CUSTOMER_PLANS = {
    "recall_due": plan_recall,
    "wedding_package_followup": plan_wedding,
    "bridal_followup": plan_wedding,
    "customer_lapsed_hard": plan_lapsed_customer,
    "customer_lapsed_soft": plan_lapsed_customer,
    "trial_followup": plan_trial_followup,
    "chronic_refill_due": plan_refill,
    "appointment_tomorrow": plan_appointment,
}


def build_plan(F: Facts) -> KindPlan:
    if F.c is not None:
        fn = CUSTOMER_PLANS.get(F.kind, plan_customer_generic)
    else:
        fn = MERCHANT_PLANS.get(F.kind, plan_generic)
    try:
        plan = fn(F)
    except Exception:
        plan = plan_customer_generic(F) if F.c is not None else plan_generic(F)
    if not plan.anchor_prefs:
        plan.anchor_prefs = ANCHOR_PREFS.get(F.kind, DEFAULT_PREFS)
    if plan.scope == "customer":
        plan.use_anchor = False  # merchant facts never go into customer messages
    return plan


# ---------------------------------------------------------------------------
# anchor ranking (shared scorer; Bot A uses it to decide, Bot B as a prior)


def rank_anchors(F: Facts, plan: KindPlan) -> list[tuple[float, Anchor]]:
    prefs = plan.anchor_prefs or DEFAULT_PREFS
    scored = []
    used_text = " ".join(plan.hooks.values()) + " " + " ".join(plan.lines) + " " + " ".join(plan.proposal)
    used_words = set(re.findall(r"[a-z0-9₹]+", used_text.lower()))
    for a in F.anchors:
        if any(a.id.startswith(x) for x in plan.exclude_anchors):
            continue
        a_words = [w for w in re.findall(r"[a-z0-9₹]+", a.text.lower()) if len(w) > 3 or w.isdigit()]
        if a_words and sum(w in used_words for w in a_words) / len(a_words) >= 0.6:
            continue
        if a.tag in prefs:
            rel = 1.0 - 0.12 * prefs.index(a.tag)
        elif a.tag in DEFAULT_PREFS:
            rel = 0.25
        else:
            rel = 0.1
        # don't repeat a fact the hook already states
        if a.text.split(" (")[0][:30].lower() in used_text.lower():
            rel *= 0.2
        scored.append((round(a.strength * rel, 4), a))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return scored


# ---------------------------------------------------------------------------
# rendering


def _cta_text(F: Facts, plan: KindPlan, lever: str) -> str:
    verb, obj = plan.proposal
    if plan.cta_type == "open_ended":
        return ""
    hing = F.hinglish and F.slug in {"salons", "restaurants", "pharmacies"}
    timed = re.search(r"\b(today|tonight|this week|now)\b", obj)
    if lever == "urgency":
        when = "" if timed else (" aaj hi" if hing else " today")
        return f"Reply YES and I'll {verb} {obj}{when}."
    if hing:
        return f"Want me to {verb} {obj}? Bas YES reply kar dijiye."
    if lever == "curiosity":
        return f"Want me to {verb} {obj}? Just reply YES."
    return f"Want me to {verb} {obj}? Reply YES."


def render(F: Facts, plan: KindPlan, lever: Optional[str] = None, anchor: Optional[Anchor] = None) -> Draft:
    lever = lever if lever in plan.hooks else plan.default_lever
    if lever not in plan.hooks:
        lever = next(iter(plan.hooks))
    hook = plan.hooks[lever]
    if plan.scope == "customer":
        emoji = {"dentists": "🦷", "gyms": "👋", "salons": "✨", "pharmacies": "", "restaurants": "🍽️"}.get(F.slug, "")
        opener = _cust_open(F, emoji)
        parts = [opener, hook] + [l for l in plan.lines if l] + [plan.customer_cta or ""]
        body = clean_body(" ".join(p for p in parts if p))
        cta = plan.cta_type
        send_as = "merchant_on_behalf"
        name = (F.c or {}).get("first") or ""
        tname = f"merchant_{F.kind}_v1"
        params = [name, F.biz_name, hook, plan.customer_cta or ""]
        anchor_id = None
    else:
        sal = F.salutation
        opener = f"{sal}, {hook}"
        anchor_line = ""
        anchor_id = None
        if plan.use_anchor and anchor is not None:
            anchor_line = _sentence(anchor.text)
            anchor_id = anchor.id
        body_lines = [l for l in plan.lines if l]
        cta_line = _cta_text(F, plan, lever)
        parts = [opener, anchor_line] + body_lines + [cta_line]
        body = " ".join(p for p in parts if p)
        if "\n" not in body and any(l.startswith("•") for l in body_lines):
            body = opener + "\n" + "\n".join(body_lines) + ("\n" + anchor_line if anchor_line else "") + ("\n" + cta_line if cta_line else "")
        if plan.signoff:
            body = body + f" {plan.signoff}"
        body = clean_body(body)
        cta = plan.cta_type
        send_as = "vera"
        tname = f"vera_{F.kind}_v1"
        params = [sal, hook, cta_line or "open_ended"]
    verb, obj = plan.proposal
    rationale = f"{humanize(plan.why) or humanize(F.kind)}; lever={lever}"
    if anchor_id:
        rationale += f"; personalised with {anchor_id}"
    rationale += f"; single {cta} CTA"
    return Draft(body=body, cta=cta, send_as=send_as, template_name=tname, template_params=params,
                 rationale=rationale, lever=lever, anchor_id=anchor_id, proposal=f"{verb} {obj}",
                 artifact=plan.artifact)


def all_variants(F: Facts, plan: KindPlan) -> list[Draft]:
    """Every (lever x top-anchor) combination — Bot B's fallback candidate pool."""
    ranked = rank_anchors(F, plan)
    anchors: list[Optional[Anchor]] = [a for _, a in ranked[:3]] or [None]
    if not plan.use_anchor:
        anchors = [None]
    out = []
    for lever in plan.hooks:
        for a in anchors:
            out.append(render(F, plan, lever, a))
    seen, uniq = set(), []
    for d in out:
        h = stable_hash(d.body)
        if h not in seen:
            seen.add(h)
            uniq.append(d)
    return uniq
