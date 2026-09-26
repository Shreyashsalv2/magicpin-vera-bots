"""Turn the four raw contexts into a grounded fact sheet.

Everything a message may say is derived here from the pushed contexts. The
NumberBank records every number that exists in (or is derived from) the
contexts, so the validator can reject any draft that invents a figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .util import (
    MONTHS,
    ctr_pct,
    days_between,
    first_name,
    fmt_int,
    humanize,
    parse_dt,
    pct,
    walk_numbers,
)

CATEGORY_WORDS = {
    "dentists": {"biz": "clinic", "people": "patients", "person": "patient", "label": "dental practices"},
    "salons": {"biz": "salon", "people": "clients", "person": "client", "label": "salons"},
    "restaurants": {"biz": "restaurant", "people": "diners", "person": "diner", "label": "restaurants"},
    "gyms": {"biz": "studio", "people": "members", "person": "member", "label": "gyms"},
    "pharmacies": {"biz": "pharmacy", "people": "customers", "person": "customer", "label": "pharmacies"},
}
DEFAULT_WORDS = {"biz": "business", "people": "customers", "person": "customer", "label": "businesses"}

MONTH_INDEX = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}

GENERIC_SMALL_NUMBERS = {15.0, 20.0, 24.0, 30.0, 45.0, 48.0, 60.0, 90.0}


class NumberBank:
    """Numbers the bot is allowed to state."""

    def __init__(self) -> None:
        self.values: set[float] = set()

    def add(self, *vals: Any) -> None:
        for v in vals:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            self.values.add(round(f, 4))
            self.values.add(round(abs(f), 4))

    def absorb(self, obj: Any) -> None:
        raw: set[float] = set()
        walk_numbers(obj, raw)
        for v in raw:
            self.add(v)
            if -1.0 <= v <= 1.0 and v != 0:
                # fractions are shown as percentages: 0.021 -> 2.1, 0.38 -> 38
                self.add(round(v * 100, 1), round(v * 100))
            if abs(v) >= 1000:
                self.add(round(v / 1000, 1))  # 2.4k style
        # year / day numbers are always fine
        for y in range(2020, 2031):
            self.add(y)

    def allows(self, x: float) -> bool:
        if x <= 12 or x in GENERIC_SMALL_NUMBERS:
            return True
        for v in self.values:
            if abs(v - x) <= max(0.051, 0.006 * abs(v)):
                return True
        return False


@dataclass
class Anchor:
    id: str
    tag: str
    text: str
    strength: float
    nums: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "tag": self.tag, "text": self.text, "strength": round(self.strength, 3)}


def _parse_signals(signals: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in signals or []:
        if not isinstance(s, str):
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            m = re.match(r"(\d+)", v)
            out[k] = int(m.group(1)) if m else v
        else:
            out[s] = True
    return out


def month_in_range(month_range: str, month: int) -> bool:
    """'Nov-Feb' / 'Apr-Jun' / 'Jan' / 'Feb 14'."""
    parts = [p.strip()[:3].lower() for p in re.split(r"[-–]", month_range or "") if p.strip()]
    idx = [MONTH_INDEX.get(p) for p in parts if MONTH_INDEX.get(p)]
    if not idx:
        return False
    if len(idx) == 1:
        return month == idx[0]
    a, b = idx[0], idx[-1]
    if a <= b:
        return a <= month <= b
    return month >= a or month <= b


class Facts:
    def __init__(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict],
        now: datetime,
    ) -> None:
        self.category = category or {}
        self.merchant = merchant or {}
        self.trigger = trigger or {}
        self.customer = customer
        self.now = now
        self.nums = NumberBank()
        for ctx in (self.category, self.merchant, self.trigger, self.customer or {}):
            self.nums.absorb(ctx)

        # --- category -------------------------------------------------------
        self.slug = self.category.get("slug") or self.merchant.get("category_slug") or "business"
        self.words = CATEGORY_WORDS.get(self.slug, DEFAULT_WORDS)
        voice = self.category.get("voice") or {}
        self.tone = voice.get("tone", "")
        self.taboos = [t.split("(")[0].strip().lower() for t in (voice.get("vocab_taboo") or voice.get("taboos") or [])]
        self.vocab = voice.get("vocab_allowed") or []
        self.peer = self.category.get("peer_stats") or {}
        self.peer_label = humanize(re.sub(r"_?\d{4}$", "", str(self.peer.get("scope") or self.words["label"])))
        self.digest = [d for d in (self.category.get("digest") or []) if isinstance(d, dict)]
        self.catalog = [o for o in (self.category.get("offer_catalog") or []) if isinstance(o, dict)]
        self.content_library = self.category.get("patient_content_library") or []
        self.seasonal_beats = self.category.get("seasonal_beats") or []
        self.trend_signals = self.category.get("trend_signals") or []

        # --- merchant -------------------------------------------------------
        m = self.merchant
        ident = m.get("identity") or {}
        self.merchant_id = m.get("merchant_id") or self.trigger.get("merchant_id") or ""
        self.biz_name = ident.get("name") or "your business"
        self.locality = ident.get("locality") or ""
        self.city = ident.get("city") or ""
        self.languages = [str(x).lower() for x in (ident.get("languages") or ["en"])]
        self.verified = ident.get("verified")
        owner = (ident.get("owner_first_name") or "").strip()
        self.owner = owner
        if self.slug == "dentists":
            bare = re.sub(r"^dr\.?\s*", "", owner, flags=re.I).strip()
            self.salutation = f"Dr. {bare}" if bare else self.biz_name
        else:
            self.salutation = owner or f"{self.biz_name} team"
        self.hinglish = "hi" in self.languages
        self.sub = m.get("subscription") or {}
        self.perf = m.get("performance") or {}
        self.d7 = self.perf.get("delta_7d") or {}
        self.offers_active = [o.get("title") for o in (m.get("offers") or []) if o.get("status") == "active" and o.get("title")]
        self.offers_inactive = [o.get("title") for o in (m.get("offers") or []) if o.get("status") != "active" and o.get("title")]
        self.agg = m.get("customer_aggregate") or {}
        self.signals = _parse_signals(m.get("signals") or [])
        self.reviews = [r for r in (m.get("review_themes") or []) if isinstance(r, dict)]
        self.history = [h for h in (m.get("conversation_history") or []) if isinstance(h, dict)]

        # --- trigger --------------------------------------------------------
        t = self.trigger
        self.trigger_id = t.get("id") or ""
        self.kind = (t.get("kind") or "generic").strip()
        self.tp = t.get("payload") or {}
        self.placeholder = bool(self.tp.get("placeholder"))
        self.urgency = int(t.get("urgency") or 2)
        self.suppression_key = t.get("suppression_key") or f"{self.kind}:{self.merchant_id}:{self.trigger_id}"
        self.expires_at = parse_dt(t.get("expires_at"))

        # --- customer -------------------------------------------------------
        self.c = None
        if self.customer:
            self.c = self._customer_facts(self.customer)

        self.digest_item = self._resolve_digest()
        self.anchors = self._build_anchors()

    # ------------------------------------------------------------------ helpers
    @property
    def people(self) -> str:
        return self.words["people"]

    def offer_or_catalog(self, prefer: Optional[list[str]] = None) -> tuple[Optional[str], bool]:
        """Best offer to reference. Returns (title, is_merchant_live_offer)."""
        if self.offers_active:
            if prefer:
                for p in prefer:
                    for o in self.offers_active:
                        if p.lower() in o.lower():
                            return o, True
            return self.offers_active[0], True
        pool = [o for o in self.catalog if o.get("type") in {"service_at_price", "free_service", "free_trial"}]
        if prefer:
            for p in prefer:
                for o in pool:
                    if p.lower() in (o.get("title") or "").lower():
                        return o["title"], False
        return (pool[0]["title"], False) if pool else (None, False)

    def current_beats(self, lookahead_months: int = 2) -> list[dict]:
        month = self.now.month
        out = []
        for b in self.seasonal_beats:
            rng = b.get("month_range") or ""
            for off in range(0, lookahead_months + 1):
                mm = (month - 1 + off) % 12 + 1
                if month_in_range(rng, mm):
                    out.append({**b, "_offset": off})
                    break
        return sorted(out, key=lambda b: b["_offset"])

    def _resolve_digest(self) -> Optional[dict]:
        ids = [self.tp.get(k) for k in ("top_item_id", "digest_item_id", "alert_id", "item_id")]
        ids = [i for i in ids if i]
        for i in ids:
            for d in self.digest:
                if d.get("id") == i:
                    return d
        top = self.tp.get("top_item")
        if isinstance(top, dict) and top.get("title"):
            return top
        kind_map = {
            "research_digest": ["research"],
            "regulation_change": ["compliance"],
            "cde_opportunity": ["cde"],
            "supply_alert": ["alert", "supply"],
            "category_seasonal": ["seasonal"],
            "category_trend_movement": ["trend"],
            "tech_update": ["tech"],
        }
        wanted = kind_map.get(self.kind)
        if wanted:
            matches = [d for d in self.digest if d.get("kind") in wanted]
            if matches:
                return matches[-1]
        return None

    def _customer_facts(self, cust: dict) -> dict:
        ident = cust.get("identity") or {}
        rel = cust.get("relationship") or {}
        prefs = cust.get("preferences") or {}
        consent = cust.get("consent") or {}
        raw_name = ident.get("name") or ""
        parent = None
        pm = re.search(r"\(parent:\s*([^)]+)\)", raw_name)
        if pm:
            parent = pm.group(1).strip()
        name = re.sub(r"\(.*?\)", "", raw_name).strip()
        if name.startswith("(") or not name:
            name = ""
        lang = str(ident.get("language_pref") or "en").lower()
        greet = None
        if lang.startswith("te"):
            greet = "Namaskaram"
        elif lang.startswith("ta"):
            greet = "Vanakkam"
        elif lang.startswith("kn"):
            greet = "Namaskara"
        elif lang.startswith("mr"):
            greet = "Namaskar"
        if lang in {"hi", "hindi"}:
            mode = "hindi"
        elif "hi" in lang and "mix" in lang:
            mode = "hinglish"
        else:
            mode = "english"
        last = parse_dt(rel.get("last_visit"))
        since = days_between(last, self.now)
        if since is not None and since < 0:
            since = None
        services = []
        for s in rel.get("services_received") or []:
            if s and s != "...":
                h = humanize(re.sub(r"_x\d+.*$", "", str(s)))
                h = re.sub(r"^chronic rx ", "", h)
                if h not in services:
                    services.append(h)
        facts = {
            "id": cust.get("customer_id"),
            "name": name,
            "first": first_name(name) if name else "",
            "parent": parent,
            "senior": bool(ident.get("senior_citizen")) or str(ident.get("age_band", "")).startswith(("60", "65", "70")),
            "age_band": ident.get("age_band"),
            "lang": lang,
            "mode": mode,
            "greet": greet,
            "state": cust.get("state") or "",
            "last_visit": last,
            "days_since": since,
            "visits": rel.get("visits_total"),
            "services": services,
            "ltv": rel.get("lifetime_value"),
            "favourite": rel.get("favourite_dish"),
            "prefs": prefs,
            "slot_pref": humanize(prefs.get("preferred_slots")) if prefs.get("preferred_slots") else "",
            "channel": prefs.get("channel") or "",
            "consent_scope": consent.get("scope") or [],
            "opted_in": bool(consent.get("opted_in_at")) and (prefs.get("reminder_opt_in", True) is not False or bool(consent.get("scope"))),
        }
        if since is not None:
            self.nums.add(since, round(since / 7), round(since / 30), round(since / 30.4))
        return facts

    # ---------------------------------------------------------------- anchors
    def _build_anchors(self) -> list[Anchor]:
        A: list[Anchor] = []
        p, peer = self.perf, self.peer
        people = self.people

        ctr, pctr = p.get("ctr"), peer.get("avg_ctr")
        if isinstance(ctr, (int, float)) and isinstance(pctr, (int, float)) and pctr:
            gap = (pctr - ctr) / pctr
            if gap >= 0.1:
                A.append(Anchor("ctr_gap", "visibility_gap",
                                f"your profile CTR is {ctr_pct(ctr)} vs the {ctr_pct(pctr)} median for {self.peer_label}",
                                min(1.0, 0.35 + gap), [ctr, pctr]))
            elif gap <= -0.1:
                A.append(Anchor("ctr_lead", "strength",
                                f"your profile CTR of {ctr_pct(ctr)} is ahead of the {ctr_pct(pctr)} median for {self.peer_label}",
                                min(0.8, 0.3 + abs(gap) / 2), [ctr, pctr]))

        for metric, peer_key, label in (("calls", "avg_calls_30d", "calls"), ("views", "avg_views_30d", "profile views")):
            mv, pv = p.get(metric), peer.get(peer_key)
            if isinstance(mv, (int, float)) and isinstance(pv, (int, float)) and pv:
                gap = (pv - mv) / pv
                if gap >= 0.25:
                    A.append(Anchor(f"{metric}_gap", "visibility_gap",
                                    f"{fmt_int(mv)} {label} in the last 30 days vs ~{fmt_int(pv)} for similar {self.words['label']}",
                                    min(0.95, 0.3 + gap / 2), [mv, pv]))
                elif gap <= -0.4:
                    A.append(Anchor(f"{metric}_lead", "strength",
                                    f"{fmt_int(mv)} {label} in 30 days, well above the ~{fmt_int(pv)} typical for {self.words['label']}",
                                    min(0.8, 0.25 + abs(gap) / 3), [mv, pv]))

        for key, label in (("calls_pct", "calls"), ("views_pct", "profile views"), ("ctr_pct", "CTR")):
            d = self.d7.get(key)
            if isinstance(d, (int, float)) and abs(d) >= 0.1:
                direction = "up" if d > 0 else "down"
                A.append(Anchor(f"d7_{key}", "momentum_up" if d > 0 else "momentum_down",
                                f"{label} {direction} {pct(d)} this week", min(1.0, 0.3 + abs(d)), [d]))

        stale = self.signals.get("stale_posts")
        if isinstance(stale, int):
            freq = peer.get("avg_post_freq_days")
            extra = f" (similar {self.words['label']} post every ~{freq} days)" if freq else ""
            A.append(Anchor("stale_posts", "content", f"your last Google post was {stale} days ago{extra}",
                            min(0.9, 0.3 + stale / 60), [stale]))
        elif self.signals.get("no_recent_post"):
            A.append(Anchor("stale_posts", "content", "no recent Google post on your profile", 0.4))

        for r in self.reviews:
            occ = r.get("occurrences_30d")
            theme = humanize(r.get("theme"))
            if not occ or not theme:
                continue
            quote = r.get("common_quote")
            if r.get("sentiment") == "neg":
                txt = f"{occ} reviews in 30 days mention {theme}" + (f" (\"{quote}\")" if quote else "")
                A.append(Anchor(f"review_neg_{r.get('theme')}", "review_neg", txt, min(0.9, 0.35 + occ / 10), [occ]))
            else:
                txt = f"{occ} reviews in 30 days praise {theme}" + (f" (\"{quote}\")" if quote else "")
                A.append(Anchor(f"review_pos_{r.get('theme')}", "review_pos", txt, min(0.7, 0.25 + occ / 25), [occ]))

        ag = self.agg
        for key, days in (("lapsed_180d_plus", 180), ("lapsed_90d_plus", 90)):
            n = ag.get(key)
            if isinstance(n, (int, float)) and n > 0:
                A.append(Anchor("lapsed", "retention", f"{fmt_int(n)} {people} haven't been back in {days}+ days",
                                min(0.9, 0.35 + n / 300), [n]))
                break
        if isinstance(ag.get("high_risk_adult_count"), (int, float)):
            n = ag["high_risk_adult_count"]
            A.append(Anchor("high_risk", "clinical_cohort", f"{fmt_int(n)} high-risk adult patients on your roster", 0.6, [n]))
        if isinstance(ag.get("chronic_rx_count"), (int, float)):
            n = ag["chronic_rx_count"]
            A.append(Anchor("chronic_rx", "clinical_cohort", f"{fmt_int(n)} chronic-Rx customers who refill with you", 0.6, [n]))
        if isinstance(ag.get("total_active_members"), (int, float)):
            n = ag["total_active_members"]
            A.append(Anchor("members", "base", f"your {fmt_int(n)} active members", 0.5, [n]))
        for key, peer_key, label in (("retention_6mo_pct", "retention_6mo_pct", "6-month retention"),
                                     ("retention_3mo_pct", "retention_3mo_pct", "3-month retention")):
            mv, pv = ag.get(key), peer.get(peer_key)
            if isinstance(mv, (int, float)):
                if isinstance(pv, (int, float)) and pv and mv < pv * 0.9:
                    A.append(Anchor("retention_gap", "retention", f"{label} at {pct(mv)} vs {pct(pv)} for peers",
                                    min(0.85, 0.35 + (pv - mv) * 2), [mv, pv]))
                elif isinstance(pv, (int, float)) and pv and mv > pv * 1.05:
                    A.append(Anchor("retention_lead", "strength", f"{label} at {pct(mv)}, above the {pct(pv)} peer level", 0.45, [mv, pv]))
        if isinstance(ag.get("repeat_customer_pct"), (int, float)):
            A.append(Anchor("repeat", "base", f"{pct(ag['repeat_customer_pct'])} of your {people} are repeat buyers", 0.35))
        if isinstance(ag.get("delivery_orders_30d"), (int, float)):
            n = ag["delivery_orders_30d"]
            A.append(Anchor("delivery_orders", "base", f"{fmt_int(n)} delivery orders in the last 30 days", 0.45, [n]))
        if isinstance(ag.get("total_unique_ytd"), (int, float)) and ag["total_unique_ytd"] > 0:
            n = ag["total_unique_ytd"]
            A.append(Anchor("unique_ytd", "base", f"{fmt_int(n)} unique {people} so far this year", 0.3, [n]))

        if self.offers_active:
            A.append(Anchor("offer_live", "offer", f"your \"{self.offers_active[0]}\" offer is live", 0.45))
        else:
            title, _ = self.offer_or_catalog()
            if title:
                A.append(Anchor("offer_gap", "offer_gap",
                                f"no active offer on your profile right now (\"{title}\" is the format that converts best for {self.words['label']})",
                                0.5))

        days_left = self.sub.get("days_remaining")
        status = self.sub.get("status")
        if status == "expired" and self.sub.get("days_since_expiry"):
            A.append(Anchor("sub_expired", "subscription", f"your {self.sub.get('plan') or 'magicpin'} plan lapsed {self.sub['days_since_expiry']} days ago", 0.55))
        elif isinstance(days_left, (int, float)) and days_left <= 30:
            label = "trial" if status == "trial" else f"{self.sub.get('plan') or 'magicpin'} plan"
            A.append(Anchor("sub_renewal", "subscription", f"your {label} has {int(days_left)} days left", 0.5, [days_left]))

        if self.verified is False:
            A.append(Anchor("unverified", "visibility_gap", "your Google profile is still unverified", 0.55))

        last_merchant = next((h for h in reversed(self.history) if h.get("from") == "merchant" and h.get("body")), None)
        if last_merchant:
            A.append(Anchor("last_ask", "history", f"you asked: \"{last_merchant['body']}\"", 0.4))

        return A

    # --------------------------------------------------------------- utilities
    def anchor(self, aid: str) -> Optional[Anchor]:
        return next((a for a in self.anchors if a.id == aid), None)

    def anchors_by_tags(self, tags: list[str]) -> list[Anchor]:
        return [a for a in self.anchors if a.tag in tags]

    def summary_for_llm(self) -> dict:
        """Compact, fact-only view of the contexts for prompts."""
        m = {
            "business_name": self.biz_name,
            "salutation": self.salutation,
            "category": self.slug,
            "locality": self.locality,
            "city": self.city,
            "languages": self.languages,
            "subscription": self.sub,
            "performance_30d": {k: v for k, v in self.perf.items() if k != "window_days"},
            "peer_benchmarks": {k: v for k, v in self.peer.items()},
            "active_offers": self.offers_active,
            "inactive_offers": self.offers_inactive,
            "customer_aggregate": self.agg,
            "signals": self.merchant.get("signals") or [],
            "review_themes": self.reviews,
            "recent_conversation": self.history[-4:],
        }
        out = {
            "merchant": m,
            "trigger": {"kind": self.kind, "urgency": self.urgency, "payload": self.tp},
            "category_voice": {"tone": self.tone, "vocab_allowed": self.vocab[:12], "taboo": self.taboos},
        }
        if self.digest_item:
            out["digest_item"] = self.digest_item
        if self.c:
            c = dict(self.c)
            c["last_visit"] = c["last_visit"].date().isoformat() if c.get("last_visit") else None
            out["customer"] = c
        beats = self.current_beats()
        if beats:
            out["seasonal_now"] = [{"month_range": b.get("month_range"), "note": b.get("note")} for b in beats[:2]]
        return out
