"""Reply state machine.

The *decision* (send / wait / end) is deterministic and shared by both bots —
auto-reply detection, opt-outs, hostility and intent transitions are too
important to leave to a model. The *wording* of `send` replies can be improved
by each bot's strategy, then re-validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .facts import Facts
from .util import fmt_int, fmt_money, norm_text, stable_hash

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching|your message|messaging)",
    r"thanks for (contacting|reaching out|your message|messaging)",
    r"(will|shall) (get back|respond|revert|reply) (to you )?(shortly|soon|asap|at the earliest)",
    r"our (team|executive|representative) will",
    r"(currently|presently) (unavailable|away|closed|out of office)",
    r"out of (the )?office",
    r"automated (message|reply|response|assistant)",
    r"auto[- ]?reply",
    r"business hours",
    r"we are (closed|away)",
    r"this is an automated",
    r"jaankari ke liye (bahut[- ]bahut )?shukriya",
    r"hamari team tak",
    r"aapka sandesh (mil gaya|prapt)",
    r"i am an? (automated|virtual) assistant",
]
OPT_OUT_PATTERNS = [
    r"\bstop\b", r"\bunsubscribe\b", r"not interested", r"no interest", r"don'?t (message|text|contact|send|bother)",
    r"do not (message|text|contact|send)", r"remove me", r"\bopt[- ]?out\b", r"band karo", r"mat bhejo",
    r"message mat", r"nahi chahiye", r"leave me alone", r"never (message|contact)",
]
HOSTILE_PATTERNS = [
    r"\buseless\b", r"\bspam\b", r"\bbothering\b", r"\bwaste of (time|my time)\b", r"\bidiot", r"\bstupid\b",
    r"\bnonsense\b", r"\bshut up\b", r"\bfraud\b", r"\bscam\b", r"\bpathetic\b", r"\bharass", r"\bannoying\b",
    r"\bbakwas\b", r"\bpagal\b", r"\bbekaar\b", r"\bget lost\b", r"\bdamn\b", r"\bf\*+", r"\bfuck", r"\bbloody\b",
    r"why are you (bothering|messaging|disturbing)",
]
LATER_PATTERNS = [
    (r"\b(tomorrow|kal)\b", 86400), (r"next week|agle hafte", 604800), (r"\b(an|1|one) hour\b|ghante", 3600),
    (r"\b(later|busy|not now|baad mein|abhi nahi|in a meeting|call you back|driving)\b", 14400),
]
COMMIT_PATTERNS = [
    r"^(yes|yeah|yep|yup|haan|han|ha|ji|ji haan|sure|ok|okay|okk|k|done|confirm|confirmed|go|go ahead|proceed|chalo|theek hai|thik hai|perfect|great|1|2)\b",
    r"let'?s (do|go|start)", r"\bdo it\b", r"go ahead", r"please (do|send|go|proceed|start|share|draft|book)",
    r"\bsend (it|me|the|over)\b", r"\bkar (do|dijiye|dena)\b", r"\bkaro\b", r"\bbook (it|me|the)\b",
    r"\bsign me up\b", r"\bi want to (join|start|go ahead|do)\b", r"\bjudna\b", r"\bjoin\b", r"\bset it up\b",
    r"\bstart\b", r"\bwhat'?s next\b", r"\bnext step",
]
OFF_TOPIC_PATTERNS = [
    r"\bgst\b", r"\bincome tax\b", r"\bitr\b", r"\btax (filing|return)", r"\bloan\b", r"\binsurance\b",
    r"\bchartered accountant\b", r"\baccountant\b", r"\blawyer\b", r"\blegal case\b", r"\bcourt\b", r"\bvisa\b",
    r"\bpassport\b", r"\belectricity bill\b", r"\brecipe\b", r"\bcricket score\b", r"\bmovie\b", r"\bpolitic",
    r"\bstock market\b", r"\bshare price\b", r"\bcrypto\b", r"\bmy (phone|laptop|car)\b", r"\bhoroscope\b",
    r"\bbank account\b", r"\bpan card\b", r"\baadhaar\b",
]
OBJECTION_PATTERNS = [
    r"expensive", r"too (costly|much)", r"no budget", r"can'?t afford", r"mehenga", r"mehnga", r"already (have|do|doing|tried)",
    r"doesn'?t work", r"didn'?t work", r"not (sure|convinced|needed|required)", r"no need", r"zaroorat nahi",
    r"what'?s the point", r"waste", r"not worth",
]
QUESTION_START = re.compile(r"^(what|how|why|when|where|which|who|can|could|is|are|does|do|will|kitna|kitne|kya|kaise|kab|kaun|kahan)\b", re.I)
HINGLISH_MARKERS = re.compile(r"\b(haan|nahi|kya|hai|hain|karo|kar do|chahiye|mujhe|aap|bhai|ji|kaise|kitna|theek|accha|acha|dijiye|karein|hoga)\b", re.I)


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def is_auto_reply(msg: str) -> bool:
    return _any(AUTO_REPLY_PATTERNS, msg)


def classify(msg: str) -> str:
    m = (msg or "").strip()
    low = m.lower()
    if not m:
        return "empty"
    if is_auto_reply(m):
        return "auto_reply"
    if _any(OPT_OUT_PATTERNS, low):
        return "opt_out"
    if _any(HOSTILE_PATTERNS, low):
        return "hostile"
    if _any(OFF_TOPIC_PATTERNS, low):
        return "off_topic"
    for p, _ in LATER_PATTERNS:
        if re.search(p, low) and not re.search(r"let'?s|go ahead|do it", low):
            return "later"
    if _any(OBJECTION_PATTERNS, low) and not re.match(r"^(yes|ok|haan|sure)", low):
        return "objection"
    if _any(COMMIT_PATTERNS, low):
        return "commit"
    if re.match(r"^(no|nope|nah|nahi|na)\b", low):
        return "negative"
    if "?" in m or QUESTION_START.match(low):
        return "question"
    if re.search(r"\b(thanks|thank you|great|nice|good|awesome|interesting|cool|wow|shukriya|dhanyavaad)\b", low):
        return "positive"
    return "neutral"


def later_seconds(msg: str) -> int:
    low = (msg or "").lower()
    for p, secs in LATER_PATTERNS:
        if re.search(p, low):
            return secs
    return 14400


@dataclass
class ReplyDecision:
    action: str  # send | wait | end
    intent: str
    body: str = ""
    cta: str = "open_ended"
    rationale: str = ""
    wait_seconds: Optional[int] = None
    polish: bool = False  # strategy may rewrite the body (re-validated)
    commit: bool = False
    stage: Optional[str] = None
    flags: dict = field(default_factory=dict)


def _hing(msg: str, F: Optional[Facts]) -> bool:
    return bool(HINGLISH_MARKERS.search(msg or "")) or bool(re.search(r"[ऀ-ॿ]", msg or ""))


def _gerund(verb: str) -> str:
    verb = verb.strip()
    special = {"set up": "setting up", "pull": "pulling", "draft": "drafting", "share": "sharing", "start": "starting",
               "block": "blocking", "book": "booking", "hold": "holding", "turn": "turning", "reactivate": "reactivating",
               "dispatch": "dispatching", "confirm": "confirming", "help": "helping", "fix": "fixing", "send": "sending"}
    return special.get(verb, verb + "ing")


def build_artifact(F: Facts, conv: dict) -> str:
    """A concrete deliverable matching what Vera offered — so a 'yes' gets work, not questions."""
    if conv.get("artifact"):
        return conv["artifact"]
    proposal = (conv.get("proposal") or "").lower()
    offer = F.offers_active[0] if F.offers_active else (F.offer_or_catalog()[0] or "")
    loc = f", {F.locality}" if F.locality else ""
    if "post" in proposal:
        lead = offer or F.biz_name
        return (f"Post 1: \"{lead} at {F.biz_name}{loc}. Message us on WhatsApp to book your slot.\"\n"
                f"Post 2: \"Why {F.people} in {F.locality or 'the area'} choose {F.biz_name} — real reviews, clear prices, no surprises.\"")
    if "offer" in proposal:
        title = re.search(r"\"([^\"]+)\"", conv.get("proposal") or "")
        t = title.group(1) if title else offer
        return f"Offer ready: \"{t}\" — shown on your Google profile and magicpin listing for 30 days, with a WhatsApp booking button."
    if "review" in proposal:
        return f"\"Thank you for visiting {F.biz_name}! If we made your day a little better, a 1-line Google review would really help us 🙏\""
    if "verification" in proposal:
        return "Steps: 1) I request the verification call from Google, 2) you answer and read back the code, 3) I confirm it on the profile. That's it."
    if "renewal" in proposal:
        amt = F.tp.get("renewal_amount")
        plan = F.tp.get("plan") or F.sub.get("plan") or "current"
        return f"Renewal: {plan} plan{(' — ' + fmt_money(amt)) if amt else ''}. Same features, no change to your listing."
    if "list" in proposal:
        return "Pulling the matching customer list now and pairing it with a ready-to-send WhatsApp note for each."
    if "whatsapp" in proposal or "note" in proposal or "abstract" in proposal:
        d = F.digest_item or {}
        if d.get("summary"):
            return f"Draft for your {F.people}: \"{d.get('title')}. {d.get('summary').split('.')[0]}. Ask us at your next visit if this applies to you.\""
        return f"Draft: \"A quick update from {F.biz_name}{loc} — reply to this message and we'll help you with the next step.\""
    if "package" in proposal or "campaign" in proposal or "challenge" in proposal:
        return f"Draft plan: 1) the package headline + price anchored on \"{offer}\", 2) a Google post, 3) a WhatsApp broadcast to your past {F.people}." if offer else f"Draft plan: 1) package headline, 2) Google post, 3) WhatsApp broadcast to your past {F.people}."
    return f"Plan: 1) {proposal or 'the change we discussed'}, 2) go live on your profile, 3) I report back on results in 7 days."


def decide(msg: str, from_role: str, conv: dict, mstate: dict, F: Optional[Facts], turn_number: int) -> ReplyDecision:
    intent = classify(msg)
    canned = mstate.setdefault("canned", {})
    key = stable_hash(norm_text(msg), 12)
    repeat_count = canned.get(key, 0) + 1 if msg and msg.strip() else 0
    canned[key] = repeat_count
    # same text 2+ times from the same sender is almost always a canned reply
    if intent not in {"opt_out", "hostile"} and repeat_count >= 2 and len(norm_text(msg).split()) >= 4:
        intent = "auto_reply"

    name = ""
    if F is not None:
        name = F.salutation if from_role == "merchant" else ((F.c or {}).get("first") or "")
    hing = _hing(msg, F)
    sent_count = len(conv.get("bodies") or [])
    stage = conv.get("stage") or "pitch"
    proposal = conv.get("proposal") or "set this up for you"

    if conv.get("status") == "ended" and intent in {"opt_out", "hostile", "auto_reply", "empty"}:
        return ReplyDecision("end", intent, rationale="Conversation already closed; not re-opening.")

    if intent == "empty":
        return ReplyDecision("wait", intent, wait_seconds=3600, rationale="Empty message; waiting for a real reply.")

    if intent == "auto_reply":
        n = conv.get("auto_count", 0) + 1
        conv["auto_count"] = n
        n = max(n, repeat_count)
        if n <= 1:
            body = ("Lagta hai yeh auto-reply hai 🙂 Owner dekhein to bas 'YES' reply kar dein — main aage ka kaam sambhal lungi."
                    if hing else "Looks like an auto-reply 🙂 When the owner sees this, just reply YES and I'll take it from there.")
            return ReplyDecision("send", intent, body=body, cta="binary_yes_no",
                                 rationale="Detected WhatsApp Business auto-reply; one short flag for the owner, no repeat pitch.")
        if n == 2:
            return ReplyDecision("wait", intent, wait_seconds=86400,
                                 rationale="Same canned auto-reply again — owner isn't on the phone; backing off 24h.")
        return ReplyDecision("end", intent, rationale=f"Auto-reply {n}x with no human response; closing to avoid wasted turns.")

    if intent == "opt_out":
        mstate["opted_out"] = True
        return ReplyDecision("end", intent, rationale="Merchant asked to stop / not interested; closing and suppressing future sends.")

    if intent == "hostile":
        h = conv.get("hostile_count", 0) + 1
        conv["hostile_count"] = h
        if h >= 2 or re.search(r"stop|don'?t|never", msg or "", re.I):
            mstate["opted_out"] = True
            return ReplyDecision("end", intent, rationale="Merchant is frustrated; exiting gracefully and pausing outreach.")
        body = (f"Maaf kijiye{', ' + name if name else ''} — aage se sirf kaam ki baat hi bhejungi. Kabhi bhi STOP likh dein, main message band kar dungi."
                if hing else f"Sorry for the bother{', ' + name if name else ''} — I'll only message when there's something specific and useful. Reply STOP anytime and I won't message again.")
        return ReplyDecision("send", intent, body=body, cta="none",
                             rationale="Acknowledged frustration with an apology and a clear opt-out; no pitch.")

    if intent == "later":
        secs = later_seconds(msg)
        mstate["snooze_seconds"] = secs
        return ReplyDecision("wait", intent, wait_seconds=secs, rationale=f"Merchant asked for time; backing off {secs // 3600}h.")

    if sent_count >= 5:
        return ReplyDecision("end", intent, rationale="Conversation has run 5+ bot turns; closing politely to avoid fatigue.")

    if intent == "off_topic":
        o = conv.get("offtopic_count", 0) + 1
        conv["offtopic_count"] = o
        topic = "GST/tax filing" if re.search(r"gst|tax|itr", msg or "", re.I) else "that"
        expert = "your CA" if re.search(r"gst|tax|itr|account", msg or "", re.I) else "a specialist"
        if o >= 2:
            return ReplyDecision("wait", intent, wait_seconds=14400,
                                 rationale="Repeated off-topic asks; pausing rather than pushing.")
        back = f"Coming back to what I can do for you: {proposal} — reply YES and I'll get started."
        if hing:
            body = f"{topic} mein main help nahi kar paungi — iske liye {expert} best rahenge. Wapas apne kaam par: {proposal} — bas YES reply karein."
        else:
            body = f"I'll have to leave {topic} to {expert} — that's outside what I can help with. {back}"
        return ReplyDecision("send", intent, body=body, cta="binary_yes_no", polish=False,
                             rationale="Politely declined out-of-scope ask; redirected to the original thread with one CTA.")

    if intent == "commit" or (intent == "positive" and stage == "pitch" and re.search(r"\b(yes|ok|sure|haan)\b", msg or "", re.I)):
        conv["commit_count"] = conv.get("commit_count", 0) + 1
        if F is None:
            body = "Great — starting now. Here's the next step: I'll set it up and send it here for your final CONFIRM."
            return ReplyDecision("send", "commit", body=body, cta="binary_confirm_cancel", commit=True, stage="drafted",
                                 rationale="Explicit go-ahead; switched to action mode.")
        if from_role == "customer":
            slot = None
            if re.match(r"^\s*1\b", msg or ""):
                slot = 0
            elif re.match(r"^\s*2\b", msg or ""):
                slot = 1
            slots = [s.get("label") for s in (F.tp.get("available_slots") or F.tp.get("next_session_options") or []) if isinstance(s, dict)]
            chosen = slots[slot] if slot is not None and slot < len(slots) else (slots[0] if len(slots) == 1 else None)
            if chosen:
                body = (f"Confirmed ✅ {chosen} at {F.biz_name}. Aapko ek din pehle reminder bhej denge."
                        if (F.c or {}).get("mode") in {"hinglish", "hindi"} else
                        f"Confirmed ✅ {chosen} at {F.biz_name}. We'll send you a reminder a day before.")
            else:
                body = (f"Done ✅ {F.biz_name} team aapki request confirm karke isi chat par bata degi."
                        if (F.c or {}).get("mode") in {"hinglish", "hindi"} else
                        f"Done ✅ The {F.biz_name} team is on it — you'll get the confirmation right here.")
            return ReplyDecision("send", "commit", body=body, cta="none", commit=True, stage="done",
                                 rationale="Customer confirmed; closing the booking loop with a clear confirmation.")
        if stage == "pitch":
            art = build_artifact(F, conv)
            verb = proposal.split(" ")[0] if proposal else "set"
            if hing and F.slug in {"salons", "restaurants", "pharmacies"}:
                body = f"Done — yeh raha draft:\n\n{art}\n\nReply CONFIRM karein aur main ise live kar dungi (changes chahiye to bas likh dein)."
            else:
                body = f"Done — here's the draft:\n\n{art}\n\nReply CONFIRM and I'll put it live (or send edits and I'll update it)."
            return ReplyDecision("send", "commit", body=body, cta="binary_confirm_cancel", commit=True, stage="drafted", polish=True,
                                 rationale=f"Merchant committed; switched from pitch to action — delivered the {verb} deliverable with a single CONFIRM step.")
        if stage == "drafted":
            body = (f"Live ✅ Sab set hai, {name}. Main 7 din mein results (views, calls) yahin share karungi."
                    if hing and F.slug in {"salons", "restaurants", "pharmacies"} else
                    f"Live ✅ All set, {name}. I'll share the results (views and calls) here in 7 days.")
            return ReplyDecision("send", "commit", body=body, cta="none", commit=True, stage="done",
                                 rationale="Merchant confirmed; executed and set a follow-up expectation.")
        return ReplyDecision("end", "commit", rationale="Task already completed; nothing more to push in this thread.")

    if intent == "objection":
        ob = conv.get("objection_count", 0) + 1
        conv["objection_count"] = ob
        if ob >= 2:
            return ReplyDecision("end", intent, rationale="Second objection; respecting the merchant's call and closing.")
        fact = ""
        if F is not None and F.anchors:
            top = sorted(F.anchors, key=lambda a: -a.strength)[0]
            fact = f" For context, {top.text}."
        body = (f"Fair point.{fact} I can prepare it as a draft only — nothing goes live without your OK. Reply YES if you want to see it."
                if not hing else f"Sahi baat hai.{fact} Main sirf draft bana deti hoon — aapke OK ke bina kuch live nahi hoga. Dekhna ho to YES reply karein.")
        return ReplyDecision("send", intent, body=body, cta="binary_yes_no", polish=True,
                             rationale="Handled objection once with a grounded fact and a zero-risk draft offer.")

    if intent == "negative":
        return ReplyDecision("end", intent, rationale="Merchant declined; closing without pushing.")

    if intent == "question":
        body = answer_question(msg, F, conv, hing)
        return ReplyDecision("send", intent, body=body, cta="binary_yes_no", polish=True,
                             rationale="Answered the merchant's question from context, then one CTA to move forward.")

    # positive / neutral
    if F is not None and stage == "pitch":
        body = (f"Great — I can {proposal} right away. Reply YES and I'll send you the draft right here."
                if not hing else f"Badhiya! Main abhi {proposal} — bas YES reply karein, draft yahin bhej dungi.")
    else:
        body = "Noted 👍 I'll keep an eye on your numbers and ping you only when there's something worth acting on."
        return ReplyDecision("send", intent, body=body, cta="none", rationale="Acknowledged; nothing pending to push.", stage=stage)
    return ReplyDecision("send", intent, body=body, cta="binary_yes_no", polish=True,
                         rationale="Positive signal; moved to a single low-effort next step.")


def answer_question(msg: str, F: Optional[Facts], conv: dict, hing: bool) -> str:
    low = (msg or "").lower()
    proposal = conv.get("proposal") or "set it up"
    if F is None:
        return f"Good question — I'll check and come back with specifics. Meanwhile, want me to {proposal}? Reply YES."
    ans = ""
    if re.search(r"price|cost|charge|fee|kitna|kitne|how much|paisa|rate", low):
        amt = F.tp.get("renewal_amount")
        if amt:
            ans = f"It's {fmt_money(amt)} for the {F.tp.get('plan') or F.sub.get('plan') or 'plan'}"
        elif F.offers_active:
            ans = f"Your live offer is \"{F.offers_active[0]}\" — drafting and posting it costs you nothing extra"
        else:
            ans = "Drafting this costs you nothing extra — it's part of what I do on your profile"
    elif re.search(r"how long|kitna time|when|kab|time", low):
        ans = "The draft takes me a few minutes; once you say CONFIRM it goes live the same day"
    elif re.search(r"source|proof|kaise pata|how do you know|data", low):
        d = F.digest_item or {}
        if d.get("source"):
            ans = f"It's from {d['source']}: {d.get('title')}"
        elif F.anchors:
            ans = f"From your own dashboard: {F.anchors[0].text}"
    elif re.search(r"what|kya|details|more|explain|samjha", low):
        d = F.digest_item or {}
        if d.get("summary"):
            ans = d["summary"].split(". ")[0]
        elif F.anchors:
            ans = f"Here's the key number: {sorted(F.anchors, key=lambda a: -a.strength)[0].text}"
    if not ans:
        top = sorted(F.anchors, key=lambda a: -a.strength)[0].text if F.anchors else f"{F.biz_name} has room to grow on Google"
        ans = f"Short answer: {top}"
    if hing and F.slug in {"salons", "restaurants", "pharmacies"}:
        return f"{ans}. Aage badhein? {proposal} — bas YES reply karein."
    return f"{ans}. Want me to go ahead and {proposal}? Reply YES."
