"""Local end-to-end harness for both bots (in-process, no network needed).

usage: python tests/harness.py signal|planner [--now 2026-04-26T10:30:00Z] [--quiet] [--url http://host]
Loads the expanded dataset, runs the 30 canonical test pairs through /v1/tick,
validates + rubric-scores each message, then replays the judge's conversation
scenarios through /v1/reply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "expanded"


def load_dataset():
    cats = {p.stem: json.loads(p.read_text()) for p in (EXP / "categories").glob("*.json")}
    ms = {json.loads(p.read_text())["merchant_id"]: json.loads(p.read_text()) for p in (EXP / "merchants").glob("*.json")}
    cs = {json.loads(p.read_text())["customer_id"]: json.loads(p.read_text()) for p in (EXP / "customers").glob("*.json")}
    ts = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in (EXP / "triggers").glob("*.json")}
    pairs = json.loads((EXP / "test_pairs.json").read_text())["pairs"]
    return cats, ms, cs, ts, pairs


def make_client(bot: str, url: str | None):
    sys.path.insert(0, str(ROOT / f"vera-{bot}"))
    if url:
        return httpx.AsyncClient(base_url=url, timeout=35)
    from vera.api import create_app
    import strategy as strat  # type: ignore
    cls = strat.SignalStrategy if bot == "signal" else strat.PlannerStrategy
    app = create_app(cls())
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bot", timeout=35), app


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot", choices=["signal", "planner"])
    ap.add_argument("--now", default="2026-04-26T10:30:00Z")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    res = make_client(args.bot, args.url)
    client, app = (res, None) if args.url else res
    cats, ms, cs, ts, pairs = load_dataset()
    if args.pairs:
        keep = set(args.pairs.split(","))
        pairs = [p for p in pairs if p["test_id"] in keep]

    r = await client.get("/v1/healthz"); print("healthz", r.status_code, r.json())
    r = await client.get("/v1/metadata"); print("metadata", r.status_code, json.dumps(r.json())[:300])

    t0 = time.time()
    async def push(scope, cid, payload, v=1):
        return await client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": v,
                                                       "payload": payload, "delivered_at": "2026-04-26T09:45:00Z"})
    for slug, c in cats.items():
        await push("category", slug, c)
    await asyncio.gather(*[push("merchant", k, v) for k, v in ms.items()])
    await asyncio.gather(*[push("customer", k, v) for k, v in cs.items()])
    await asyncio.gather(*[push("trigger", k, v) for k, v in ts.items()])
    print(f"pushed all contexts in {time.time() - t0:.1f}s")
    r = await client.get("/v1/healthz"); print("healthz", r.json())
    # idempotency checks
    r = await push("category", "dentists", cats["dentists"]); print("re-push same version ->", r.status_code, r.json())
    r = await push("category", "dentists", cats["dentists"], 2); print("push v2 ->", r.status_code, r.json())
    r = await client.post("/v1/context", json={"scope": "bogus", "context_id": "x", "version": 1, "payload": {}})
    print("bad scope ->", r.status_code, r.json())

    # scoring helpers from the bot's own package
    from vera.facts import Facts
    from vera.util import parse_dt
    from vera.validate import rubric, validate
    from vera.kinds import build_plan

    totals, issues_all, latencies, sources, records = [], 0, [], {}, []
    for p in pairs:
        tr = ts[p["trigger_id"]]
        t1 = time.time()
        r = await client.post("/v1/tick", json={"now": args.now, "available_triggers": [p["trigger_id"]]})
        latencies.append(time.time() - t1)
        acts = r.json().get("actions", [])
        if not acts:
            print(f"\n[{p['test_id']}] {tr['kind']} -> NO ACTION")
            continue
        a = acts[0]
        records.append({"test_id": p["test_id"], "body": a["body"], "cta": a["cta"], "send_as": a["send_as"],
                        "suppression_key": a["suppression_key"], "rationale": a["rationale"]})
        m = ms[p["merchant_id"]]
        F = Facts(cats[m["category_slug"]], m, tr, cs.get(p.get("customer_id")) if p.get("customer_id") else None,
                  parse_dt(args.now))
        scope = "customer" if a["send_as"] == "merchant_on_behalf" else "merchant"
        build_plan(F)  # registers numbers the plan legitimately derives (tiers, day counts)
        iss = validate(F, a["body"], a["cta"], scope=scope)
        rb = rubric(F, a["body"], a["cta"], scope)
        totals.append(rb["total"])
        issues_all += bool(iss)
        src = "llm" if "groq" in a["rationale"] or "plan:" in a["rationale"] or "signal=" in a["rationale"] else "template"
        sources[src] = sources.get(src, 0) + 1
        if not args.quiet:
            print(f"\n[{p['test_id']}] {tr['kind']} | {m['identity']['name']} | cta={a['cta']} send_as={a['send_as']} "
                  f"| {latencies[-1]:.2f}s | rubric={rb['total']} | {src}")
            print("  " + a["body"].replace("\n", "\n  "))
            print("  rationale:", a["rationale"][:220])
            if iss:
                print("  ISSUES:", iss)
    print(f"\n== {len(totals)}/{len(pairs)} messages, avg rubric {sum(totals) / max(1, len(totals)):.1f}/50, "
          f"with issues: {issues_all}, max tick latency {max(latencies):.2f}s, sources {sources}")

    if args.jsonl:
        Path(args.jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        print("wrote", args.jsonl)

    # replay: re-tick same trigger must be suppressed
    r = await client.post("/v1/tick", json={"now": args.now, "available_triggers": [pairs[0]["trigger_id"]]})
    print("re-tick same trigger ->", len(r.json()["actions"]), "actions (expect 0)")

    # full multi-turn on a real tick conversation
    print("\n-- multi-turn on research digest conversation --")
    await push("trigger", "trg_001_research_digest_dentists", ts["trg_001_research_digest_dentists"], 5)
    await push("merchant", "m_001_drmeera_dentist_delhi", ms["m_001_drmeera_dentist_delhi"], 5)
    r = await client.post("/v1/tick", json={"now": args.now, "available_triggers": ["trg_001_research_digest_dentists", "trg_022_cde_webinar_dentists"]})
    acts = r.json()["actions"]
    print("tick ->", [(a["trigger_id"], a["conversation_id"]) for a in acts])
    if acts:
        conv = acts[0]["conversation_id"]
        for turn, msg in enumerate(["What's the source for this?", "Btw can you also help me with my GST filing this month?",
                                    "Yes please send the abstract. Also draft the patient WhatsApp.", "CONFIRM",
                                    "thanks"], start=2):
            r = await client.post("/v1/reply", json={"conversation_id": conv, "merchant_id": "m_001_drmeera_dentist_delhi", "from_role": "merchant",
                                                     "message": msg, "received_at": "2026-04-26T10:45:00Z", "turn_number": turn})
            print(f"  M: {msg}\n  B: {json.dumps(r.json(), ensure_ascii=False)[:600]}")
    # conversation scenarios
    mid = "m_001_drmeera_dentist_delhi"
    print("\n-- auto-reply hell (judge simulator style: new conv each turn) --")
    for i in range(1, 5):
        r = await client.post("/v1/reply", json={"conversation_id": f"conv_auto_{i}", "merchant_id": mid, "customer_id": None,
                                                 "from_role": "merchant", "message": "Thank you for contacting us! Our team will respond shortly.",
                                                 "received_at": "2026-04-26T10:42:00Z", "turn_number": i + 1})
        print(i, r.json())
    print("\n-- intent transition --")
    r = await client.post("/v1/reply", json={"conversation_id": "conv_intent_1", "merchant_id": "m_003_studio11_salon_hyderabad",
                                             "from_role": "merchant", "message": "Ok lets do it. Whats next?",
                                             "received_at": "2026-04-26T10:42:00Z", "turn_number": 2})
    print(r.json())
    print("\n-- hostile --")
    r = await client.post("/v1/reply", json={"conversation_id": "conv_hostile", "merchant_id": "m_005_pizzajunction_restaurant_delhi",
                                             "from_role": "merchant", "message": "Stop messaging me. This is useless spam.",
                                             "received_at": "2026-04-26T10:42:00Z", "turn_number": 2})
    print(r.json())

    print("\n-- customer booking reply --")
    r = await client.post("/v1/tick", json={"now": args.now, "available_triggers": ["trg_003_recall_due_priya"]})
    print("tick recall (already sent earlier? expect 0 or 1):", len(r.json()["actions"]))
    r = await client.post("/v1/reply", json={"conversation_id": "conv_priya_x", "merchant_id": mid, "customer_id": "c_001_priya_for_m001",
                                             "from_role": "customer", "message": "1", "received_at": "2026-04-26T11:05:00Z", "turn_number": 2})
    print(r.json())
    r = await client.get("/v1/healthz"); print("\nhealthz", r.json())
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
