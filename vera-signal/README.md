# vera-signal — "rank the signal, write once"

Public bot URL: https://vera-signal.vercel.app. It exposes `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`, and optionally `/v1/teardown`.

## Approach
1. **Facts, not prose.** The four contexts (category, merchant, trigger, customer) are turned into a fact sheet. Every number in the contexts, plus every number derived from them (day counts, %, tier prices), goes into a *NumberBank*. Merchant "anchors" are computed with a strength score: CTR vs peer median, 7-day deltas, stale posts, review themes, lapsed and high-risk cohorts, live offer or offer gap, renewal, and verification.
2. **Trigger plan.** Each trigger kind (research digest, compliance, perf dip/spike, competitor, festival, IPL, recall, refill, winback, curious-ask …, plus a generic handler for unknown kinds) has a plan. A plan holds a why-now hook, optional detail lines, one concrete thing Vera offers to do, and the CTA type.
3. **Rank.** A deterministic ranker scores `anchor strength × relevance to the trigger kind`. It drops anchors that only repeat the hook, then keeps the single strongest one.
4. **Write once.** One Groq `openai/gpt-oss-120b` call writes the message from only that trigger + anchor + proposal. It runs at temperature 0 with a fixed seed and a response cache, so the same input gives the same output.
5. **Gate.** The grounding validator rejects any draft that has an invented number or proper noun, a URL, a taboo word, more than one CTA, a preamble, or a missing salutation. It allows one repair retry, then falls back to the ranked template.

## Conversations (`/v1/reply`)
A deterministic state machine decides send/wait/end:
- **Auto-reply:** flag once → wait 24h → end. Detection works per merchant and across conversations.
- **Opt-out or hostile:** end, and suppress the merchant from future ticks.
- **"Yes / let's do it":** switch to action mode immediately and deliver the draft, with a single CONFIRM step. No re-qualifying.
- **Off-topic (e.g. GST):** decline politely and redirect.
- **Objection:** reply once with a grounded fact and a zero-risk "draft only" offer.
- **Later:** wait.
- **Questions:** answered from the contexts.

The LLM may only reword `send` bodies, and its rewrite is re-validated.

## Reliability and fallback
- If Groq is missing, rejects the key, rate-limits, times out or returns bad JSON, the bot drops to the deterministic template engine within the same request.
- A circuit breaker stops calling Groq after 3 failures (60s cooldown; 10 min on an auth error).
- An 8s tick budget and a 7.5s reply budget keep responses well under the judge's 10s/30s limits.
- State lives in Upstash Redis on Vercel, because instances are not shared, with an in-memory fallback. Context pushes use an atomic compare-and-set for version idempotency.
- Suppression keys, per-recipient dedupe (one message per merchant/customer per tick), consent checks and snooze-after-wait are all enforced.

## Tradeoffs
- One LLM call per message keeps latency and rate-limit usage low. Quality therefore depends on the ranker choosing the right signal.
- Templates are conservative, so they never invent a fact. The LLM adds fluency, not facts.

## What extra context would help most
Real slot availability, merchant price lists beyond the active offers, and per-merchant historical reply rates by message type.
