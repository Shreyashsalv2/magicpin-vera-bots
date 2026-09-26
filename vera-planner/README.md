# vera-planner — "plan, draft three ways, judge, pick"

Public bot URL: https://vera-planner-two.vercel.app. It exposes `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`, and optionally `/v1/teardown`.

## Approach
1. **Facts, not prose.** Shared with the sister bot: a grounded fact sheet, a NumberBank of every allowed number, and merchant anchors with strength scores. Each trigger kind has a plan (why-now hooks for three levers, detail lines, the concrete proposal, and the CTA type).
2. **Plan.** One Groq `openai/gpt-oss-120b` call sees *all* the top candidate merchant signals. It writes an explicit plan (which signal, why now, hook, what Vera will do, CTA) and then **three drafts**, one per compulsion lever: proof, urgency and curiosity.
3. **Filter.** Every draft passes the grounding validator. Drafts are dropped for invented numbers or names, URLs, taboo words, more than one CTA, a preamble, or a missing salutation.
4. **Judge and pick.** Survivors are scored by a deterministic rubric that mirrors the judge's five dimensions. A second Groq call then acts as a critic on the same dimensions and picks the winner. The critic is overruled only if the rubric rates its pick clearly worse.
5. **Deterministic.** Temperature 0, a fixed seed and a response cache mean the same input gives the same output.

## Conversations (`/v1/reply`)
The shared deterministic state machine decides send/wait/end:
- **Auto-reply:** flag once → wait 24h → end.
- **Opt-out or hostile:** end.
- **Commit:** go straight to action mode and deliver the draft.
- **Off-topic:** decline and redirect.
- **Objection:** reply once.
- **Later:** wait.

For `send` bodies the planner writes two candidate replies. Both are validated, and the rubric picks one.

## Reliability and fallback
- If Groq is missing, rejects the key, rate-limits, times out or returns bad JSON, the bot runs the **same process deterministically**. Every (lever × top-3 signal) template variant is generated, validated and rubric-scored, and the best one is sent.
- Circuit breaker, secondary model (`openai/gpt-oss-20b`), an 8s tick budget, Upstash Redis state with atomic version CAS, suppression, consent and per-recipient dedupe are all shared with vera-signal.

## Tradeoffs
- It makes 2 LLM calls per message, which means more reasoning but more latency and quota. The deadline logic falls back to the rubric-picked template whenever time runs short.
- Lever diversity sometimes wins on engagement, even when it scores lower on specificity.

## What extra context would help most
Real slot availability, merchant price lists beyond the active offers, and per-merchant historical reply rates by lever.
