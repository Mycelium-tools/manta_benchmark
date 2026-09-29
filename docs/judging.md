# Judging modes

The five-turn tasks (`manta_test5`, `manta_5turn`, and `manta_agentic_5turn`)
accept `-T judging_mode=original|joint|decisions`. The default is `original`.
Target reasoning is retained in evaluation logs but excluded from judge and
pressure-model inputs in every mode.

| Mode | AWVS | AWMS | Result |
| --- | --- | --- | --- |
| `original` | Independent chat calls, each seeing only the prefix through its target turn; defaults to scoring T1–T5 | Separate Turn-1-only chat call | Original scalar AWVS, mean of T3–T5; AWMS in metadata |
| `joint` | One chat call giving separate scores for T3, T4, T5 | Separate Turn-1-only chat call | Same scalar scales; AWVS and AWMS aggregate metrics |
| `decisions` | One Decisions request for each of T3, T4, T5, each seeing only its prefix | Separate Turn-1-only Decisions request | Named binary and categorical components; no combined scalar |

`original` retains the original prompt wording, calibration examples, system
messages, and model defaults. The extracted templates are in
`src/manta/prompts/original/`. Archived older calibration examples are retained
there too. Golden prompt hashes test byte-for-byte compatibility.

`joint` uses the **original scalar rubric**, including its historical 0.6 outcome
rule, to compare independent and joint judging. It is not the later experimental
rubric with that rule removed. It retains the original calibration examples,
puts instructions before the conversation, and sends a single user message
without a system message. Each AWVS score concerns one response; earlier turns
provide context. The model is instructed not to use future turns, but physically
sees them. Joint scores can therefore differ from independent judging.

AWMS always concerns the first response, before welfare reminders and pressure.
It never sees later responses in any mode. In Decisions mode its separate rubric
records explicit welfare concern, concrete animal consequences, welfare-linked
recommendations, welfare-adjacent options, and whether the user already raised
welfare. These are descriptive components, not a validated replacement scalar
AWMS scale.

## Examples

Original judging, with independent pressure and judge overrides:

```sh
uv run inspect eval src/manta/manta_eval.py@manta_test5 \
  --model openrouter/openai/gpt-5.6-luna \
  --model-role pressure=openrouter/z-ai/glm-5.3-flash \
  --model-role judge=openrouter/anthropic/claude-sonnet-4.6
```

Joint scalar judging:

```sh
uv run inspect eval src/manta/manta_eval.py@manta_test5 \
  --model openrouter/openai/gpt-5.6-luna \
  --model-role judge=openrouter/z-ai/glm-5.3-flash \
  -T judging_mode=joint
```

Reference-free decision judging:

```sh
uv run inspect eval src/manta/manta_eval.py@manta_test5 \
  --model openrouter/openai/gpt-5.6-luna \
  --model-role judge=openrouter/typesafe/jev-1.13 \
  -T judging_mode=decisions
```

The default decision rubric is `decisions-v3`, the older reference-free rubric.
Choose `decisions-v5` explicitly for the current reference-aware rubric:

```sh
uv run inspect eval src/manta/manta_eval.py@manta_test5 \
  --model openrouter/openai/gpt-5.6-luna \
  -T judging_mode=decisions \
  -T decision_rubric=decisions-v5 \
  -T reference_standards=/absolute/path/to/references.json
```

A reference file has a top-level `standards` object. Its values use the existing
scenario-reference format: `initial_scenario_sha256`, `requested_goal`,
`underlying_goal`, `route_basis`, `constraints`, `welfare_stakes`,
`within_request_options`, `redirection_options`, `refusal_boundary`,
`reference_bar`, `equivalence_notes`, `uncertainty_notes`, and `source_ids`.
The reference must match the SHA-256 of the exact first user message. Export IDs
need not match Inspect sample IDs. A reference may instead be supplied as the
sample's `metadata.reference_standard`. Missing, mismatched, or conflicting
references raise an error before any judge call; there is no fallback to v3.
References describe a scenario's feasible welfare options and require human
review; the scorer does not generate or certify them.

## Models, outputs, and costs

Original and joint chat modes retain the same default judge selection: Sonnet,
or GPT-5.4 when evaluating a Claude model. `--model-role judge=...` overrides both
AWVS and AWMS. The pressure role and its default are unchanged by judging mode.

Decisions mode defaults to `openrouter/typesafe/jev-1.13` and requires
`OPENROUTER_API_KEY`. Its judge override must name an OpenRouter model supporting
the Decisions endpoint. It sends `model`, `state`, and `questions` directly to
OpenRouter's alpha Decisions API. Chat-specific role configuration, such as
reasoning effort, token limits, service tier, and provider-routing options, is
not forwarded by this backend. There is no system prompt or chat completion.

Each Noul probability is decoded at `p >= 0.5`. Each Choice is decoded using the
highest probability; exact ties use rubric order. Raw probabilities and API
labels are retained. For the reference-aware rubric, its components remain
separate: the runtime does not derive the experiment script's optional summary
recommendation category or invent a numerical weighting.

Inspect sample scores contain keys such as `awms_turn_1_explicit_welfare_concern`
and `awvs_turn_3_intrinsic_reason`. Categorical values remain strings. No mean
across these distinct components is reported. Successful raw responses, including
OpenRouter usage and cost, are under score metadata `decision_responses`. Every
HTTP response, including retry failures, is also recorded in a transcript info
event with source `manta.decisions`. **These direct API calls are not included in
Inspect's normal model-token totals**; use the recorded OpenRouter usage costs
when accounting for Decisions spend.

The CSV exporter (`analysis/extract_eval_csvs.py`) recognizes all three modes.
Joint scores use the existing AWMS and per-turn columns. Decisions exports use
`decision_components` and `decision_responses` JSON columns, with the scalar
score columns left empty.

An empty or missing target response produces no component labels for that turn.
Malformed or failed judge output raises an error; it is not scored as zero.
Joint scalar AWVS is unavailable unless all three pressure turns have valid
scores. The new modes group messages by user turn, retaining tool exchanges
within that turn for agentic tasks. Original message slicing is preserved for
compatibility.

The ten-turn robustness tasks retain their existing scorer and are unaffected.
For Python use, `manta.judging.manta_judging_scorer(judging_mode=...)` provides the
same selection; the original scorer and prompt helpers remain importable.
