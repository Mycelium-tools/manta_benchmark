# Experiments

Standalone research tools; the benchmark entry points and defaults remain in
`src/manta`. Run these commands from the repository root with `OPENROUTER_API_KEY`
in the environment or root `.env`. Add `--dry-run` to judging commands to inspect
exact inputs without paying for calls.

## Judge existing conversations

```bash
uv run python -m experiments.judge_experiment \
  --model openrouter/z-ai/glm-5.3-flash --items-v2 --reasoning-effort high

uv run python -m experiments.judge_experiment \
  --model openrouter/typesafe/jev-1.13 --jev-rubric decisions-v5 \
  --decision-context prefix
```

Chat judges default to 50 conversations, three repeats, concurrency 32, and a
32,768-token output cap. The default rubric has three AWVS items; `--items-v2`
selects the revised four-item rubric. Both score Turns 3–5 in one call, with
visible text only and no system prompt. Each item is kept separate.

Jev defaults to the reference-based v4 rubric, 100 conversations (IDs 051–150),
and one repeat. `decisions-v5` uses the shorter reference-based rubric tested
most recently. `--decision-context prefix` sends only Turns 1–3, 1–4, or 1–5 for
each target; the default is all five turns. Binary decisions use a 0.5 threshold;
choices use the highest probability. Raw probabilities are saved. Uncertain or
conflicting components are not silently converted into a scalar score.

Chat requests accept `--providers baseten fireworks` and `--reasoning-effort high`.
Flex is requested by default, without changing the model ID; `--service-tier default`
opts out. Models without a flex endpoint may use standard rates. Claude requests
through OpenRouter include a cache breakpoint before the conversation. These chat
settings do not apply to the Decisions endpoint.

Use `--input-file PATH` to judge any frozen conversation JSON with the same schema:
an object containing `conversations`, each with a unique `conversation_id`,
comma-separated `pressure_types`, and string `user_turn_1` / `assistant_turn_1`
through `user_turn_5` / `assistant_turn_5` fields. For a reference-based Jev rubric,
also pass the matching `--reference-standards PATH`.

```bash
uv run python -m experiments.judge_experiment \
  --model openrouter/typesafe/jev-1.13 --jev-rubric decisions-v5 \
  --input-file judge_experiments/leaderboard100_seed1_fresh_scenarios.json \
  --reference-standards experiments/reference_standards/leaderboard100_seed1_fresh_scenarios_standards_v1.json \
  --decision-context prefix
```

The last command requires the locally saved fresh-100 sample. The two committed
reference files are provisional, manually reviewed scenario annotations, not
expert-validated gold answers. They are bound to exact initial-scenario hashes.

The default sampler has `SEED = 1` hardcoded and reads the verified non-trial logs
in [leaderboard_sources.json](leaderboard_sources.json). This is the September 8,
2026 leaderboard snapshot, not a live website query. All source logs are already
in the repository. Generated samples and outputs stay in the ignored
`judge_experiments/` directory; existing frozen samples are reused unchanged.

Each run saves exact inputs/settings, `results.jsonl`, `summary.json`, and an
OpenRouter cost ledger. Failed or incomplete responses are not zero scores.
Decisions runs support `--resume OUTPUT_DIRECTORY`, preserving completed requests
and archived references. Historical rubric variants remain selectable for
reproduction; `decisions-v6` and `decisions-v7` are experimental candidates, not
recommended replacements for v5.

For older comparisons, `--v2` selects scalar AWMS plus joint AWVS;
`--v2 --no-calibration` removes examples. `--humanjudges` uses the local 12-case
validation export; `--humanjudges --v1` replays its original logged prompts,
including any original reasoning exposure. Those collaborator files are not
committed here. Experiment-only prompts live in `experiments/prompts`; v3 and v5
Jev rubrics are shared with the benchmark under `src/manta/prompts`.

## Reports and balance

```bash
uv run python -m experiments.judge_experiment_report
uv run python -m experiments.openrouter_balance
```

The report tool reads the original 50-sample, three-item runs and writes
`judge_experiments/awvs_items_comparison.md`, with repeats listed after each target
turn. Other rubric results are available in their run's JSON files. Neither
command generates model completions.

## English / simplified-Chinese pilot

```bash
uv run python -m experiments.language_experiment                     # prepare only
uv run python -m experiments.language_experiment --run --phase smoke # one pair
uv run python -m experiments.language_experiment --run --phase main
uv run python -m experiments.language_experiment --run --phase control
```

[chinese_inputs.json](chinese_inputs.json) contains the 12 paired scenarios.
The target, pressure model, providers and price caps are explicit constants in
`language_experiment.py`. Outputs default to `language_experiments/chinese_pilot/`;
use a new `--output` directory when changing inputs or the protocol. `--budget`
sets that pilot directory's cumulative spending cap (default $8). Existing calls
are reused, and reasoning is excluded from subsequent conversation turns.

## Offline checks

```bash
uv run python -m pytest tests experiments/tests
```

The tests cover original prompt compatibility, reasoning exclusion, parsing,
Inspect execution/export, reference matching, and paid-request reuse. Saved
research outputs and historical scripts in `notes/` are local artifacts, not
required by these tools.
