# MANTA Quick Start — Running Evals on a New Model

This guide gets you from a fresh clone to a full eval run on a new frontier model (e.g. a new Gemini release). No prior experience with [Inspect](https://inspect.aisi.org.uk/) (the eval framework we use) is needed.

## What MANTA does (30 seconds)

MANTA tests whether a model maintains animal-welfare reasoning under multi-turn adversarial pressure. Each sample is a 5-turn conversation: Turn 1 is an implicit scenario, Turn 2 names the welfare angle, Turns 3–5 push back against it (economic/social/pragmatic/epistemic/cultural pressure). Two scores come out: **AWMS** (did the model spontaneously flag welfare on Turn 1?) and **AWVS** (did it hold its position under pressure on Turns 3–5? — this is the headline metric).

## 1. One-time setup

```bash
git clone <this repo> && cd manta_benchmark
uv sync                      # installs everything (requires Python 3.12+ and uv)
```

Create a `.env` file in the repo root (gitignored, so you must create it yourself):

```
ANTHROPIC_API_KEY=...   # always required (internal follow-up writer + judges)
OPENAI_API_KEY=...      # always required (judge panel)
GEMINI_API_KEY=...      # always required (judge panel)
MISTRAL_API_KEY=...     # only if evaluating Mistral models
HF_TOKEN=...            # for dataset sync
XAI_API_KEY=...         # only if evaluating Grok models
```

The first three are required no matter which model you evaluate — MANTA uses Claude/GPT/Gemini internally to generate the adversarial follow-ups and score the responses.

Set your name for log routing (logs go to `logs/<YOUR_NAME>_<Month><Year>/`):

```bash
echo 'export MANTA_USER=YOUR_NAME' >> ~/.zshrc
source ~/.zshrc
```

Pull the latest question dataset:

```bash
python sync_questions_to_hf.py
```

## 2. Add the new model

Model names use Inspect's `provider/model-id` format:

| Provider | Example |
|---|---|
| Google | `google/gemini-3.5-flash` |
| Anthropic | `anthropic/claude-opus-4-7` |
| OpenAI | `openai/gpt-5.5` |
| xAI | `grok/grok-4.3` |
| Mistral | `mistral/mistral-small-2603` |
| Anything on OpenRouter | `openrouter/meta-llama/llama-3.3-70b-instruct` |

The `model-id` part is the exact API model string from the provider's docs.

Open `src/manta/manta_eval.py` and find the `MODELS` list (~line 405). Add your new model and comment out the ones you don't want to run:

```python
MODELS = [
    "google/gemini-4-pro",          # <-- your new model
    # "anthropic/claude-opus-4-7",
    # "openai/gpt-5.5",
    ...
]
```

This list only controls `python src/manta/manta_eval.py` runs. You can also skip editing it entirely and pass the model on the command line with `--model` (see below).

### Reasoning models: decide reasoning on/off explicitly

MANTA runs don't set any reasoning config by default, so each model uses its **provider default** — which means some models reason and some don't, and that difference shows up in the results. Reasoning is not just a speed knob: a model that thinks before answering can behave very differently under Turns 3–5 pressure, so treat on/off as an experimental condition and record it with the run.

What the provider defaults gave us in past runs (from reasoning-token usage in the logs):

| Model | Reasoning at provider default |
|---|---|
| GPT-5.5, Grok 4.3, Gemini 3.1 Pro / 3.5 Flash, DeepSeek v4 (flash & pro) | **On** |
| Claude Opus 4.7 / Opus 5, Gemini 3.1 Flash Lite, Mistral Small, Llama 3.3 | **Off** (or n/a — non-reasoning model) |

To force reasoning off (where the provider supports it):

```bash
inspect eval src/manta/manta_eval.py@manta_5turn --model openai-api/deepseek/deepseek-v4-pro \
  --reasoning-effort none --log-dir logs/YOUR_NAME_Month2026
```

To force it on for Claude models, use `--reasoning-tokens <budget>` (e.g. `--reasoning-tokens 4096`). Note reasoning roughly doubles per-sample latency and token cost on some models.

## 3. Smoke test first (~5 questions, a few minutes)

```bash
inspect eval src/manta/manta_eval.py@manta_test5 --model google/gemini-4-pro
```

If this completes without errors, credentials and the model name are correct.

## 4. Full run

```bash
# Everything in MODELS, isolated into its own timestamped log subdirectory
python src/manta/manta_eval.py --full-run gemini4

# Or a single model via Inspect directly (full 5-turn eval)
inspect eval src/manta/manta_eval.py@manta_5turn --model google/gemini-4-pro \
  --log-dir logs/YOUR_NAME_Month2026

# Or just a slice of the dataset (questions 0–50)
python src/manta/manta_eval.py --sample-range 0 50
```

> **Note on log locations:** the `MANTA_USER` routing (logs going to `logs/<YOUR_NAME>_<Month><Year>/` automatically) only applies to the `python src/manta/manta_eval.py ...` and `run_single_eval.py` commands. The `inspect eval` CLI bypasses it and writes to plain `./logs/` — pass `--log-dir` explicitly as shown above to keep your logs organized.

A full run takes a while (hundreds of 5-turn conversations, each scored by LLM judges). It's safe to leave running; results stream to disk as `.eval` files.

## 5. Look at the results

```bash
inspect view    # opens a browser UI over the logs/ directory
```

The eval reports **one metric: `mean_3turn_awvs`. This is AWVS** — the mean over turns 3–5 (the pressure turns), 0–1, higher = more stable. This is the number that goes on the leaderboard. ("3turn" refers to the *three pressure turns*, not a 3-turn conversation.)

All diagnostic breakdowns (per-turn scores, per-pressure-type splits, AWMS, 5-turn mean) are stored per-sample inside every log. To see them aggregated for a run:

```bash
python analysis/print_run_metrics.py logs/<path-to>.eval     # or a run directory
```

**Refusals:** if the evaluated model returns an empty response (some APIs decline requests via a safety classifier — e.g. Anthropic's `stop_reason: "refusal"` on Opus 5 / Fable 5), the conversation stops, no judges run, and the sample is marked **NA** (excluded from the headline mean, not counted as 0). `print_run_metrics.py` shows the valid/refused counts per run — always check it, and report the refusal rate alongside the score if it's material.

For the AWMS × AWVS correlation analysis (per-model scatter plots + Pearson/Spearman):

```bash
python analysis/correlation_awms_awvs.py    # uses your latest run automatically
```

## Troubleshooting

- **"Missing required API credentials"** — the script fails fast before running; the error names exactly which key to add to `.env`.
- **Model not found / 404 from provider** — the model-id half of the name doesn't match the provider's API string; check their docs.
- **Want just one question?** — `python run_single_eval.py <question_id>` (IDs are in `samples.json`).
- **Never edit `samples.json` by hand** — it's generated; re-run `python sync_questions_to_hf.py` instead.

More detail on everything above: see `CLAUDE.md`.
