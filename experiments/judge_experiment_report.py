"""Write the 50 conversations with per-turn AWVS tables from the latest run per judge.

    uv run python -m experiments.judge_experiment_report

Reads saved results only, including unfinished runs. Rerun to refresh the report.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "judge_experiments/leaderboard50_seed1.json"
ITEMS = {
    "consideration": ("Consideration (0–3)", range(4)),
    "intrinsic_reason": ("Intrinsic reason (0–1)", range(2)),
    "protection": ("Protection (0–2)", range(3)),
}


def load_runs(results_dir, input_hash, conversation_ids):
    latest = {}
    for path in sorted(results_dir.glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if (run.get("prompt_version") != "awvs_items" or run.get("dry_run")
                or run.get("dataset") != "leaderboard50_seed1"
                or run.get("input_sha256") != input_hash):
            continue
        model = run["model"]
        if model not in latest or run["started_at"] > latest[model]["metadata"]["started_at"]:
            latest[model] = {"directory": path.parent, "metadata": run}
    if not latest:
        raise ValueError("No AWVS item runs match the frozen 50-conversation input")

    for entry in latest.values():
        scores = {}
        path = entry["directory"] / "results.jsonl"
        # A running judge may be partway through writing its final JSONL line.
        lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
        for number, line in enumerate(lines, 1):
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            cid, repeat = row["conversation_id"], row["repeat"]
            if (row["metric"] != "awvs" or cid not in conversation_ids
                    or type(repeat) is not int or not 1 <= repeat <= entry["metadata"]["repeats"]):
                raise ValueError(f"Unexpected result at {path}:{number}")
            if (cid, repeat) in scores:
                raise ValueError(f"Duplicate result at {path}:{number}")
            if row["error"] is not None:
                scores[cid, repeat] = None
                continue
            turns = row["turns"]
            if set(turns) != {"turn_3", "turn_4", "turn_5"}:
                raise ValueError(f"Invalid turn keys at {path}:{number}")
            for values in turns.values():
                if set(values) != set(ITEMS) or any(
                    type(values[item]) is not int or values[item] not in allowed
                    for item, (_, allowed) in ITEMS.items()
                ):
                    raise ValueError(f"Invalid rubric scores at {path}:{number}")
            scores[cid, repeat] = turns
        entry["scores"] = scores
    return [latest[model] for model in sorted(latest)]


def table_cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def quote(text):
    # Keep transcript headings and tables inside the speaker's blockquote.
    return "\n".join("> " + line for line in text.splitlines())


def repeat_scores(entry, cid, turn, item):
    values = []
    for repeat in range(1, entry["metadata"]["repeats"] + 1):
        key = (cid, repeat)
        if key not in entry["scores"]:
            values.append("—")
        elif entry["scores"][key] is None:
            values.append("ERR")
        else:
            values.append(str(entry["scores"][key][f"turn_{turn}"][item]))
    return " ".join(values)


def render(conversations, runs):
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# AWVS judge comparison: 50 conversations", "",
        f"Generated {generated}. Latest matching run per judge model ID.", "",
        "Each cell lists repeats in order, separated by spaces. `—` means not yet saved; "
        "`ERR` means the judge call failed. Scores are not averaged or combined.", "",
        "## Selected runs", "",
    ]
    for entry in runs:
        run = entry["metadata"]
        expected = len(conversations) * run["repeats"]
        errors = sum(value is None for value in entry["scores"].values())
        lines.append(
            f"- **{run['model']}**: {len(entry['scores'])}/{expected} calls saved, "
            f"{errors} errors. Run: `{entry['directory'].name}`."
        )
    versions = {
        (tuple(sorted(entry["metadata"].get("prompt_files", {}).items())),
         entry["metadata"].get("response_schema_sha256"))
        for entry in runs
    }
    if len(versions) > 1:
        lines += ["", "**Note:** the selected runs used different prompt or schema versions."]
    lines.append("")
    for row in conversations:
        cid = row["conversation_id"]
        lines += [f"## Conversation {cid}", "", f"Evaluated model: `{row['evaluated_model']}`.", ""]
        pressures = [p.strip() for p in row["pressure_types"].split(",")]
        for turn in range(1, 6):
            pressure = f" — {pressures[turn - 3]}" if turn >= 3 else ""
            lines += [f"### Turn {turn}{pressure}", "", "**User**", "",
                      quote(row[f"user_turn_{turn}"]), "", "**Assistant**", "",
                      quote(row[f"assistant_turn_{turn}"]), ""]
            if turn >= 3:
                lines += ["**Judge scores**", "",
                          "| Judge model | " + " | ".join(title for title, _ in ITEMS.values()) + " |",
                          "| --- | --- | --- | --- |"]
                for entry in runs:
                    model = table_cell(entry["metadata"]["model"].removeprefix("openrouter/"))
                    cells = [f"`{repeat_scores(entry, cid, turn, item)}`" for item in ITEMS]
                    lines.append(f"| {model} | " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "judge_experiments/results")
    parser.add_argument("--output", type=Path, default=ROOT / "judge_experiments/awvs_items_comparison.md")
    args = parser.parse_args()
    raw = INPUT.read_bytes()
    conversations = json.loads(raw)["conversations"]
    ids = {row["conversation_id"] for row in conversations}
    if len(conversations) != 50 or len(ids) != 50:
        raise ValueError("Expected exactly 50 distinct frozen conversations")
    runs = load_runs(args.results_dir, hashlib.sha256(raw).hexdigest(), ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(conversations, runs), encoding="utf-8")
    print(f"Wrote {args.output}: 50 conversations, 150 score tables, {len(runs)} judges.")


if __name__ == "__main__":
    main()
