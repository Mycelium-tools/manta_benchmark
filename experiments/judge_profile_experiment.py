"""Apply a Decisions rubric with a chat judge, one call per target turn.

Each complete sample takes three calls, seeing turns 1-3, 1-4 and 1-5 respectively. All
selected items are returned together, each with a brief explanation, with no repeats or
automatic retries. Optional targets supplies conversation_id and turn; reviewer labels are
never sent. Default is preparation only; --run sends requests.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import httpx
import jsonschema

from experiments import judge_experiment_jev as jev

PROMPT = Path(__file__).with_name("prompts") / "judge_profile_direct.txt"
CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


async def run(args):
    sample = jev.experiment.load_frozen_sample(args.input_file)
    conversations = {r["conversation_id"]: r for r in sample["conversations"]}
    if args.targets:
        panel = json.loads(args.targets.read_text())
        targets = [(t["conversation_id"], t["turn"]) for t in panel.get("targets", panel.get("cases", []))]
    else:
        targets = [(cid, turn) for cid in conversations for turn in (3, 4, 5)]
    if args.limit:
        targets = targets[:args.limit]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Targets must be distinct conversation/turn pairs")
    references = jev.load_reference_standards(args.reference_standards, sample["conversations"])
    rubric_path, version = jev.RUBRICS[args.rubric]
    rubric = json.loads(rubric_path.read_text())
    questions = jev.make_questions(rubric)
    # Default: the items that feed AWVS (role "core"), or every item of a rubric without roles.
    selected_items = args.items or [name for name, item in rubric["items"].items() if item.get("role", "core") == "core"]
    if set(selected_items) - rubric["items"].keys() or any(q["type"] not in ("noul", "choice") for q in questions.values()):
        raise ValueError("Select supported Noul/Choice rubric items")
    out = args.output_dir
    cached = {}
    metadata = {
        "model": args.model, "prompt_version": version, "decision_context": "prefix",
        "input_sha256": jev.experiment.file_sha256(args.input_file),
        "reference_standards_sha256": jev.experiment.file_sha256(args.reference_standards),
        "repeats": 1, "dry_run": not args.run, "targets": targets,
        "questions": questions, "selected_items": selected_items, "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort, "prompt": PROMPT.read_text(),
        "script_sha256": jev.experiment.file_sha256(Path(__file__)),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": "Chat judge: all selected questions in one call per prefix; no reviewer labels or evaluated-model reasoning.",
    }
    if args.resume:
        saved = json.loads((out / "run.json").read_text())
        for key in ("model", "input_sha256", "reference_standards_sha256", "targets", "questions",
                    "selected_items", "max_tokens", "reasoning_effort", "prompt", "dry_run"):
            if json.loads(json.dumps(metadata[key])) != saved[key]:
                raise ValueError(f"Cannot resume: {key} differs")
        source = out / "target_results.jsonl"
        for row in (json.loads(line) for line in source.read_text().splitlines()) if source.exists() else ():
            if row.get("error") is None:
                cached[(row["conversation_id"], row["turn"])] = row
        print(f"Reusing {len(cached)} completed targets", flush=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "run.json").write_text(json.dumps(metadata, indent=2))
        (out / "conversations.json").write_bytes(args.input_file.read_bytes())
        (out / "reference_standards.json").write_bytes(args.reference_standards.read_bytes())

    def append(name, value):
        with (out / name).open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    if args.resume:
        append("resume.jsonl", {"script_sha256": metadata["script_sha256"],
                               "started_at": metadata["started_at"], "reused_targets": len(cached)})

    load_dotenv(jev.experiment.ROOT / ".env")
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"} if args.run else {}
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        async def score(cid, target):
            if (cid, target) in cached:
                return cached[(cid, target)]
            row = {"conversation_id": cid, "turn": target, "repeat": 1, "error": None}
            batch = {k: v for k, v in questions.items()
                     if k.startswith(f"turn_{target}_") and k.split("_", 2)[2] in selected_items}
            labels = {key.split("_", 2)[2]: (
                {"type": "boolean"} if q["type"] == "noul" else
                {"type": "string", "enum": list(q["criteria"])}) for key, q in batch.items()}
            schema = {"type": "object", "additionalProperties": False,
                      "properties": {item: {"type": "object", "additionalProperties": False,
                          "properties": {"explanation": {"type": "string"}, "label": label},
                          "required": ["explanation", "label"]} for item, label in labels.items()},
                      "required": list(labels)}
            source = {"questions": batch, "state": jev.state_for_batch(
                jev.make_state(conversations[cid], references[cid]), batch, "prefix")}
            payload = {
                "model": args.model.removeprefix("openrouter/"),
                "messages": [{"role": "user", "content": PROMPT.read_text() + "\n\n" + json.dumps(source, ensure_ascii=False)}],
                "reasoning": {"effort": args.reasoning_effort}, "max_tokens": args.max_tokens, "service_tier": "flex",
                "provider": {"require_parameters": True, "max_price": {"prompt": 0.30, "completion": 1}},
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "direct", "strict": True, "schema": schema}},
            }
            append("requests.jsonl", {"conversation_id": cid, "stage": "direct", "payload": payload})
            if not args.run:
                return
            try:
                async with semaphore:
                    response = await client.post(CHAT_ENDPOINT, json=payload)
                body = response.json()
                append("attempts.jsonl", {"conversation_id": cid, "stage": "direct",
                                         "http_status": response.status_code, "response": body})
                if body.get("id") or body.get("usage"):
                    append("openrouter_usage.jsonl", {"conversation_id": cid, "stage": "direct",
                        **{k: body.get(k) for k in ("id", "model", "provider", "usage", "error")}})
                response.raise_for_status()
                choice = body["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError(f"direct finish reason: {choice['finish_reason']}")
                result = json.loads(choice["message"]["content"])
                jsonschema.validate(result, schema)
                values = {item: (int(answer["label"]) if type(answer["label"]) is bool else answer["label"])
                          for item, answer in result.items()}
                row.update(output=body, explanations={item: answer["explanation"] for item, answer in result.items()},
                           turns={f"turn_{target}": values})
            except (ValueError, KeyError, TypeError, httpx.HTTPError, jsonschema.ValidationError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            append("target_results.jsonl", row)
            print(cid, target, row["error"] or "OK", flush=True)
            return row

        rows = await asyncio.gather(*(score(cid, turn) for cid, turn in targets))
    if args.run:
        combined = {}
        for row in rows:
            record = combined.setdefault(row["conversation_id"], {
                "conversation_id": row["conversation_id"], "repeat": 1,
                "error": None, "turns": {}, "target_errors": {},
            })
            record["turns"].update(row.get("turns", {}))
            if row["error"]:
                record["target_errors"][str(row["turn"])] = row["error"]
        for row in combined.values():
            if not row["turns"]:
                row["error"] = str(row["target_errors"])
        (out / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in combined.values()))
        return int(any(row["target_errors"] for row in combined.values()))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--targets", type=Path, help="Optional review panel; default is all three target turns of every conversation")
    parser.add_argument("--reference-standards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--rubric", choices=tuple(jev.RUBRICS), default="decisions-v17")
    parser.add_argument("--items", nargs="+", help="Rubric items to ask (default: the items that feed AWVS)")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--reasoning-effort", choices=("high", "medium", "low"), default="low")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed targets in output-dir, preserving attempts and costs")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Concurrency, token limit and target limit must be positive")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
