"""Re-score frozen MANTA conversations with chat or Decisions judges.

Run from the repository root:
    uv run python -m experiments.judge_experiment --model openrouter/z-ai/glm-5.3-flash

See experiments/README.md for datasets, rubric variants, and output files.
Use --dry-run to save exact requests without API calls.
"""

import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from string import Template
import sys
import zipfile

if sys.version_info < (3, 14):
    import zipfile_zstd  # noqa: F401 -- Inspect's dependency adds zstd ZIP support.

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)

from manta import manta_scorer


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = Path(__file__).resolve().parent / "prompts"
HUMAN_INPUT = ROOT / "humanjudges/manta_judge_validation_12_questions_gemini_3.1_flash_lite.csv"
EXPERIMENTS = ROOT / "judge_experiments"
LEADERBOARD_SOURCES = Path(__file__).with_name("leaderboard_sources.json")
LEADERBOARD_INPUT = EXPERIMENTS / "leaderboard50_seed1.json"
SEED = 1
SAMPLE_SIZE = 50
ITEM_PROMPT = PROMPTS / "awvs_joint_items.txt"
ITEM_SCHEMA = PROMPTS / "awvs_joint_items.schema.json"
ITEM_V2_PROMPT = PROMPTS / "awvs_items_v2.txt"
ITEM_V2_SCHEMA = PROMPTS / "awvs_items_v2.schema.json"
V2_PROMPT_FILES = {
    "awms": (
        PROMPTS / "awms_v2_system.txt",
        PROMPTS / "awms_v2.txt",
    ),
    "awvs": (
        PROMPTS / "awvs_joint_v2_system.txt",
        PROMPTS / "awvs_joint_v2.txt",
    ),
}
V1_PROMPTS = ROOT / "humanjudges/v1_judge_prompts.json"
REPEATS = 3


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def visible_text(content):
    # Never stringify ContentReasoning (including its summary), tool data, etc.
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block["type"] == "text")


def leaderboard_input_path(sample_size=SAMPLE_SIZE, offset=0):
    suffix = f"_offset{offset}" if offset else ""
    return EXPERIMENTS / f"leaderboard{sample_size}_seed{SEED}{suffix}.json"


def load_frozen_sample(path):
    """Load an explicitly selected, visible-text conversation sample."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["conversations"]
    ids = [row["conversation_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Frozen sample must contain distinct conversation IDs")
    for row in rows:
        for turn in range(1, 6):
            for role in ("user", "assistant"):
                value = row[f"{role}_turn_{turn}"]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Frozen sample requires five complete visible-text turns")
    return data


def load_leaderboard(sample_size=SAMPLE_SIZE, offset=0):
    """Freeze the sample once; later judges use the same saved text, offline."""
    if sample_size < 1 or offset < 0:
        raise ValueError("Sample size must be positive and offset nonnegative")
    input_path = leaderboard_input_path(sample_size, offset)
    if input_path.exists():
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if (data["seed"] != SEED or len(data["conversations"]) != sample_size
                or data.get("offset", 0) != offset):
            raise ValueError("Saved leaderboard sample does not match seed, size or offset")
        if data["source_manifest_sha256"] != file_sha256(LEADERBOARD_SOURCES):
            raise ValueError("The source allowlist changed; preserve the frozen experiment before resampling")
        return data

    sources = json.loads(LEADERBOARD_SOURCES.read_text(encoding="utf-8"))
    pool, seen, source_hashes = [], set(), {}
    for source in sources["logs"]:
        path = ROOT / source["path"]
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
            spec = header["eval"]
            if (header["status"] != "success" or spec["task"] != "manta_5turn"
                    or spec["model"] != source["model"] or spec["run_id"] != source["run_id"]):
                raise ValueError(f"Source run differs from the verified allowlist: {path}")
            summaries = json.loads(archive.read("summaries.json"))
        count = 0
        for sample in summaries:
            value = sample.get("scores", {}).get("manta_per_turn_scorer", {}).get("value")
            # Only eligibility is used, never score magnitude or explanations.
            if sample.get("error") or type(value) not in (int, float) or not 0 <= value <= 1:
                continue
            key = (spec["model"], str(sample["id"]), sample["epoch"])
            if key in seen:
                raise ValueError(f"Duplicate source conversation: {key}")
            seen.add(key)
            pool.append({"evaluated_model": key[0], "sample_id": key[1], "epoch": key[2],
                         "source_log": source["path"]})
            count += 1
        if count != source["valid_samples"]:
            raise ValueError(f"Eligible sample count changed in {path}")
        source_hashes[source["path"]] = file_sha256(path)
    pool.sort(key=lambda row: (row["evaluated_model"], row["sample_id"], row["epoch"]))
    selected = pool.copy()
    random.Random(SEED).shuffle(selected)
    conversations = []
    skipped = []
    complete_count = 0
    for entry in selected:
        with zipfile.ZipFile(ROOT / entry["source_log"]) as archive:
            member = f"samples/{entry['sample_id']}_epoch_{entry['epoch']}.json"
            raw = archive.read(member)
        sample = json.loads(raw)
        messages = [m for m in sample["messages"] if m["role"] != "system"]
        texts = [visible_text(m["content"]) for m in messages]
        if (sample.get("error") or [m["role"] for m in messages] != ["user", "assistant"] * 5
                or not all(text.strip() for text in texts)):
            # Some old scored samples contain reasoning-only assistant turns.
            # A fixed slice of complete entries in a random order remains uniform.
            skipped.append(entry)
            continue
        metadata = sample.get("metadata", {})
        pressures = metadata.get("pressure_types") or metadata.get("pressure")
        if not isinstance(pressures, list) or len(pressures) != 3:
            raise ValueError(f"Missing pressure plan: {entry}")
        complete_count += 1
        if complete_count <= offset:
            continue
        row = {"conversation_id": f"{complete_count:03d}", **entry, "sample_uuid": sample.get("uuid"),
               "source_sample_sha256": hashlib.sha256(raw).hexdigest(),
               "pressure_types": ",".join(pressures)}
        for turn in range(1, 6):
            for message_offset, role in enumerate(("user", "assistant")):
                row[f"{role}_turn_{turn}"] = texts[2 * (turn - 1) + message_offset]
        conversations.append(row)
        if len(conversations) == sample_size:
            break
    if len(conversations) != sample_size:
        raise ValueError(f"Only {len(conversations)} complete visible conversations available")
    data = {"seed": SEED, "sampling": "Uniform without replacement over eligible conversations",
            "offset": offset,
            "algorithm": f"Sort candidates by model/sample ID/epoch; random.Random(SEED).shuffle; skip {offset} complete visible transcripts, take next {sample_size}",
            "skipped_incomplete_candidates": skipped,
            "pool_size": len(pool), "pool_sha256": hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
            "source_manifest_sha256": file_sha256(LEADERBOARD_SOURCES),
            "source_log_sha256": source_hashes, "conversations": conversations}
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Deliberately omit model identities and all original scores from this packet.
    blind = ["# Leaderboard sample: blind transcripts", "", f"Seed {SEED}; {sample_size} conversations; offset {offset}. No judge scores or model labels.", ""]
    for row in conversations:
        blind.extend([f"## Conversation {row['conversation_id']}", ""])
        for turn in range(1, 6):
            pressure = f" ({row['pressure_types'].split(',')[turn - 3]})" if turn >= 3 else ""
            blind.extend([f"### Turn {turn}{pressure}", "", "USER:", row[f"user_turn_{turn}"],
                          "", "ASSISTANT:", row[f"assistant_turn_{turn}"], ""])
    input_path.with_stem(input_path.stem + "_blind").with_suffix(".md").write_text(
        "\n".join(blind), encoding="utf-8")
    return data


def prompt_files(no_calibration=False, v2=False, items_v2=False):
    if not v2:
        return {"awvs": (None, ITEM_V2_PROMPT if items_v2 else ITEM_PROMPT)}
    return {
        metric: (system, user.with_stem(user.stem + "_no_calibration") if no_calibration else user)
        for metric, (system, user) in V2_PROMPT_FILES.items()
    }


def load_item_schema(items_v2=False):
    path = ITEM_V2_SCHEMA if items_v2 else ITEM_SCHEMA
    schema = json.loads(path.read_text(encoding="utf-8"))
    # Inspect's JSONSchema omits $ref/$defs, so inline the shared turn schema.
    turn_schema = schema.pop("$defs")["turn_scores"]
    schema["properties"] = {turn: turn_schema for turn in schema["required"]}
    return schema


def load_prompts(path, v1=False, no_calibration=False, v2=False, items_v2=False):
    if v1 and no_calibration:
        raise ValueError("--no-calibration cannot change the archived --v1 prompts")
    if v1:
        snapshot = json.loads(V1_PROMPTS.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["input_sha256"]:
            raise ValueError("The validation CSV differs from the recorded v1 prompt snapshot")
        return snapshot["prompts"]
    templates = {
        metric: (system.read_text(encoding="utf-8").rstrip() if system else None,
                 Template(user.read_text(encoding="utf-8")))
        for metric, (system, user) in prompt_files(no_calibration, v2, items_v2).items()
    }
    prompts = {}
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))["conversations"]
    for row in rows:
        cid = row["conversation_id"]
        if cid in prompts:
            raise ValueError(f"Duplicate conversation ID: {cid}")
        # Both dataset formats contain visible text only, without reasoning.
        messages = []
        for turn in range(1, 6):
            for role, cls in [("user", ChatMessageUser), ("assistant", ChatMessageAssistant)]:
                content = row[f"{role}_turn_{turn}"]
                if not content.strip():
                    raise ValueError(f"Empty {role} response: conversation {cid}, turn {turn}")
                messages.append(cls(content=content))
        pressures = [p.strip() for p in row["pressure_types"].split(",")]
        if len(pressures) != 3 or not all(pressures):
            raise ValueError(f"Expected three pressure types for conversation {cid}")
        format_conv = manta_scorer.format_conversation
        conversation = "\n\n".join(
            f"--- Turn {turn}"
            + (f" (pressure: {pressures[turn - 3]})" if turn >= 3 else " (context only)")
            + f" ---\n{format_conv(messages[2 * (turn - 1):2 * turn])}"
            for turn in range(1, 6)
        )
        prompts[cid] = {}
        for metric, (system, template) in templates.items():
            text = format_conv(messages[:2]) if metric == "awms" else conversation
            prompts[cid][metric] = (system, template.substitute(conversation=text))
    expected = 12 if path.suffix == ".csv" else len(rows)
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} conversations, found {len(prompts)}")
    return prompts


def judge_messages(prompt):
    system, user = prompt
    messages = [ChatMessageSystem(content=system)] if system else []
    return messages + [ChatMessageUser(content=user)]


def openrouter_cache_messages(prompt):
    """Add a Claude cache breakpoint without changing the prompt's text/order."""
    system, user = prompt
    # Both current and archived v1 prompts mark the start of conversation data.
    boundary = re.search(r"^--- CONVERSATION (?:TO EVALUATE|CONTEXT)\b", user, re.MULTILINE)
    if boundary is None or boundary.start() == 0:
        raise ValueError("Expected a shared prompt prefix before the conversation")
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [
        {"role": "user", "content": [
            {"type": "text", "text": user[:boundary.start()],
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user[boundary.start():]},
        ]},
    ]


def parse_response(text):
    # Unlike the general scorer's permissive parser, never turn missing output into 0.5.
    match = re.fullmatch(
        r"\s*SCORE:\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*"
        r"EXPLANATION:\s*(\S.*?)\s*", text, re.IGNORECASE | re.DOTALL,
    )
    if not match or not 0 <= float(match[1]) <= 1:
        raise ValueError("Expected SCORE: 0–1 followed by a nonempty EXPLANATION")
    return float(match[1]), match[2]


def parse_turns_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)[:-3].strip()
    try:
        turns = json.loads(text)
    except json.JSONDecodeError:
        # Some judges prepend commentary to an otherwise complete JSON answer.
        # Accept only one complete candidate; never guess between two answers.
        candidates = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and set(candidate) == {"turn_3", "turn_4", "turn_5"}:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("Expected one unambiguous JSON object with scores for turns 3–5")
        turns = candidates[0]
    if not isinstance(turns, dict) or set(turns) != {"turn_3", "turn_4", "turn_5"}:
        raise ValueError("Expected JSON scores for turn_3, turn_4, turn_5")
    return turns


def parse_item_response(text, schema):
    turns = parse_turns_json(text)
    for turn, rules in schema["properties"].items():
        result = turns[turn]
        if not isinstance(result, dict) or set(result) != set(rules["required"]):
            raise ValueError(f"Expected exactly {', '.join(rules['required'])} for {turn}")
        for item, rule in rules["properties"].items():
            score = result[item]
            if type(score) is not int or score not in rule["enum"]:
                raise ValueError(f"Invalid {turn}.{item}: expected an integer in {rule['enum']}")
    return turns


def parse_joint_response(text):
    turns = parse_turns_json(text)
    for turn, result in turns.items():
        if not isinstance(result, dict):
            raise ValueError(f"Invalid result for {turn}")
        score, explanation = result.get("score"), result.get("explanation")
        # The original prompt example quoted its score placeholder. Preserve
        # those valid judgments by accepting numeric strings as well as numbers.
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                pass
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score for {turn}: expected a number from 0 to 1")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Missing explanation for {turn}")
        result["score"] = score
    return turns


def mean_if_complete(values):
    return statistics.mean(values) if all(v is not None for v in values) else None


def describe(values):
    complete = all(v is not None for v in values)
    return {
        "repeats": values,
        "n_valid": sum(v is not None for v in values),
        "mean": mean_if_complete(values),
        "sd": statistics.stdev(values) if complete and len(values) > 1 else None,
    }


def summarize_items(results, prompts, schema, repeats=REPEATS):
    scores = {
        (r["conversation_id"], r["repeat"]): r.get("turns", {}) if not r["error"] else {}
        for r in results
    }
    conversations = {}
    for cid in prompts:
        conversations[cid] = {}
        for turn, rules in schema["properties"].items():
            conversations[cid][turn] = {
                item: describe([
                    scores.get((cid, repeat), {}).get(turn, {}).get(item)
                    for repeat in range(1, repeats + 1)
                ])
                for item in rules["required"]
            }
    overall = {}
    for turn, rules in schema["properties"].items():
        overall[turn] = {
            item: describe([
                mean_if_complete([c[turn][item]["repeats"][i] for c in conversations.values()])
                for i in range(repeats)
            ])
            for item in rules["required"]
        }
    return {
        "sd_definition": "Sample standard deviation across repeats, not standard error; null with fewer than two repeats.",
        "missing_scores": "An incomplete set has null mean/SD; errors are not scored as zero.",
        "aggregation": "Each rubric item and turn is summarized separately; no combined AWVS score.",
        "errors": sum(r["error"] is not None for r in results),
        "overall": overall,
        "conversations": conversations,
    }


def summarize(results, prompts):
    scores = {(r["conversation_id"], r["metric"], r["repeat"]): r["score"] for r in results}
    for r in results:
        if r["metric"] == "awvs":
            for turn in (3, 4, 5):
                scores[r["conversation_id"], f"awvs_t{turn}", r["repeat"]] = (
                    r.get("turns", {}).get(f"turn_{turn}", {}).get("score")
                )
    conversations = {}
    for cid in prompts:
        values = {
            metric: [scores[cid, metric, repeat] for repeat in range(1, REPEATS + 1)]
            for metric in ("awms", "awvs_t3", "awvs_t4", "awvs_t5")
        }
        values["awvs"] = [
            mean_if_complete([values[f"awvs_t{t}"][i] for t in (3, 4, 5)])
            for i in range(REPEATS)
        ]
        conversations[cid] = {metric: describe(v) for metric, v in values.items()}
    overall = {
        metric: describe([
            mean_if_complete([c[metric]["repeats"][i] for c in conversations.values()])
            for i in range(REPEATS)
        ])
        for metric in ("awms", "awvs")
    }
    return {
        "sd_definition": "Sample standard deviation across three repeats, not standard error.",
        "missing_scores": "An incomplete set has null mean/SD; errors are not scored as zero or 0.5.",
        "errors": sum(r["error"] is not None for r in results),
        "overall": overall,
        "conversations": conversations,
    }


async def run(args):
    custom_input = getattr(args, "input_file", None)
    input_path = custom_input or (HUMAN_INPUT if args.humanjudges else LEADERBOARD_INPUT)
    selection = (load_frozen_sample(input_path) if custom_input else
                 None if args.humanjudges else load_leaderboard())
    prompts = load_prompts(input_path, v1=args.v1, no_calibration=args.no_calibration,
                           v2=args.v2, items_v2=args.items_v2)
    item_schema = None if args.v1 or args.v2 else load_item_schema(args.items_v2)
    item_schema_path = ITEM_V2_SCHEMA if args.items_v2 else ITEM_SCHEMA
    item_version = "awvs_items_v2" if args.items_v2 else "awvs_items"
    calls = sum(map(len, prompts.values())) * REPEATS
    # Validate and prepare all breakpoints before making any model calls.
    cached_messages = {
        (cid, metric): openrouter_cache_messages(prompt)
        for cid, metrics in prompts.items() for metric, prompt in metrics.items()
    } if args.model.startswith("openrouter/anthropic/") else {}
    extra_body = {}
    model_args = {}
    if args.model.startswith("openrouter/"):
        # This is a top-level API field, not a model suffix or provider option.
        extra_body["service_tier"] = args.service_tier
        if args.service_tier == "flex":
            model_args["client_timeout"] = 900
    elif args.model.startswith("openai/"):
        # Inspect's native OpenAI adapter takes the tier as a model argument.
        model_args["service_tier"] = args.service_tier
    if cached_messages:
        # Inspect's OpenRouter adapter does not implement cache_prompt. Claude
        # needs this request-body setting to cache the full repeated prompt.
        extra_body["cache_control"] = {"type": "ephemeral"}
    config = GenerateConfig(
        reasoning_effort=args.reasoning_effort, temperature=args.temperature,
        max_tokens=args.max_tokens, max_retries=2, max_connections=args.concurrency,
        extra_body=extra_body or None,
        response_schema=ResponseSchema(name="awvs_items", json_schema=item_schema, strict=True)
        if item_schema else None,
    )
    if args.provider:
        if not args.model.startswith("openrouter/"):
            raise ValueError("--provider is only supported for openrouter/ models")
        model_args["provider"] = {"only": args.provider, "allow_fallbacks": True}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model)
    if item_schema:
        slug += f"_{item_version}"
    elif args.no_calibration:
        slug += "_no_calibration"
    results_dir = ROOT / "humanjudges/results" if args.humanjudges else EXPERIMENTS / "results"
    out = args.output_dir or results_dir / f"{slug}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps({
        "model": args.model, "model_args": model_args, "repeats": REPEATS,
        "dataset": "humanjudges12" if args.humanjudges else input_path.stem,
        "dry_run": args.dry_run,
        "script_sha256": file_sha256(Path(__file__)),
        "selection": {k: v for k, v in selection.items() if k != "conversations"} if selection else None,
        "mode": "v1_recorded_prompts" if args.v1 else "awvs_items_visible_text" if item_schema else "joint_visible_text",
        "prompt_version": "v1_recorded" if args.v1 else item_version if item_schema else "v2",
        "calibration_examples": (args.v1 or args.v2) and not args.no_calibration,
        "prompt_files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for paths in prompt_files(args.no_calibration, args.v2, args.items_v2).values() for path in paths if path
        } if not args.v1 else {},
        "response_schema_file": str(item_schema_path.relative_to(ROOT)) if item_schema else None,
        "response_schema_sha256": file_sha256(item_schema_path) if item_schema else None,
        "target_reasoning": "Original logged judge input" if args.v1 else "Excluded: visible text only",
        "config": config.model_dump(exclude_none=True), "concurrency": args.concurrency,
        "started_at": stamp, "input_file": str(input_path),
        "input_csv": str(input_path) if args.humanjudges else None,
        "input_sha256": file_sha256(input_path),
        "scorer_sha256": hashlib.sha256(Path(manta_scorer.__file__).read_bytes()).hexdigest(),
        "prompts": prompts,
    }, indent=2), encoding="utf-8")
    if selection:
        (out / "conversations.json").write_bytes(input_path.read_bytes())
        print("Evaluated models:", dict(sorted(Counter(row["evaluated_model"] for row in selection["conversations"]).items())))
    print(f"Scoring {len(prompts)} conversations, {REPEATS} repeats, {calls} calls. Output: {out}")
    if args.dry_run:
        print("Dry run: saved inputs and exact prompts; no model calls made.")
        return 0
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    async with get_model(args.model, config=config, **model_args) as judge:
        if args.model.startswith("openrouter/"):
            # Inspect's normalized output omits OpenRouter's billed cost and ID.
            # Preserve response metadata before the adapter handles any API error,
            # so responses from retried calls also remain in the cost ledger.
            original_on_response = judge.api.on_response

            def record_openrouter_usage(response):
                billing = {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    **{key: response.get(key) for key in
                       ("id", "model", "provider", "usage", "error")},
                }
                with (out / "openrouter_usage.jsonl").open("a", encoding="utf-8") as ledger:
                    ledger.write(json.dumps(billing, ensure_ascii=False) + "\n")
                original_on_response(response)

            judge.api.on_response = record_openrouter_usage
        with (out / "results.jsonl").open("w", encoding="utf-8") as output:
            async def score_prompt(cid, metric, prompt):
                # Reuse exactly the same prompt; repeat IDs never enter the model input.
                messages = judge_messages(prompt)
                request_config = GenerateConfig()
                if args.model.startswith("openrouter/"):
                    # Share routing across conversations with the same rubric.
                    # Group v1's awvs_t3/t4/t5 together too. This is metadata;
                    # cache reuse still requires identical prompt prefixes.
                    session_id = f"manta-judge-{metric.split('_')[0]}"
                    request_config.extra_body = {
                        **(config.extra_body or {}), "session_id": session_id,
                    }
                    if cached_messages:
                        # Inspect drops cache_control on ContentText. Supply the
                        # same messages as raw OpenRouter JSON via extra_body;
                        # the SDK overrides its serialized messages with these.
                        request_config.extra_body["messages"] = cached_messages[cid, metric]
                for repeat in range(1, REPEATS + 1):
                    record = {
                        "conversation_id": cid, "metric": metric, "repeat": repeat,
                        "error": None,
                        "request_extra_body": request_config.extra_body,
                    }
                    if not item_schema:
                        record.update(score=None, explanation=None)
                    try:
                        async with semaphore:
                            # Disable Inspect's response cache, preserving fresh generations.
                            # Provider-side prompt caching is unaffected.
                            result = await judge.generate(messages, config=request_config, cache=False)
                        record["output"] = result.model_dump(mode="json", exclude_none=True)
                        if result.stop_reason == "max_tokens":
                            raise ValueError("Judge output was truncated; increase --max-tokens")
                        if item_schema:
                            record["turns"] = parse_item_response(result.completion, item_schema)
                        elif metric == "awvs":
                            turns = parse_joint_response(result.completion)
                            record["turns"] = turns
                            record["score"] = statistics.mean(t["score"] for t in turns.values())
                            record["explanation"] = "\n".join(
                                f"{turn}: {turns[turn]['explanation']}" for turn in sorted(turns)
                            )
                        else:
                            record["score"], record["explanation"] = parse_response(result.completion)
                    except Exception as error:
                        record["error"] = f"{type(error).__name__}: {error}"
                    results.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    status = "ERROR" if record["error"] else "OK" if item_schema else f"{record['score']:.3f}"
                    print(f"[{len(results)}/{calls}] {cid} {metric} repeat {repeat}: {status}", flush=True)

            await asyncio.gather(*[
                score_prompt(cid, metric, prompt)
                for cid, metrics in prompts.items() for metric, prompt in metrics.items()
            ])
    summary = summarize_items(results, prompts, item_schema) if item_schema else summarize(results, prompts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results and summary to {out} ({summary['errors']} errors).")
    return 1 if summary["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Inspect model ID, e.g. openrouter/z-ai/glm-5.3-flash")
    parser.add_argument("--humanjudges", action="store_true", help="Use the previous 12 human-validation conversations instead of the frozen leaderboard sample")
    parser.add_argument("--input-file", type=Path, help="Use an existing frozen conversation JSON with either chat or Decisions judges")
    parser.add_argument("--dry-run", action="store_true", help="Save the selected conversations and exact prompts without calling a model")
    parser.add_argument("--resume", type=Path, help="Decisions models only: resume a saved run, retaining valid completed judgments and the cost ledger")
    parser.add_argument("--repeats", type=int, help="Decisions models only: judgments per conversation (default: 1; resume uses the saved count)")
    parser.add_argument("--sample-size", type=int, help="Decisions models only: number of leaderboard conversations (v4/v5 default: 100; earlier rubrics: 50; resume uses saved size)")
    parser.add_argument("--sample-offset", type=int, help="Decisions models only: skip complete conversations in the seed-1 shuffle (v4/v5 default: 50; earlier rubrics: 0; resume uses saved offset)")
    parser.add_argument("--jev-rubric", choices=("decisions", "decisions-v7", "decisions-v6", "decisions-v5", "decisions-v4", "decisions-v3", "decisions-v2", "decisions-v1", "score"), help="Jev/Solar Decide: reference-grounded v4 (default), concise decisions-v5, atomic decisions-v6, clarified decisions-v7, previous decisions-v3/v2/v1, or original Score rubric; resume infers saved version")
    parser.add_argument("--reference-standards", type=Path, help="Decisions v4/v5: scenario reference JSON (default: frozen 100-scenario standards; resume uses archived copy)")
    parser.add_argument("--request-timeout", type=float, help="Decisions models only: timeout in seconds per API attempt (Solar default: 180; Jev: 120)")
    parser.add_argument("--questions-per-call", type=int, help="Decisions models only: split each conversation's questions into batches (Solar default: 9; Jev: all; state is unchanged)")
    parser.add_argument("--decision-context", choices=("full", "prefix"), help="Decisions models only: full five-turn context (default), or only turns through the target; prefix groups questions by target turn")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--v1", action="store_true", help="Replay recorded AWMS and separate T3/4/5 AWVS prompts, including original context/formatting")
    variant.add_argument("--v2", action="store_true", help="Use the previous AWMS + single-scale AWVS prompts, system messages, and output format")
    variant.add_argument("--items-v2", action="store_true", help="Use awvs_items_v2.txt: four AWVS items including disparagement, scored jointly for T3–T5")
    parser.add_argument("--no-calibration", action="store_true", help="Omit examples with --v2; the default item rubric already has none")
    parser.add_argument("--output-dir", type=Path, help="New output directory; default: judge_experiments/results/<model>_awvs_items_<UTC time> (humanjudges/results with --humanjudges)")
    parser.add_argument("--reasoning-effort", help="Optional Inspect reasoning effort, e.g. low, high, max")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature; otherwise the provider default")
    parser.add_argument(
        "--service-tier", choices=("flex", "default", "auto", "priority"), default="flex",
        help="OpenRouter/OpenAI service tier (default: flex); other adapters are unchanged. "
             "OpenRouter models without flex endpoints can still use standard rates.",
    )
    parser.add_argument("--max-tokens", type=int, default=32768, help="Per-call output limit, including reasoning (default: 32768)")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum simultaneous calls (default: 32)")
    parser.add_argument(
        "--provider", "--providers", nargs="+", action="extend",
        help="Allowed OpenRouter provider slugs; routing and fallback stay within this list",
    )
    args = parser.parse_args()
    if args.input_file and (args.humanjudges or args.sample_size is not None or args.sample_offset is not None):
        parser.error("--input-file cannot be combined with --humanjudges, --sample-size or --sample-offset")
    decision_model = (args.model.startswith("openrouter/typesafe/jev-")
                      or args.model == "openrouter/upstage/solar-decide")
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("--concurrency and --max-tokens must be positive")
    if args.request_timeout is not None and (args.request_timeout <= 0 or not decision_model):
        parser.error("--request-timeout must be positive and is for Decisions models only")
    if args.questions_per_call is not None and (args.questions_per_call < 1 or not decision_model):
        parser.error("--questions-per-call must be positive and is for Decisions models only")
    if args.decision_context and not decision_model:
        parser.error("--decision-context is for Decisions models only")
    if ((args.repeats is not None and args.repeats < 1)
            or (args.sample_size is not None and args.sample_size < 1)
            or (args.sample_offset is not None and args.sample_offset < 0)):
        parser.error("--repeats and --sample-size must be positive; --sample-offset must be nonnegative")
    if (any(value is not None for value in (args.repeats, args.sample_size, args.sample_offset))
            and not decision_model):
        parser.error("--repeats, --sample-size and --sample-offset are currently for Decisions models only")
    if args.v1 and not args.humanjudges:
        parser.error("--v1 replays the archived 12-conversation prompts; use --humanjudges --v1")
    if args.v1 and args.no_calibration:
        parser.error("--no-calibration cannot change the archived --v1 prompts")
    if args.resume and (not decision_model or args.output_dir or args.dry_run):
        parser.error("--resume is for Decisions models only; do not combine it with --output-dir or --dry-run")
    if (args.jev_rubric or args.reference_standards) and not decision_model:
        parser.error("--jev-rubric and --reference-standards are for Decisions models only")
    load_dotenv(ROOT / ".env")
    if decision_model:
        from experiments.judge_experiment_jev import run as run_jev
        raise SystemExit(asyncio.run(run_jev(args)))
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
