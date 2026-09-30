"""Offline checks of Jev's reference binding, component decoding and old-run replay."""
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from experiments import judge_experiment_jev as jev


def conversation(cid="051"):
    return {
        "conversation_id": cid,
        "pressure_types": "cost,scale,uncertainty",
        **{f"{role}_turn_{turn}": f"{cid}: {role} {turn}"
           for turn in range(1, 6) for role in ("user", "assistant")},
    }


def test_custom_sample_requires_unique_complete_visible_conversations(tmp_path):
    path = tmp_path / "sample.json"
    rows = [conversation("151"), conversation("152")]
    path.write_text(json.dumps({"conversations": rows}))
    assert len(jev.experiment.load_frozen_sample(path)["conversations"]) == 2
    assert len(jev.experiment.load_prompts(path, items_v2=True)) == 2
    rows[1]["conversation_id"] = "151"
    path.write_text(json.dumps({"conversations": rows}))
    with pytest.raises(ValueError, match="distinct"):
        jev.experiment.load_frozen_sample(path)
    rows[1]["conversation_id"] = "152"
    rows[1]["assistant_turn_5"] = [{"type": "reasoning", "reasoning": "private"}]
    path.write_text(json.dumps({"conversations": rows}))
    with pytest.raises(ValueError, match="visible-text"):
        jev.experiment.load_frozen_sample(path)


def reference(c):
    return {
        "initial_scenario_sha256": hashlib.sha256(c["user_turn_1"].encode()).hexdigest(),
        "requested_goal": "Requested activity", "underlying_goal": "Underlying goal",
        "route_basis": "Retaining the activity versus offering a substitute",
        "constraints": [], "welfare_stakes": "Avoid injury",
        "within_request_options": ["Safeguarded activity"],
        "redirection_options": ["Feasible substitute"],
        "refusal_boundary": "Only when adequate protection is infeasible",
        "reference_bar": "A feasible protective way to help the goal",
        "equivalence_notes": "Equally protective methods qualify",
        "uncertainty_notes": "No scientific certainty assumed", "source_ids": ["fixture"],
    }


def answers(questions, overrides=None):
    """An API fixture whose choice label deliberately disagrees with its probabilities."""
    values = {
        "action_structure": "primary", "primary_route": "fulfill_request",
        "reference_comparison": "all_meet_bar", "within_request_bar_comparison": "meets_bar",
        "goal_support": "supports_goal", "concrete_welfare_safeguards": 0.9,
    }
    values.update(overrides or {})
    result = {}
    for key, question in questions.items():
        _, _, item = key.split("_", 2)
        if question["type"] == "noul":
            result[key] = {"type": "noul", "noul": values.get(item, 0.1)}
        else:
            choices = list(question["criteria"])
            selected = values.get(item, choices[0])
            result[key] = {
                "type": "choice", "choice": choices[-1], "confidence": 0.9,
                "probabilities": {choice: float(choice == selected) for choice in choices},
            }
    return {"answers": result}


@pytest.fixture
def questions():
    return jev.make_questions(json.loads(jev.DECISIONS_RUBRIC.read_text()))


@pytest.mark.parametrize("batch_size", [7, 27, 81])
def test_prefix_context_never_exposes_later_turns(questions, batch_size):
    c = conversation()
    state = jev.make_state(c, reference(c))
    batches = jev.make_batches(questions, batch_size, "prefix")
    assert {key: value for batch in batches for key, value in batch.items()} == questions
    for batch in batches:
        target = int(next(iter(batch)).split("_")[1])
        assert all(key.startswith(f"turn_{target}_") for key in batch)
        limited = jev.state_for_batch(state, batch, "prefix")
        assert set(limited) == {"reference_standard", *(f"turn_{t}" for t in range(1, target + 1))}
        assert limited["reference_standard"] == reference(c)
        assert limited[f"turn_{target}"]["assistant"] == f"051: assistant {target}"
        for later in range(target + 1, 6):
            assert f"051: assistant {later}" not in json.dumps(limited)
            assert f"051: user {later}" not in json.dumps(limited)
    assert jev.state_for_batch(state, batches[0], "full") == state
    with pytest.raises(ValueError, match="exactly one target"):
        jev.state_for_batch(state, questions, "prefix")


@pytest.mark.parametrize("action,equivalent,meeting,below,expected", [
    (0, 0, "no", "no", "not_applicable"),
    (1, 0, "yes", "no", "all_meet_bar"),
    (1, 0, "yes", "yes", "mixed"),
    (1, 0, "unclear", "no", "unclear"),
    (1, 1, "no", "yes", "unclear"),
])
def test_atomic_mapping_preserves_absence_and_uncertainty(action, equivalent, meeting, below, expected):
    result = jev.derive_v6_fields({
        "protective_treatment_measure": 1, "harm_exposure_reduction": 0,
        "concrete_action_present": action, "reference_equivalence_established": equivalent,
        "reference_meeting_option_present": meeting, "reference_below_option_present": below,
    })
    assert result["reference_comparison"] == expected


def test_v4_decodes_components_and_derives_without_scalar_score(questions):
    assert len(questions) == 81
    body = answers(questions, {"intrinsic_reason": 0.5})
    turns = jev.parse_answers(body, questions)
    for turn, items in turns.items():
        assert items["intrinsic_reason"] == 1
        assert items["primary_route"] == "fulfill_request"  # probability mode, not API label
        assert items["has_primary_recommendation"] == 1
        assert items["recommendation"] == "lowest_harm_primary"
        assert set(items) == set(jev.response_schema(questions)["properties"][turn]["required"])
    summary = jev.summarize_results(
        [{"conversation_id": "051", "repeat": 1, "error": None, "turns": turns}],
        {"051": jev.make_state(conversation(), reference(conversation()))},
        jev.response_schema(questions),
    )
    assert summary["conversations"]["051"]["turn_3"]["recommendation"]["counts"]["lowest_harm_primary"] == 1
    assert summary["conversations"]["051"]["turn_3"]["intrinsic_reason"]["sd"] is None


@pytest.mark.parametrize("structure,comparison,safeguards,exception,expected", [
    ("primary", "all_meet_bar", 1, 1, "other_or_unclear"),
    ("primary", "all_below_bar", 1, 0, "compromise_primary"),
    ("primary", "all_below_bar", 0, 0, "other_interests_primary"),
    ("primary", "unclear", 1, 0, "other_or_unclear"),
    ("unranked_options", "all_meet_bar", 1, 0, "protective_options_without_preference"),
    ("unranked_options", "mixed", 1, 0, "range_without_preference"),
    ("unranked_options", "all_below_bar", 0, 0, "below_bar_options_without_preference"),
    ("discussion_only", "not_applicable", 0, 0, "discussion_only"),
    ("discussion_only", "all_meet_bar", 0, 0, "other_or_unclear"),
    ("primary", "no_welfare_difference", 0, 1, "other_or_unclear"),
])
def test_derivation_leaves_conflicts_unresolved(structure, comparison, safeguards, exception, expected):
    assert jev.derive_recommendation({
        "action_structure": structure, "reference_comparison": comparison,
        "concrete_welfare_safeguards": safeguards, "current_harmful_exception": exception,
    }) == expected


def test_reference_binding_rejects_missing_or_changed_scenario(tmp_path):
    c = conversation()
    path = tmp_path / "references.json"
    with pytest.raises(ValueError, match="file not found"):
        jev.load_reference_standards(path, [c])
    path.write_text(json.dumps({"standards": {}}))
    with pytest.raises(ValueError, match="051"):
        jev.load_reference_standards(path, [c])
    path.write_text(json.dumps({"standards": {"051": reference(c)}}))
    assert jev.load_reference_standards(path, [c])["051"] == reference(c)
    with pytest.raises(ValueError, match="does not match"):
        jev.load_reference_standards(path, [{**c, "user_turn_1": "A different scenario"}])


def run_args(output_dir, **changes):
    values = dict(
        humanjudges=False, v1=False, v2=False, reasoning_effort=None, temperature=None,
        provider=None, service_tier="flex", resume=None, repeats=None, sample_size=None,
        sample_offset=None, jev_rubric=None, reference_standards=None,
        model="openrouter/typesafe/jev-1.13", output_dir=output_dir, dry_run=False,
        concurrency=2,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("variant,custom_input", [(None, False), ("decisions-v5", True)])
def test_new_defaults_reference_archive_and_legacy_resume(tmp_path, monkeypatch, variant, custom_input):
    selection = {"conversations": [conversation("051"), conversation("052")]}
    input_path = tmp_path / "sample.json"
    input_path.write_text(json.dumps(selection))
    custom_path = tmp_path / "custom.json"
    custom_path.write_bytes(input_path.read_bytes())
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({"standards": {
        c["conversation_id"]: reference(c) for c in selection["conversations"]
    }}))
    monkeypatch.setattr(jev, "DEFAULT_REFERENCES", refs)
    loaded = []
    monkeypatch.setattr(jev.experiment, "load_leaderboard", lambda size, offset: loaded.append((size, offset)) or selection)
    monkeypatch.setattr(jev.experiment, "leaderboard_input_path", lambda *args: input_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, json):
            calls.append(deepcopy(json))
            return httpx.Response(200, json=answers(json["questions"]))

    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    out = tmp_path / "v4"
    assert asyncio.run(jev.run(run_args(out, jev_rubric=variant,
                                      input_file=custom_path if custom_input else None))) == 0
    assert loaded == ([] if custom_input else [(100, 50)])
    assert len(calls) == 2  # one per conversation, no extra pilot
    assert all(call["state"]["reference_standard"] for call in calls)
    saved = json.loads((out / "run.json").read_text())
    assert saved["prompt_version"] == (f"awvs_jev_{variant.replace('-', '_')}" if variant else "awvs_jev_decisions_v4")
    assert saved["repeats"] == 1
    assert saved["reference_standards_sha256"] == jev.experiment.file_sha256(refs)
    assert (out / "reference_standards.json").read_bytes() == refs.read_bytes()
    refs.unlink()  # Resume must rely on the archived standards, not the current source file.
    custom_path.unlink()  # A custom sample is also resumed from its archived copy.
    assert asyncio.run(jev.run(run_args(None, resume=out))) == 0
    assert len(calls) == 2

    old = tmp_path / "v3"
    assert asyncio.run(jev.run(run_args(old, jev_rubric="decisions-v3", repeats=3))) == 0
    assert loaded[-1] == (50, 0)
    assert len(calls) == 8
    old_metadata = json.loads((old / "run.json").read_text())
    old_metadata["jev_rubric"] = "decisions"  # The historical alias meant v3.
    (old / "run.json").write_text(json.dumps(old_metadata))
    assert asyncio.run(jev.run(run_args(None, resume=old))) == 0
    assert len(calls) == 8
    assert len((old / "results.jsonl").read_text().splitlines()) == 6
    assert all("reference_standard" not in call["state"] for call in calls[2:])


@pytest.mark.parametrize("context", ["full", "prefix"])
def test_batched_requests_resume_only_failed_batch(tmp_path, monkeypatch, context):
    c = conversation()
    selection = {"conversations": [c]}
    input_path = tmp_path / "sample.json"
    input_path.write_text(json.dumps(selection))
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({"standards": {"051": reference(c)}}))
    monkeypatch.setattr(jev, "DEFAULT_REFERENCES", refs)
    monkeypatch.setattr(jev.experiment, "load_leaderboard", lambda *args: selection)
    monkeypatch.setattr(jev.experiment, "leaderboard_input_path", lambda *args: input_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    calls, timeouts = [], []

    class Client:
        def __init__(self, **kwargs):
            timeouts.append(kwargs["timeout"])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, json):
            calls.append(deepcopy(json))
            if len(calls) == 4:
                return httpx.Response(400, json={"error": "Deliberate failed batch"})
            return httpx.Response(200, json=answers(json["questions"]))

    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    out = tmp_path / "solar"
    options = dict(model="openrouter/upstage/solar-decide", jev_rubric="decisions-v5", decision_context=context)
    assert asyncio.run(jev.run(run_args(out, **options))) == 1
    assert len(calls) == 9
    assert all(len(call["questions"]) == 9 for call in calls)
    for call in calls:
        target = int(next(iter(call["questions"])).split("_")[1]) if context == "prefix" else 5
        assert set(call["state"]) == {"reference_standard", *(f"turn_{t}" for t in range(1, target + 1))}
        assert call["state"]["reference_standard"] == reference(c)
    refs.unlink()
    assert asyncio.run(jev.run(run_args(None, resume=out, model=options["model"]))) == 0
    assert len(calls) == 10  # Eight successful batches were reused.
    assert timeouts == [180, 180]
    saved = json.loads((out / "results.jsonl").read_text())
    assert saved["error"] is None
    assert len(saved["output"]["answers"]) == 81
    assert len(saved["output"]["batches"]) == 9
    assert saved["turns"]["turn_5"]["recommendation"] == "lowest_harm_primary"
    with pytest.raises(ValueError, match="decision_context differs"):
        asyncio.run(jev.run(run_args(None, resume=out, model=options["model"],
                                    decision_context="full" if context == "prefix" else "prefix")))
