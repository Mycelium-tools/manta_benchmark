"""Offline contracts for prompt compatibility and the three judging modes."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.scorer import Target
from inspect_ai.scorer._scorer import as_scorer_spec

from manta import judging, judging_decisions, judging_prompts


def messages(empty=()):
    result = [ChatMessageSystem(content="Task setup")]
    for turn in range(1, 6):
        result.append(ChatMessageUser(content=f"USER_MARKER_{turn}"))
        result.append(ChatMessageAssistant(content=[
            ContentReasoning(reasoning=f"PRIVATE_REASONING_{turn}"),
            ContentText(text="" if turn in empty else f"ANSWER_MARKER_{turn}"),
        ]))
    return result


def state(empty=()):
    return SimpleNamespace(messages=messages(empty), metadata={"pressure_types": ["economic", "social", "cultural"]})


def test_original_prompts_match_first_commit_byte_for_byte():
    fixture = json.loads((Path(__file__).parent / "fixtures/original_prompt_hashes.json").read_text())
    for case in fixture["cases"]:
        prompts = getattr(judging_prompts, case["function"])(*case["args"])
        assert [hashlib.sha256(value.encode()).hexdigest() for value in prompts] == case["sha256"]


@pytest.mark.parametrize("mode,scorer_name", [
    ("original", "manta_per_turn_scorer"), ("joint", "manta_joint_scorer"), ("decisions", "manta_decisions_scorer"),
])
def test_mode_selects_distinct_inspect_scorer(mode, scorer_name):
    spec = as_scorer_spec(judging.manta_judging_scorer(mode))
    assert spec.scorer.endswith(scorer_name)
    if mode == "decisions":
        assert not spec.metrics  # Do not manufacture a headline from components.


def test_unknown_mode_fails():
    with pytest.raises(ValueError, match="judging_mode"):
        judging.manta_judging_scorer("typo")


@pytest.mark.parametrize("empty", [(), (4,), (1, 3, 4, 5)])
def test_joint_routing_visible_text_and_missingness(monkeypatch, empty):
    calls, roles = [], []
    targets = [turn for turn in (3, 4, 5) if turn not in empty]

    class Judge:
        async def generate(self, prompt):
            calls.append(prompt)
            assert len(prompt) == 1 and prompt[0].role == "user"
            assert "PRIVATE_REASONING" not in prompt[0].text
            if "--- TURN 3;" in prompt[0].text:
                return SimpleNamespace(completion=json.dumps({f"turn_{turn}": {"score": 0.8, "explanation": "Evidence."} for turn in targets}))
            assert "ANSWER_MARKER_2" not in prompt[0].text
            assert "USER_MARKER_2" not in prompt[0].text
            return SimpleNamespace(completion="SCORE: 0.7\nEXPLANATION: Welfare surfaced.")

        def __str__(self):
            return "openrouter/chosen/judge"

    def get_model(**kwargs):
        if not kwargs:
            return SimpleNamespace(name="anthropic/claude-target")
        roles.append(kwargs)
        return Judge()

    monkeypatch.setattr(judging, "get_model", get_model)
    result = asyncio.run(judging.manta_judging_scorer("joint")(state(empty), Target("")))
    assert result.value == (0.8 if len(targets) == 3 else "NA")
    assert result.metadata["awms_score"] == (None if 1 in empty else 0.7)
    assert result.metadata["missing_or_empty_turns"] == [turn for turn in (3, 4, 5) if turn in empty]
    assert len(calls) == bool(targets) + (1 not in empty)
    assert all(role == {"role": "judge", "default": "openai/gpt-5.4"} for role in roles)


@pytest.mark.parametrize("answer", ["", "SCORE: invalid", "SCORE: NaN", "SCORE: 2", "SCORE: 0.2\nSCORE: 0.3"])
def test_joint_awms_malformed_is_error(monkeypatch, answer):
    class Judge:
        async def generate(self, prompt):
            return SimpleNamespace(completion=answer)
        def __str__(self):
            return "openrouter/test"
    monkeypatch.setattr(judging, "get_model", lambda **kw: Judge() if kw else SimpleNamespace(name="target"))
    with pytest.raises(ValueError):
        asyncio.run(judging.manta_joint_scorer()(state(empty=(3, 4, 5)), Target("")))


@pytest.mark.parametrize("bad", [
    {}, {"turn_3": {"score": 0.7, "explanation": "e"}},
    {"turn_3": {"score": True, "explanation": "e"}, "turn_4": {"score": 0.7, "explanation": "e"}},
    {"turn_3": {"score": 2, "explanation": "e"}, "turn_4": {"score": 0.7, "explanation": "e"}},
])
def test_joint_schema_rejects_missing_or_invalid_scores(bad):
    with pytest.raises(ValueError):
        judging.parse_joint_response(json.dumps(bad), [3, 4])


def test_agentic_turns_stay_with_user_and_exclude_reasoning():
    source = messages()
    source[2:2] = [ChatMessageAssistant(content="Searching"), ChatMessageTool(content="SEARCH_RESULT", tool_call_id="tool")]
    turns = judging.conversation_turns(source)
    assert len(turns) == 5
    assert "Searching" in judging.assistant_text(turns[0])
    assert "ANSWER_MARKER_1" in judging.assistant_text(turns[0])
    assert "ANSWER_MARKER_2" not in judging.assistant_text(turns[0])
    payload = judging_decisions.make_state(turns[:3], [])
    assert payload["turn_1"]["tool_context"] == ["SEARCH_RESULT"]
    assert "PRIVATE_REASONING" not in json.dumps(payload)
    assert "turn_4" not in payload


def decision_response(questions):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.5}
        else:
            labels = list(question["criteria"])
            answers[key] = {"type": "choice", "choice": labels[-1], "probabilities": {label: float(i == 0) for i, label in enumerate(labels)}}
    return {"answers": answers, "usage": {"cost": 0.01}, "model": "typesafe/mock"}


@pytest.fixture
def decisions_api(monkeypatch):
    calls, events, roles = [], [], []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json=decision_response(payload["questions"]))
    original_client = httpx.AsyncClient
    monkeypatch.setattr(judging_decisions.httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(judging_decisions, "transcript", lambda: SimpleNamespace(info=lambda data, **kwargs: events.append(data)))
    class Judge:
        def __str__(self):
            return "openrouter/typesafe/chosen"
    def get_model(**kwargs):
        roles.append(kwargs)
        return Judge()
    monkeypatch.setattr(judging_decisions, "get_model", get_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    return calls, events, roles


@pytest.mark.parametrize("empty", [(), (4,), (1, 3, 4, 5)])
def test_decisions_end_to_end_prefixes_and_component_scores(decisions_api, empty):
    calls, events, roles = decisions_api
    result = asyncio.run(judging.manta_judging_scorer("decisions")(state(empty), Target("")))
    targets = [turn for turn in (1, 3, 4, 5) if turn not in empty]
    assert len(calls) == len(targets)
    assert len(events) == len(targets)
    assert roles == [{"role": "judge", "default": "openrouter/typesafe/jev-1.13"}]
    for turn, payload in zip(targets, calls):
        assert payload["model"] == "typesafe/chosen"
        assert set(payload["state"]) == {f"turn_{t}" for t in range(1, turn + 1)}
        assert all(key.startswith(f"turn_{turn}_") for key in payload["questions"])
        assert "PRIVATE_REASONING" not in json.dumps(payload)
        assert "messages" not in payload
    if targets:
        assert isinstance(result.value, dict)
        assert result.metadata["decision_responses"][f"turn_{targets[0]}"]["usage"]["cost"] == 0.01
        assert result.value.get("awms_turn_1_explicit_welfare_concern") == (1 if 1 not in empty else None)
        if 3 not in empty:
            assert result.value["awvs_turn_3_recommendation"] == "lowest_harm_primary"  # mode, not API choice
    else:
        assert result.value == "NA"
    assert "3turn_mean" not in result.metadata and "awms_score" not in result.metadata
    assert result.metadata["missing_or_empty_turns"] == list(empty)


def reference():
    fields = ["requested_goal", "underlying_goal", "route_basis", "constraints", "welfare_stakes", "within_request_options", "redirection_options", "refusal_boundary", "reference_bar", "equivalence_notes", "uncertainty_notes", "source_ids"]
    return {**dict.fromkeys(fields, "reference"), "initial_scenario_sha256": hashlib.sha256(b"USER_MARKER_1").hexdigest()}


def test_reference_aware_mode_requires_exact_reference_before_calls(decisions_api):
    calls, _, _ = decisions_api
    with pytest.raises(ValueError, match="reference_standard"):
        asyncio.run(judging.manta_judging_scorer("decisions", decision_rubric="decisions-v5")(state(), Target("")))
    assert not calls
    sample = state()
    sample.metadata["reference_standard"] = {**reference(), "initial_scenario_sha256": "wrong"}
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(judging.manta_judging_scorer("decisions", decision_rubric="decisions-v5")(sample, Target("")))
    assert not calls


def test_reference_file_matches_text_not_sample_id(decisions_api, tmp_path):
    calls, _, _ = decisions_api
    path = tmp_path / "references.json"
    path.write_text(json.dumps({"standards": {"arbitrary_export_id": reference()}}))
    result = asyncio.run(judging.manta_judging_scorer("decisions", decision_rubric="decisions-v5", reference_standards=str(path))(state(), Target("")))
    assert len(calls) == 4
    assert "reference_standard" not in calls[0]["state"]
    assert all(call["state"]["reference_standard"] == reference() for call in calls[1:])
    assert result.metadata["reference_standard"] == reference()


@pytest.mark.parametrize("probability", [None, True, -1, 1.5, float("nan")])
def test_invalid_probabilities_do_not_become_zero(probability):
    question = {"turn_3_flag": {"type": "noul"}}
    with pytest.raises(ValueError):
        judging_decisions.parse_answers({"answers": {"turn_3_flag": {"type": "noul", "noul": probability}}}, question)


def test_missing_decision_answer_is_error():
    with pytest.raises(ValueError, match="exactly"):
        judging_decisions.parse_answers({"answers": {}}, {"turn_3_flag": {"type": "noul"}})


@pytest.mark.parametrize("task_name", ["manta_test5", "manta_5turn", "manta_agentic_5turn"])
@pytest.mark.parametrize("mode", ["original", "joint", "decisions"])
def test_five_turn_tasks_expose_judging_modes(monkeypatch, task_name, mode):
    from inspect_ai.dataset import Sample
    from manta import manta_eval
    monkeypatch.setattr(manta_eval, "load_samples", lambda: [Sample(input="Example", target="{}")])
    task = getattr(manta_eval, task_name)(judging_mode=mode)
    assert len(task.scorer) == 1
    expected = {"original": "manta_per_turn_scorer", "joint": "manta_joint_scorer", "decisions": "manta_decisions_scorer"}[mode]
    assert as_scorer_spec(task.scorer[0]).scorer.endswith(expected)


def test_awms_aggregate_does_not_count_missing_as_zero():
    import math
    from manta.manta_scorer import mean_awms_turn1
    metric = mean_awms_turn1()
    missing = SimpleNamespace(metadata={"awms_score": None})
    present = SimpleNamespace(metadata={"awms_score": 0.8})
    assert math.isnan(metric([missing]))
    assert metric([missing, present]) == 0.8
