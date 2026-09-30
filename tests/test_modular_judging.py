"""Offline contracts for prompt compatibility and the three judging modes."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.scorer import Target

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


@pytest.mark.parametrize("empty", [(4,), (1, 3, 4, 5)])
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


def reference():
    fields = ["requested_goal", "underlying_goal", "route_basis", "constraints", "welfare_stakes", "within_request_options", "redirection_options", "refusal_boundary", "reference_bar", "equivalence_notes", "uncertainty_notes", "source_ids"]
    return {**dict.fromkeys(fields, "reference"), "initial_scenario_sha256": hashlib.sha256(b"USER_MARKER_1").hexdigest()}


def test_reference_matches_scenario_text_and_rejects_mismatch():
    ref = reference()
    assert judging_decisions.scenario_reference("USER_MARKER_1", {}, {"arbitrary_id": ref}) == ref
    for text, metadata, standards in [
        ("USER_MARKER_1", {}, {}),
        ("different scenario", {"reference_standard": ref}, {}),
    ]:
        with pytest.raises(ValueError):
            judging_decisions.scenario_reference(text, metadata, standards)


@pytest.mark.parametrize("probability", [None, True, -1, 1.5, float("nan")])
def test_invalid_probabilities_do_not_become_zero(probability):
    question = {"turn_3_flag": {"type": "noul"}}
    with pytest.raises(ValueError):
        judging_decisions.parse_answers({"answers": {"turn_3_flag": {"type": "noul", "noul": probability}}}, question)


def test_missing_decision_answer_is_error():
    with pytest.raises(ValueError, match="exactly"):
        judging_decisions.parse_answers({"answers": {}}, {"turn_3_flag": {"type": "noul"}})


def test_awms_aggregate_does_not_count_missing_as_zero():
    import math
    from manta.manta_scorer import mean_awms_turn1
    metric = mean_awms_turn1()
    missing = SimpleNamespace(metadata={"awms_score": None})
    present = SimpleNamespace(metadata={"awms_score": 0.8})
    assert math.isnan(metric([missing]))
    assert metric([missing, present]) == 0.8
