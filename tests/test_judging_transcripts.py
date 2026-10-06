"""Only visible responses reach MANTA's downstream models."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.scorer import Target

from manta import dynamic_multiturn_solver as pressure
from manta import manta_scorer as scoring


def messages():
    return [
        ChatMessageUser(content="Visible question"),
        ChatMessageAssistant(content=[
            ContentReasoning(reasoning="PRIVATE_THOUGHT", summary="PRIVATE_SUMMARY",
                             signature="PRIVATE_SIGNATURE"),
            ContentText(text="Visible answer"),
            ContentReasoning(reasoning="REDACTED_PRIVATE", redacted=True),
            ContentText(text="More visible text"),
        ]),
    ]


@pytest.mark.parametrize("formatter", [scoring.format_conversation,
                                       pressure.format_conversation_for_prompt])
def test_reasoning_removed_without_mutating_log(formatter):
    original = messages()
    saved = deepcopy(original)
    rendered = formatter(original)
    assert "Visible question" in rendered
    assert "Visible answer\nMore visible text" in rendered
    assert "PRIVATE" not in rendered
    assert original == saved


@pytest.mark.parametrize("formatter, expected", [
    (scoring.format_conversation, "[USER]: Hello\n\n[ASSISTANT]: Answer\n"),
    (pressure.format_conversation_for_prompt, "USER: Hello\n\nASSISTANT: Answer"),
])
def test_plain_text_format_unchanged(formatter, expected):
    assert formatter([ChatMessageUser(content="Hello"),
                      ChatMessageAssistant(content="Answer")]) == expected


@pytest.mark.parametrize("target_model, default_judge", [
    ("openai/gpt-5.5", "anthropic/claude-sonnet-4-6"),
    ("anthropic/claude-opus-4-7", "openai/gpt-5.4"),
])
def test_both_metrics_use_judge_role_and_preserve_defaults(monkeypatch, target_model, default_judge):
    requests, roles = [], []

    class Judge:
        async def generate(self, prompt):
            requests.append(prompt)
            return SimpleNamespace(completion="SCORE: 0.8\nEXPLANATION: Visible answer.")

        def __str__(self):
            return "chosen/judge"

    def get_model(*args, **kwargs):
        if not kwargs:
            return SimpleNamespace(name=target_model)
        roles.append(kwargs)
        return Judge()

    monkeypatch.setattr(scoring, "get_model", get_model)
    state = SimpleNamespace(messages=messages() * 3, metadata={})
    result = asyncio.run(scoring.manta_per_turn_scorer(turns_to_score=[3])(state, Target("")))
    assert roles == [{"role": "judge", "default": default_judge}] * 2
    assert len(requests) == 2
    assert all("PRIVATE" not in msg.text for request in requests for msg in request)
    assert result.metadata["per_turn_judge"] == "chosen/judge"
    assert result.metadata["awms_judge"] == "chosen/judge"


def test_pressure_role_and_visible_transcript(monkeypatch):
    roles, prompts = [], []

    class Pressure:
        async def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(completion="Follow-up")

    def get_model(**kwargs):
        roles.append(kwargs)
        return Pressure()

    async def generate(state):
        state.messages.append(messages()[1])
        return state

    monkeypatch.setattr(pressure, "get_model", get_model)
    monkeypatch.setattr(pressure, "transcript", lambda: SimpleNamespace(info=lambda _: None))
    state = SimpleNamespace(messages=[messages()[0]], metadata={}, sample_id="fixture")
    asyncio.run(pressure.dynamic_multi_turn_conversation(turn_count=3, epoch_store=False)(state, generate))
    assert roles == [{"role": "pressure", "default": pressure.FOLLOWUP_GENERATOR_MODEL}]
    assert len(prompts) == 2
    assert all("PRIVATE" not in prompt and "Visible answer" in prompt for prompt in prompts)
