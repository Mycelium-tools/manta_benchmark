"""Exercise real Inspect task execution and .eval serialization, entirely offline."""

import hashlib
import io
import json
from pathlib import Path

import httpx
import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, ModelUsage, get_model

from manta.dynamic_multiturn_solver import dynamic_multi_turn_conversation
from manta.judging import manta_judging_scorer


def output(model, content):
    result = ModelOutput.from_content(model=model, content=content)
    result.usage = ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    return result


def reference():
    fields = ["requested_goal", "underlying_goal", "route_basis", "constraints", "welfare_stakes", "within_request_options", "redirection_options", "refusal_boundary", "reference_bar", "equivalence_notes", "uncertainty_notes", "source_ids"]
    return {**dict.fromkeys(fields, "fixture reference"), "initial_scenario_sha256": hashlib.sha256(b"USER_MARKER_1").hexdigest()}


def decisions_output(questions):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.5}
        else:
            categories = list(question["criteria"])
            answers[key] = {
                "type": "choice", "choice": categories[-1], "confidence": 0.9,
                "probabilities": {category: float(index == 0) for index, category in enumerate(categories)},
            }
    return {"answers": answers, "model": "typesafe/integration-fixture", "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.001}}


@pytest.mark.parametrize("variant", ["original", "joint", "decisions-v3", "decisions-v5"])
def test_inspect_execution_and_log_roundtrip(monkeypatch, tmp_path, variant):
    """Use actual named roles, solver, scorer, Inspect runner, and binary log reader."""
    # Optional installed plugins (e.g. tracking integrations) are outside this
    # runtime test and can create background telemetry/network clients.
    from inspect_ai._util import entrypoints
    monkeypatch.setattr(entrypoints, "entry_points", lambda **kwargs: [])
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("INSPECT_TRACE_FILE", str(tmp_path / "inspect-trace.log"))
    pressure_calls, target_calls, judge_calls, decision_calls = [], [], [], []

    def pressure_response(messages, tools, choice, config):
        pressure_calls.append(messages)
        assert all("PRIVATE_REASONING" not in message.text for message in messages)
        return output("mockllm/pressure-fixture", f"USER_MARKER_{len(pressure_calls) + 1}")

    def target_response(messages, tools, choice, config):
        target_calls.append(messages)
        turn = len(target_calls)
        return output("mockllm/target-fixture", [
            ContentReasoning(reasoning=f"PRIVATE_REASONING_{turn}"),
            ContentText(text=f"ANSWER_MARKER_{turn}"),
        ])

    def judge_response(messages, tools, choice, config):
        judge_calls.append(messages)
        text = "\n".join(message.text for message in messages)
        assert "PRIVATE_REASONING" not in text
        if "Animal Welfare Moral Sensitivity" in text:
            assert "ANSWER_MARKER_1" in text
            assert all(f"ANSWER_MARKER_{turn}" not in text for turn in (2, 3, 4, 5))
            return output("mockllm/judge-fixture", "SCORE: 0.7\nEXPLANATION: Welfare recognition.")
        if variant == "joint":
            return output("mockllm/judge-fixture", json.dumps({f"turn_{turn}": {"score": 0.8, "explanation": "Maintained welfare."} for turn in (3, 4, 5)}))
        return output("mockllm/judge-fixture", "SCORE: 0.8\nEXPLANATION: Maintained welfare.")

    def handle_request(request):
        assert request.method == "POST" and str(request.url) == "https://openrouter.ai/api/alpha/decisions", f"Unexpected HTTP request: {request.method} {request.url}"
        assert request.headers["authorization"] == "Bearer integration-test-only"
        payload = json.loads(request.content)
        decision_calls.append(payload)
        assert "PRIVATE_REASONING" not in json.dumps(payload)
        return httpx.Response(200, json=decisions_output(payload["questions"]))

    transport = httpx.MockTransport(handle_request)

    async def fake_async_request(self, request):
        return await transport.handle_async_request(request)

    def forbid_sync_request(self, request):
        raise AssertionError(f"Unexpected synchronous HTTP request: {request.method} {request.url}")

    # Intercept the HTTP transport itself, preserving real AsyncClient and role
    # initialization. No HTTP request can reach an external provider.
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_async_request)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_sync_request)
    monkeypatch.setenv("OPENROUTER_API_KEY", "integration-test-only")
    target = get_model("mockllm/target-fixture", custom_outputs=target_response, memoize=False)
    pressure = get_model("mockllm/pressure-fixture", custom_outputs=pressure_response, memoize=False)
    is_decisions = variant.startswith("decisions-")
    judge = (get_model("openrouter/typesafe/integration-fixture", memoize=False) if is_decisions else
             get_model("mockllm/judge-fixture", custom_outputs=judge_response, memoize=False))
    metadata = {"pressure": ["economic", "social", "cultural"]}
    if variant == "decisions-v5":
        metadata["reference_standard"] = reference()
    task = Task(
        name=f"manta_integration_{variant}",
        dataset=[Sample(input="USER_MARKER_1", id="fixture", metadata=metadata)],
        solver=dynamic_multi_turn_conversation(turn_count=5, epoch_store=False),
        scorer=manta_judging_scorer("decisions" if is_decisions else variant, decision_rubric=variant if is_decisions else "decisions-v3"),
    )
    artifacts = tmp_path / variant
    artifacts.mkdir(parents=True, exist_ok=True)
    logs = inspect_eval(
        task, model=target, model_roles={"pressure": pressure, "judge": judge},
        log_dir=str(artifacts), log_format="eval", log_model_api=True,
        max_samples=1, fail_on_error=True, debug_errors=True,
    )
    assert len(logs) == 1 and logs[0].status == "success"
    # Read the actual persisted binary bytes. This avoids reusing Inspect's
    # filesystem singleton across the separate event loops of eval() and the
    # synchronous log reader, while exercising the same .eval decoder.
    saved = read_eval_log(io.BytesIO(Path(logs[0].location).read_bytes()), resolve_attachments=True)
    assert saved.status == "success" and len(saved.samples) == 1
    sample = saved.samples[0]
    assert sample.error is None and len(sample.messages) == 10
    assert len(target_calls) == 5 and len(pressure_calls) == 4
    assert any("PRIVATE_REASONING_1" in message.model_dump_json() for message in sample.messages)
    assert saved.eval.model_roles["pressure"].model == "mockllm/pressure-fixture"
    score = next(iter(sample.scores.values()))
    events = [event for event in sample.events if event.event == "info" and event.source == "manta.decisions"]

    if is_decisions:
        assert not judge_calls and len(decision_calls) == 4
        assert saved.eval.model_roles["judge"].model == "openrouter/typesafe/integration-fixture"
        assert len(events) == 4
        assert sum(event.data["response"]["usage"]["cost"] for event in events) == 0.004
        assert isinstance(score.value, dict) and score.value["awms_turn_1_explicit_welfare_concern"] == 1
        assert all(isinstance(value, (int, str)) for value in score.value.values())
        assert any(isinstance(value, str) for value in score.value.values())
        assert "3turn_mean" not in score.metadata and "awms_score" not in score.metadata
        assert all(not result.metrics for result in saved.results.scores)
        for turn, payload in zip((1, 3, 4, 5), decision_calls):
            assert payload["model"] == "typesafe/integration-fixture"
            assert {key for key in payload["state"] if key.startswith("turn_")} == {f"turn_{number}" for number in range(1, turn + 1)}
            assert all(key.startswith(f"turn_{turn}_") for key in payload["questions"])
            assert ("reference_standard" in payload["state"]) == (variant == "decisions-v5" and turn >= 3)
        assert score.metadata["decision_responses"]["turn_3"]["usage"]["cost"] == 0.001
    else:
        assert not decision_calls and not events
        assert len(judge_calls) == (6 if variant == "original" else 2)
        assert saved.eval.model_roles["judge"].model == "mockllm/judge-fixture"
        assert score.value == 0.8 and score.metadata["awms_score"] == 0.7
        if variant == "original":
            for turn, request in enumerate(judge_calls[:5], 1):
                text = "\n".join(message.text for message in request)
                assert f"ANSWER_MARKER_{turn}" in text
                assert all(f"ANSWER_MARKER_{later}" not in text for later in range(turn + 1, 6))
        else:
            assert all(len(request) == 1 and request[0].role == "user" for request in judge_calls)
