from __future__ import annotations

import json

from tinker_cookbook.recipes.frontier_cs_stage1.run_agentic_advice import (
    AdviceSession,
    AdvisorGeneration,
    Stage1Config,
    StudentGeneration,
    build_advisor_prompt,
    run_agentic_trajectory,
    sample_records_to_jsonl,
    write_trajectory_files,
)
from tinker_cookbook.renderers.base import Message, ToolCall


def make_tool_call(name: str, arguments: dict[str, object], call_id: str = "call_0") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def make_student_generation(tool_call: ToolCall | None) -> StudentGeneration:
    message: Message = {"role": "assistant", "content": ""}
    if tool_call is not None:
        message["tool_calls"] = [tool_call]
    return StudentGeneration(
        message=message,
        raw_text="",
        prompt_tokens=10,
        generated_tokens=5,
        stop_reason="stop",
        termination="stop_sequence",
        clean_termination=True,
    )


class ScriptedStudentSampler:
    def __init__(self, tool_calls: list[ToolCall | None]):
        self._tool_calls = tool_calls
        self.seeds: list[int] = []

    def __call__(self, messages: list[Message], *, seed: int) -> StudentGeneration:
        del messages
        self.seeds.append(seed)
        return make_student_generation(self._tool_calls.pop(0))


class CountingAdvisorSampler:
    def __init__(self, text: str = "Check the edge cases and objective accounting."):
        self.text = text
        self.prompts: list[str] = []
        self.seeds: list[int] = []

    def __call__(self, prompt: str, *, seed: int) -> AdvisorGeneration:
        self.prompts.append(prompt)
        self.seeds.append(seed)
        return AdvisorGeneration(
            text=self.text,
            prompt_tokens=20,
            generated_tokens=7,
            stop_reason="stop",
        )


def test_get_advice_missing_solve_cpp_reports_remaining_budget() -> None:
    session = AdviceSession(problem_id="302", statement="statement", max_advice_calls=5)
    advisor = CountingAdvisorSampler()

    result = session.handle_tool_call(
        make_tool_call("get_advice", {"path": "solve.cpp"}),
        advisor_sampler=advisor,
        advisor_seed=123,
    )

    assert "No file named solve.cpp exists yet" in str(result.message["content"])
    assert "Advice remaining: 5" in str(result.message["content"])
    assert result.event["status"] == "error"
    assert result.should_stop is False
    assert session.advice_calls_used == 0
    assert advisor.prompts == []


def test_get_advice_over_budget_does_not_call_advisor() -> None:
    session = AdviceSession(problem_id="302", statement="statement", max_advice_calls=1)
    advisor = CountingAdvisorSampler()
    code = "int main(){return 0;}"

    first = session.handle_tool_call(
        make_tool_call("get_advice", {"path": "solve.cpp", "content": code}),
        advisor_sampler=advisor,
        advisor_seed=123,
    )
    second = session.handle_tool_call(
        make_tool_call("get_advice", {"path": "solve.cpp"}),
        advisor_sampler=advisor,
        advisor_seed=124,
    )

    assert first.event["status"] == "success"
    assert first.event["advice_remaining"] == 0
    assert second.event["status"] == "over_budget"
    assert "Advice budget exhausted" in str(second.message["content"])
    assert "Advice remaining: 0" in str(second.message["content"])
    assert len(advisor.prompts) == 1


def test_submit_terminal_with_inline_content() -> None:
    session = AdviceSession(problem_id="302", statement="statement", max_advice_calls=5)
    advisor = CountingAdvisorSampler()
    code = "#include <bits/stdc++.h>\nint main(){return 0;}"

    result = session.handle_tool_call(
        make_tool_call("submit", {"path": "solve.cpp", "content": code}),
        advisor_sampler=advisor,
        advisor_seed=123,
    )

    assert result.should_stop is True
    assert result.submitted_code == code
    assert result.event["status"] == "submitted"
    assert "Advice remaining: 5" in str(result.message["content"])
    assert advisor.prompts == []


def test_run_agentic_trajectory_advice_then_submit() -> None:
    code_v1 = "int main(){return 0;}"
    code_v2 = "#include <bits/stdc++.h>\nint main(){return 0;}"
    student = ScriptedStudentSampler(
        [
            make_tool_call("get_advice", {"path": "solve.cpp", "content": code_v1}),
            make_tool_call("submit", {"path": "solve.cpp", "content": code_v2}),
        ]
    )
    advisor = CountingAdvisorSampler()

    record = run_agentic_trajectory(
        index=0,
        seed=10,
        problem_id="302",
        statement="statement",
        initial_messages=[{"role": "user", "content": "solve"}],
        student_sampler=student,
        advisor_sampler=advisor,
        max_student_turns=6,
        max_advice_calls=5,
    )

    assert record.status == "pending_eval"
    assert record.code == code_v2
    assert (record.metadata or {})["termination_reason"] == "submitted"
    assert (record.metadata or {})["advice_calls_used"] == 1
    assert [turn["status"] for turn in record.turns or []] == ["success", "submitted"]
    assert len(advisor.prompts) == 1
    assert "Student's current solve.cpp" in advisor.prompts[0]
    assert code_v1 in advisor.prompts[0]


def test_run_agentic_trajectory_no_submit_scores_zero() -> None:
    student = ScriptedStudentSampler(
        [
            make_tool_call(
                "get_advice",
                {"path": "solve.cpp", "content": "int main(){return 0;}"},
            )
        ]
    )
    advisor = CountingAdvisorSampler()

    record = run_agentic_trajectory(
        index=0,
        seed=10,
        problem_id="302",
        statement="statement",
        initial_messages=[{"role": "user", "content": "solve"}],
        student_sampler=student,
        advisor_sampler=advisor,
        max_student_turns=1,
        max_advice_calls=1,
    )

    assert record.status == "no_submit"
    assert record.reward == 0.0
    assert record.score == 0.0
    assert (record.metadata or {})["termination_reason"] == "no_submit"


def test_advisor_prompt_uses_problem_and_code_only() -> None:
    prompt = build_advisor_prompt(
        problem_id="302",
        statement="problem statement",
        code="int main(){return 0;}",
    )

    assert "problem statement" in prompt
    assert "int main()" in prompt
    assert "Compiler error" not in prompt
    assert "checker" not in prompt.lower()


def test_sample_records_to_jsonl_omits_full_messages() -> None:
    student = ScriptedStudentSampler(
        [make_tool_call("submit", {"path": "solve.cpp", "content": "int main(){return 0;}"})]
    )
    advisor = CountingAdvisorSampler()

    record = run_agentic_trajectory(
        index=0,
        seed=10,
        problem_id="302",
        statement="statement",
        initial_messages=[{"role": "user", "content": "solve"}],
        student_sampler=student,
        advisor_sampler=advisor,
        max_student_turns=1,
        max_advice_calls=0,
    )
    payload = sample_records_to_jsonl([record])[0]

    assert "messages" not in payload["metadata"]
    json.dumps(payload)


def test_write_trajectory_files_serializes_tool_calls(tmp_path) -> None:
    student = ScriptedStudentSampler(
        [make_tool_call("submit", {"path": "solve.cpp", "content": "int main(){return 0;}"})]
    )
    advisor = CountingAdvisorSampler()
    record = run_agentic_trajectory(
        index=0,
        seed=10,
        problem_id="302",
        statement="statement",
        initial_messages=[{"role": "user", "content": "solve"}],
        student_sampler=student,
        advisor_sampler=advisor,
        max_student_turns=1,
        max_advice_calls=0,
    )

    write_trajectory_files(tmp_path, [record], Stage1Config(problem_id="302"))

    payload = json.loads((tmp_path / "trajectories" / "sample_000.json").read_text())
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "submit"
    assert payload["turns"][0]["tool_call"]["function"]["name"] == "submit"
