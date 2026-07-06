"""Run Stage 1 agentic advice rollouts on Frontier-CS algorithmic tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import chz

from tinker_cookbook import model_info, renderers, tokenizer_utils
from tinker_cookbook.recipes.frontier_cs_stage0.run_baseline import (
    DEFAULT_FRONTIER_CS_ROOT,
    DEFAULT_QWEN_REPO,
    SampleRecord,
    _reject_home_output,
    _write_json,
    _write_jsonl,
    load_problem_statement,
    sample_record_to_dict,
    score_samples,
    write_sample_files,
    write_score_artifacts,
)
from tinker_cookbook.renderers.base import Message, ToolCall, ToolSpec, format_content_as_string
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.ml_log import dump_config

SCRATCH_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524")
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "tinker_runs" / "frontier_cs_stage1"
DEFAULT_RENDERER_NAME = "qwen3"
DEFAULT_SOLVE_PATH = "solve.cpp"

STUDENT_SYSTEM_PROMPT = """You are a strong competitive programmer solving Frontier-CS algorithmic tasks.

You have a virtual workspace that may contain `solve.cpp`.
Call exactly one tool at a time.
Every assistant response must be a structured tool call, not plain text.
Use `get_advice` when you want critique of your current `solve.cpp`.
Use `submit` only when your final C++17 solution is ready.
To write or update `solve.cpp`, pass the complete file contents in the tool call's `content` argument.
Tool responses will tell you how many advice calls remain.
The final answer is graded only after a valid `submit` call."""

STUDENT_USER_PROMPT_TEMPLATE = """Solve Frontier-CS algorithmic problem {problem_id}.

Write a complete C++17 program in `solve.cpp`. The program must read from stdin and write the required output to stdout.
You may ask for advice before submitting, but advice calls are limited. When ready, call `submit` with `path` set to `solve.cpp`.
{advice_requirement}
Do not answer in prose or Markdown. Your assistant turns must be tool calls.

Problem statement:

{statement}
"""

ADVISOR_SYSTEM_PROMPT = """You are an advisor for a competitive-programming student.

Give critique and improvement advice for the student's current `solve.cpp`.
Do not provide a full replacement solution.
Do not output a complete source file.
Do not use a code block containing a full program.
Focus on likely bugs, missed constraints, misunderstood objective details, complexity issues, and concrete next steps."""

ADVISOR_USER_PROMPT_TEMPLATE = """Frontier-CS algorithmic problem {problem_id}.

Problem statement:

{statement}

Student's current solve.cpp:

```cpp
{code}
```

Give concise advice to improve this solution. Do not write a complete replacement solution."""

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "get_advice",
        "description": (
            "Ask for critique of the current solve.cpp. Provide content to create or replace "
            "solve.cpp before asking. This consumes one advice call when solve.cpp exists and "
            "advice budget remains."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file to critique. Use solve.cpp.",
                },
                "content": {
                    "type": "string",
                    "description": "Optional full contents to write to solve.cpp before critique.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit the final solve.cpp for grading. Provide content to create or replace "
            "solve.cpp before submitting. A valid submit ends the trajectory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file to submit. Use solve.cpp.",
                },
                "content": {
                    "type": "string",
                    "description": "Optional full contents to write to solve.cpp before submitting.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


@chz.chz
class Stage1Config:
    """Configuration for Frontier-CS Stage 1 agentic advice evaluation."""

    frontier_cs_root: str = str(DEFAULT_FRONTIER_CS_ROOT)
    problem_id: str = "302"
    model_name: str = DEFAULT_QWEN_REPO
    tokenizer_name: str | None = None
    renderer_name: str | None = DEFAULT_RENDERER_NAME
    checkpoint_url: str | None = None
    base_url: str | None = "http://127.0.0.1:8000/"
    output_dir: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str | None = None
    num_samples: int = 1
    max_tokens: int = 8192
    max_prompt_tokens: int = 24576
    max_student_turns: int = 6
    max_advice_calls: int = 5
    min_advice_calls: int = 0
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    advisor_model_name: str | None = None
    advisor_tokenizer_name: str | None = None
    advisor_renderer_name: str | None = None
    advisor_checkpoint_url: str | None = None
    advisor_max_tokens: int = 1024
    advisor_temperature: float = 0.2
    advisor_top_p: float = 1.0
    advisor_top_k: int = -1
    save_trajectories: bool = True
    evaluate: bool = True
    evaluator: str = "direct_checker"
    judge_url: str = "http://127.0.0.1:8081"
    auto_start_judge: bool = False
    score_scale: float = 100.0
    compile_timeout: int = 60
    case_timeout: int = 20
    eval_workers: int = 1


@dataclass(frozen=True)
class StudentGeneration:
    message: Message
    raw_text: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str | None
    termination: str
    clean_termination: bool


@dataclass(frozen=True)
class AdvisorGeneration:
    text: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str | None


class StudentSampler(Protocol):
    def __call__(self, messages: list[Message], *, seed: int) -> StudentGeneration:
        ...


class AdvisorSampler(Protocol):
    def __call__(self, prompt: str, *, seed: int) -> AdvisorGeneration:
        ...


@dataclass
class VirtualWorkspace:
    files: dict[str, str] = field(default_factory=dict)

    def update_and_resolve(self, *, path: str, content: str | None) -> tuple[str | None, str | None]:
        if path != DEFAULT_SOLVE_PATH:
            return None, f"Expected path `{DEFAULT_SOLVE_PATH}`, got `{path}`."
        if content is not None:
            self.files[path] = content
        if path not in self.files:
            return None, f"No file named {path} exists yet."
        return self.files[path], None


@dataclass
class ToolExecution:
    message: Message
    event: dict[str, Any]
    should_stop: bool
    submitted_code: str | None = None


@dataclass
class AdviceSession:
    problem_id: str
    statement: str
    max_advice_calls: int
    min_advice_calls: int = 0
    workspace: VirtualWorkspace = field(default_factory=VirtualWorkspace)
    advice_calls_used: int = 0

    @property
    def advice_remaining(self) -> int:
        return max(0, self.max_advice_calls - self.advice_calls_used)

    def handle_tool_call(
        self,
        tool_call: ToolCall,
        *,
        advisor_sampler: AdvisorSampler,
        advisor_seed: int,
    ) -> ToolExecution:
        name = tool_call.function.name
        arguments, parse_error = parse_tool_arguments(tool_call.function.arguments)
        if parse_error is not None:
            return self._tool_error(
                tool_call,
                f"Could not parse tool arguments: {parse_error}",
                error_type="argument_parse_error",
            )
        if name == "get_advice":
            return self._handle_get_advice(tool_call, arguments, advisor_sampler, advisor_seed)
        if name == "submit":
            return self._handle_submit(tool_call, arguments)
        return self._tool_error(tool_call, f"Unknown tool `{name}`.", error_type="unknown_tool")

    def _handle_get_advice(
        self,
        tool_call: ToolCall,
        arguments: dict[str, Any],
        advisor_sampler: AdvisorSampler,
        advisor_seed: int,
    ) -> ToolExecution:
        code, error = self._resolve_code(arguments)
        if error is not None:
            return self._tool_error(tool_call, error, error_type="missing_or_invalid_file")
        if self.advice_remaining <= 0:
            content = (
                "Advice budget exhausted. Advice remaining: 0. "
                "Please submit your final solution with submit."
            )
            return self._tool_message(
                tool_call,
                content,
                {
                    "tool": "get_advice",
                    "status": "over_budget",
                    "advice_remaining": 0,
                },
                should_stop=False,
            )

        self.advice_calls_used += 1
        assert code is not None
        prompt = build_advisor_prompt(
            problem_id=self.problem_id,
            statement=self.statement,
            code=code,
        )
        advisor_generation = advisor_sampler(prompt, seed=advisor_seed)
        content = (
            f"Advice for {DEFAULT_SOLVE_PATH}:\n\n"
            f"{advisor_generation.text.strip()}\n\n"
            f"Advice remaining: {self.advice_remaining}"
        )
        return self._tool_message(
            tool_call,
            content,
            {
                "tool": "get_advice",
                "status": "success",
                "advice_remaining": self.advice_remaining,
                "advice_calls_used": self.advice_calls_used,
                "advisor_prompt": prompt,
                "advisor_text": advisor_generation.text,
                "advisor_prompt_tokens": advisor_generation.prompt_tokens,
                "advisor_generated_tokens": advisor_generation.generated_tokens,
                "advisor_stop_reason": advisor_generation.stop_reason,
            },
            should_stop=False,
        )

    def _handle_submit(self, tool_call: ToolCall, arguments: dict[str, Any]) -> ToolExecution:
        code, error = self._resolve_code(arguments)
        if error is not None:
            return self._tool_error(tool_call, error, error_type="missing_or_invalid_file")
        if self.advice_calls_used < self.min_advice_calls:
            content = (
                f"Submit is not allowed yet. Call get_advice at least "
                f"{self.min_advice_calls} time(s) before submit. "
                f"Advice remaining: {self.advice_remaining}"
            )
            return self._tool_message(
                tool_call,
                content,
                {
                    "tool": "submit",
                    "status": "advice_required",
                    "advice_remaining": self.advice_remaining,
                    "advice_calls_used": self.advice_calls_used,
                    "min_advice_calls": self.min_advice_calls,
                },
                should_stop=False,
            )
        assert code is not None
        return self._tool_message(
            tool_call,
            f"Submitted {DEFAULT_SOLVE_PATH}. Advice remaining: {self.advice_remaining}",
            {
                "tool": "submit",
                "status": "submitted",
                "advice_remaining": self.advice_remaining,
                "code_chars": len(code),
            },
            should_stop=True,
            submitted_code=code,
        )

    def _resolve_code(self, arguments: dict[str, Any]) -> tuple[str | None, str | None]:
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or path.strip() == "":
            return None, "Tool argument `path` must be the string `solve.cpp`."
        if content is not None and not isinstance(content, str):
            return None, "Tool argument `content` must be a string when provided."
        code, error = self.workspace.update_and_resolve(path=path, content=content)
        if error is None:
            return code, None
        return (
            None,
            (
                f"{error} Provide complete source in the `content` field when calling "
                f"`get_advice` or `submit`. Advice remaining: {self.advice_remaining}"
            ),
        )

    def _tool_error(
        self,
        tool_call: ToolCall,
        content: str,
        *,
        error_type: str,
    ) -> ToolExecution:
        if "Advice remaining:" not in content:
            content = f"{content} Advice remaining: {self.advice_remaining}"
        return self._tool_message(
            tool_call,
            content,
            {
                "tool": tool_call.function.name,
                "status": "error",
                "error_type": error_type,
                "advice_remaining": self.advice_remaining,
            },
            should_stop=False,
        )

    def _tool_message(
        self,
        tool_call: ToolCall,
        content: str,
        event: dict[str, Any],
        *,
        should_stop: bool,
        submitted_code: str | None = None,
    ) -> ToolExecution:
        return ToolExecution(
            message={
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call.id or "",
                "name": tool_call.function.name,
            },
            event=event,
            should_stop=should_stop,
            submitted_code=submitted_code,
        )


def parse_tool_arguments(arguments_json: str) -> tuple[dict[str, Any], str | None]:
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(arguments, dict):
        return {}, "tool arguments must be a JSON object"
    return arguments, None


def build_advisor_prompt(*, problem_id: str, statement: str, code: str) -> str:
    return ADVISOR_USER_PROMPT_TEMPLATE.format(
        problem_id=problem_id,
        statement=statement,
        code=code,
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, ToolCall):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sample_record_without_full_messages(record: SampleRecord) -> dict[str, Any]:
    payload = sample_record_to_dict(record)
    payload["metadata"] = {
        key: value
        for key, value in (record.metadata or {}).items()
        if key != "messages"
    }
    return to_jsonable(payload)


def make_initial_messages(
    *,
    renderer: renderers.Renderer,
    problem_id: str,
    statement: str,
    min_advice_calls: int = 0,
) -> list[Message]:
    prefix = renderer.create_conversation_prefix_with_tools(
        TOOL_SPECS,
        system_prompt=STUDENT_SYSTEM_PROMPT,
    )
    return prefix + [
        {
            "role": "user",
            "content": STUDENT_USER_PROMPT_TEMPLATE.format(
                problem_id=problem_id,
                statement=statement,
                advice_requirement=advice_requirement_text(min_advice_calls),
            ),
        }
    ]


def advice_requirement_text(min_advice_calls: int) -> str:
    if min_advice_calls <= 0:
        return ""
    return (
        f"For this run, call `get_advice` at least {min_advice_calls} time(s) "
        "before your final `submit` call."
    )


def run_agentic_trajectory(
    *,
    index: int,
    seed: int,
    problem_id: str,
    statement: str,
    initial_messages: list[Message],
    student_sampler: StudentSampler,
    advisor_sampler: AdvisorSampler,
    max_student_turns: int,
    max_advice_calls: int,
    min_advice_calls: int = 0,
) -> SampleRecord:
    session = AdviceSession(
        problem_id=problem_id,
        statement=statement,
        max_advice_calls=max_advice_calls,
        min_advice_calls=min_advice_calls,
    )
    messages = list(initial_messages)
    turns: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    submitted_code: str | None = None
    termination_reason = "max_turns"
    started = time.monotonic()

    for turn in range(1, max_student_turns + 1):
        turn_seed = seed + turn - 1
        generation = student_sampler(messages, seed=turn_seed)
        messages.append(generation.message)

        turn_summary: dict[str, Any] = {
            "turn": turn,
            "seed": turn_seed,
            "prompt_tokens": generation.prompt_tokens,
            "generated_tokens": generation.generated_tokens,
            "stop_reason": generation.stop_reason,
            "termination": generation.termination,
            "clean_termination": generation.clean_termination,
        }
        turn_event: dict[str, Any] = {
            **turn_summary,
            "assistant_message": generation.message,
            "raw_text": generation.raw_text,
        }

        if not generation.clean_termination:
            termination_reason = "parse_error"
            turn_summary["status"] = "parse_error"
            turns.append(turn_summary)
            trajectory.append(turn_event)
            break

        tool_calls = list(generation.message.get("tool_calls") or [])
        if not tool_calls:
            termination_reason = "no_tool_call"
            turn_summary["status"] = "no_tool_call"
            turns.append(turn_summary)
            trajectory.append(turn_event)
            break

        if len(tool_calls) > 1:
            turn_summary["ignored_tool_calls"] = len(tool_calls) - 1
            turn_event["ignored_tool_calls"] = [
                tool_call.model_dump(mode="json") for tool_call in tool_calls[1:]
            ]

        tool_execution = session.handle_tool_call(
            tool_calls[0],
            advisor_sampler=advisor_sampler,
            advisor_seed=seed + 10_000 + turn,
        )
        messages.append(tool_execution.message)
        turn_summary.update(
            {
                "status": tool_execution.event["status"],
                "tool": tool_execution.event["tool"],
                "advice_remaining": tool_execution.event["advice_remaining"],
            }
        )
        turn_event.update(
            {
                "tool_call": tool_calls[0].model_dump(mode="json"),
                "tool_result_message": tool_execution.message,
                "tool_event": tool_execution.event,
            }
        )
        turns.append(turn_summary)
        trajectory.append(turn_event)

        if tool_execution.should_stop:
            termination_reason = "submitted"
            submitted_code = tool_execution.submitted_code
            break
    else:
        termination_reason = "no_submit"

    duration_seconds = round(time.monotonic() - started, 3)
    code = submitted_code or session.workspace.files.get(DEFAULT_SOLVE_PATH, "")
    text = code if submitted_code is not None else ""
    metadata = {
        "termination_reason": termination_reason,
        "submitted": submitted_code is not None,
        "advice_calls_used": session.advice_calls_used,
        "advice_remaining": session.advice_remaining,
        "min_advice_calls": session.min_advice_calls,
        "virtual_files": sorted(session.workspace.files),
        "messages": messages,
    }
    return SampleRecord(
        index=index,
        seed=seed,
        text=text,
        code=code,
        status="pending_eval" if submitted_code is not None else "no_submit",
        reward=None if submitted_code is not None else 0.0,
        score=None if submitted_code is not None else 0.0,
        score_unbounded=None if submitted_code is not None else 0.0,
        message=None if submitted_code is not None else termination_reason,
        duration_seconds=duration_seconds,
        metadata=metadata,
        turns=turns,
        trajectory=trajectory,
    )


def resolve_renderer_name(model_name: str, explicit_renderer_name: str | None) -> str:
    if explicit_renderer_name is not None:
        return explicit_renderer_name
    try:
        return model_info.get_recommended_renderer_name(model_name)
    except Exception as exc:
        if "Qwen3" in model_name:
            return DEFAULT_RENDERER_NAME
        raise ValueError(
            f"No renderer_name provided and no recommendation known for {model_name!r}"
        ) from exc


def make_run_dir(config: Stage1Config) -> Path:
    output_root = Path(config.output_dir).expanduser()
    _reject_home_output(output_root)
    run_name = config.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"problem_{config.problem_id}_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_sampling_client(service_client: Any, *, model_name: str, checkpoint_url: str | None) -> Any:
    if checkpoint_url is not None:
        return service_client.create_sampling_client(
            model_path=checkpoint_url,
            base_model=model_name,
        )
    return service_client.create_sampling_client(base_model=model_name)


def make_student_sampler(
    *,
    sampling_client: Any,
    renderer: renderers.Renderer,
    tokenizer: Any,
    config: Stage1Config,
) -> StudentSampler:
    import tinker

    stop = renderer.get_stop_sequences()

    def sample(messages: list[Message], *, seed: int) -> StudentGeneration:
        prompt = renderer.build_generation_prompt(messages)
        if prompt.length > config.max_prompt_tokens:
            return StudentGeneration(
                message={
                    "role": "assistant",
                    "content": f"Prompt exceeded max_prompt_tokens={config.max_prompt_tokens}.",
                },
                raw_text="",
                prompt_tokens=prompt.length,
                generated_tokens=0,
                stop_reason="context_overflow",
                termination="context_overflow",
                clean_termination=False,
            )
        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=seed,
            ),
        ).result()
        sequence = result.sequences[0]
        raw_text = str(tokenizer.decode(sequence.tokens, skip_special_tokens=False))
        message, termination = renderer.parse_response(sequence.tokens)
        return StudentGeneration(
            message=message,
            raw_text=raw_text,
            prompt_tokens=prompt.length,
            generated_tokens=len(sequence.tokens),
            stop_reason=str(sequence.stop_reason) if sequence.stop_reason is not None else None,
            termination=str(termination),
            clean_termination=termination.is_clean,
        )

    return sample


def make_advisor_sampler(
    *,
    sampling_client: Any,
    renderer: renderers.Renderer,
    tokenizer: Any,
    config: Stage1Config,
) -> AdvisorSampler:
    import tinker

    stop = renderer.get_stop_sequences()

    def sample(prompt: str, *, seed: int) -> AdvisorGeneration:
        messages: list[Message] = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        model_input = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=config.advisor_max_tokens,
                temperature=config.advisor_temperature,
                top_p=config.advisor_top_p,
                top_k=config.advisor_top_k,
                seed=seed,
            ),
        ).result()
        sequence = result.sequences[0]
        parsed_message, termination = renderer.parse_response(sequence.tokens)
        if termination.is_clean:
            text = format_content_as_string(parsed_message["content"])
        else:
            text = str(tokenizer.decode(sequence.tokens, skip_special_tokens=True))
        return AdvisorGeneration(
            text=text.strip(),
            prompt_tokens=model_input.length,
            generated_tokens=len(sequence.tokens),
            stop_reason=str(sequence.stop_reason) if sequence.stop_reason is not None else None,
        )

    return sample


def sample_from_tinker(config: Stage1Config, *, statement: str) -> list[SampleRecord]:
    import tinker

    if config.max_student_turns < 1:
        raise ValueError("max_student_turns must be at least 1")
    if config.max_advice_calls < 0:
        raise ValueError("max_advice_calls must be non-negative")
    if config.min_advice_calls < 0:
        raise ValueError("min_advice_calls must be non-negative")
    if config.min_advice_calls > config.max_advice_calls:
        raise ValueError("min_advice_calls cannot exceed max_advice_calls")
    if config.max_student_turns < config.max_advice_calls + 1:
        raise ValueError("max_student_turns must allow advice calls plus final submit")

    tokenizer = tokenizer_utils.get_tokenizer(config.tokenizer_name or config.model_name)
    renderer_name = resolve_renderer_name(config.model_name, config.renderer_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=config.model_name)

    advisor_model_name = config.advisor_model_name or config.model_name
    advisor_tokenizer = tokenizer_utils.get_tokenizer(config.advisor_tokenizer_name or advisor_model_name)
    advisor_renderer_name = resolve_renderer_name(
        advisor_model_name,
        config.advisor_renderer_name or config.renderer_name,
    )
    advisor_renderer = renderers.get_renderer(
        advisor_renderer_name,
        advisor_tokenizer,
        model_name=advisor_model_name,
    )

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("frontier_cs_stage1"),
    )
    student_client = create_sampling_client(
        service_client,
        model_name=config.model_name,
        checkpoint_url=config.checkpoint_url,
    )
    advisor_checkpoint_url = config.advisor_checkpoint_url or config.checkpoint_url
    advisor_client = (
        student_client
        if advisor_model_name == config.model_name and advisor_checkpoint_url == config.checkpoint_url
        else create_sampling_client(
            service_client,
            model_name=advisor_model_name,
            checkpoint_url=advisor_checkpoint_url,
        )
    )

    student_sampler = make_student_sampler(
        sampling_client=student_client,
        renderer=renderer,
        tokenizer=tokenizer,
        config=config,
    )
    advisor_sampler = make_advisor_sampler(
        sampling_client=advisor_client,
        renderer=advisor_renderer,
        tokenizer=advisor_tokenizer,
        config=config,
    )
    initial_messages = make_initial_messages(
        renderer=renderer,
        problem_id=config.problem_id,
        statement=statement,
        min_advice_calls=config.min_advice_calls,
    )

    records: list[SampleRecord] = []
    for index in range(config.num_samples):
        sample_seed = config.seed + index * config.max_student_turns
        records.append(
            run_agentic_trajectory(
                index=index,
                seed=sample_seed,
                problem_id=config.problem_id,
                statement=statement,
                initial_messages=initial_messages,
                student_sampler=student_sampler,
                advisor_sampler=advisor_sampler,
                max_student_turns=config.max_student_turns,
                max_advice_calls=config.max_advice_calls,
                min_advice_calls=config.min_advice_calls,
            )
        )
    return records


def write_trajectory_files(
    run_dir: Path,
    samples: list[SampleRecord],
    config: Stage1Config,
) -> None:
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    index_records: list[dict[str, Any]] = []
    for record in samples:
        prefix = f"sample_{record.index:03d}"
        relative_path = Path("trajectories") / f"{prefix}.json"
        metadata = record.metadata or {}
        payload = {
            "problem_id": config.problem_id,
            "model_name": config.model_name,
            "renderer_name": config.renderer_name,
            "tokenizer_name": config.tokenizer_name,
            "checkpoint_url": config.checkpoint_url,
            "advisor": {
                "model_name": config.advisor_model_name or config.model_name,
                "renderer_name": config.advisor_renderer_name or config.renderer_name,
                "checkpoint_url": config.advisor_checkpoint_url or config.checkpoint_url,
                "temperature": config.advisor_temperature,
                "max_tokens": config.advisor_max_tokens,
            },
            "sampling": {
                "max_tokens": config.max_tokens,
                "max_student_turns": config.max_student_turns,
                "max_advice_calls": config.max_advice_calls,
                "min_advice_calls": config.min_advice_calls,
                "max_prompt_tokens": config.max_prompt_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "sample_seed": record.seed,
            },
            "record": sample_record_without_full_messages(record),
            "messages": to_jsonable(metadata.get("messages", [])),
            "turns": to_jsonable(record.trajectory or []),
        }
        _write_json(run_dir / relative_path, payload)
        index_records.append(
            {
                "index": record.index,
                "seed": record.seed,
                "trajectory_path": str(relative_path),
                "num_turns": len(record.turns or []),
                "termination_reason": metadata.get("termination_reason"),
                "advice_calls_used": metadata.get("advice_calls_used"),
                "advice_remaining": metadata.get("advice_remaining"),
                "submitted": metadata.get("submitted"),
                "status": record.status,
                "reward": record.reward,
                "score": record.score,
                "best_curve_score": record.curve_score,
            }
        )

    _write_jsonl(trajectories_dir / "index.jsonl", index_records)


def sample_records_to_jsonl(samples: list[SampleRecord]) -> list[dict[str, Any]]:
    return [sample_record_without_full_messages(sample) for sample in samples]


def run(config: Stage1Config) -> Path:
    if config.eval_workers < 1:
        raise ValueError(f"eval_workers must be positive: {config.eval_workers}")

    frontier_cs_root = Path(config.frontier_cs_root).expanduser().resolve()
    statement = load_problem_statement(frontier_cs_root, config.problem_id)
    run_dir = make_run_dir(config)

    _write_json(run_dir / "config.json", dump_config(config))
    _write_json(
        run_dir / "tool_specs.json",
        {
            "tools": TOOL_SPECS,
            "student_system_prompt": STUDENT_SYSTEM_PROMPT,
            "advisor_system_prompt": ADVISOR_SYSTEM_PROMPT,
        },
    )
    (run_dir / "prompt.md").write_text(
        STUDENT_USER_PROMPT_TEMPLATE.format(
            problem_id=config.problem_id,
            statement=statement,
            advice_requirement=advice_requirement_text(config.min_advice_calls),
        ),
        encoding="utf-8",
    )

    samples = sample_from_tinker(config, statement=statement)
    write_sample_files(run_dir, samples)
    if config.save_trajectories:
        write_trajectory_files(run_dir, samples, config)
    _write_jsonl(run_dir / "samples.jsonl", sample_records_to_jsonl(samples))

    if config.evaluate:
        submitted_samples = [sample for sample in samples if (sample.metadata or {}).get("submitted")]
        if submitted_samples:
            score_samples(
                frontier_cs_root,
                config.problem_id,
                submitted_samples,
                work_dir=run_dir / "direct_eval",
                evaluator=config.evaluator,
                judge_url=config.judge_url,
                auto_start_judge=config.auto_start_judge,
                score_scale=config.score_scale,
                compile_timeout=config.compile_timeout,
                case_timeout=config.case_timeout,
                eval_workers=config.eval_workers,
            )
        if config.save_trajectories:
            write_trajectory_files(run_dir, samples, config)
        _write_jsonl(run_dir / "evaluations.jsonl", sample_records_to_jsonl(samples))

    write_score_artifacts(run_dir, samples)

    scores = [sample.curve_score for sample in samples]
    print(f"Stage 1 run written to {run_dir}")
    print(f"Best score: {max(scores) if scores else 'n/a'}")
    print(f"Best-of-k CSV: {run_dir / 'best_of_k.csv'}")
    print(f"Best-of-k plot: {run_dir / 'best_of_k.svg'}")
    return run_dir


if __name__ == "__main__":
    run(chz.entrypoint(Stage1Config))
