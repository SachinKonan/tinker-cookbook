"""Run the Stage 0 best-of-k baseline on Frontier-CS algorithmic tasks.

This runner intentionally talks to Frontier-CS's judge API directly. The judge
can be hosted by Docker, Apptainer, or a later Ray actor layer as long as it
exposes the standard algorithmic API on ``judge_url``.
"""

from __future__ import annotations

import csv
import html
import importlib
import json
import math
import re
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.ml_log import dump_config

SCRATCH_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524")
DEFAULT_FRONTIER_CS_ROOT = SCRATCH_ROOT / "Frontier-CS"
DEFAULT_OUTPUT_ROOT = SCRATCH_ROOT / "tinker_runs" / "frontier_cs_stage0"
DEFAULT_QWEN_REPO = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_QWEN_PATH = (
    SCRATCH_ROOT
    / "hf"
    / "hub"
    / "models--Qwen--Qwen3-4B-Thinking-2507"
    / "snapshots"
    / "768f209d9ea81521153ed38c47d515654e938aea"
)

SYSTEM_PROMPT = """You are a strong competitive programmer. Produce a complete C++17 program for the given optimization problem. The program must read from stdin and write the required output to stdout."""

USER_PROMPT_TEMPLATE = """Solve Frontier-CS algorithmic problem {problem_id}.

Return only a complete C++17 solution. Prefer robust, scalable heuristics over explanation. Do not write prose, analysis, or Markdown.

Problem statement:

{statement}
"""

CXX_COMPLETION_PREFIX = "#include <bits/stdc++.h>\nusing namespace std;\n"
REPAIR_USER_PROMPT_TEMPLATE = """Solve Frontier-CS algorithmic problem {problem_id}.

The previous candidate did not compile. Produce a complete replacement C++17 source file now. Do not explain. Do not output Markdown. The file must contain a complete `int main()` and must read stdin and write stdout.

Compiler error from the previous candidate:

```text
{compile_error}
```

Previous candidate, possibly truncated:

```cpp
{candidate_code}
```

Problem statement:

{statement}
"""

CODE_BLOCK_RE = re.compile(
    r"```(?:\s*(?:cpp|c\+\+|cxx|cc))?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


@chz.chz
class Stage0Config:
    """Configuration for the Frontier-CS Stage 0 baseline."""

    frontier_cs_root: str = str(DEFAULT_FRONTIER_CS_ROOT)
    problem_id: str = "302"
    model_name: str = DEFAULT_QWEN_REPO
    tokenizer_name: str | None = None
    checkpoint_url: str | None = None
    base_url: str | None = "http://127.0.0.1:8000/"
    output_dir: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str | None = None
    num_samples: int = 50
    samples_per_request: int = 1
    max_tokens: int = 8192
    max_turns: int = 20
    max_prompt_tokens: int = 24576
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    save_trajectories: bool = True
    evaluate: bool = True
    evaluator: str = "direct_checker"
    judge_url: str = "http://127.0.0.1:8081"
    auto_start_judge: bool = False
    score_scale: float = 100.0
    compile_timeout: int = 60
    case_timeout: int = 20


@dataclass
class SampleRecord:
    index: int
    seed: int
    text: str
    code: str
    reward: float | None = None
    score: float | None = None
    score_unbounded: float | None = None
    status: str | None = None
    message: str | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] | None = None
    turns: list[dict[str, Any]] | None = None
    trajectory: list[dict[str, Any]] | None = None

    @property
    def curve_score(self) -> float:
        if self.reward is None or self.status != "success":
            return 0.0
        return float(self.reward)


def sample_record_to_dict(record: SampleRecord, *, include_trajectory: bool = False) -> dict[str, Any]:
    return {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if include_trajectory or field.name != "trajectory"
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _reject_home_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise ValueError(f"Refusing to write Stage 0 artifacts under home: {resolved}")


def make_run_dir(config: Stage0Config) -> Path:
    output_root = Path(config.output_dir).expanduser()
    _reject_home_output(output_root)
    run_name = config.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"problem_{config.problem_id}_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_problem_statement(frontier_cs_root: Path, problem_id: str) -> str:
    statement_path = (
        frontier_cs_root / "algorithmic" / "problems" / str(problem_id) / "statement.txt"
    )
    if not statement_path.exists():
        raise FileNotFoundError(f"Frontier-CS statement not found: {statement_path}")
    return strip_outer_markdown_fence(statement_path.read_text(encoding="utf-8"))


def strip_outer_markdown_fence(statement: str) -> str:
    """Remove Frontier-CS's file-level markdown fence without touching examples."""
    lines = statement.splitlines()
    if lines and lines[0].strip().lower() == "```markdown" and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return statement


def build_prompt(problem_id: str, statement: str) -> tuple[str, str]:
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(problem_id=problem_id, statement=statement)


def render_messages(tokenizer: Any, messages: list[dict[str, str]], assistant_prefix: str = "") -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            rendered = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return close_empty_qwen_think(rendered) + assistant_prefix
        except Exception:
            pass

    rendered = ""
    for message in messages:
        rendered += f"{message['role'].title()}:\n{message['content']}\n\n"
    return f"{rendered}Assistant:\n{assistant_prefix}"


def render_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    return render_messages(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        assistant_prefix=CXX_COMPLETION_PREFIX,
    )


def close_empty_qwen_think(rendered_prompt: str) -> str:
    """Start completion after Qwen's explicit think block for code-only baselines."""
    stripped = rendered_prompt.rstrip()
    if stripped.endswith("<think>"):
        return f"{stripped}\n</think>\n\n"
    return rendered_prompt


def extract_cpp_code(text: str) -> str:
    matches = [match.group("code").strip() for match in CODE_BLOCK_RE.finditer(text)]
    if matches:
        with_main = [code for code in matches if "main(" in code or "int main" in code]
        return max(with_main or matches, key=len)
    return text.strip()


def token_count(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def tail_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[-max_tokens:], skip_special_tokens=False)


def compact_compile_error(error: str | None, max_chars: int = 4000) -> str:
    if not error:
        return "No compiler error was captured."
    return error[-max_chars:]


def render_repair_prompt(
    tokenizer: Any,
    *,
    problem_id: str,
    statement: str,
    candidate_code: str,
    compile_error: str | None,
    max_prompt_tokens: int,
) -> str:
    compile_error_text = compact_compile_error(compile_error)
    rendered = ""
    for candidate_budget in (20000, 16000, 12000, 8000, 4000, 2000, 1000, 0):
        candidate_tail = tail_by_tokens(tokenizer, candidate_code, candidate_budget)
        user_prompt = REPAIR_USER_PROMPT_TEMPLATE.format(
            problem_id=problem_id,
            statement=statement,
            compile_error=compile_error_text,
            candidate_code=candidate_tail,
        )
        rendered = render_messages(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            assistant_prefix=CXX_COMPLETION_PREFIX,
        )
        if token_count(tokenizer, rendered, add_special_tokens=True) <= max_prompt_tokens:
            return rendered
    return rendered


def expected_best_of_k(scores: list[float]) -> list[float]:
    """Exact E[max] for k draws without replacement from the observed scores."""
    if not scores:
        return []

    sorted_scores = sorted(float(score) for score in scores)
    n_scores = len(sorted_scores)
    curve: list[float] = []
    for k in range(1, n_scores + 1):
        denominator = math.comb(n_scores, k)
        expected = 0.0
        for rank, score in enumerate(sorted_scores, start=1):
            if rank >= k:
                expected += score * math.comb(rank - 1, k - 1) / denominator
        curve.append(expected)
    return curve


def prefix_best(scores: list[float]) -> list[float]:
    best = 0.0
    curve = []
    for score in scores:
        best = max(best, float(score))
        curve.append(best)
    return curve


def write_best_of_k_csv(path: Path, expected_curve: list[float], prefix_curve: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "expected_best", "prefix_best"])
        writer.writeheader()
        for k, expected in enumerate(expected_curve, start=1):
            writer.writerow(
                {
                    "k": k,
                    "expected_best": f"{expected:.10f}",
                    "prefix_best": f"{prefix_curve[k - 1]:.10f}",
                }
            )


def write_svg_plot(path: Path, expected_curve: list[float], prefix_curve: list[float]) -> None:
    width = 900
    height = 520
    margin_left = 70
    margin_right = 30
    margin_top = 38
    margin_bottom = 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_values = expected_curve + prefix_curve + [0.0, 1.0]
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def point(k_idx: int, value: float) -> tuple[float, float]:
        x = margin_left + (k_idx / max(1, len(expected_curve) - 1)) * plot_width
        y = margin_top + (y_max - value) / (y_max - y_min) * plot_height
        return x, y

    def polyline(values: list[float]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(i, v) for i, v in enumerate(values)))

    x_axis_y = margin_top + plot_height
    title = html.escape("Frontier-CS Stage 0: E[best of k]")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.0f}" y="24" text-anchor="middle" font-family="Arial, sans-serif" font-size="18">{title}</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{x_axis_y}" stroke="#222" stroke-width="1"/>
  <line x1="{margin_left}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" stroke="#222" stroke-width="1"/>
  <text x="{width / 2:.0f}" y="{height - 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">k samples</text>
  <text x="18" y="{height / 2:.0f}" transform="rotate(-90 18,{height / 2:.0f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">reward</text>
  <text x="{margin_left - 10}" y="{margin_top + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{y_max:.3f}</text>
  <text x="{margin_left - 10}" y="{x_axis_y + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{y_min:.3f}</text>
  <text x="{margin_left}" y="{x_axis_y + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">1</text>
  <text x="{width - margin_right}" y="{x_axis_y + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{len(expected_curve)}</text>
  <polyline fill="none" stroke="#0f766e" stroke-width="3" points="{polyline(expected_curve)}"/>
  <polyline fill="none" stroke="#b45309" stroke-width="2" stroke-dasharray="6 5" points="{polyline(prefix_curve)}"/>
  <line x1="{width - 270}" y1="54" x2="{width - 232}" y2="54" stroke="#0f766e" stroke-width="3"/>
  <text x="{width - 224}" y="58" font-family="Arial, sans-serif" font-size="13">E[best of k], without replacement</text>
  <line x1="{width - 270}" y1="76" x2="{width - 232}" y2="76" stroke="#b45309" stroke-width="2" stroke-dasharray="6 5"/>
  <text x="{width - 224}" y="80" font-family="Arial, sans-serif" font-size="13">generation-order prefix best</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _load_tokenizer(model_name: str, tokenizer_name: str | None) -> Any:
    from tinker_cookbook import tokenizer_utils

    return tokenizer_utils.get_tokenizer(tokenizer_name or model_name)


def sample_from_tinker(
    config: Stage0Config, system_prompt: str, user_prompt: str, *, turn_work_dir: Path
) -> list[SampleRecord]:
    import tinker

    if config.max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    tokenizer = _load_tokenizer(config.model_name, config.tokenizer_name)
    initial_prompt = render_prompt(tokenizer, system_prompt, user_prompt)
    statement = user_prompt.split("Problem statement:\n\n", maxsplit=1)[-1]

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("frontier_cs_stage0"),
    )
    if config.checkpoint_url:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_url,
            base_model=config.model_name,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    records: list[SampleRecord] = []
    for index in range(config.num_samples):
        sample_seed = config.seed + index * config.max_turns
        prompt = initial_prompt
        candidate_text = ""
        candidate_code = ""
        compile_error: str | None = None
        turns: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = []
        compiled = False

        for turn in range(1, config.max_turns + 1):
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            turn_seed = sample_seed + turn - 1
            result = sampling_client.sample(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    seed=turn_seed,
                ),
            ).result()

            sequence = result.sequences[0]
            generated_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            candidate_text = CXX_COMPLETION_PREFIX + generated_text
            candidate_code = extract_cpp_code(candidate_text)
            turn_dir = turn_work_dir / f"sample_{index:03d}" / f"turn_{turn:02d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            _, compile_error = _compile_solution(candidate_code, turn_dir, config.compile_timeout)
            stop_reason = getattr(sequence, "stop_reason", None)
            turn_summary = {
                "turn": turn,
                "seed": turn_seed,
                "prompt_tokens": len(prompt_tokens),
                "generated_tokens": len(sequence.tokens),
                "stop_reason": str(stop_reason) if stop_reason is not None else None,
                "compile_ok": compile_error is None,
                "compile_error": compile_error,
                "code_chars": len(candidate_code),
            }
            turns.append(turn_summary)
            trajectory.append(
                {
                    **turn_summary,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "candidate_text": candidate_text,
                    "candidate_code": candidate_code,
                }
            )
            if compile_error is None:
                compiled = True
                break

            prompt = render_repair_prompt(
                tokenizer,
                problem_id=config.problem_id,
                statement=statement,
                candidate_code=candidate_code,
                compile_error=compile_error,
                max_prompt_tokens=config.max_prompt_tokens,
            )

        records.append(
            SampleRecord(
                index=index,
                seed=sample_seed,
                text=candidate_text,
                code=candidate_code,
                turns=turns,
                trajectory=trajectory,
                metadata={"compiled_during_sampling": compiled},
            )
        )
    return records


def score_samples(
    frontier_cs_root: Path,
    problem_id: str,
    samples: list[SampleRecord],
    *,
    work_dir: Path,
    evaluator: str,
    judge_url: str,
    auto_start_judge: bool,
    score_scale: float,
    compile_timeout: int,
    case_timeout: int,
) -> None:
    if evaluator == "direct_checker":
        score_samples_with_direct_checker(
            frontier_cs_root,
            problem_id,
            samples,
            work_dir=work_dir,
            compile_timeout=compile_timeout,
            case_timeout=case_timeout,
        )
        return
    if evaluator != "judge_api":
        raise ValueError(f"Unknown evaluator: {evaluator}")

    frontier_src = frontier_cs_root / "src"
    if not frontier_src.exists():
        raise FileNotFoundError(f"Frontier-CS src/ not found: {frontier_src}")
    sys.path.insert(0, str(frontier_src))
    try:
        runner_module = importlib.import_module("frontier_cs.runner.algorithmic_local")
    finally:
        with suppress(ValueError):
            sys.path.remove(str(frontier_src))

    runner = runner_module.AlgorithmicLocalRunner(
        judge_url=judge_url,
        base_dir=frontier_cs_root,
        auto_start=auto_start_judge,
    )
    for record in samples:
        started = time.monotonic()
        result = runner.evaluate(problem_id, record.code, lang="cpp")
        existing_metadata = record.metadata or {}
        record.score = result.score
        if result.score is not None:
            record.reward = max(0.0, min(1.0, float(result.score) / score_scale))
        record.score_unbounded = result.score_unbounded
        record.status = result.status.value
        record.message = result.message
        record.duration_seconds = result.duration_seconds or round(time.monotonic() - started, 3)
        record.metadata = {**existing_metadata, **(result.metadata or {})}


POINTS_RE = re.compile(r"\bpoints\s+([0-9]+(?:\.[0-9]+)?)")
RATIO_RE = re.compile(r"\bRatio:\s*([0-9]+(?:\.[0-9]+)?)")


def _run_command(
    args: list[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    stdin_path: Path | None = None,
    stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    stdin = stdin_path.open("r", encoding="utf-8") if stdin_path else None
    stdout_file = stdout_path.open("w", encoding="utf-8") if stdout_path else None
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            stdin=stdin,
            stdout=stdout_file if stdout_file else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
    finally:
        if stdin is not None:
            stdin.close()
        if stdout_file is not None:
            stdout_file.close()


def _parse_checker_ratio(output: str) -> float:
    if match := POINTS_RE.search(output):
        return max(0.0, min(1.0, float(match.group(1))))
    if match := RATIO_RE.search(output):
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0


def _compile_checker(
    frontier_cs_root: Path, problem_id: str, build_dir: Path, timeout: int
) -> Path:
    problem_dir = frontier_cs_root / "algorithmic" / "problems" / str(problem_id)
    checker_src = problem_dir / "chk.cc"
    testlib_include = frontier_cs_root / "algorithmic" / "judge" / "include"
    checker_bin = build_dir / "checker"
    result = _run_command(
        [
            "g++",
            str(checker_src),
            "-O2",
            "-pipe",
            "-std=gnu++17",
            "-I",
            str(testlib_include),
            "-o",
            str(checker_bin),
        ],
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Checker compile failed: {result.stderr}")
    return checker_bin


def _compile_solution(code: str, sample_dir: Path, timeout: int) -> tuple[Path | None, str | None]:
    source_path = sample_dir / "solution.cpp"
    binary_path = sample_dir / "solution"
    source_path.write_text(code + "\n", encoding="utf-8")
    result = _run_command(
        [
            "g++",
            str(source_path),
            "-O2",
            "-pipe",
            "-std=gnu++17",
            "-o",
            str(binary_path),
        ],
        timeout=timeout,
    )
    if result.returncode != 0:
        return None, result.stderr[-4000:]
    return binary_path, None


def score_samples_with_direct_checker(
    frontier_cs_root: Path,
    problem_id: str,
    samples: list[SampleRecord],
    *,
    work_dir: Path,
    compile_timeout: int,
    case_timeout: int,
) -> None:
    problem_dir = frontier_cs_root / "algorithmic" / "problems" / str(problem_id)
    testdata_dir = problem_dir / "testdata"
    input_paths = sorted(testdata_dir.glob("*.in"), key=lambda path: int(path.stem))
    if not input_paths:
        raise FileNotFoundError(f"No test cases found in {testdata_dir}")

    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    checker_bin = _compile_checker(frontier_cs_root, problem_id, build_dir, compile_timeout)

    for record in samples:
        existing_metadata = record.metadata or {}
        start = time.monotonic()
        sample_dir = work_dir / f"sample_{record.index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        binary_path, compile_error = _compile_solution(record.code, sample_dir, compile_timeout)
        if binary_path is None:
            record.reward = 0.0
            record.score = 0.0
            record.status = "error"
            record.message = "Solution compile failed"
            record.duration_seconds = round(time.monotonic() - start, 3)
            record.metadata = {**existing_metadata, "compile_error": compile_error}
            continue

        case_results: list[dict[str, Any]] = []
        ratios: list[float] = []
        for input_path in input_paths:
            case_name = input_path.stem
            answer_path = input_path.with_suffix(".ans")
            output_path = sample_dir / f"{case_name}.out"
            try:
                run_result = _run_command(
                    [str(binary_path)],
                    stdin_path=input_path,
                    stdout_path=output_path,
                    timeout=case_timeout,
                    cwd=sample_dir,
                )
            except subprocess.TimeoutExpired:
                case_results.append({"case": case_name, "ratio": 0.0, "status": "timeout"})
                ratios.append(0.0)
                continue

            if run_result.returncode != 0:
                case_results.append(
                    {
                        "case": case_name,
                        "ratio": 0.0,
                        "status": "runtime_error",
                        "stderr": run_result.stderr[-1000:],
                    }
                )
                ratios.append(0.0)
                continue

            checker_result = _run_command(
                [str(checker_bin), str(input_path), str(output_path), str(answer_path)],
                timeout=case_timeout,
                cwd=sample_dir,
            )
            checker_output = f"{checker_result.stdout or ''}\n{checker_result.stderr or ''}"
            ratio = _parse_checker_ratio(checker_output)
            ratios.append(ratio)
            case_results.append(
                {
                    "case": case_name,
                    "ratio": ratio,
                    "status": "checked",
                    "checker_returncode": checker_result.returncode,
                    "checker_output": checker_output.strip()[-1000:],
                }
            )

        reward = sum(ratios) / len(ratios)
        record.reward = reward
        record.score = reward * 100.0
        record.score_unbounded = record.score
        record.status = "success"
        record.message = None
        record.duration_seconds = round(time.monotonic() - start, 3)
        record.metadata = {**existing_metadata, "cases": case_results}


def write_sample_files(run_dir: Path, samples: list[SampleRecord]) -> None:
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for record in samples:
        prefix = f"sample_{record.index:03d}"
        (samples_dir / f"{prefix}.cpp").write_text(record.code + "\n", encoding="utf-8")
        (samples_dir / f"{prefix}.txt").write_text(record.text, encoding="utf-8")


def write_trajectory_files(
    run_dir: Path, samples: list[SampleRecord], config: Stage0Config
) -> None:
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    index_records: list[dict[str, Any]] = []
    for record in samples:
        prefix = f"sample_{record.index:03d}"
        relative_path = Path("trajectories") / f"{prefix}.json"
        payload = {
            "problem_id": config.problem_id,
            "model_name": config.model_name,
            "tokenizer_name": config.tokenizer_name,
            "checkpoint_url": config.checkpoint_url,
            "sampling": {
                "max_tokens": config.max_tokens,
                "max_turns": config.max_turns,
                "max_prompt_tokens": config.max_prompt_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "sample_seed": record.seed,
                "samples_per_request": config.samples_per_request,
            },
            "record": sample_record_to_dict(record),
            "turns": record.trajectory or [],
        }
        _write_json(run_dir / relative_path, payload)
        index_records.append(
            {
                "index": record.index,
                "seed": record.seed,
                "trajectory_path": str(relative_path),
                "num_turns": len(record.turns or []),
                "compiled_during_sampling": (record.metadata or {}).get(
                    "compiled_during_sampling"
                ),
                "status": record.status,
                "reward": record.reward,
                "score": record.score,
                "best_curve_score": record.curve_score,
            }
        )

    _write_jsonl(trajectories_dir / "index.jsonl", index_records)


def summarize(samples: list[SampleRecord], expected_curve: list[float]) -> dict[str, Any]:
    scores = [sample.curve_score for sample in samples]
    successful = [sample for sample in samples if sample.status == "success"]
    best_index = max(range(len(samples)), key=lambda idx: scores[idx]) if samples else None
    return {
        "num_samples": len(samples),
        "num_successful_evals": len(successful),
        "best_score": max(scores) if scores else None,
        "best_sample_index": best_index,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "expected_best_at_g": expected_curve[-1] if expected_curve else None,
        "scores": scores,
    }


def run(config: Stage0Config) -> Path:
    frontier_cs_root = Path(config.frontier_cs_root).expanduser().resolve()
    statement = load_problem_statement(frontier_cs_root, config.problem_id)
    system_prompt, user_prompt = build_prompt(config.problem_id, statement)
    run_dir = make_run_dir(config)

    _write_json(run_dir / "config.json", dump_config(config))
    (run_dir / "prompt.md").write_text(
        f"# System\n\n{system_prompt}\n\n# User\n\n{user_prompt}\n",
        encoding="utf-8",
    )

    samples = sample_from_tinker(
        config, system_prompt, user_prompt, turn_work_dir=run_dir / "turn_compile"
    )
    write_sample_files(run_dir, samples)
    if config.save_trajectories:
        write_trajectory_files(run_dir, samples, config)
    _write_jsonl(run_dir / "samples.jsonl", [sample_record_to_dict(sample) for sample in samples])

    if config.evaluate:
        score_samples(
            frontier_cs_root,
            config.problem_id,
            samples,
            work_dir=run_dir / "direct_eval",
            evaluator=config.evaluator,
            judge_url=config.judge_url,
            auto_start_judge=config.auto_start_judge,
            score_scale=config.score_scale,
            compile_timeout=config.compile_timeout,
            case_timeout=config.case_timeout,
        )
        if config.save_trajectories:
            write_trajectory_files(run_dir, samples, config)
        _write_jsonl(
            run_dir / "evaluations.jsonl", [sample_record_to_dict(sample) for sample in samples]
        )

    scores = [sample.curve_score for sample in samples]
    expected_curve = expected_best_of_k(scores)
    prefix_curve = prefix_best(scores)
    write_best_of_k_csv(run_dir / "best_of_k.csv", expected_curve, prefix_curve)
    write_svg_plot(run_dir / "best_of_k.svg", expected_curve, prefix_curve)
    _write_json(run_dir / "summary.json", summarize(samples, expected_curve))

    print(f"Stage 0 run written to {run_dir}")
    print(f"Best score: {max(scores) if scores else 'n/a'}")
    print(f"Best-of-k CSV: {run_dir / 'best_of_k.csv'}")
    print(f"Best-of-k plot: {run_dir / 'best_of_k.svg'}")
    return run_dir


if __name__ == "__main__":
    run(chz.entrypoint(Stage0Config))
