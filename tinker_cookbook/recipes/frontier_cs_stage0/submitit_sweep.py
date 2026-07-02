"""Submitit controller for dynamic Frontier-CS Stage 0 sweeps.

The controller submits independent chunks per problem and decides whether to
continue after each chunk. The default schedule is:

* +10 samples, total G=10
* +15 samples, total G=25
* +25 samples, total G=50

Each chunk reuses ``slurm/run_qwen_stage0.sbatch`` inside a Submitit allocation,
so it follows the same local Tinker startup path as the validated single-problem
run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import submitit

SCRATCH_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524")
DEFAULT_REPO_ROOT = SCRATCH_ROOT / "tinker-cookbook"
DEFAULT_SKYRL_TX_ROOT = SCRATCH_ROOT / "SkyRL" / "skyrl-tx"
DEFAULT_FRONTIER_CS_ROOT = SCRATCH_ROOT / "Frontier-CS"
DEFAULT_RUN_ROOT = SCRATCH_ROOT / "tinker_runs" / "frontier_cs_stage0"
DEFAULT_MODEL_SNAPSHOT = (
    SCRATCH_ROOT
    / "hf"
    / "hub"
    / "models--Qwen--Qwen3-4B-Thinking-2507"
    / "snapshots"
    / "768f209d9ea81521153ed38c47d515654e938aea"
)
DEFAULT_PROBLEMS = ("46", "308", "314", "302", "48", "306", "303", "307", "309", "313", "159")
DEFAULT_CHUNK_SIZES = (10, 15, 25)


@dataclass(frozen=True)
class SweepChunk:
    problem_id: str
    stage_index: int
    start_sample: int
    num_samples: int
    total_after: int
    seed: int
    run_name: str

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.num_samples - 1


@dataclass(frozen=True)
class WorkerSettings:
    repo_root: str = str(DEFAULT_REPO_ROOT)
    skyrl_tx_root: str = str(DEFAULT_SKYRL_TX_ROOT)
    frontier_cs_root: str = str(DEFAULT_FRONTIER_CS_ROOT)
    run_root: str = str(DEFAULT_RUN_ROOT)
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    model_snapshot: str = str(DEFAULT_MODEL_SNAPSHOT)
    max_tokens: int = 8192
    max_turns: int = 20
    max_prompt_tokens: int = 24576
    temperature: float = 0.8
    top_p: float = 1.0
    samples_per_request: int = 1
    start_judge: bool = False


@dataclass(frozen=True)
class ChunkResult:
    problem_id: str
    stage_index: int
    run_name: str
    run_dir: str
    num_samples: int
    seed: int
    scores: list[float]
    best_score: float
    mean_score: float
    successful_evals: int
    duration_seconds: float


@dataclass(frozen=True)
class ContinuationDecision:
    should_continue: bool
    reason: str


@dataclass
class ProblemState:
    problem_id: str
    results: list[ChunkResult]

    @property
    def scores(self) -> list[float]:
        scores: list[float] = []
        for result in sorted(self.results, key=lambda item: item.stage_index):
            scores.extend(result.scores)
        return scores

    @property
    def next_stage_index(self) -> int:
        return len(self.results)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    chunks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not chunks or any(chunk <= 0 for chunk in chunks):
        raise ValueError(f"Chunk sizes must be positive integers: {value!r}")
    return chunks


def make_sweep_name(prefix: str = "fcs_stage0_sweep") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_chunk(
    *,
    problem_id: str,
    stage_index: int,
    chunk_sizes: tuple[int, ...],
    sweep_name: str,
    max_turns: int,
) -> SweepChunk:
    if stage_index < 0 or stage_index >= len(chunk_sizes):
        raise IndexError(f"Invalid stage_index={stage_index} for {chunk_sizes=}")
    start_sample = sum(chunk_sizes[:stage_index])
    num_samples = chunk_sizes[stage_index]
    total_after = start_sample + num_samples
    seed = start_sample * max_turns
    run_name = f"{sweep_name}_p{problem_id}_g{start_sample:03d}_{total_after - 1:03d}"
    return SweepChunk(
        problem_id=problem_id,
        stage_index=stage_index,
        start_sample=start_sample,
        num_samples=num_samples,
        total_after=total_after,
        seed=seed,
        run_name=run_name,
    )


def last_improvement_position(scores: list[float], eps: float = 1e-12) -> int:
    best = float("-inf")
    last = 0
    for idx, score in enumerate(scores, start=1):
        if score > best + eps:
            best = score
            last = idx
    return last


def decide_continuation(
    scores: list[float],
    *,
    next_stage_index: int,
    chunk_sizes: tuple[int, ...] = DEFAULT_CHUNK_SIZES,
    plateau_window: int = 8,
    min_repeated_best: int = 2,
    continue_zero_until: int = 25,
    eps: float = 1e-12,
) -> ContinuationDecision:
    if next_stage_index >= len(chunk_sizes):
        return ContinuationDecision(False, "maximum scheduled total reached")
    if not scores:
        return ContinuationDecision(True, "no completed scores yet")

    total = len(scores)
    best = max(scores)
    if best <= eps:
        if total < continue_zero_until:
            return ContinuationDecision(True, f"all scores are zero before G={continue_zero_until}")
        return ContinuationDecision(False, "all scores stayed zero through exploration budget")

    repeated_best = sum(1 for score in scores if abs(score - best) <= eps)
    since_improvement = total - last_improvement_position(scores, eps=eps)
    if repeated_best >= min_repeated_best and since_improvement >= plateau_window:
        return ContinuationDecision(
            False,
            (
                f"plateau: best repeated {repeated_best} times and no improvement "
                f"in last {since_improvement} samples"
            ),
        )

    return ContinuationDecision(True, "best-of-k curve is still exploratory")


def load_chunk_result(run_dir: Path, chunk: SweepChunk, duration_seconds: float) -> ChunkResult:
    summary_path = run_dir / "summary.json"
    eval_path = run_dir / "evaluations.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Chunk summary not found: {summary_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Chunk evaluations not found: {eval_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scores = [float(score) for score in summary.get("scores", [])]
    return ChunkResult(
        problem_id=chunk.problem_id,
        stage_index=chunk.stage_index,
        run_name=chunk.run_name,
        run_dir=str(run_dir),
        num_samples=int(summary["num_samples"]),
        seed=chunk.seed,
        scores=scores,
        best_score=float(summary["best_score"] or 0.0),
        mean_score=float(summary["mean_score"] or 0.0),
        successful_evals=int(summary["num_successful_evals"]),
        duration_seconds=duration_seconds,
    )


def run_chunk(chunk: SweepChunk, settings: WorkerSettings) -> ChunkResult:
    repo_root = Path(settings.repo_root)
    script = repo_root / "tinker_cookbook/recipes/frontier_cs_stage0/slurm/run_qwen_stage0.sbatch"
    run_dir = Path(settings.run_root) / chunk.run_name

    env = os.environ.copy()
    env.update(
        {
            "SCRATCH_BASE": str(SCRATCH_ROOT),
            "REPO_ROOT": settings.repo_root,
            "SKYRL_TX_ROOT": settings.skyrl_tx_root,
            "FRONTIER_CS_ROOT": settings.frontier_cs_root,
            "RUN_ROOT": settings.run_root,
            "MODEL_NAME": settings.model_name,
            "MODEL_SNAPSHOT": settings.model_snapshot,
            "TOKENIZER_NAME": settings.model_snapshot,
            "PROBLEM_ID": chunk.problem_id,
            "SAMPLES": str(chunk.num_samples),
            "SEED": str(chunk.seed),
            "RUN_NAME": chunk.run_name,
            "SAMPLES_PER_REQUEST": str(settings.samples_per_request),
            "MAX_TOKENS": str(settings.max_tokens),
            "MAX_TURNS": str(settings.max_turns),
            "MAX_PROMPT_TOKENS": str(settings.max_prompt_tokens),
            "TEMPERATURE": str(settings.temperature),
            "TOP_P": str(settings.top_p),
            "START_JUDGE": "1" if settings.start_judge else "0",
            "SHARD_ATTENTION_HEADS": "1",
        }
    )

    started = time.monotonic()
    subprocess.run(["bash", str(script)], cwd=repo_root, env=env, check=True)
    duration_seconds = round(time.monotonic() - started, 3)
    return load_chunk_result(run_dir, chunk, duration_seconds)


def run_mock_chunk(chunk: SweepChunk, settings: WorkerSettings) -> ChunkResult:
    """Cheap Slurm smoke worker that does not start Tinker or request GPUs."""
    started = time.monotonic()
    run_dir = Path(settings.run_root) / chunk.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    problem_bias = (sum(ord(char) for char in chunk.problem_id) % 7) / 100.0
    scores = [
        round(problem_bias + 0.01 * (chunk.start_sample + sample_idx + 1), 12)
        for sample_idx in range(chunk.num_samples)
    ]
    summary = {
        "best_sample_index": scores.index(max(scores)) if scores else None,
        "best_score": max(scores) if scores else None,
        "expected_best_at_g": max(scores) if scores else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "num_samples": chunk.num_samples,
        "num_successful_evals": chunk.num_samples,
        "scores": scores,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "evaluations.jsonl").open("w", encoding="utf-8") as f:
        for sample_idx, score in enumerate(scores):
            f.write(
                json.dumps(
                    {
                        "index": sample_idx,
                        "reward": score,
                        "score": score * 100.0,
                        "status": "success",
                        "turns": [],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    duration_seconds = round(time.monotonic() - started, 3)
    return load_chunk_result(run_dir, chunk, duration_seconds)


def write_state(
    state_path: Path,
    *,
    sweep_name: str,
    problem_states: dict[str, ProblemState],
    decisions: dict[str, str],
) -> None:
    payload: dict[str, Any] = {
        "sweep_name": sweep_name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "problems": {
            problem_id: {
                "scores": state.scores,
                "results": [asdict(result) for result in state.results],
                "next_stage_index": state.next_stage_index,
                "decision": decisions.get(problem_id),
            }
            for problem_id, state in sorted(problem_states.items())
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_executor(args: argparse.Namespace, submitit_dir: Path) -> submitit.SlurmExecutor:
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    params: dict[str, Any] = {
        "partition": args.partition,
        "account": args.account,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
        "time": args.timeout_min,
        "job_name": args.job_name,
        "ntasks_per_node": 1,
    }
    if args.qos:
        params["qos"] = args.qos
    if args.gpus_per_node > 0:
        params["gpus_per_node"] = args.gpus_per_node
    executor.update_parameters(**params)
    return executor


def print_dry_run(
    *,
    problems: tuple[str, ...],
    chunk_sizes: tuple[int, ...],
    sweep_name: str,
    max_turns: int,
) -> None:
    for problem_id in problems:
        chunks = [
            make_chunk(
                problem_id=problem_id,
                stage_index=stage_index,
                chunk_sizes=chunk_sizes,
                sweep_name=sweep_name,
                max_turns=max_turns,
            )
            for stage_index in range(len(chunk_sizes))
        ]
        print(problem_id, [asdict(chunk) for chunk in chunks])


def run_controller(args: argparse.Namespace) -> None:
    problems = parse_csv(args.problems)
    chunk_sizes = parse_int_csv(args.chunk_sizes)
    sweep_name = args.sweep_name or make_sweep_name()
    worker_settings = WorkerSettings(
        repo_root=args.repo_root,
        skyrl_tx_root=args.skyrl_tx_root,
        frontier_cs_root=args.frontier_cs_root,
        run_root=args.run_root,
        model_snapshot=args.model_snapshot,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        max_prompt_tokens=args.max_prompt_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        samples_per_request=args.samples_per_request,
    )

    if args.dry_run:
        print_dry_run(
            problems=problems,
            chunk_sizes=chunk_sizes,
            sweep_name=sweep_name,
            max_turns=args.max_turns,
        )
        return

    sweep_dir = Path(args.run_root) / "sweeps" / sweep_name
    state_path = sweep_dir / "state.json"
    submitit_dir = sweep_dir / "submitit"
    executor = configure_executor(args, submitit_dir)
    problem_states = {problem_id: ProblemState(problem_id=problem_id, results=[]) for problem_id in problems}
    decisions: dict[str, str] = {}
    ready = [
        make_chunk(
            problem_id=problem_id,
            stage_index=0,
            chunk_sizes=chunk_sizes,
            sweep_name=sweep_name,
            max_turns=args.max_turns,
        )
        for problem_id in problems
    ]
    active: dict[str, tuple[submitit.Job[ChunkResult], SweepChunk]] = {}
    worker = run_mock_chunk if args.mock_worker else run_chunk

    while ready or active:
        while ready and len(active) < args.max_active_jobs:
            chunk = ready.pop(0)
            job = executor.submit(worker, chunk, worker_settings)
            active[job.job_id] = (job, chunk)
            print(f"submitted job={job.job_id} problem={chunk.problem_id} run={chunk.run_name}", flush=True)

        completed_job_ids = [job_id for job_id, (job, _) in active.items() if job.done()]
        if not completed_job_ids:
            time.sleep(args.poll_seconds)
            continue

        for job_id in completed_job_ids:
            job, chunk = active.pop(job_id)
            result = job.result()
            state = problem_states[chunk.problem_id]
            state.results.append(result)
            decision = decide_continuation(
                state.scores,
                next_stage_index=state.next_stage_index,
                chunk_sizes=chunk_sizes,
                plateau_window=args.plateau_window,
                min_repeated_best=args.min_repeated_best,
                continue_zero_until=args.continue_zero_until,
            )
            decisions[chunk.problem_id] = decision.reason
            print(
                (
                    f"completed job={job_id} problem={chunk.problem_id} "
                    f"stage={chunk.stage_index} best={result.best_score:.6f} "
                    f"total_g={len(state.scores)} decision={decision.reason}"
                ),
                flush=True,
            )
            if decision.should_continue:
                ready.append(
                    make_chunk(
                        problem_id=chunk.problem_id,
                        stage_index=state.next_stage_index,
                        chunk_sizes=chunk_sizes,
                        sweep_name=sweep_name,
                        max_turns=args.max_turns,
                    )
                )
            write_state(
                state_path,
                sweep_name=sweep_name,
                problem_states=problem_states,
                decisions=decisions,
            )

    write_state(
        state_path,
        sweep_name=sweep_name,
        problem_states=problem_states,
        decisions=decisions,
    )
    print(f"sweep complete: {state_path}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic Submitit sweep for Frontier-CS Stage 0")
    parser.add_argument("--problems", default=",".join(DEFAULT_PROBLEMS))
    parser.add_argument("--chunk-sizes", default=",".join(str(x) for x in DEFAULT_CHUNK_SIZES))
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--max-active-jobs", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-worker", action="store_true")

    parser.add_argument("--partition", default="pli")
    parser.add_argument("--account", default="llm_explore")
    parser.add_argument("--qos", default="pli-low")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--timeout-min", type=int, default=360)
    parser.add_argument("--job-name", default="fcs-stage0-sweep")

    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--skyrl-tx-root", default=str(DEFAULT_SKYRL_TX_ROOT))
    parser.add_argument("--frontier-cs-root", default=str(DEFAULT_FRONTIER_CS_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--model-snapshot", default=str(DEFAULT_MODEL_SNAPSHOT))

    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=24576)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-request", type=int, default=1)

    parser.add_argument("--plateau-window", type=int, default=8)
    parser.add_argument("--min-repeated-best", type=int, default=2)
    parser.add_argument("--continue-zero-until", type=int, default=25)
    return parser


def main() -> None:
    run_controller(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
