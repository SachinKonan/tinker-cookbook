import json
from pathlib import Path

from tinker_cookbook.recipes.frontier_cs_stage0.submitit_sweep import (
    DEFAULT_CHUNK_SIZES,
    ChunkResult,
    ProblemState,
    WorkerSettings,
    collect_job_result,
    decide_continuation,
    last_improvement_position,
    load_chunk_result,
    load_problem_states_from_state,
    make_chunk,
    make_next_continuation_chunk,
    parse_csv,
    parse_int_csv,
    run_mock_chunk,
)


def test_make_chunk_uses_10_15_25_schedule_and_seed_offsets() -> None:
    chunks = [
        make_chunk(
            problem_id="302",
            stage_index=stage_index,
            chunk_sizes=DEFAULT_CHUNK_SIZES,
            sweep_name="sweep",
            max_turns=20,
        )
        for stage_index in range(3)
    ]

    assert [(chunk.start_sample, chunk.num_samples, chunk.total_after) for chunk in chunks] == [
        (0, 10, 10),
        (10, 15, 25),
        (25, 25, 50),
    ]
    assert [chunk.seed for chunk in chunks] == [0, 200, 500]
    assert [chunk.run_name for chunk in chunks] == [
        "sweep_p302_g000_009",
        "sweep_p302_g010_024",
        "sweep_p302_g025_049",
    ]


def test_make_next_continuation_chunk_resumes_from_current_total() -> None:
    state = ProblemState(
        problem_id="302",
        results=[
            ChunkResult("302", 0, "sweep_p302_g000_009", "/tmp/a", 10, 0, [0.0] * 10, 0.0, 0.0, 10, 1.0),
            ChunkResult("302", 1, "sweep_p302_g010_024", "/tmp/b", 15, 200, [0.1] * 15, 0.1, 0.1, 15, 1.0),
        ],
    )

    chunk = make_next_continuation_chunk(
        state=state,
        sweep_name="sweep",
        max_turns=20,
        continuation_chunk_sizes=(65, 60),
        target_total_samples=150,
        initial_stage_index=state.next_stage_index,
    )

    assert chunk is not None
    assert chunk.stage_index == 2
    assert chunk.start_sample == 25
    assert chunk.num_samples == 65
    assert chunk.total_after == 90
    assert chunk.seed == 500
    assert chunk.run_name == "sweep_p302_g025_089"


def test_make_next_continuation_chunk_caps_at_target() -> None:
    state = ProblemState(
        problem_id="309",
        results=[
            ChunkResult("309", 0, "a", "/tmp/a", 10, 0, [0.0] * 10, 0.0, 0.0, 10, 1.0),
            ChunkResult("309", 1, "b", "/tmp/b", 15, 200, [0.0] * 15, 0.0, 0.0, 15, 1.0),
            ChunkResult("309", 2, "c", "/tmp/c", 25, 500, [0.0] * 25, 0.0, 0.0, 25, 1.0),
            ChunkResult("309", 3, "d", "/tmp/d", 65, 1000, [0.0] * 65, 0.0, 0.0, 65, 1.0),
        ],
    )

    chunk = make_next_continuation_chunk(
        state=state,
        sweep_name="sweep",
        max_turns=20,
        continuation_chunk_sizes=(65, 60),
        target_total_samples=150,
        initial_stage_index=3,
    )

    assert chunk is not None
    assert chunk.stage_index == 4
    assert chunk.start_sample == 115
    assert chunk.num_samples == 35
    assert chunk.total_after == 150
    assert chunk.run_name == "sweep_p309_g115_149"


def test_last_improvement_position() -> None:
    assert last_improvement_position([0.1, 0.2, 0.2, 0.3, 0.25]) == 4


def test_decide_continuation_stops_converged_g10_like_problem_302() -> None:
    scores = [
        0.2450327804,
        0.36680568537,
        0.0,
        0.0,
        0.36680568537,
        0.36680568537,
        0.36680568537,
        0.0,
        0.36680568537,
        0.24405779301,
    ]

    decision = decide_continuation(scores, next_stage_index=1)

    assert not decision.should_continue
    assert "plateau" in decision.reason


def test_decide_continuation_gives_zero_runs_one_more_stage() -> None:
    g10 = [0.0] * 10
    g25 = [0.0] * 25

    assert decide_continuation(g10, next_stage_index=1).should_continue
    assert not decide_continuation(g25, next_stage_index=2).should_continue


def test_decide_continuation_continues_when_recent_best_improved() -> None:
    scores = [0.0, 0.1, 0.1, 0.15, 0.16, 0.2, 0.21, 0.23, 0.24, 0.25]

    decision = decide_continuation(scores, next_stage_index=1)

    assert decision.should_continue
    assert "exploratory" in decision.reason


def test_parse_csv_helpers() -> None:
    assert parse_csv("46, 308,,302") == ("46", "308", "302")
    assert parse_int_csv("10, 15,25") == (10, 15, 25)


def test_load_problem_states_from_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "problems": {
                    "302": {
                        "results": [
                            {
                                "problem_id": "302",
                                "stage_index": 1,
                                "run_name": "sweep_p302_g010_024",
                                "run_dir": "/tmp/b",
                                "num_samples": 15,
                                "seed": 200,
                                "scores": [0.2],
                                "best_score": 0.2,
                                "mean_score": 0.2,
                                "successful_evals": 1,
                                "duration_seconds": 2.0,
                            },
                            {
                                "problem_id": "302",
                                "stage_index": 0,
                                "run_name": "sweep_p302_g000_009",
                                "run_dir": "/tmp/a",
                                "num_samples": 10,
                                "seed": 0,
                                "scores": [0.1],
                                "best_score": 0.1,
                                "mean_score": 0.1,
                                "successful_evals": 1,
                                "duration_seconds": 1.0,
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    states = load_problem_states_from_state(state_path, ("302", "306"))

    assert states["302"].scores == [0.1, 0.2]
    assert states["302"].next_stage_index == 2
    assert states["306"].scores == []


def test_run_mock_chunk_writes_loadable_artifacts(tmp_path: Path) -> None:
    chunk = make_chunk(
        problem_id="302",
        stage_index=0,
        chunk_sizes=(2,),
        sweep_name="mock",
        max_turns=20,
    )
    settings = WorkerSettings(run_root=str(tmp_path))

    result = run_mock_chunk(chunk, settings)
    loaded = load_chunk_result(tmp_path / chunk.run_name, chunk, duration_seconds=0.0)

    assert result.num_samples == 2
    assert result.successful_evals == 2
    assert result.best_score == loaded.best_score
    assert (tmp_path / chunk.run_name / "summary.json").exists()
    assert (tmp_path / chunk.run_name / "evaluations.jsonl").exists()


def test_collect_job_result_falls_back_to_artifacts(tmp_path: Path) -> None:
    chunk = make_chunk(
        problem_id="302",
        stage_index=0,
        chunk_sizes=(2,),
        sweep_name="mock",
        max_turns=20,
    )
    settings = WorkerSettings(run_root=str(tmp_path))
    expected = run_mock_chunk(chunk, settings)

    class MissingResultJob:
        job_id = "123"

        def result(self) -> ChunkResult:
            raise RuntimeError("missing result pickle")

    result = collect_job_result(MissingResultJob(), chunk, settings)

    assert result.best_score == expected.best_score
    assert result.duration_seconds == 0.0
