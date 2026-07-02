from pathlib import Path

from tinker_cookbook.recipes.frontier_cs_stage0.submitit_sweep import (
    DEFAULT_CHUNK_SIZES,
    WorkerSettings,
    decide_continuation,
    last_improvement_position,
    load_chunk_result,
    make_chunk,
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
