import json
from pathlib import Path

from tinker_cookbook.recipes.frontier_cs_stage0.run_baseline import (
    DEFAULT_QWEN_REPO,
    SampleRecord,
    Stage0Config,
    _parse_checker_ratio,
    close_empty_qwen_think,
    compact_compile_error,
    expected_best_of_k,
    extract_cpp_code,
    prefix_best,
    render_repair_prompt,
    sample_record_to_dict,
    strip_outer_markdown_fence,
    tail_by_tokens,
    write_svg_plot,
    write_trajectory_files,
)


class CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str], *, skip_special_tokens: bool = False) -> str:
        return "".join(tokens)


def test_expected_best_of_k_two_scores() -> None:
    assert expected_best_of_k([0.0, 1.0]) == [0.5, 1.0]


def test_default_model_is_qwen() -> None:
    assert Stage0Config().model_name == DEFAULT_QWEN_REPO


def test_default_stage0_generation_controls() -> None:
    config = Stage0Config()
    assert config.max_tokens == 8192
    assert config.max_turns == 20
    assert config.max_prompt_tokens == 24576
    assert config.save_trajectories is True


def test_parse_checker_ratio() -> None:
    output = "points 0.25 Ratio: 0.250000000000 B=10 X=7 denom=10"
    assert _parse_checker_ratio(output) == 0.25


def test_expected_best_of_k_with_duplicate_max() -> None:
    assert expected_best_of_k([0.0, 1.0, 1.0]) == [2 / 3, 1.0, 1.0]


def test_prefix_best() -> None:
    assert prefix_best([0.1, 0.0, 0.4, 0.3]) == [0.1, 0.1, 0.4, 0.4]


def test_extract_cpp_code_prefers_cpp_fence() -> None:
    text = "Here is code:\n```cpp\n#include <bits/stdc++.h>\nint main(){return 0;}\n```\n"
    assert extract_cpp_code(text) == "#include <bits/stdc++.h>\nint main(){return 0;}"


def test_strip_outer_markdown_fence_preserves_inner_fences() -> None:
    statement = "```markdown\n# Title\n```\ninput\n```\n```"
    assert strip_outer_markdown_fence(statement) == "# Title\n```\ninput\n```\n"


def test_close_empty_qwen_think() -> None:
    rendered = "<|im_start|>assistant\n<think>\n"
    assert close_empty_qwen_think(rendered) == "<|im_start|>assistant\n<think>\n</think>\n\n"


def test_tail_by_tokens_keeps_suffix() -> None:
    assert tail_by_tokens(CharTokenizer(), "abcdef", 3) == "def"


def test_compact_compile_error_keeps_suffix() -> None:
    assert compact_compile_error("0123456789", max_chars=4) == "6789"
    assert compact_compile_error(None) == "No compiler error was captured."


def test_render_repair_prompt_respects_budget() -> None:
    prompt = render_repair_prompt(
        CharTokenizer(),
        problem_id="302",
        statement="short statement",
        candidate_code=("A" * 5000) + "TAIL",
        compile_error="missing main",
        max_prompt_tokens=2000,
    )

    assert len(prompt) <= 2000
    assert "TAIL" in prompt
    assert "missing main" in prompt


def test_write_svg_plot(tmp_path: Path) -> None:
    out = tmp_path / "plot.svg"
    write_svg_plot(out, [0.2, 0.4], [0.1, 0.5])
    assert out.read_text().startswith("<svg")


def test_sample_record_to_dict_excludes_trajectory_by_default() -> None:
    record = SampleRecord(
        index=0,
        seed=123,
        text="text",
        code="int main(){return 0;}",
        trajectory=[{"prompt": "full prompt"}],
    )

    assert "trajectory" not in sample_record_to_dict(record)
    assert sample_record_to_dict(record, include_trajectory=True)["trajectory"] == [
        {"prompt": "full prompt"}
    ]


def test_write_trajectory_files(tmp_path: Path) -> None:
    record = SampleRecord(
        index=0,
        seed=123,
        text="raw text",
        code="int main(){return 0;}",
        reward=0.5,
        score=50.0,
        status="success",
        metadata={"compiled_during_sampling": True},
        turns=[{"turn": 1, "compile_ok": True}],
        trajectory=[
            {
                "turn": 1,
                "prompt": "rendered prompt",
                "generated_text": "generated",
                "candidate_code": "int main(){return 0;}",
                "compile_ok": True,
            }
        ],
    )

    write_trajectory_files(tmp_path, [record], Stage0Config())

    trajectory = json.loads((tmp_path / "trajectories" / "sample_000.json").read_text())
    assert trajectory["turns"][0]["prompt"] == "rendered prompt"
    assert "trajectory" not in trajectory["record"]

    index = (tmp_path / "trajectories" / "index.jsonl").read_text().splitlines()
    assert len(index) == 1
    assert json.loads(index[0])["trajectory_path"] == "trajectories/sample_000.json"
