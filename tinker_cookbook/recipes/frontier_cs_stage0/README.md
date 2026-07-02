# Frontier-CS Stage 0 Baseline

This recipe runs the raw parallel-sampling baseline from `IDEA.md` on
Frontier-CS algorithmic tasks. It samples `G` C++17 solutions through Tinker,
scores them with the Frontier-CS algorithmic judge API, and writes the observed
reward distribution plus `E[best of k]` for `1 <= k <= G`.

Default local paths are scratch-backed:

```bash
FRONTIER_CS_ROOT=/scratch/gpfs/ZHUANGL/sk7524/Frontier-CS
RUN_ROOT=/scratch/gpfs/ZHUANGL/sk7524/tinker_runs/frontier_cs_stage0
MODEL_NAME=Qwen/Qwen3-4B-Thinking-2507
MODEL_SNAPSHOT=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea
```

The runner assumes a local Tinker-compatible server at `base_url` and a running
Frontier-CS judge at `judge_url`:

```bash
uv run --extra dev python -m tinker_cookbook.recipes.frontier_cs_stage0.run_baseline \
    problem_id=302 \
    num_samples=50 \
    samples_per_request=1 \
    base_url=http://127.0.0.1:8000/ \
    judge_url=http://127.0.0.1:8081
```

Outputs are written under one run directory:

- `samples/*.cpp` and `samples/*.txt`
- `samples.jsonl`
- `evaluations.jsonl`
- `best_of_k.csv`
- `best_of_k.svg`
- `summary.json`

The judge boundary is deliberately the Frontier-CS judge HTTP API. Docker,
Apptainer, or a later Ray actor can host that API without changing the
generation and plotting code.

Frontier-CS returns bounded scores on a `0..100` scale. `evaluations.jsonl`
keeps that raw `score`, while `best_of_k.csv`, `best_of_k.svg`, and
`summary.json` use normalized `reward = score / 100`.

## Local SkyRL caveat

The current local `/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-tx` server accepts
the HF repo id as `--base-model` and uses `HF_HOME` for cached weights:

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-tx
HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
uv run --extra tinker -m tx.tinker.api \
    --host 127.0.0.1 \
    --port 8000 \
    --base-model Qwen/Qwen3-4B-Thinking-2507 \
    --database-url sqlite:////scratch/gpfs/ZHUANGL/sk7524/tx_state/tinker.db \
    --checkpoints-base /scratch/gpfs/ZHUANGL/sk7524/tx_checkpoints
```

`Qwen/Qwen3-4B-Thinking-2507` reports `Qwen3ForCausalLM`, which matches the
current SkyRL tx Qwen3 model dispatch path.
