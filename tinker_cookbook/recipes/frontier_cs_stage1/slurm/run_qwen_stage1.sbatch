#!/usr/bin/env bash
#SBATCH --job-name=fcs-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/tinker_runs/frontier_cs_stage1/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/tinker_runs/frontier_cs_stage1/slurm/%x-%j.err

set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/ZHUANGL/sk7524}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_BASE}/tinker-cookbook-stage1-advice}"
SKYRL_TX_ROOT="${SKYRL_TX_ROOT:-${SCRATCH_BASE}/SkyRL/skyrl-tx}"
FRONTIER_CS_ROOT="${FRONTIER_CS_ROOT:-${SCRATCH_BASE}/Frontier-CS}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_BASE}/tinker_runs/frontier_cs_stage1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-${SCRATCH_BASE}/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
SLURM_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
TINKER_PORT="${TINKER_PORT:-$((9000 + (SLURM_ID_FOR_PORT % 1000)))}"
SAMPLES="${SAMPLES:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-4}}"

export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_BASE}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH_BASE}/.local/share/uv/python}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_BASE}/.cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_BASE}/.tmp/apptainer}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

RUN_DIR="${RUN_ROOT}/slurm/${SLURM_JOB_ID:-local_$$}"
mkdir -p "$RUN_DIR" "${RUN_ROOT}/slurm" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

cleanup() {
    if [[ -n "${TINKER_PID:-}" ]]; then
        kill "$TINKER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$SKYRL_TX_ROOT"
TINKER_CMD=(
    uv run --offline --frozen --extra tinker -m tx.tinker.api
    --host 127.0.0.1
    --port "$TINKER_PORT"
    --base-model "$MODEL_NAME"
    --max-lora-adapters "${MAX_LORA_ADAPTERS:-1}"
    --max-lora-rank "${MAX_LORA_RANK:-1}"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --checkpoints-base "${RUN_DIR}/tx_checkpoints"
    --database-url "sqlite:///${RUN_DIR}/tinker.db"
    --sample-max-num-sequences 1
)
if [[ "${SHARD_ATTENTION_HEADS:-1}" == "0" ]]; then
    TINKER_CMD+=(--no-shard-attention-heads)
fi
if [[ -n "${EXTERNAL_INFERENCE_URL:-}" ]]; then
    TINKER_CMD+=(--external-inference-url "$EXTERNAL_INFERENCE_URL")
fi
"${TINKER_CMD[@]}" >"${RUN_DIR}/tinker_server.log" 2>&1 &
TINKER_PID=$!

TINKER_READY=0
for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${TINKER_PORT}/" >/dev/null 2>&1; then
        TINKER_READY=1
        break
    fi
    if ! kill -0 "$TINKER_PID" 2>/dev/null; then
        echo "Tinker server process exited before becoming ready" >&2
        tail -n 200 "${RUN_DIR}/tinker_server.log" >&2 || true
        exit 1
    fi
    sleep 5
done
if [[ "$TINKER_READY" != "1" ]]; then
    echo "Tinker server did not become ready on port ${TINKER_PORT}" >&2
    tail -n 200 "${RUN_DIR}/tinker_server.log" >&2 || true
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -x "${SCRATCH_BASE}/tinker-cookbook/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRATCH_BASE}/tinker-cookbook/.venv/bin/python"
fi

cd "$REPO_ROOT"
RUN_NAME="${RUN_NAME:-slurm_${SLURM_JOB_ID:-local_$$}_problem_${PROBLEM_ID:-302}_stage1_advice}"
"$PYTHON_BIN" -m tinker_cookbook.recipes.frontier_cs_stage1.run_agentic_advice \
    frontier_cs_root="$FRONTIER_CS_ROOT" \
    problem_id="${PROBLEM_ID:-302}" \
    model_name="$MODEL_NAME" \
    tokenizer_name="${TOKENIZER_NAME:-$MODEL_SNAPSHOT}" \
    renderer_name="${RENDERER_NAME:-qwen3}" \
    base_url="http://127.0.0.1:${TINKER_PORT}/" \
    output_dir="$RUN_ROOT" \
    run_name="$RUN_NAME" \
    num_samples="$SAMPLES" \
    max_tokens="${MAX_TOKENS:-8192}" \
    max_student_turns="${MAX_STUDENT_TURNS:-6}" \
    max_advice_calls="${MAX_ADVICE_CALLS:-5}" \
    min_advice_calls="${MIN_ADVICE_CALLS:-0}" \
    max_malformed_turns="${MAX_MALFORMED_TURNS:-2}" \
    max_prompt_tokens="${MAX_PROMPT_TOKENS:-24576}" \
    temperature="${TEMPERATURE:-0.8}" \
    top_p="${TOP_P:-1.0}" \
    seed="${SEED:-0}" \
    advisor_model_name="${ADVISOR_MODEL_NAME:-$MODEL_NAME}" \
    advisor_tokenizer_name="${ADVISOR_TOKENIZER_NAME:-$MODEL_SNAPSHOT}" \
    advisor_renderer_name="${ADVISOR_RENDERER_NAME:-qwen3}" \
    advisor_max_tokens="${ADVISOR_MAX_TOKENS:-1024}" \
    advisor_temperature="${ADVISOR_TEMPERATURE:-0.2}" \
    advisor_top_p="${ADVISOR_TOP_P:-1.0}" \
    evaluate="${EVALUATE:-true}" \
    compile_timeout="${COMPILE_TIMEOUT:-60}" \
    case_timeout="${CASE_TIMEOUT:-20}" \
    eval_workers="${EVAL_WORKERS:-1}"
