#!/usr/bin/env bash
set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/ZHUANGL/sk7524}"
FRONTIER_CS_ROOT="${FRONTIER_CS_ROOT:-${SCRATCH_BASE}/Frontier-CS}"
RUN_DIR="${RUN_DIR:-${SCRATCH_BASE}/tinker_runs/frontier_cs_stage0/judge}"
JUDGE_PORT="${JUDGE_PORT:-8081}"
JUDGE_WORKERS="${JUDGE_WORKERS:-7}"
GJ_PARALLELISM="${GJ_PARALLELISM:-7}"
FRONTIER_CS_JUDGE_IMAGE="${FRONTIER_CS_JUDGE_IMAGE:-docker://yanagiorigami/frontier-cs-harbor-judge:latest}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_BASE}/.cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_BASE}/.tmp/apptainer}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RUN_DIR"/{images,submissions,data}

SIF_PATH="${SIF_PATH:-${SCRATCH_BASE}/tinker_runs/frontier_cs_stage0/images/frontier-cs-judge.sif}"
if [[ ! -f "$SIF_PATH" ]]; then
    apptainer pull "$SIF_PATH" "$FRONTIER_CS_JUDGE_IMAGE"
fi

exec apptainer run \
    --cleanenv \
    --containall \
    --bind "${FRONTIER_CS_ROOT}/algorithmic/problems:/app/problems:ro" \
    --bind "${RUN_DIR}/submissions:/app/submissions" \
    --bind "${RUN_DIR}/data:/app/data" \
    --env "PORT=${JUDGE_PORT}" \
    --env "GJ_ADDR=http://127.0.0.1:5050" \
    --env "JUDGE_WORKERS=${JUDGE_WORKERS}" \
    --env "GJ_PARALLELISM=${GJ_PARALLELISM}" \
    --env "SAVE_OUTPUTS=false" \
    "$SIF_PATH"
