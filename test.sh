#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

TASK="${TASK:-OpenDuck-Tracking-No-State-Estimation}"
MOTION_FILE="${MOTION_FILE:-src/assets/motions/open_duck/A2_-_Sway_t2_stageii_50hz.npz}"
NUM_ENVS="${NUM_ENVS:-1}"
DEVICE="${DEVICE:-cuda:0}"
VIEWER="${VIEWER:-auto}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl/open_duck_tracking}"

# An explicit checkpoint can be passed as the first argument. Otherwise, use
# the numerically latest checkpoint from the latest run directory.
CHECKPOINT_FILE="${1:-${CHECKPOINT_FILE:-logs/rsl_rl/open_duck_tracking/2026-07-14_15-02-49_A2_-_Sway_t2_stageii_realxml/model_7500.pt}}"
if [[ -z "${CHECKPOINT_FILE}" ]]; then
  CHECKPOINT_FILE="$(
    find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' \
      | sort -V \
      | tail -n 1
  )"
fi

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "Motion file not found: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT_FILE}" || ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "Checkpoint not found under: ${LOG_ROOT}" >&2
  echo "Usage: $0 [path/to/model_<iteration>.pt]" >&2
  exit 1
fi

echo "Task:       ${TASK}"
echo "Motion:     ${MOTION_FILE}"
echo "Checkpoint: ${CHECKPOINT_FILE}"
echo "Device:     ${DEVICE}"
echo "Viewer:     ${VIEWER}"

exec python scripts/play.py \
  "${TASK}" \
  --motion-file="${MOTION_FILE}" \
  --checkpoint-file="${CHECKPOINT_FILE}" \
  --num-envs="${NUM_ENVS}" \
  --device="${DEVICE}" \
  --viewer="${VIEWER}"
