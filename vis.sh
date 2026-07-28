#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-OpenDuck-Tracking-No-State-Estimation}"
MOTION_FILE="${MOTION_FILE:-src/assets/motions/open_duck/A2_-_Sway_t2_stageii_realxml.npz}"
NUM_ENVS="${NUM_ENVS:-1}"
DEVICE="${DEVICE:-cuda:0}"
VIEWER="${VIEWER:-auto}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl/open_duck_tracking}"
CHECKPOINT_FILE="${1:-${CHECKPOINT_FILE:-}}"

if [[ -z "${CHECKPOINT_FILE}" && -d "${LOG_ROOT}" ]]; then
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
  echo "Checkpoint not found. Pass one as the first argument or set CHECKPOINT_FILE." >&2
  exit 1
fi

exec python scripts/play.py \
  "${TASK}" \
  "--motion-file=${MOTION_FILE}" \
  "--checkpoint-file=${CHECKPOINT_FILE}" \
  "--num-envs=${NUM_ENVS}" \
  "--device=${DEVICE}" \
  "--viewer=${VIEWER}"
