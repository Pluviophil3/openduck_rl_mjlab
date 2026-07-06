#!/usr/bin/env bash
# Record every existing and newly created model_*.pt checkpoint.
#
# Usage:
#   ./watch_checkpoint.sh [run_dir] [num_envs] [video_length]
#
# Environment overrides:
#   MOTION_FILE, TASK, DEVICE, VIDEO_HEIGHT, VIDEO_WIDTH, POLL_INTERVAL,
#   CONDA_ENV, ONCE=1

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-logs/rsl_rl/open_duck_tracking}"
TASK="${TASK:-OpenDuck-Tracking-No-State-Estimation}"
DEVICE="${DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-unitree}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
VIDEO_WIDTH="${VIDEO_WIDTH:-640}"
ONCE="${ONCE:-0}"

detect_latest_run() {
  find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort \
    | tail -n 1
}

RUN_DIR="${1:-$(detect_latest_run)}"
NUM_ENVS="${2:-1}"
VIDEO_LENGTH="${3:-500}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[watch] ERROR: no run directory found under ${LOG_ROOT}" >&2
  exit 1
fi

# Prefer an explicit MOTION_FILE. Otherwise recover the exact absolute path
# written into this training run's frozen env configuration.
MOTION_FILE="${MOTION_FILE:-}"
if [[ -z "${MOTION_FILE}" && -f "${RUN_DIR}/params/env.yaml" ]]; then
  MOTION_FILE="$(
    awk '$1 == "motion_file:" {print $2; exit}' "${RUN_DIR}/params/env.yaml"
  )"
fi
if [[ -z "${MOTION_FILE}" || ! -f "${MOTION_FILE}" ]]; then
  echo "[watch] ERROR: motion file could not be resolved for ${RUN_DIR}" >&2
  echo "[watch] Set it explicitly, for example:" >&2
  echo "  MOTION_FILE=src/assets/motions/open_duck/motion.npz $0 ${RUN_DIR}" >&2
  exit 1
fi

VIDEO_DIR="${RUN_DIR}/videos/checkpoints"
mkdir -p "${VIDEO_DIR}"

echo "[watch] Run:      ${RUN_DIR}"
echo "[watch] Motion:   ${MOTION_FILE}"
echo "[watch] Videos:   ${VIDEO_DIR}"
echo "[watch] Task:     ${TASK}"
echo "[watch] Length:   ${VIDEO_LENGTH} steps"
echo "[watch] Interval: ${POLL_INTERVAL} seconds"

declare -A recorded

checkpoint_is_stable() {
  local checkpoint="$1"
  local size_before size_after
  size_before=$(stat -c '%s' "${checkpoint}")
  sleep 2
  size_after=$(stat -c '%s' "${checkpoint}")
  [[ "${size_before}" -gt 0 && "${size_before}" -eq "${size_after}" ]]
}

record_checkpoint() {
  local checkpoint="$1"
  local stem
  stem=$(basename "${checkpoint}" .pt)

  [[ -n "${recorded[${checkpoint}]+_}" ]] && return 0
  if compgen -G "${VIDEO_DIR}/${stem}-*.mp4" >/dev/null; then
    echo "[watch] Skip ${stem}: video already exists"
    recorded["${checkpoint}"]=1
    return 0
  fi
  if ! checkpoint_is_stable "${checkpoint}"; then
    echo "[watch] Wait for ${stem}: checkpoint is still being written"
    return 0
  fi

  echo "[watch] Recording ${stem}"
  if conda run -n "${CONDA_ENV}" --no-capture-output \
    python scripts/record_checkpoint.py "${TASK}" \
      --checkpoint-file="${checkpoint}" \
      --motion-file="${MOTION_FILE}" \
      --video-folder="${VIDEO_DIR}" \
      --num-envs="${NUM_ENVS}" \
      --video-length="${VIDEO_LENGTH}" \
      --device="${DEVICE}" \
      --video-height="${VIDEO_HEIGHT}" \
      --video-width="${VIDEO_WIDTH}"; then
    recorded["${checkpoint}"]=1
    echo "[watch] Finished ${stem}"
  else
    echo "[watch] WARNING: recording failed for ${stem}; it will be retried" >&2
  fi
}

scan_checkpoints() {
  local checkpoint
  while IFS= read -r checkpoint; do
    record_checkpoint "${checkpoint}"
  done < <(
    find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' \
      | sort -V
  )
}

scan_checkpoints
if [[ "${ONCE}" == "1" ]]; then
  echo "[watch] ONCE=1, all existing checkpoints processed"
  exit 0
fi

echo "[watch] Watching for new checkpoints; press Ctrl+C to stop"
while true; do
  sleep "${POLL_INTERVAL}"
  scan_checkpoints
done
