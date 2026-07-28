#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-OpenDuck-Tracking-No-State-Estimation}"
MOTION_FILE="${MOTION_FILE:-src/assets/motions/open_duck/A2_-_Sway_t2_stageii_realxml.npz}"
NUM_ENVS="${NUM_ENVS:-4096}"
RUN_NAME="${RUN_NAME:-A2_-_Sway_t2_stageii_realxml}"
RANDOMIZATION_PROFILE="${RANDOMIZATION_PROFILE:-}"

CMD=(
  python scripts/train.py
  "${TASK}"
  "--motion-file=${MOTION_FILE}"
  "--env.scene.num-envs=${NUM_ENVS}"
  "--agent.run-name=${RUN_NAME}"
)

if [[ -n "${RANDOMIZATION_PROFILE}" ]]; then
  CMD+=("--randomization-profile=${RANDOMIZATION_PROFILE}")
fi

exec "${CMD[@]}" "$@"
