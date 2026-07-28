#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q scripts src
python scripts/list_envs.py OpenDuck
python -c "import numpy as np; p='src/assets/motions/open_duck/new_motion_realxml_backlash.npz'; d=np.load(p); print(p, d['joint_pos'].shape, d['body_pos_w'].shape)"
