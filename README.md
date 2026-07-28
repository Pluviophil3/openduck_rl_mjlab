# OpenDuck RL MJLab

[中文说明](README_zh.md)

This repository is a clean fork of upstream `openduck_rl_mjlab` with the
OpenDuck Mini V2 motion-tracking work migrated in small commits.  The upstream
base is `origin/main` at commit `1425b15` (`Fix the warnings during
rough-terrain training.`).

The main new task IDs are:

```text
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
```

`OpenDuck-Tracking-No-State-Estimation` is the default training target used by
the helper scripts.  It removes privileged actor observations for base linear
velocity and motion-anchor position, while the critic still receives privileged
tracking state.

## What changed from upstream

The fork adds OpenDuck-specific source changes without copying local training
logs, W&B runs, or checkpoint artifacts.

| Area | Change |
| --- | --- |
| Robot asset | Added `src/assets/robots/open_duck_mini_v2/` with STL/PNG assets, `open_duck_mini_v2.xml`, `open_duck_mini_v2_real.xml`, scene XMLs, and `open_duck_constants.py`. |
| Task config | Added `src/tasks/tracking/config/open_duck/` with task registration, PPO config, OpenDuck environment config, and domain-randomization profiles. |
| Tracking MDP | Added OpenDuck action processing, backlash-aware motion loading, effective non-backlash joint observations, foot-contact observation, and clipped action history. |
| Runtime imports | Training/play/export code uses the local `src.tasks.tracking.mdp` instead of the upstream `mjlab.tasks.tracking.mdp` so the OpenDuck MDP extensions are used everywhere. |
| Motion data | Added OpenDuck NPZ motions under `src/assets/motions/open_duck/`, including 16-joint legacy clips and 26-joint real-XML/backlash clips. |
| Data tools | Added `scripts/duck_json_to_npz.py` and `scripts/resample_motion_npz.py` for constructing motion NPZ files. |
| Run scripts | Added `train.sh`, `forward_train.sh`, `vis.sh`, and `test.sh`. |
| Dependencies | `setup.py` now includes `mujoco==3.5.0`, `mujoco-warp==3.5.0`, `warp-lang==1.12.0`, and `scipy`. |

## Install

Create and activate your Python environment, then install this package editable:

```bash
pip install -e .
```

For headless training, the scripts set MuJoCo through the Python runtime.  If
your machine needs an explicit renderer, set:

```bash
export MUJOCO_GL=egl
```

## List Tasks

```bash
python scripts/list_envs.py OpenDuck
```

Expected task IDs:

```text
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
```

## Train

Default sway training:

```bash
bash train.sh
```

Default forward/head-motion training:

```bash
bash forward_train.sh
```

Both scripts accept environment variable overrides:

```bash
TASK=OpenDuck-Tracking-No-State-Estimation \
MOTION_FILE=src/assets/motions/open_duck/new_motion_realxml_backlash.npz \
NUM_ENVS=4096 \
RUN_NAME=forward_new_obs_safety \
RANDOMIZATION_PROFILE=baseline \
bash forward_train.sh
```

You can also call `scripts/train.py` directly:

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_t2_stageii_realxml.npz \
  --env.scene.num-envs=4096 \
  --agent.run-name=A2_-_Sway_t2_stageii_realxml
```

Useful direct flags:

| Flag | Meaning |
| --- | --- |
| `--motion-file` | Required for tracking tasks. Points to an OpenDuck NPZ clip. |
| `--randomization-profile` | Optional OpenDuck profile override: `nominal`, `baseline`, `sensor`, `dynamics`, `actuator`, `latency`, `light`, `full`. |
| `--checkpoint-file` | Optional local `.pt` checkpoint to initialize or resume from directly. |
| `--env.scene.num-envs` | Number of parallel MuJoCo environments. |
| `--agent.run-name` | Suffix for the run directory under `logs/rsl_rl/open_duck_tracking/`. |

## Play A Checkpoint

Pass a checkpoint explicitly:

```bash
bash vis.sh logs/rsl_rl/open_duck_tracking/<run>/model_7500.pt
```

Or set variables:

```bash
MOTION_FILE=src/assets/motions/open_duck/new_motion_realxml_backlash.npz \
CHECKPOINT_FILE=logs/rsl_rl/open_duck_tracking/<run>/model_7500.pt \
VIEWER=auto \
bash vis.sh
```

`vis.sh` tries to find the newest `model_*.pt` under
`logs/rsl_rl/open_duck_tracking/` if no checkpoint argument is provided.

## Smoke Test

```bash
bash test.sh
```

This compiles Python files, lists OpenDuck tasks, and prints the shape of the
default forward motion NPZ.

## Robot Asset

The main robot config is:

```text
src/assets/robots/open_duck_mini_v2/open_duck_constants.py
```

Important XML files:

| File | Purpose |
| --- | --- |
| `open_duck_mini_v2.xml` | Nominal OpenDuck model. |
| `open_duck_mini_v2_real.xml` | Real/backlash-aware model used by training and data tools. |
| `open_duck_mini_v2_no_head.xml` | Variant without the head assembly. |
| `scene_mjx_flat_terrain.xml` | Flat terrain scene. |
| `scene_mjx_rough_terrain.xml` | Rough terrain scene. |
| `scene_training_neutral.xml` | Training scene with neutral setup. |

`OPEN_DUCK_XML` points to `open_duck_mini_v2_real.xml`.

The real XML has 26 MuJoCo joints after the free root:

```text
16 non-backlash joints + 10 passive *_backlash joints
```

The effective tracking joint state sums each main joint with its matching
`*_backlash` joint when present, and drops the passive backlash columns from
policy-visible joint observations and reference commands.

## Actions

The policy action is joint-position target offset.

| Item | Value |
| --- | --- |
| Action term | `joint_pos` |
| Action class | `OpenDuckJointPositionAction` |
| Policy action dimension | `14` |
| Controlled joints | 10 leg joints plus `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` |
| Not controlled by policy | `left_antenna`, `right_antenna`, passive `*_backlash` joints |
| Scale | `0.25` rad for each controlled joint pattern |
| Offset | `use_default_offset=True` |
| Raw action clip | `[-20, 20]` before scaling |
| Joint limit safety | Clips targets inside MuJoCo joint limits with `0.02` rad margin |
| Optional rate limit | `max_target_step`, disabled by default |
| Optional low-pass filter | `cutoff_frequency`, disabled by default |

The action sent to MuJoCo is:

```text
target = clipped_raw_action * OPEN_DUCK_ACTION_SCALE + default_joint_position
```

Then optional rate limiting, joint-limit safety clipping, and optional low-pass
filtering are applied.

## Actor Observations

Actor observations are concatenated in the order defined by
`open_duck_flat_tracking_env_cfg`.

With state estimation (`OpenDuck-Tracking`):

| Term | Dim | Meaning |
| --- | ---: | --- |
| `command` | 32 | Reference effective joint position and velocity, `16 + 16`. |
| `motion_anchor_pos_b` | 3 | Desired trunk anchor position in the robot anchor frame. |
| `base_lin_vel` | 3 | IMU linear velocity. |
| `base_ang_vel` | 3 | IMU angular velocity. |
| `base_lin_acc` | 3 | IMU linear acceleration. |
| `joint_pos` | 16 | Effective non-backlash joint position relative to default. |
| `joint_vel` | 16 | Effective non-backlash joint velocity. |
| `actions` | 42 | Current and previous two clipped raw action vectors, `14 * 3`. |
| `feet_contact` | 2 | Binary contact for left and right TPU foot bottoms. |
| **Total** | **120** | Actor observation width. |

Without state estimation (`OpenDuck-Tracking-No-State-Estimation`):

| Removed term | Dim |
| --- | ---: |
| `motion_anchor_pos_b` | 3 |
| `base_lin_vel` | 3 |

The no-state actor width is therefore `114`.

## Critic Observations

The critic keeps privileged information:

| Term | Dim | Meaning |
| --- | ---: | --- |
| `command` | 32 | Reference effective joint position and velocity. |
| `motion_anchor_pos_b` | 3 | Desired trunk anchor position in robot frame. |
| `motion_anchor_ori_b` | 6 | Desired anchor orientation as first two rotation-matrix columns. |
| `body_pos` | 24 | 8 tracked body positions, `8 * 3`. |
| `body_ori` | 48 | 8 tracked body orientations, `8 * 6`. |
| `base_lin_vel` | 3 | IMU linear velocity. |
| `base_ang_vel` | 3 | IMU angular velocity. |
| `base_lin_acc` | 3 | IMU linear acceleration. |
| `joint_pos` | 16 | Effective non-backlash joint position relative to default. |
| `joint_vel` | 16 | Effective non-backlash joint velocity. |
| `actions` | 42 | 3-frame clipped raw action history. |
| `feet_contact` | 2 | Binary left/right foot contact. |
| **Total** | **198** | Critic observation width. |

Tracked bodies:

```text
trunk_assembly
left_roll_to_pitch_assembly
knee_and_ankle_assembly_2
foot_assembly
right_roll_to_pitch_assembly
knee_and_ankle_assembly_4
foot_assembly_2
head_assembly
```

## Rewards

The OpenDuck task starts from the shared motion-tracking reward set and tightens
position tolerances for the smaller robot.

| Reward | Weight | Std / Params |
| --- | ---: | --- |
| `motion_global_root_pos` | `0.5` | `std=0.08` for OpenDuck. |
| `motion_global_root_ori` | `0.5` | `std=0.4`. |
| `motion_body_pos` | `1.0` | `std=0.08` for OpenDuck. |
| `motion_body_ori` | `1.0` | `std=0.4`. |
| `motion_body_lin_vel` | `1.0` | `std=1.0`. |
| `motion_body_ang_vel` | `1.0` | `std=3.14`. |
| `action_rate_l2` | `-0.1` | Penalizes action changes. |
| `joint_limit` | `-10.0` | Excludes passive `*_backlash` joints. |
| `self_collisions` | `-10.0` | Uses `self_collision` contact sensor with threshold `10.0`. |

Terminations are also tighter than the shared humanoid defaults:

| Termination | OpenDuck value |
| --- | --- |
| `anchor_pos` | `0.08` z-only threshold. |
| `anchor_ori` | Shared threshold `0.8`. |
| `ee_body_pos` | `0.08` z-only threshold on both feet and head. |

## Domain Randomization

Randomization is implemented in:

```text
src/tasks/tracking/config/open_duck/randomization.py
```

Profiles are composable named presets.  `play=True` forces `nominal`.
Training defaults to `baseline`, and `scripts/train.py` can override it with
`--randomization-profile`.

| Profile | Purpose |
| --- | --- |
| `nominal` | No randomization. Used for play/evaluation. |
| `baseline` | Observation corruption, reset perturbation, pushes, encoder bias, foot friction, trunk COM offset. |
| `sensor` | Observation corruption, encoder bias, gyro bias. |
| `dynamics` | Foot friction, trunk COM, body mass/inertia, damping, friction, armature. |
| `actuator` | PD gain and effort-limit scaling. |
| `latency` | Action and observation delay. |
| `light` | Moderate sensor, dynamics, actuator, and latency randomization. |
| `full` | Stronger combined randomization. |

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `observation_corruption` | Enables actor observation noise/corruption. |
| `reset_perturbation` | Enables reference-state initialization pose, velocity, and joint perturbations. |
| `push_robot` | Adds interval base velocity pushes every `1.0` to `3.0` seconds. |
| `encoder_bias_rad` | Startup joint encoder bias range in radians. |
| `gyro_bias_rad_s` | Additive gyroscope bias range in rad/s. |
| `foot_friction` | Absolute friction range for `left/right_foot_bottom_tpu`, shared across foot geoms. |
| `trunk_com_offset_m` | Additive COM offset range for `trunk_assembly`. |
| `body_mass_scale` | Body mass/inertia scale through pseudo-inertia randomization. |
| `joint_damping_scale` | Multiplicative joint damping scale. |
| `joint_friction_scale` | Multiplicative joint frictionloss scale. |
| `joint_armature_scale` | Multiplicative joint armature scale. |
| `kp_scale` | Multiplicative position actuator proportional gain scale. |
| `effort_scale` | Multiplicative actuator force range scale. |
| `action_delay_control_steps` | Min/max action delay in control steps, converted to physics steps by `cfg.decimation`. |
| `observation_delay_control_steps` | Min/max observation delay for `base_ang_vel`, `joint_pos`, and `joint_vel`. |

The reset perturbation ranges are scaled for the approximately 22 cm robot:

```text
pose x/y: +/-0.02 m
pose z: +/-0.005 m
roll/pitch: +/-0.05 rad
yaw: +/-0.1 rad
linear velocity x/y: +/-0.2 m/s
linear velocity z: +/-0.1 m/s
angular velocity roll/pitch: +/-0.25 rad/s
angular velocity yaw: +/-0.4 rad/s
joint position: +/-0.05 rad
```

## Motion Data

Motion files live in:

```text
src/assets/motions/open_duck/
```

| File | FPS | Frames | Joint columns | Notes |
| --- | ---: | ---: | ---: | --- |
| `A2_-_Sway_stageii_50hz.npz` | 50 | 600 | 16 | Legacy 16-joint sway clip. |
| `A2_-_Sway_t2_stageii.npz` | 50 | 798 | 16 | Legacy sway clip without `joint_names`. |
| `A2_-_Sway_t2_stageii_realxml.npz` | 50 | 798 | 16 | Sway clip with joint names for real XML alignment. |
| `forward_headshake_40deg_04hz_50hz.npz` | 50 | 500 | 16 | Legacy forward/head-shake clip. |
| `forward_headshake_40deg_04hz_50hz_realxml_backlash.npz` | 50 | 500 | 26 | Real XML/backlash clip. |
| `forward_headshake_real_50hz.npz` | 50 | 500 | 26 | Real XML clip. |
| `new_motion_realxml_backlash.npz` | 50 | 400 | 26 | Current default forward training clip. |

Required NPZ arrays:

```text
fps
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
joint_names
body_names
```

The motion loader accepts either full 26-joint real XML arrays or older 16-joint
arrays.  If an older clip omits passive backlash columns, the loader aligns by
`joint_names` and fills backlash columns with zeros.

## Build Motion Data

Convert an OpenDuck generator JSON recording:

```bash
python scripts/duck_json_to_npz.py \
  --input input_recording.json \
  --output src/assets/motions/open_duck/my_motion_realxml_backlash.npz
```

Resample an existing OpenDuck NPZ and rebuild body kinematics:

```bash
python scripts/resample_motion_npz.py \
  src/assets/motions/open_duck/input.npz \
  src/assets/motions/open_duck/output_50hz.npz \
  --output-fps 50
```

## PPO Settings

The OpenDuck PPO runner config is:

```text
src/tasks/tracking/config/open_duck/rl_cfg.py
```

Important defaults:

```text
actor hidden dims: 512, 256, 128
critic hidden dims: 512, 256, 128
activation: elu
actor observation normalization: enabled
critic observation normalization: enabled
distribution: GaussianDistribution, scalar std, init_std=1.0
learning rate: 1e-3, adaptive schedule
gamma: 0.99
lambda: 0.95
entropy coef: 0.005
steps per env: 24
save interval: 500
max iterations: 30001
logger: wandb
wandb project: openduck_rl_mjlab
```

## One-Time Debug Scripts Not Migrated

The source development workspace contained several one-off or generated
debugging artifacts.  They are intentionally not part of this clean migration:

| Path in old workspace | Reason |
| --- | --- |
| `checker/hardware_joint_checker.py` | Hardware joint probing/checking helper. Useful during bring-up, not required for training. |
| `checker/mujoco_joint_checker.py` | MuJoCo joint probing helper. One-off model inspection. |
| `checker/joint_checker_common.py` | Shared helper only used by checker scripts. |
| `checker/plot_joint_logs.py` | Plotting helper for joint debug logs. |
| `checker/README.md` | Documentation for the checker-only workflow. |
| `scripts/inspect_open_duck_mjcf.py` | MJCF inspection/report generation script. Its output is not needed at runtime. |
| `scripts/evaluate_open_duck.py` | Batch checkpoint evaluation script tied to local experiment artifacts. |
| `scripts/record_checkpoint.py` | Video recording helper for local checkpoints. |
| `scripts/replay_open_duck_reference_motion.py` | Viewer replay utility used to validate motion files interactively. |
| `watch_checkpoint.sh` | Local watcher for checkpoint/video generation. |
| `goal.md` | Development planning notes from the old workspace. |
| `doc/open_duck_mjcf_report.json` | Generated MJCF inspection report. |
| `doc/open_duck_sim2real_randomization.md` | Earlier randomization design note; superseded by this README and source comments. |
| `doc/openduck_training.md` | Earlier training note; superseded by this README. |
| `models/Forward_headshake/`, `models/Sway_t1/`, `models/Sway_t2/` | Local checkpoints, ONNX exports, videos, TensorBoard events, and params from experiments. |
| `logs/`, `wandb/`, `MUJOCO_LOG.TXT`, `unitree_rl_mjlab.egg-info/`, `__pycache__/` | Generated runtime/build/cache artifacts. |

The reusable parts of that workflow are preserved as source:

```text
scripts/duck_json_to_npz.py
scripts/resample_motion_npz.py
src/assets/motions/open_duck/*.npz
```

## Commit Migration Order

This branch was built from a clean clone in staged commits:

```text
2cee8a1 Add OpenDuck Mini robot assets
b3673dc Add OpenDuck tracking task config
c962bfc Adapt tracking runtime for OpenDuck
0828d01 Add OpenDuck motion data tools
ec154b6 Add OpenDuck reference motions
7558991 Add OpenDuck training and play scripts
ab822d6 Use local tracking MDP imports
44eb49f Document OpenDuck MJLab fork
938d422 Add OpenDuck randomization tests
```

The README documents the resulting clean fork and the migration choices.
