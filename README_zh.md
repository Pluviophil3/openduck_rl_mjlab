# OpenDuck RL MJLab 中文说明

这个仓库是从上游 `openduck_rl_mjlab` 重新克隆出来的干净迁移版本，基于
`origin/main` 的 `1425b15` 提交：`Fix the warnings during rough-terrain training.`。

本分支把旧 `unitree_rl_mjlab` 工作区里和 OpenDuck Mini V2 相关的可复用修改，按功能拆分后逐步迁移过来；本地训练日志、W&B 运行记录、checkpoint、视频和一次性调试脚本没有混入核心仓库。

新增的核心任务 ID：

```text
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
```

默认推荐使用：

```text
OpenDuck-Tracking-No-State-Estimation
```

这个任务会从 actor 输入里去掉更偏 privileged 的 `motion_anchor_pos_b` 和 `base_lin_vel`，用于更接近无完整状态估计的部署设置；critic 仍保留 privileged tracking 信息以帮助训练。

## 相比上游修改了什么

| 模块 | 修改内容 |
| --- | --- |
| 机器人资产 | 新增 `src/assets/robots/open_duck_mini_v2/`，包含 STL/PNG mesh、`open_duck_mini_v2.xml`、`open_duck_mini_v2_real.xml`、scene XML 和 `open_duck_constants.py`。 |
| 任务配置 | 新增 `src/tasks/tracking/config/open_duck/`，注册 OpenDuck tracking task，定义 OpenDuck 环境配置、PPO 配置和域随机化 profile。 |
| Tracking MDP | 新增 backlash-aware motion loader、effective non-backlash joint 观测、脚底接触观测、action history、OpenDuck action safety processing。 |
| 本地 tracking 路径 | 训练、播放和导出逻辑改为使用 `src.tasks.tracking.mdp`，而不是上游 `mjlab.tasks.tracking.mdp`，确保 OpenDuck 的扩展 MDP 生效。 |
| 动作数据 | 新增 `src/assets/motions/open_duck/*.npz`，同时支持旧 16-joint motion 和 real XML/backlash 26-joint motion。 |
| 数据工具 | 新增 `scripts/duck_json_to_npz.py` 和 `scripts/resample_motion_npz.py`，用于构造和重采样 OpenDuck motion。 |
| 运行脚本 | 新增 `train.sh`、`forward_train.sh`、`vis.sh`、`test.sh`。 |
| 依赖 | `setup.py` 增加 `mujoco==3.5.0`、`mujoco-warp==3.5.0`、`warp-lang==1.12.0`、`scipy`。 |
| 测试 | 新增 `tests/test_open_duck_randomization.py`，覆盖随机化 profile 的行为。 |

## 安装

进入仓库后，安装为 editable package：

```bash
pip install -e .
```

如果是无显示器训练，通常需要：

```bash
export MUJOCO_GL=egl
```

## 查看任务

```bash
python scripts/list_envs.py OpenDuck
```

应能看到：

```text
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
```

## 训练

默认 sway motion 训练：

```bash
bash train.sh
```

默认 forward/head motion 训练：

```bash
bash forward_train.sh
```

这两个脚本都支持环境变量覆盖：

```bash
TASK=OpenDuck-Tracking-No-State-Estimation \
MOTION_FILE=src/assets/motions/open_duck/new_motion_realxml_backlash.npz \
NUM_ENVS=4096 \
RUN_NAME=forward_new_obs_safety \
RANDOMIZATION_PROFILE=baseline \
bash forward_train.sh
```

也可以直接调用 `scripts/train.py`：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_t2_stageii_realxml.npz \
  --env.scene.num-envs=4096 \
  --agent.run-name=A2_-_Sway_t2_stageii_realxml
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--motion-file` | tracking 任务必填，指向 OpenDuck NPZ motion。 |
| `--randomization-profile` | 可选随机化 profile：`nominal`、`baseline`、`sensor`、`dynamics`、`actuator`、`latency`、`light`、`full`。 |
| `--checkpoint-file` | 可选本地 `.pt` checkpoint，用于初始化或恢复训练。 |
| `--env.scene.num-envs` | 并行 MuJoCo 环境数量。 |
| `--agent.run-name` | 训练 run 名称后缀，日志保存在 `logs/rsl_rl/open_duck_tracking/` 下。 |

## 播放 checkpoint

显式传入 checkpoint：

```bash
bash vis.sh logs/rsl_rl/open_duck_tracking/<run>/model_7500.pt
```

也可以通过环境变量传入：

```bash
MOTION_FILE=src/assets/motions/open_duck/new_motion_realxml_backlash.npz \
CHECKPOINT_FILE=logs/rsl_rl/open_duck_tracking/<run>/model_7500.pt \
VIEWER=auto \
bash vis.sh
```

如果没有显式传入 checkpoint，`vis.sh` 会尝试在
`logs/rsl_rl/open_duck_tracking/` 下查找最新的 `model_*.pt`。

## 轻量检查

```bash
bash test.sh
```

这个脚本会：

1. 编译 `scripts/` 和 `src/` 下的 Python 文件。
2. 如果依赖已安装，列出 OpenDuck task。
3. 检查默认 forward motion NPZ 的 shape。

如果当前 Python 环境还没有安装 `tyro/mjlab`，task registry 检查会自动跳过。

## OpenDuck 机器人资产

主配置文件：

```text
src/assets/robots/open_duck_mini_v2/open_duck_constants.py
```

重要 XML：

| 文件 | 用途 |
| --- | --- |
| `open_duck_mini_v2.xml` | 名义 OpenDuck 模型。 |
| `open_duck_mini_v2_real.xml` | 带真实结构/backlash 的模型，训练和数据构造默认使用它。 |
| `open_duck_mini_v2_no_head.xml` | 去掉头部组件的模型变体。 |
| `scene_mjx_flat_terrain.xml` | 平地场景。 |
| `scene_mjx_rough_terrain.xml` | 粗糙地形场景。 |
| `scene_training_neutral.xml` | 训练用 neutral 场景。 |

`OPEN_DUCK_XML` 指向：

```text
src/assets/robots/open_duck_mini_v2/xmls/open_duck_mini_v2_real.xml
```

`open_duck_mini_v2_real.xml` 在 free root 之后有 26 个 MuJoCo joint：

```text
16 个非 backlash joint + 10 个 passive *_backlash joint
```

训练时的 effective joint state 会把主关节和对应 `*_backlash` 关节相加，然后从策略可见 joint 观测和 reference command 中去掉 passive backlash 列。

## 动作输出

策略输出是 joint position target offset。

| 项 | 值 |
| --- | --- |
| action term | `joint_pos` |
| action 类 | `OpenDuckJointPositionAction` |
| 策略动作维度 | `14` |
| 控制关节 | 10 个腿部关节，加 `neck_pitch`、`head_pitch`、`head_yaw`、`head_roll` |
| 不由策略控制 | `left_antenna`、`right_antenna`、所有 passive `*_backlash` joint |
| action scale | 每个受控关节 pattern 都是 `0.25` rad |
| offset | `use_default_offset=True` |
| 原始 action clip | `[-20, 20]` |
| joint safety clip | 在 MuJoCo joint limit 内加 `0.02` rad margin 后裁剪 |
| 可选 rate limit | `max_target_step`，默认关闭 |
| 可选低通滤波 | `cutoff_frequency`，默认关闭 |

实际发送给 MuJoCo 的 target：

```text
target = clipped_raw_action * OPEN_DUCK_ACTION_SCALE + default_joint_position
```

之后会依次经过可选 rate limit、joint limit safety clip 和可选 low-pass filter。

## Actor 观测

Actor 观测在 `open_duck_flat_tracking_env_cfg` 中按固定顺序拼接。

带状态估计的 `OpenDuck-Tracking`：

| 观测项 | 维度 | 含义 |
| --- | ---: | --- |
| `command` | 32 | reference effective joint position 和 velocity，`16 + 16`。 |
| `motion_anchor_pos_b` | 3 | 期望 trunk anchor 在机器人 anchor 坐标系下的位置。 |
| `base_lin_vel` | 3 | IMU 线速度。 |
| `base_ang_vel` | 3 | IMU 角速度。 |
| `base_lin_acc` | 3 | IMU 线加速度。 |
| `joint_pos` | 16 | effective non-backlash joint position relative to default。 |
| `joint_vel` | 16 | effective non-backlash joint velocity。 |
| `actions` | 42 | 当前和前两帧 clipped raw action，`14 * 3`。 |
| `feet_contact` | 2 | 左右 TPU 脚底是否接触。 |
| **总维度** | **120** | Actor observation width。 |

无状态估计的 `OpenDuck-Tracking-No-State-Estimation` 会移除：

| 移除项 | 维度 |
| --- | ---: |
| `motion_anchor_pos_b` | 3 |
| `base_lin_vel` | 3 |

因此 no-state actor 维度是：

```text
120 - 3 - 3 = 114
```

## Critic 观测

Critic 保留 privileged tracking 信息：

| 观测项 | 维度 | 含义 |
| --- | ---: | --- |
| `command` | 32 | reference effective joint position 和 velocity。 |
| `motion_anchor_pos_b` | 3 | 期望 trunk anchor 在机器人坐标系下的位置。 |
| `motion_anchor_ori_b` | 6 | anchor 姿态，用旋转矩阵前两列表示。 |
| `body_pos` | 24 | 8 个 tracked body 的位置，`8 * 3`。 |
| `body_ori` | 48 | 8 个 tracked body 的姿态，`8 * 6`。 |
| `base_lin_vel` | 3 | IMU 线速度。 |
| `base_ang_vel` | 3 | IMU 角速度。 |
| `base_lin_acc` | 3 | IMU 线加速度。 |
| `joint_pos` | 16 | effective non-backlash joint position relative to default。 |
| `joint_vel` | 16 | effective non-backlash joint velocity。 |
| `actions` | 42 | 三帧 clipped raw action history。 |
| `feet_contact` | 2 | 左右脚接触。 |
| **总维度** | **198** | Critic observation width。 |

Tracked bodies：

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

## 奖励函数

OpenDuck 任务从通用 motion tracking reward 继承，然后针对小尺寸机器人缩小部分 position tolerance。

| 奖励项 | 权重 | 参数 |
| --- | ---: | --- |
| `motion_global_root_pos` | `0.5` | OpenDuck 中 `std=0.08`。 |
| `motion_global_root_ori` | `0.5` | `std=0.4`。 |
| `motion_body_pos` | `1.0` | OpenDuck 中 `std=0.08`。 |
| `motion_body_ori` | `1.0` | `std=0.4`。 |
| `motion_body_lin_vel` | `1.0` | `std=1.0`。 |
| `motion_body_ang_vel` | `1.0` | `std=3.14`。 |
| `action_rate_l2` | `-0.1` | 惩罚 action 变化。 |
| `joint_limit` | `-10.0` | 排除 passive `*_backlash` joint。 |
| `self_collisions` | `-10.0` | 使用 `self_collision` contact sensor，阈值 `10.0`。 |

终止条件：

| 终止项 | OpenDuck 设置 |
| --- | --- |
| `anchor_pos` | z-only threshold `0.08`。 |
| `anchor_ori` | 共享 threshold `0.8`。 |
| `ee_body_pos` | 两个脚和头部的 z-only threshold `0.08`。 |

## 域随机化

实现位置：

```text
src/tasks/tracking/config/open_duck/randomization.py
```

训练默认 profile 是 `baseline`；播放/evaluation 中 `play=True` 时强制使用 `nominal`。也可以在训练时使用：

```bash
python scripts/train.py ... --randomization-profile=full
```

可用 profile：

| Profile | 用途 |
| --- | --- |
| `nominal` | 无随机化。播放和确定性评估使用。 |
| `baseline` | 复现 Sway_t2 训练风格：观测噪声、reset perturbation、push、encoder bias、脚底摩擦、trunk COM offset。 |
| `sensor` | 传感器噪声与 bias。 |
| `dynamics` | 摩擦、COM、质量/惯量、damping、frictionloss、armature。 |
| `actuator` | PD gain 和 effort limit 缩放。 |
| `latency` | action delay 和 observation delay。 |
| `light` | 中等强度组合随机化。 |
| `full` | 更强的组合随机化。 |

主要参数含义：

| 参数 | 含义 |
| --- | --- |
| `observation_corruption` | 是否启用 actor observation noise/corruption。 |
| `reset_perturbation` | 是否启用 reference-state initialization 的 pose、velocity、joint perturbation。 |
| `push_robot` | 每隔 `1.0` 到 `3.0` 秒给机器人施加速度扰动。 |
| `encoder_bias_rad` | 启动时 joint encoder bias 范围，单位 rad。 |
| `gyro_bias_rad_s` | 陀螺仪 additive bias 范围，单位 rad/s。 |
| `foot_friction` | 左右 TPU 脚底 geom 的绝对摩擦范围，并且左右脚共享随机值。 |
| `trunk_com_offset_m` | `trunk_assembly` COM offset 范围，单位 m。 |
| `body_mass_scale` | body mass/inertia 缩放，通过 pseudo-inertia randomization 实现。 |
| `joint_damping_scale` | joint damping 乘法缩放。 |
| `joint_friction_scale` | joint frictionloss 乘法缩放。 |
| `joint_armature_scale` | joint armature 乘法缩放。 |
| `kp_scale` | position actuator 比例增益缩放。 |
| `effort_scale` | actuator force range 缩放。 |
| `action_delay_control_steps` | action delay 的控制步数范围，会乘以 `cfg.decimation` 转成 physics lag。 |
| `observation_delay_control_steps` | `base_ang_vel`、`joint_pos`、`joint_vel` 的观测 delay 范围。 |

OpenDuck reset perturbation 的默认尺度：

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

## Motion 数据

Motion 文件位置：

```text
src/assets/motions/open_duck/
```

| 文件 | FPS | 帧数 | joint 列数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `A2_-_Sway_stageii_50hz.npz` | 50 | 600 | 16 | 旧 16-joint sway clip。 |
| `A2_-_Sway_t2_stageii.npz` | 50 | 798 | 16 | 旧 sway clip，无 `joint_names`。 |
| `A2_-_Sway_t2_stageii_realxml.npz` | 50 | 798 | 16 | 带 joint names，可按 real XML 对齐。 |
| `forward_headshake_40deg_04hz_50hz.npz` | 50 | 500 | 16 | 旧 forward/head-shake clip。 |
| `forward_headshake_40deg_04hz_50hz_realxml_backlash.npz` | 50 | 500 | 26 | real XML/backlash clip。 |
| `forward_headshake_real_50hz.npz` | 50 | 500 | 26 | real XML clip。 |
| `new_motion_realxml_backlash.npz` | 50 | 400 | 26 | 当前 `forward_train.sh` 默认 motion。 |

NPZ 必需字段：

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

Motion loader 同时支持：

1. 完整的 26-joint real XML/backlash motion。
2. 旧的 16-joint motion。

如果旧 motion 不包含 passive backlash 列，loader 会根据 `joint_names` 对齐主关节，并将 backlash 列填 0。

## 构造 Motion 数据

把 OpenDuck generator JSON 转成训练用 NPZ：

```bash
python scripts/duck_json_to_npz.py \
  --input input_recording.json \
  --output src/assets/motions/open_duck/my_motion_realxml_backlash.npz
```

批量转换目录：

```bash
python scripts/duck_json_to_npz.py \
  --input-dir path/to/json_dir \
  --output-dir src/assets/motions/open_duck
```

重采样已有 OpenDuck NPZ，并重新计算 body kinematics：

```bash
python scripts/resample_motion_npz.py \
  src/assets/motions/open_duck/input.npz \
  src/assets/motions/open_duck/output_50hz.npz \
  --output-fps 50
```

## PPO 配置

配置文件：

```text
src/tasks/tracking/config/open_duck/rl_cfg.py
```

默认值：

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

## 没有迁入的无用/一次性调试脚本和产物

旧工作区里有一些只服务于 bring-up、检查、视频录制、实验管理的脚本和产物。它们没有进入这个干净仓库：

| 旧路径 | 不迁入原因 |
| --- | --- |
| `checker/hardware_joint_checker.py` | 硬件 joint probing/checking 脚本，属于实机 bring-up 调试。 |
| `checker/mujoco_joint_checker.py` | MuJoCo joint probing 脚本，属于一次性模型检查。 |
| `checker/joint_checker_common.py` | 只服务 checker 脚本的共享 helper。 |
| `checker/plot_joint_logs.py` | 调试 joint log 的绘图脚本。 |
| `checker/README.md` | checker 工作流说明。 |
| `scripts/inspect_open_duck_mjcf.py` | MJCF 检查/报告生成脚本，运行时不需要。 |
| `scripts/evaluate_open_duck.py` | 绑定本地实验产物的 batch evaluation 脚本。 |
| `scripts/record_checkpoint.py` | 本地 checkpoint 录视频工具。 |
| `scripts/replay_open_duck_reference_motion.py` | 用 viewer 交互检查 motion 的工具，不属于核心训练路径。 |
| `watch_checkpoint.sh` | 本地自动监视 checkpoint/video 的脚本。 |
| `goal.md` | 旧工作区开发规划笔记。 |
| `doc/open_duck_mjcf_report.json` | 自动生成的 MJCF 检查报告。 |
| `doc/open_duck_sim2real_randomization.md` | 早期随机化设计文档，已被 README 和源码取代。 |
| `doc/openduck_training.md` | 早期训练说明，已被 README 取代。 |
| `models/Forward_headshake/`、`models/Sway_t1/`、`models/Sway_t2/` | 本地 checkpoint、ONNX、视频、TensorBoard event 和参数文件。 |
| `logs/`、`wandb/`、`MUJOCO_LOG.TXT`、`unitree_rl_mjlab.egg-info/`、`__pycache__/` | 运行、构建、缓存产物。 |

保留下来的可复用数据工作流是：

```text
scripts/duck_json_to_npz.py
scripts/resample_motion_npz.py
src/assets/motions/open_duck/*.npz
```

## 迁移提交顺序

当前分支从干净上游 clone 开始，按下面的顺序提交：

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
f9bc0c5 Update migration commit log
9014a2d Document non-migrated debug artifacts
```

中文文档提交会追加在这些迁移提交之后。
