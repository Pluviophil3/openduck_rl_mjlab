# OpenDuck 参考轨迹训练

本文档记录 OpenDuck Mini V2 在本仓库中的资产、动作数据、训练参数和完整训练流程。该任务使用逐帧全身参考轨迹跟踪和 PPO，不包含 AMP 判别器或 AMP reward。

## 1. 环境与依赖

推荐使用仓库安装时锁定的版本：

| 依赖 | 版本 |
|---|---:|
| Python | 3.11 |
| mjlab | 1.2.0 |
| MuJoCo | 3.5.0 |
| MuJoCo Warp | 3.5.0 |
| Warp | 1.12.0 |

```bash
conda activate unitree
cd /home/luolinfeng/duck_ws/duck_amp/unitree_rl_mjlab
pip install -e .
```

验证 GPU 和任务注册：

```bash
python -c \
  "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/list_envs.py --keyword OpenDuck
```

应当能看到：

```text
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
```

推荐训练 `OpenDuck-Tracking-No-State-Estimation`，因为它不依赖实机难以可靠获得的全局位置和基座线速度。

## 2. Asset 与参考动作

机器人模型：

```text
src/assets/robots/open_duck_mini_v2/
```

模型规模：

| 项目 | 数量 |
|---|---:|
| 非 world body | 22 |
| revolute joint | 16 |
| MJCF position actuator | 16 |
| 策略控制 actuator | 14 |
| sensor | 14 |

策略控制双腿 10 个关节以及颈部/头部 4 个关节。左右 antenna 保留在机器人状态和参考轨迹中，但不属于策略 action。

当前参考动作：

```text
src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz
```

| 项目 | 值 |
|---|---:|
| 原始帧数 | 360 |
| 原始 FPS | 约 29.9376 |
| 输出帧数 | 600 |
| 输出 FPS | 50 |
| 输出时长 | 11.98 s |
| joint_pos/joint_vel | `[600, 16]` |
| body 状态 | `[600, 22, ...]` |

重新生成动作：

```bash
python scripts/resample_motion_npz.py \
  ../AMP_mjlab/src/assets/motions/open_duck/amp/WalkandRun/A2_-_Sway_stageii.npz \
  src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz \
  --output-fps 50
```

脚本会插值 root pose 和 joint position，重新计算速度，并通过当前 OpenDuck MJCF 重新执行全身正向运动学。不要只修改 NPZ 的 `fps` 字段。

## 3. 仿真与控制参数

| 参数 | 值 |
|---|---:|
| MuJoCo timestep | 0.005 s |
| control decimation | 4 |
| policy/control period | 0.02 s |
| policy/control frequency | 50 Hz |
| episode length | 10 s |
| terrain | plane |
| `nconmax` | 48 |
| `njmax` | 640 |
| `contact_sensor_maxmatch` | 128 |
| CCD iterations | 50 |
| solver iterations | 10 |
| line-search iterations | 20 |

动作采用关节位置控制：

```text
q_target = q_default + action_scale * policy_action
```

Action 共 14 维，scale 为：

| 关节组 | scale |
|---|---:|
| hip yaw/roll/pitch | 0.13 |
| knee/ankle | 0.13 |
| neck pitch | 0.10 |
| head pitch/yaw/roll | 0.10 |

## 4. Reference command

Anchor body：

```text
trunk_assembly
```

跟踪 body：

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

训练默认使用 adaptive reference-state sampling。失败较多的动作区间会获得更高的 reset 采样概率。

| 参数 | 值 |
|---|---:|
| sampling mode | adaptive |
| adaptive kernel size | 1 |
| adaptive lambda | 0.8 |
| uniform sampling ratio | 0.1 |
| adaptive alpha | 0.001 |
| joint reset noise | `[-0.05, 0.05]` rad |
| root x/y reset noise | `[-0.02, 0.02]` m |
| root z reset noise | `[-0.005, 0.005]` m |
| root roll/pitch noise | `[-0.05, 0.05]` rad |
| root yaw noise | `[-0.1, 0.1]` rad |
| root linear velocity x/y | `[-0.2, 0.2]` m/s |
| root linear velocity z | `[-0.1, 0.1]` m/s |
| root angular velocity roll/pitch | `[-0.25, 0.25]` rad/s |
| root angular velocity yaw | `[-0.4, 0.4]` rad/s |

Play 模式固定从第 0 帧开始，并关闭 RSI 扰动、push 和 observation corruption。

## 5. Observation

### 5.1 No-State-Estimation actor

推荐任务的 actor observation 共 87 维：

| Term | 维度 | 内容 |
|---|---:|---|
| command | 32 | reference joint position + velocity |
| motion anchor orientation | 6 | anchor 相对朝向的 6D rotation |
| base angular velocity | 3 | IMU gyro |
| joint position | 16 | 相对默认关节位置 |
| joint velocity | 16 | 当前关节速度 |
| last action | 14 | 上一步策略动作 |

Actor 启用 observation corruption：

| Term | 均匀噪声 |
|---|---:|
| anchor orientation | `[-0.05, 0.05]` |
| base angular velocity | `[-0.2, 0.2]` |
| joint position | `[-0.01, 0.01]` |
| joint velocity | `[-0.5, 0.5]` |

完整版 `OpenDuck-Tracking` 还增加 3 维 anchor position 和 3 维 base linear velocity。

### 5.2 Critic

Critic observation 共 165 维，并使用以下 privileged information：

- anchor 相对位置和朝向
- 8 个被跟踪 body 的位置与朝向
- base linear/angular velocity
- 完整 joint position/velocity
- reference command 和 last action

Actor 和 critic 都使用独立 empirical observation normalization。

## 6. Reward 与终止条件

跟踪奖励形式为：

```text
exp(-error² / std²)
```

| Reward | 权重 | std |
|---|---:|---:|
| anchor global position | 0.5 | 0.08 m |
| anchor global orientation | 0.5 | 0.4 rad |
| relative body position | 1.0 | 0.08 m |
| relative body orientation | 1.0 | 0.4 rad |
| body linear velocity | 1.0 | 1.0 m/s |
| body angular velocity | 1.0 | 3.14 rad/s |
| action rate L2 | -0.1 | — |
| joint limit | -10.0 | — |
| self collision | -10.0 | force threshold 10 N |

终止条件：

| 条件 | 参数 |
|---|---:|
| timeout | 10 s |
| anchor z error | 0.08 m |
| anchor orientation error | 0.8 |
| feet/head z error | 0.08 m |

`ee_body_pos` 检查 `foot_assembly`、`foot_assembly_2` 和 `head_assembly`。

## 7. Domain randomization

| Randomization | 参数 |
|---|---|
| push interval | 1–3 s |
| push linear x/y | `[-0.2, 0.2]` m/s |
| push linear z | `[-0.1, 0.1]` m/s |
| push angular roll/pitch | `[-0.25, 0.25]` rad/s |
| push angular yaw | `[-0.4, 0.4]` rad/s |
| trunk COM offset | 每轴 `[-0.005, 0.005]` m |
| encoder bias | `[-0.01, 0.01]` rad |
| foot friction | `[0.3, 1.2]`，双脚共享采样值 |

## 8. PPO 参数

Actor 与 critic 均为：

```text
MLP: 512 → 256 → 128
activation: ELU
```

| 参数 | 值 |
|---|---:|
| seed | 42 |
| rollout steps/environment | 24 |
| max iterations | 30001 |
| save interval | 500 |
| observation groups | actor=`actor`，critic=`critic` |
| action clipping | none |
| action distribution | Gaussian |
| initial action std | 1.0，scalar |
| PPO epochs | 5 |
| mini-batches | 4 |
| optimizer | Adam |
| learning rate | 1e-3 |
| LR schedule | adaptive |
| clip parameter | 0.2 |
| value loss coefficient | 1.0 |
| clipped value loss | true |
| entropy coefficient | 0.005 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| desired KL | 0.01 |
| maximum gradient norm | 1.0 |
| advantage normalization | rollout-wide |
| multi-GPU | disabled by default |
| resume | false by default |
| load run pattern | `.*` |
| load checkpoint pattern | `model_.*.pt` |

日志参数：

| 参数 | 值 |
|---|---|
| logger | W&B |
| project | `openduck_rl_mjlab` |
| tags | `openduck`, `motion-tracking`, `mjlab` |
| local log root | `logs/rsl_rl/open_duck_tracking` |
| upload model | true |

## 9. 完整训练流程

### 9.1 W&B 登录

```bash
wandb login
wandb status
```

W&B 会在第一次训练时自动创建 `openduck_rl_mjlab` project。

### 9.2 Smoke test

首次修改 asset、动作或 reward 后先运行：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz \
  --env.scene.num-envs=16 \
  --agent.max-iterations=2 \
  --agent.save-interval=1 \
  --agent.run-name=migration-smoke
```

已验证该命令可以完成 GPU 环境创建、rollout、PPO update、checkpoint 保存、ONNX 导出和 W&B 同步。

### 9.3 正式训练

RTX 3090 可以从 4096 个环境开始；若显存不足则降至 2048 或 1024：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz \
  --env.scene.num-envs=4096 \
  --agent.run-name=sway-stageii-v1
```

保存目录：

```text
logs/rsl_rl/open_duck_tracking/<timestamp>_<run-name>/
├── model_<iteration>.pt
├── policy.onnx
├── <run-name>.onnx
└── params/
    ├── agent.yaml
    └── env.yaml
```

### 9.4 续训

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz \
  --env.scene.num-envs=4096 \
  --agent.resume \
  --agent.load-run='.*sway-stageii-v1' \
  --agent.load-checkpoint='model_.*.pt'
```

### 9.5 仿真回放

```bash
python scripts/play.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_stageii_50hz.npz \
  --checkpoint-file=logs/rsl_rl/open_duck_tracking/<run>/model_<iteration>.pt
```

## 10. 训练监控

优先观察：

- `Metrics/motion/error_body_pos`
- `Metrics/motion/error_body_rot`
- `Metrics/motion/error_joint_pos`
- `Episode_Termination/ee_body_pos`
- `Episode_Termination/anchor_pos`
- `Mean episode length`
- `Mean action std`

健康趋势是 body/joint error 下降、episode length 上升、termination rate 下降。若训练早期几乎所有 episode 都由 `ee_body_pos` 终止，可先将阈值从 0.08 m 放宽，再随着策略稳定逐步收紧。

所有运行最终采用的参数都会保存到该次 run 的 `params/env.yaml` 和 `params/agent.yaml`；排查结果时应以这两个快照为准。
