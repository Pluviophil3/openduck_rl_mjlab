下面把“任务一：面向 Sim2Real 的训练准备”拆成 7 个阶段。核心原则是：先冻结基线，再测量硬件，再分组加入随机化，最后做组合压力测试。

## 总体目标

训练输出需要满足：

```text
策略在 nominal 仿真中表现不明显退化
+ 在合理参数偏差下稳定
+ 观测仅依赖实机可获得的数据
+ 控制频率、延迟、动作限制与实机一致
+ 所有随机化范围有测量依据或明确假设
```

建议第一条部署动作选择 `A2_-_Sway_t2_stageii`，比带行走和快速头部动作的策略更适合作为首次上机候选。

---

# Phase 0：冻结并量化当前基线

目标：回答“加入随机化后到底变好了还是变差了”。

### TODO

- [ ] 固定训练 motion、随机种子、环境数、PPO 参数

- [ ] 保存当前配置快照

- [ ] 明确模型输入输出：

  ```text
  actor observation: 87
  action: 14
  control frequency: 50 Hz
  ```

- [ ] 新增 nominal evaluation 配置，关闭所有随机化：

  - observation corruption
  - encoder bias
  - COM offset
  - friction randomization
  - push
  - RSI pose/velocity perturbation

- [ ] 记录以下指标：

  - episode completion rate
  - MPKPE
  - root-relative MPKPE
  - root orientation error
  - foot/head position error
  - joint velocity error
  - action rate
  - joint-limit penalty
  - termination reason
  - action 最大值和 P95/P99

- [ ] 至少使用 3 个 seed 评估

### 注意

当前 `play=True` 只关闭了 observation corruption、push 和 RSI；`base_com`、`foot_friction`、`encoder_bias` 仍可能执行，因此现在的 play 不等于严格 nominal evaluation。

### 验收

- 10 秒 episode 完成率建议 ≥95%
- 多个 seed 表现稳定
- 没有 NaN/Inf
- action 没有长期饱和
- 保存一份 baseline 报告和 baseline ONNX

### 输出

```text
eval_profiles/nominal.yaml
reports/baseline_<run>.json
plots/baseline_<run>.png
```

---

# Phase 1：OpenDuck 实机参数标定

目标：随机化范围尽量来自真实机器人，而不是拍脑袋。

## 1.1 静态参数

- [ ] 称量整机质量

- [ ] 估算 trunk、头部、电池等主要质量分布

- [ ] 检查实际脚底材料

- [ ] 保存所有关节 soft offset

- [ ] 检查 IMU 安装方向和静态姿态偏差

## 1.2 执行器测试

吊装机器人，逐关节低幅度测试：

- [ ] 阶跃目标位置

- [ ] 正弦扫频

- [ ] 慢速正反向运动

- [ ] 记录：

  ```text
  timestamp
  target_position
  measured_position
  measured_velocity
  motor voltage
  control loop duration
  IMU
  ```

- [ ] 估算：

  - 命令到响应延迟
  - 最大稳定速度
  - 等效 stiffness/damping
  - deadband
  - backlash/hysteresis
  - 正反向摩擦差异
  - 不同电池电压下的响应变化

## 1.3 传感器测试

- [ ] 静止 30–60 秒，测量 gyro bias/noise

- [ ] 测量 encoder 静态抖动

- [ ] 测量 IMU 与电机数据更新时间

- [ ] 测量控制循环 P50/P95/P99 周期

### 输出

建议形成：

```yaml
control_frequency: 50
action_delay_ms:
  p50: ...
  p95: ...
encoder:
  bias_rad: ...
  noise_std_rad: ...
imu:
  gyro_bias: [...]
  gyro_noise_std: [...]
actuator:
  velocity_limit: ...
  gain_scale_range: [...]
mass_scale_range: [...]
```

### 验收

- 每个随机化范围都有“实测值”或“保守假设”标签
- 训练 MJCF 的 `kp=6.55` 与实机 PID 的等效关系得到初步确认
- 明确实机真实延迟覆盖几个 20 ms policy step

---

# Phase 2：随机化基础设施

目标：让随机化可开关、可分组、可复现。

### 建议配置结构

```text
none       # 完全关闭，用于 nominal
sensor     # 传感器和标定误差
dynamics   # 质量、COM、摩擦
actuator   # gain、阻尼、速度限制
latency    # action/observation delay
full       # 全部启用
eval       # 固定压力测试参数
```

### TODO

- [ ] 不再把所有范围直接写死在 `env_cfgs.py`

- [ ] 为 OpenDuck 增加 randomization profile

- [ ] 每个 profile 支持独立启停

- [ ] 支持固定随机种子

- [ ] 输出每个环境实际采样到的参数

- [ ] 保证 critic 使用无噪声 privileged observation

- [ ] actor 只使用实机可获得的数据

- [ ] 为每项随机化写最小测试：

  - 参数确实发生变化
  - 变化范围正确
  - 不同环境可独立采样
  - nominal profile 完全不变化
  - play/eval 行为符合预期

### 建议文件

```text
src/tasks/tracking/config/open_duck/randomization_cfg.py
src/tasks/tracking/mdp/randomization.py
scripts/evaluate_robustness.py
```

---

# Phase 3：分组加入随机化

不要一次全部加入。每组先独立训练/微调，确认收益和副作用。

## 3A：传感器与标定误差

已有：

```text
joint position noise: ±0.01 rad
joint velocity noise: ±0.5 rad/s
gyro noise: ±0.2 rad/s
encoder bias: ±0.01 rad
```

TODO：

- [ ] 根据实测调整现有范围

- [ ] 添加 per-joint residual offset

- [ ] 添加 gyro 固定 bias，与逐帧 noise 区分

- [ ] 添加 IMU 安装姿态偏差

- [ ] 必要时添加 joint velocity 低通效果

验收：

- nominal tracking 退化不超过 5–10%
- sensor stress test completion rate 明显优于 baseline

## 3B：接触与刚体参数

建议初始范围：

```text
total/link mass scale: 0.90–1.10
COM offset: ±5 mm
foot friction: 先 0.5–1.0，再扩展到 0.3–1.2
joint damping scale: 0.8–1.2
joint friction scale: 0.8–1.2
armature scale: 0.8–1.2
```

TODO：

- [ ] 增加质量随机化

- [ ] 保留当前 trunk COM 随机化

- [ ] 增加主要 link 的 COM/质量变化

- [ ] 随机化 damping、frictionloss、armature

- [ ] 检查随机质量是否破坏模型物理合理性

## 3C：执行器随机化

这是 OpenDuck 最关键的一组。

TODO：

- [ ] actuator gain scale

- [ ] 最大输出/torque scale

- [ ] 电池电压统一 scale

- [ ] 最大关节速度限制

- [ ] target position rate limit

- [ ] action low-pass filter

建议起步：

```text
gain scale:        0.85–1.15
output scale:      0.85–1.05
velocity limit:    围绕实测值 ±10%
filter cutoff:     围绕实测范围采样
```

注意训练动作 scale 继续保持：

```text
legs: 0.13
head: 0.10
```

不能改成官方 runtime 默认的 `0.25`。

## 3D：延迟随机化

TODO：

- [ ] action FIFO delay

- [ ] observation delay

- [ ] IMU 与 encoder 不同延迟

- [ ] 可选控制周期 jitter

建议第一轮：

```text
action delay:       0–1 step
observation delay:  0–1 step
```

稳定后扩展为：

```text
action delay:       0–2 steps，即 0–40 ms
observation delay:  0–2 steps
```

验收：

- 0 step 时与原环境严格一致
- FIFO reset 时不会带入上一 episode 的动作
- delay 环境没有 observation/action 时间错位 bug

## 3E：扰动与初始化

已有 push 和 RSI。

TODO：

- [ ] 降低 push 频率，避免10秒内持续被推

- [ ] 区分训练扰动和评估扰动

- [ ] 保留当前小尺度 RSI

- [ ] 增加轻微初始关节 offset

- [ ] 评估 push 是否真的提升恢复能力

建议 push 后期再开启；它不是执行器建模的替代品。

---

# Phase 4：消融实验

目标：知道哪种随机化真正有用。

### 实验矩阵

| Run | Sensor | Dynamics | Actuator | Latency | Push |
|---|---:|---:|---:|---:|---:|
| B0 | × | × | × | × | × |
| B1 | ✓ | × | × | × | × |
| B2 | × | ✓ | × | × | × |
| B3 | × | × | ✓ | × | × |
| B4 | × | × | × | ✓ | × |
| B5 | ✓ | ✓ | ✓ | ✓ | × |
| B6 | ✓ | ✓ | ✓ | ✓ | ✓ |

### TODO

- [ ] 每组先从同一 baseline checkpoint 微调

- [ ] 每组使用相同 iteration 数

- [ ] 至少两个 seed；最终候选使用三个 seed

- [ ] 同时在 nominal 和对应 stress profile 上评估

- [ ] 删除只造成 nominal 退化、没有 stress 收益的随机化

### 判断标准

每组随机化至少满足一个条件：

- 对对应 stress case 有明显提升
- 对 nominal 退化低于 5–10%
- 没有明显增加 action 抖动和关节限位触发

---

# Phase 5：组合训练与随机化课程

建议分三档训练：

## Stage A：Light

```text
sensor noise
encoder bias
COM ±2 mm
mass ±5%
gain ±5%
delay 0–1 step
```

## Stage B：Medium

```text
COM ±5 mm
mass ±10%
gain ±10%
damping/friction ±15%
delay 0–2 steps
轻微 push
```

## Stage C：Full

使用实测区间加安全 margin。

### TODO

- [ ] 从 baseline checkpoint 开始 Light

- [ ] Light 收敛后继续 Medium

- [ ] Medium 收敛后继续 Full

- [ ] 始终保留 10–20% nominal environments

- [ ] 每 500 iteration 做固定 evaluation suite

- [ ] 根据评估选择最佳 checkpoint，不默认选择最后一个

- [ ] 最终再从 scratch 训练一次，确认不是微调偶然性

---

# Phase 6：鲁棒性验收

至少建立以下测试场景：

```text
nominal
low/high friction
low/high mass
COM extreme
weak actuator
slow actuator
20 ms latency
40 ms latency
sensor bias
combined random
push recovery
```

### 推荐门槛

- nominal completion rate ≥95%
- nominal 指标相对 baseline 退化 ≤10%
- 单项边界测试 completion rate ≥90%
- combined Monte Carlo completion rate ≥85–90%
- 没有 action 长期饱和
- joint-limit termination 接近零
- 40 ms latency 下不发生立即失稳
- 3 个 seed 都达到门槛

Monte Carlo 建议至少：

```text
1000 episodes
3 policy seeds
固定一套 evaluation seeds
```

---

# Phase 7：冻结 Sim2Real 候选

### TODO

- [ ] 选择最佳 checkpoint

- [ ] 导出 `policy.onnx`

- [ ] 验证 PyTorch 与 ONNX 输出一致

- [ ] 保存 motion NPZ

- [ ] 保存 observation schema

- [ ] 保存 joint/action 顺序

- [ ] 保存 action scale

- [ ] 保存 randomization ranges

- [ ] 保存训练 Git commit 和完整命令

- [ ] 生成 deployment manifest

建议产物：

```text
artifacts/open_duck/sway_v1/
├── policy.onnx
├── motion.npz
├── training_config.yaml
├── randomization.yaml
├── hardware_profile.yaml
├── observation_schema.json
├── joint_mapping.json
├── evaluation_report.json
└── manifest.json
```

完成标准是：这个目录足以让任务二的独立 Sim2Sim runtime 重建策略输入，不再依赖“训练时脑子里记得怎么配的”。

优先执行顺序是：

```text
Phase 0 nominal评估
→ Phase 1硬件测量
→ Phase 2随机化框架
→ Phase 3C执行器
→ Phase 3D延迟
→ 其余随机化
→ 消融
→ 组合训练
→ 冻结候选
```