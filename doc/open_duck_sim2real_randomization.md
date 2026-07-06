# OpenDuck Mini Sim2Real：基线、模型核对与随机化训练

本文档对应 `models/Sway_t2`，覆盖 nominal 基线记录、MJCF 与实机核对项、随机化 profile、训练顺序和消融实验记录。

## 1. Sway_t2 nominal 基线

基线固定为：

| 项目 | 值 |
|---|---|
| checkpoint | `models/Sway_t2/model_2500.pt` |
| motion | `src/assets/motions/open_duck/A2_-_Sway_t2_stageii.npz` |
| task | `OpenDuck-Tracking-No-State-Estimation` |
| profile | `nominal` |
| motion 帧数 | 798 |
| motion FPS | 50 Hz |
| policy/control FPS | 50 Hz |
| actor observation | 87 |
| policy action | 14 |
| seed | 42 |

重新生成基线：

```bash
conda run -n unitree python scripts/evaluate_open_duck.py
```

指定其他 checkpoint：

```bash
conda run -n unitree python scripts/evaluate_open_duck.py \
  --checkpoint-file=models/Sway_t2/model_2000.pt \
  --output-dir=models/Sway_t2/nominal_model_2000
```

输出：

```text
models/Sway_t2/nominal/
├── summary.json
└── trace.npz
```

`trace.npz` 保存第一台环境的 observation、action、关节状态、关节目标、reference 和 reward；`summary.json` 保存输入文件哈希、profile、误差统计、动作统计和 termination。

### 1.1 当前结果

当前完整 798 帧运行的主要结果：

| 指标 | Mean | P95 | Max |
|---|---:|---:|---:|
| anchor position error | 0.0252 m | 0.0294 m | 0.0305 m |
| anchor rotation error | 0.1083 rad | 0.1475 rad | 0.2849 rad |
| body position error | 0.0279 m | 0.0311 m | 0.2194 m |
| body rotation error | 0.0868 rad | 0.1059 rad | 2.2045 rad |
| root-relative MPKPE | 0.0162 m | 0.0198 m | 0.0223 m |
| joint position L2 error | 0.5452 rad | 0.6469 rad | 0.7804 rad |
| joint velocity L2 error | 0.5194 rad/s | 1.0160 rad/s | 6.1384 rad/s |

需要特别保留的基线事实：

- 首步触发一次 `ee_body_pos` termination，之后重新 reset 并完成剩余轨迹。
- 88.35% 的 raw policy action 绝对值大于 1。
- raw action 最大绝对值为 9.42。
- `clip_actions` 当前为 `null`。
- MuJoCo 最终依靠 position actuator 的 control range 限制目标；实机 runtime 不能直接下发未限幅目标。

因此，当前模型可以作为训练比较 baseline，但不能原样作为实机 release candidate。随机化训练之外，还必须完成 action/joint/rate safety layer。

## 2. MJCF 物理参数

生成完整 JSON：

```bash
conda run -n unitree python scripts/inspect_open_duck_mjcf.py
```

输出为 `doc/open_duck_mjcf_report.json`。

### 2.1 重量

编译后的 MJCF 总质量：

```text
2.061549284 kg
```

| 分组 | 质量 | 占比 |
|---|---:|---:|
| trunk | 0.698526 kg | 33.88% |
| head + neck + antennas | 0.535943 kg | 26.00% |
| left leg | 0.413540 kg | 20.06% |
| right leg | 0.413540 kg | 20.06% |

这只是 MJCF inertial 的总和，不代表实物已经校准。尤其需要确认电池、Raspberry Pi、控制板、线束、外壳版本和头部附件是否与模型一致。

### 2.2 当前统一关节模型

16 个 MJCF 关节全部使用：

```text
damping     = 0.65
frictionloss = 0.083
armature    = 0.027
```

16 个 XML position actuator 全部使用：

```text
kp          = 6.55
kv          = 0
force range = [-3.57, 3.57]
```

策略只控制14个关节；两根 antenna 在 observation 中存在，但不属于 policy action。

### 2.3 足底接触

```text
sliding friction = 0.8
torsional         = 0.02
rolling           = 0.01
solref            = [0.02, 1.0]
solimp            = [0.015, 1.0, 0.031, 0.5, 2.0]
```

训练环境覆盖 MJCF 内的 `timestep=0.002 s`，实际采用：

```text
physics timestep = 0.005 s
decimation       = 4
control timestep = 0.020 s
```

## 3. 必须与实机核对的参数

下面表格需要通过称重、规格书或低幅度吊装实验填写。未确认前，profile 中的范围只是保守起点。

### 3.1 刚体和接触

| 参数 | MJCF | 实机值 | 测量方法 | 状态 |
|---|---:|---:|---|---|
| 整机质量 | 2.06155 kg |  | 电子秤 | TODO |
| trunk 质量/COM | 0.69853 kg |  | 分部称重/悬挂法 | TODO |
| head+neck 质量 | 0.52751 kg（不含天线） |  | 分部称重 | TODO |
| 左腿质量 | 0.41354 kg |  | 分部称重 | TODO |
| 右腿质量 | 0.41354 kg |  | 分部称重 | TODO |
| TPU-地面摩擦 | 0.8 |  | 斜面临界角 | TODO |
| 脚底接触刚度/阻尼 | MuJoCo solref/solimp |  | 落脚/压缩实验 | TODO |

### 3.2 关节和传动

| 参数 | MJCF | 实机值 | 测量方法 | 状态 |
|---|---:|---:|---|---|
| damping | 所有关节 0.65 |  | 断电自由衰减 | TODO |
| dry friction | 所有关节 0.083 |  | 低速正反向扫动 | TODO |
| armature | 所有关节 0.027 |  | 电机/减速比规格或辨识 | TODO |
| backlash | 未建模 |  | 正反向小步进 | TODO |
| deadband | 未建模 |  | 小幅命令扫描 | TODO |
| static zero offset | 仿真为0 |  | `find_soft_offsets.py` | TODO |
| encoder noise | 训练±0.01 rad |  | 静止日志 | TODO |

### 3.3 电机和控制

| 参数 | MJCF/runtime 假设 | 实机值 | 测量方法 | 状态 |
|---|---:|---:|---|---|
| position kp | MJCF 6.55 |  | 低幅阶跃响应 | TODO |
| head kp | MJCF 6.55；参考 runtime 常用8 |  | 阶跃响应 | TODO |
| leg kp | MJCF 6.55；参考 runtime 常用30 |  | 阶跃响应 | TODO |
| kv/kd | MJCF 0 |  | 阶跃/自由衰减 | TODO |
| effort limit | ±3.57 |  | 电机规格/电流限制 | TODO |
| max velocity | MJCF 未限制；runtime 参考5.24 rad/s |  | 规格/空载测试 | TODO |
| action latency | 未标定 |  | command/encoder 时间戳 | TODO |
| encoder latency | 未标定 |  | 时间戳对齐 | TODO |
| IMU latency | 未标定 |  | 时间戳对齐 | TODO |
| control-loop P95/P99 | 20 ms目标 |  | runtime profiling | TODO |
| battery gain drop | 未建模 |  | 满电/低电压阶跃 | TODO |

### 3.4 当前模型的主要限制

- 所有腿、颈部和头部关节共享同一套动力学参数。
- 没有显式 servo 内环、速度饱和、电流限制、热降额和电池模型。
- 没有 backlash、gear compliance、deadband 和结构柔性。
- 没有标定真实 IMU 安装旋转、固定 bias 和温漂。
- 接触参数只针对当前 MuJoCo mesh/solver，不保证等价于真实 TPU。
- 质量和惯量来自当前资产，部分电子元件 mesh 被注释，不代表真实装配。
- 训练使用50 Hz policy，但 actuator delay 在 physics step 上实现，profile 已自动乘以 decimation。

## 4. 随机化基础设施

实现位置：

```text
src/tasks/tracking/config/open_duck/randomization.py
```

训练命令新增：

```text
--randomization-profile
```

可用 profile：

| Profile | 用途 |
|---|---|
| `nominal` | 关闭全部随机化，做基准评估 |
| `baseline` | 精确复现 Sway_t2 原训练随机化 |
| `sensor` | observation noise、encoder bias、gyro固定bias |
| `dynamics` | mass/inertia、COM、摩擦、damping、friction、armature |
| `actuator` | position gain、effort limit |
| `latency` | action delay、gyro/encoder observation delay |
| `light` | 第一阶段组合随机化 |
| `full` | 完整范围与push |

### 4.1 范围

| 参数 | Baseline | Light | Full |
|---|---:|---:|---:|
| encoder bias | ±0.01 rad | ±0.01 rad | ±0.01 rad |
| gyro fixed bias | — | ±0.03 rad/s | ±0.03 rad/s |
| foot friction | 0.3–1.2 | 0.5–1.0 | 0.3–1.2 |
| trunk COM | ±5 mm | ±3 mm | ±5 mm |
| body mass + inertia | — | 0.95–1.05× | 0.90–1.10× |
| damping/friction/armature | — | 0.90–1.10× | 0.80–1.20× |
| kp | — | 0.90–1.10× | 0.85–1.15× |
| effort limit | — | 0.90–1.00× | 0.85–1.05× |
| action delay | — | 0–1 control step | 0–2 control steps |
| observation delay | — | 0–1 control step | 0–2 control steps |
| push | 开启 | 关闭 | 开启 |

质量随机化通过 pseudo-inertia 同时修改 mass 和 inertia，避免只改变质量造成物理不一致。

Action delay 包装 position actuator。配置单位是 control step，内部转换为 physics step：

```text
1 control step = decimation × physics step = 4 × 5 ms = 20 ms
```

Observation delay 应用于：

```text
base_ang_vel
joint_pos
joint_vel
```

reference command、reference orientation 和 last action 不延迟。

### 4.2 使用方式

复现原始 Sway_t2 配置：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_t2_stageii.npz \
  --env.scene.num-envs=4096 \
  --randomization-profile=baseline \
  --agent.run-name=sway-t2-baseline-reproduce
```

Light：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_t2_stageii.npz \
  --checkpoint-file=models/Sway_t2/model_2500.pt \
  --env.scene.num-envs=4096 \
  --randomization-profile=light \
  --agent.run-name=sway-t2-light
```

Full：

```bash
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/A2_-_Sway_t2_stageii.npz \
  --env.scene.num-envs=4096 \
  --randomization-profile=full \
  --agent.run-name=sway-t2-full
```

单因素微调同样使用 `--checkpoint-file=models/Sway_t2/model_2500.pt`，只需将 profile 换成 `sensor`、`dynamics`、`actuator` 或 `latency`。显式 checkpoint 会同时恢复网络、normalizer、optimizer和迭代状态；新结果仍写入新的 run 目录。

评估任意 profile：

```bash
conda run -n unitree python scripts/evaluate_open_duck.py \
  --checkpoint-file=path/to/model_XXXX.pt \
  --profile=latency \
  --num-envs=64 \
  --output-dir=reports/model_XXXX_latency
```

## 5. 必做训练

先做相同训练预算的单因素消融，再做组合训练：

| ID | Profile | 目的 | 建议初始化 |
|---|---|---|---|
| T0 | baseline | 复现实验、检查代码迁移 | scratch |
| T1 | sensor | 传感器鲁棒性 | 同一baseline checkpoint |
| T2 | dynamics | 质量/接触/关节动力学 | 同一baseline checkpoint |
| T3 | actuator | gain和输出能力 | 同一baseline checkpoint |
| T4 | latency | 0–40 ms延迟 | 同一baseline checkpoint |
| T5 | light | 第一阶段组合 | baseline或scratch |
| T6 | full | 完整组合 | T5最佳checkpoint |
| T7 | full | 排除微调偶然性 | scratch |

在实机参数尚未测量前，先完成 T0–T5。完成硬件标定后修改 profile 范围，再运行 T6/T7。

每个候选 checkpoint 至少评估：

```text
nominal
sensor
dynamics
actuator
latency
full
```

## 6. 消融实验记录

### 6.1 训练记录

| Run ID | Profile | Init checkpoint | Seed | Iterations | Best checkpoint | Notes |
|---|---|---|---:|---:|---|---|
| T0 | baseline | scratch |  |  |  |  |
| T1 | sensor |  |  |  |  |  |
| T2 | dynamics |  |  |  |  |  |
| T3 | actuator |  |  |  |  |  |
| T4 | latency |  |  |  |  |  |
| T5 | light |  |  |  |  |  |
| T6 | full |  |  |  |  |  |
| T7 | full | scratch |  |  |  |  |

### 6.2 Nominal 结果

| Run ID | Completion | Anchor pos | Anchor rot | R-MPKPE | Joint pos | Action P99 | Terminations |
|---|---:|---:|---:|---:|---:|---:|---|
| Sway_t2 baseline |  |  |  |  |  |  |  |
| T0 |  |  |  |  |  |  |  |
| T1 |  |  |  |  |  |  |  |
| T2 |  |  |  |  |  |  |  |
| T3 |  |  |  |  |  |  |  |
| T4 |  |  |  |  |  |  |  |
| T5 |  |  |  |  |  |  |  |
| T6 |  |  |  |  |  |  |  |
| T7 |  |  |  |  |  |  |  |

### 6.3 Stress profile 结果

填写 completion rate；每格使用相同 evaluation seeds 和 episode 数。

| Policy | Nominal | Sensor | Dynamics | Actuator | Latency | Full |
|---|---:|---:|---:|---:|---:|---:|
| Sway_t2 baseline |  |  |  |  |  |  |
| T0 |  |  |  |  |  |  |
| T1 |  |  |  |  |  |  |
| T2 |  |  |  |  |  |  |
| T3 |  |  |  |  |  |  |
| T4 |  |  |  |  |  |  |
| T5 |  |  |  |  |  |  |
| T6 |  |  |  |  |  |  |
| T7 |  |  |  |  |  |  |

### 6.4 进入 Sim2Sim 的门禁

- nominal 不允许首步 termination。
- nominal tracking 指标相比当前稳定段退化不超过10%。
- action 必须经过明确的 joint/rate clipping 验证。
- single-factor stress completion ≥90%。
- full Monte Carlo completion ≥85%。
- 至少3个训练 seed 中有2个通过。
- ONNX 与 PyTorch action parity 通过。
- observation、joint、action 顺序写入 deployment manifest。
