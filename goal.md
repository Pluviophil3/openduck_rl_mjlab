Phase 1：迁移 OpenDuck asset
TODO：

复制 open_duck_mini_v2/xmls

复制所有 mesh、texture

复制并重命名 open_duck_amp_constants.py

注册 get_open_duck_robot_cfg

保持原始 joint/body/actuator 顺序

检查默认 root 高度 0.22 m

检查 IMU sensor 名称为 imu_ang_vel、imu_lin_vel

检查 feet geom 和 site

检查 16 joints / 22 bodies / 14 policy actuators
验收：
Scene 可以 compile
reset 后机器人站在地面上
zero action 不出现 NaN
body、joint 名称全部能被 find_bodies/find_joints 找到
Phase 2：建立动作数据转换和验证工具
TODO：

为 OpenDuck 增加 50 Hz 转换器

使用迁移后的 XML 重新做 FK

输出 unitree tracking 所需六个数组

验证四元数顺序为 wxyz

验证 joint 顺序与 Entity 完全一致

验证 body 顺序与 MJCF 完全一致

增加 joint_names、body_names metadata

增加 NPZ schema validator

添加 reference replay/ghost visualization
NPZ 至少包含：
joint_pos          [T, 16]
joint_vel          [T, 16]
body_pos_w         [T, 22, 3]
body_quat_w        [T, 22, 4]
body_lin_vel_w     [T, 22, 3]
body_ang_vel_w     [T, 22, 3]
fps                50
验收：
reference 单独播放速度正确
第一帧和最后一帧没有速度尖峰
FK 重算后的双脚位置合理
reference 初始化后，robot/reference body error 接近零
Phase 3：新增 OpenDuck tracking task
TODO：

创建 tracking/config/open_duck/env_cfgs.py

配置 robot entity

配置 trunk_assembly anchor

配置 8 个跟踪 body

配置双脚/head termination

配置 self-collision 和 body-ground contact

配置 foot friction

配置 COM randomization

配置 OpenDuck action scale

创建普通和 No-State-Estimation 两个环境

创建 PPO 配置

在 __init__.py 注册任务
任务注册结构参考 [G1 registration (line 7)](/home/luolinfeng/duck_ws/duck_amp/unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py:7)。
验收：
python scripts/list_envs.py --keyword OpenDuck
能够看到：
OpenDuck-Tracking
OpenDuck-Tracking-No-State-Estimation
Phase 4：小规模 smoke training
先不要直接 4096 环境长训。
TODO：

1 环境运行 zero/random policy

16 环境运行 10 次 PPO iteration

256 环境运行 100～500 iteration

检查 observation/action shape

检查 reward 各分量

检查 termination 原因分布

检查自适应 frame sampling

检查 ONNX 导出
建议命令形态：
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/motion_50hz.npz \
  --env.scene.num-envs=256 \
  --agent.max-iterations=500
验收：
无 NaN/Inf
episode length 持续上升
body position/orientation error 下降
不集中在同一个 reference frame 失败
policy 输出没有长期饱和到 ±1
Phase 5：奖励和随机化调参
OpenDuck 尺寸远小于 G1，不能直接照搬全部 tracking 参数。
优先调整：

anchor position std 从 G1 的 0.3 降到约 0.05～0.12

body position std 降到约 0.04～0.10

anchor 高度终止阈值从 0.25 降到约 0.05～0.10

ee body 高度阈值缩小

root pose/velocity RSI 扰动按机器人尺寸缩小

push velocity 减小

COM randomization 从厘米级进一步缩小

action-rate 和 self-collision 权重重新标定
第一轮建议先关闭或减弱：
push_robot
大范围 COM randomization
过强 encoder noise
严格 end-effector termination
先证明动作能学会，再逐步加回鲁棒性训练。
Phase 6：完整新机器人支持
如果“作为一个新机器人构型加入”不只指 tracking，而是想做到仓库的一等公民，还需要：

OpenDuck velocity task

flat/rough terrain 配置

OpenDuck motion converter 参数

reference replay 工具

文档与训练命令

sample NPZ

ONNX metadata

部署 observation builder

实机 joint ID 映射

实机 action scale、offset、KP/KD

50 Hz 实时 reference player

FSM：Passive/FixStand/Mimic

bad orientation/contact safety

sim-to-sim 验证

实机限位和急停
八、推荐的最小可行版本
第一版只完成以下内容：
复制 OpenDuck XML、mesh、EntityCfg
把一条 OpenDuck 动作转成 50 Hz NPZ
新增 OpenDuck-Tracking-No-State-Estimation
使用 trunk_assembly anchor 和现有 8 个 AMP body
保留 14 维 action 配置
256 环境完成 smoke training
确认策略能稳定复现参考动作
这条路径不需要迁移 AMP discriminator、定制 RSL-RL 或 AMP replay buffer，工作量会小很多，也更容易定位问题。