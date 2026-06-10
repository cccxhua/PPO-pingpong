# Sim2Sim & 部署代码说明

## 目录结构

```
unitree_rl_lab/
├── unitree_rl_lab/
│   ├── scripts/sim2sim/           ← Sim2Sim (MuJoCo 验证)
│   │   ├── a1_play_mujoco.py     ← A1 乒乓球 MuJoCo 回放主脚本
│   │   ├── x1_play_mujoco.py     ← X1 乒乓球
│   │   └── play_mujoco.py        ← 通用 MuJoCo 回放
│   ├── logs/rsl_rl/a1_tabletennis/
│   │   ├── 2026-06-08_08-04-26/  ← 训练输出 (model, obs, videos，含导出的策略)
│   │   └── sim2sim/              ← Sim2Sim 验证记录 (~76 runs)
│   └── source/.../tasks/table_tennis/  ← Isaac Lab 训练环境定义
│       └── robots/a1/forehand/
│           ├── env_cfg.py        ← 环境配置 (obs/act/reward/DR)
│           └── forehand_middle_a1_whip.npz  ← 参考动作文件
│
└── worobots/a1_robot_ppo/         ← 真机部署代码 (ROS2)
    ├── a1_table_tennis_deploy.py  ← 核心部署库 (策略/观测/phase)
    ├── ppo_inference_pingpong.py  ← ROS2 推理节点 (50Hz)
    ├── play_action_pingpong_ppo.py ← 动作播放节点 (→ /record_data)
    ├── record_state_pingpong_ppo.py ← 关节状态转发节点
    └── logs/                      ← 部署日志 (CSV)
```

---

## 一、Sim2Sim (MuJoCo 验证)

### 用途

在 MuJoCo 物理引擎中回放 Isaac Lab 训练得到的策略，验证 sim-to-sim 迁移效果。MuJoCo 与 Isaac Lab 使用不同物理引擎，通过对比两者的关节轨迹来定位仿真器差异。

### 运行命令

```bash
cd unitree_rl_lab/scripts/sim2sim

# 可视化回放 (带 GUI)，如果觉得太快可以继续调整时间
python a1_play_mujoco.py \
    --policy ../../logs/rsl_rl/a1_tabletennis/exported/policy.pt \
    --real-time

# 固定发球 (调试单次挥拍)
python a1_play_mujoco.py \
    --policy ../../logs/rsl_rl/a1_tabletennis/exported/policy.pt \
    --fixed-serve --real-time

# 安全发球 (中心区域较慢球)
python a1_play_mujoco.py \
    --policy ../../logs/rsl_rl/a1_tabletennis/exported/policy.pt \
    --safe-serve

# headless模式 (仅记录数据)
python a1_play_mujoco.py \
    --policy ../../logs/rsl_rl/a1_tabletennis/exported/policy.pt \
    --headless --episodes 20
```

### 输出

运行后自动保存到 `logs/rsl_rl/a1_tabletennis/sim2sim/<timestamp>/`：
- `sim2sim_joint_pos.npz` — 关节位置时序数据
- `sim2sim_joint_pos.png` — 关节位置对比图
- `sim2sim_joint_vel.png` — 关节速度对比图
- `sim2sim_joint_torque.png` — 关节力矩对比图

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SIM_DT | 0.005s | MuJoCo 仿真步长 |
| DECIMATION | 4 | 每4步执行一次策略 (→50Hz) |
| BALL_ARRIVE_TIME_EST | 动态计算 | Sim2Sim 中根据球速动态估计 |
| ACTION_SMOOTHING_ALPHA | 0.7 | 动作 EMA 平滑 |
| RESIDUAL_SCALE | 0.15 | 残差动作缩放 |

### 已知问题 & TODO

#### MuJoCo 与 Isaac Lab 未完全对齐

1. **PD 控制器实现差异**
   - Isaac Lab 使用隐式弹簧-阻尼器 (implicit actuator)
   - MuJoCo 使用显式力矩计算 `τ = Kp*(q_target - q) - Kd*dq`
   - 表现：MuJoCo 中关节超调更大，高速运动时轨迹偏差明显（可对比pos，torque记录图）

2. **球体碰撞模型**
   - Isaac Lab: 球与桌面碰撞使用 `restitution` 参数，弹跳高度一致
   - MuJoCo: 默认碰撞参数不同，需要手动调 `solref`/`solimp` 来匹配
   - TODO: 对齐弹跳系数，当前 MuJoCo 弹跳偏低

3. **球空气阻力**
   - 训练 (`env_cfg.py`): `linear_damping=0.0`（无空气阻力，球匀速飞行）
   - MuJoCo: 默认有微量阻尼
   - 真实: 有明显加速度（弹跳后 vx 从 -1.2 加速到 -2.5 m/s）

4. **关节阻尼/摩擦**
   - Isaac Lab 中 `joint_damping` 和 `joint_friction` 设为特定值
   - MuJoCo URDF 导入后需手动设置 `dof_damping`
   - TODO: 逐关节对齐阻尼参数

5. **Motion file 索引**
   - 目前只使用 `forehand_middle_a1_whip.npz` (motion_id=0)
   - 左右区域动作文件未训练使用
   - TODO: 支持多动作选择

6. **坐标系**
   - 已对齐: +X 指向对手, 机器人在 -X 侧
   - FK 计算球拍位置与 Isaac 中 body transform 有微小偏移

---

## 二、真机部署

### 启动流程

需要启动两个 ROS2 节点（从 `unitree_rl_lab/` 目录运行）：

```bash
# 终端 1: PPO 推理节点 (50Hz 策略推理 + 日志记录)
python worobots/a1_robot_ppo/ppo_inference_pingpong.py \
    --ros-args -p policy_path:=unitree_rl_lab/logs/rsl_rl/a1_tabletennis/exported/policy.pt

# 终端 2: 动作播放节点 (将 a1/sent_actions → /record_data 发送给底层)
python worobots/a1_robot_ppo/play_action_pingpong_ppo.py
```

### 节点架构

```
[Hardware Driver]
     │
     ├── /right_joint_states (JointState, name: "joint1-a1_r" ~ "joint7-a1_r")
     │
     ▼
[ppo_inference_pingpong.py]  ← 策略推理 + 状态机
     │
     ├── subscribe: /right_joint_states, /kalman/pingpong_pos, /kalman/pingpong_vel
     ├── publish:   a1/sent_actions (JointState, 7 DOF targets)
     └── log:       logs/a1_deploy_*.csv, logs/a1_obs_*.csv
              │
              ▼
[play_action_pingpong_ppo.py]  ← 动作转发
     │
     └── publish: /record_data (Float64MultiArray[28], idx 14-20 = right arm)
              │
              ▼
[Hardware Controller]
```

### 状态机

| State | 值 | 行为 |
|-------|---|------|
| IDLE | 0 | Hold current + 缓慢回归 default_pos |
| BLENDING | 1 | 余弦平滑过渡到策略输出 (0.08s) |
| TRACKING | 2 | 策略推理 + 闭环 phase 校正 |
| SWINGING | 3 | 策略推理 (phase > HIT_PHASE) |
| RECOVERING | 4 | 余弦平滑回归 default_pos (0.5s) |

**TODO**： BLENDING 和 SWINGING 感觉不够平滑，后续继续改善。

### 关键参数

| 参数 | 值 | 位置 | 说明 |
|------|-----|------|------|
| CONTROL_HZ | 50.0 | ppo_inference | 控制频率 |
| BALL_ARRIVE_TIME_EST | 0.33 | a1_table_tennis_deploy | 触发时 phase 起始偏移 |
| HIT_PHASE | 0.54 | a1_table_tennis_deploy | 击球对应的 phase |
| RESIDUAL_SCALE | 0.05 | a1_table_tennis_deploy | 部署残差缩放 (训练=0.15) |
| PHASE_CORRECTION_GAIN | 0.10 | ppo_inference | 闭环 phase 校正增益 |
| PHASE_CORRECTION_MAX | 0.006 | ppo_inference | 每帧最大 phase 校正 |
| IDLE_RETURN_ALPHA | 0.02 | ppo_inference | IDLE 回归速度 |
| ball_incoming_vx_threshold | -1.0 | ROS param | 触发挥拍的球的 vx 阈值 |

---

## 三、日志文件

### 部署日志 (a1_deploy_*.csv)

每次运行自动生成，50Hz 采样。

**列说明 (49列)：**

| 列范围 | 名称 | 说明 |
|--------|------|------|
| 1 | time_s | 相对启动时间 (s) |
| 2 | state | 状态机状态 (0-4) |
| 3-9 | jp_1~7 | 关节实际位置 (rad) |
| 10-16 | jv_1~7 | 关节速度 (rad/s, 差分计算) |
| 17-23 | je_1~7 | 关节力矩 (Nm) |
| 24-30 | tgt_1~7 | 目标关节位置 (rad) |
| 31-33 | ball_x/y/z | 球 3D 位置 (m) |
| 34-36 | ball_vx/vy/vz | 球 3D 速度 (m/s) |
| 37-39 | racket_x/y/z | 球拍 3D 位置 (m, FK计算) |
| 40 | phase | 当前动作 phase [0,1] |
| 41 | phase_speed | 策略输出的 phase 速度 |
| 42 | motion_id | 动作文件编号 |
| 43 | ball_time_to_arrive | 球到达估计时间 [0,3]s |
| 44-46 | pred_hit_y/z/t | 预测击球点 (y, z, t) |
| 47-49 | bounce_has/rising/urgency | 弹跳状态 |

### 观测日志 (a1_obs_*.csv)

与 deploy CSV 同步生成，记录策略实际输入/输出，方便 sim2real 对比。

**列说明 (67列)：**

| 列范围 | 名称 | 说明 |
|--------|------|------|
| 1 | time_s | 时间 |
| 2 | state | 状态 |
| 3-59 | obs_00~56 | 57D 观测向量 (策略实际输入) |
| 60-67 | act_0~7 | 8D 动作输出 (7 residual + 1 phase_speed) |

**观测向量布局 (57D)：**

| obs index | 维度 | 含义 |
|-----------|------|------|
| 0-14 | 15 | motion_command: ref_dof(7) + ref_dof_vel(7) + ref_base_y(1) |
| 15-21 | 7 | joint_pos_rel: 当前位置 - 默认位置 |
| 22-28 | 7 | joint_vel: 关节速度 |
| 29-31 | 3 | ball_pos_relative: 球相对机器人位置 |
| 32-34 | 3 | ball_vel_relative: 球速度 |
| 35-37 | 3 | ball_spin: 球角速度 (部署中为0) |
| 38-40 | 3 | racket_pos: 球拍世界坐标 |
| 41 | 1 | motion_phase: 当前 phase [0,1] |
| 42 | 1 | ball_time_to_arrive: 球到达时间 [0,3] |
| 43-45 | 3 | ball_predicted_hit: (pred_y, pred_z, pred_t) |
| 46-48 | 3 | ball_bounce: (has_bounced, is_rising, urgency) |
| 49-56 | 8 | last_action: 上一帧动作输出 |

**TODO**：对比实际部署的观测日志和isaac play的观测日志（obs.npz）的差距，优化下一步的训练或者部署。

### 可视化生成（对比图）

```bash
# 使用 conda 环境 (numpy/matplotlib 兼容)
conda run -n base python your_plot_script.py
```

日志目录中已有的可视化：
- `*_ball_vel_accel.png` — 球速度 + 加速度
- `*_ball_vx_rallies.png` — 每次回合 vx 曲线
- `*_flight_accel.png` — 加速度对时间预测的影响
- `*_joints.png` — 关节轨迹
- `phase_analysis_latest.png` — Phase 演化分析

---

## 四、Sim2Real 已知差异

| 维度 | 训练 (Isaac) | 真机 | 影响 |
|------|-------------|------|------|
| 球速度模型 | 匀速 (linear_damping=0) | 有加速度 (~-1.5 m/s²) | 到达时间预测偏大 |
| 球击球高度 | z≈0.9m | z≈0.1-0.4m | 挥拍高度不匹配 |
| ACTION_SMOOTHING | 0.7 (EMA) | 已关闭 (=0) | 动作更激进 |
| RESIDUAL_SCALE | 0.15 | 0.05 | 残差幅度更小 |
| 关节速度 | 精确 (sim state) | 差分计算 (50Hz) | 可能有噪声 |
| 球角速度 | 有随机化 | 始终为0 | 缺少旋转信息（目前训练可以克服，如有更高难度要求则需要优化） |
| Phase speed | 策略输出 [0.85, 1.15] | 实测多为0.85-0.89 | 挥拍偏慢 |

### 部署端补偿措施

1. **加速度补偿**：`_correct_phase()` 使用匀加速模型估算 `t_remain`（二次方程求解），实时估计球 x 方向加速度，实机尝试时有部分改善。
2. **闭环 Phase 校正**：每帧根据球实时位置修正 phase，最大 0.006/frame
3. **Hold-current IDLE**：避免 reset 时关节突变
4. **速率限制**：MAX_DELTA=0.08 rad/step 防止目标跳变

---

## 五、TODO

### Sim2Sim 对齐

- [ ] PD 控制器参数对齐 (Kp/Kd 匹配 Isaac implicit actuator)
- [ ] 球碰撞参数对齐 (restitution, solref/solimp)
- [ ] 关节阻尼/摩擦逐关节标定
- [ ] 验证 FK 球拍位置与 Isaac body transform 一致性
- [ ] 添加 Isaac vs MuJoCo 定量对比指标 (轨迹 RMSE)

### 部署优化

- [ ] 球速度和高度适配 (训练中 z≈0.9m vs 真实 z≈0.1-0.4m)，考虑发球回球带加速度的情况
- [ ] 改善优化击球延迟问题（考虑滤波和训练时所加的延迟随机化之间的关系）
- [ ] 支持多 motion_id 选择 (目前只用 motion_id=0)
- [ ] 评估是否恢复 ACTION_SMOOTHING (当前已关闭)
- [ ] 增加击球成功率统计和自动记录

### 工具链

- [ ] 自动化 sim2real 对比脚本 (读 a1_obs CSV vs play_obs.npz)
- [ ] 实时可视化工具 (rqt/rviz 插件)
- [ ] 部署参数热加载 (无需重启节点)
