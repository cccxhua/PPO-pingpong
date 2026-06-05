# 乒乓球任务训练史 (X1 → A1)

本文档梳理了 `unitree_rl_lab` 乒乓球任务从最初的参考动作生成、X1 迭代训练、到 A1 复用与真人数据驱动的完整流程，重点记录每个阶段的**改动、踩过的坑、解决方案**。

> 时间跨度：**2026-05-13 → 2026-06-04**（约 3 周）
> 机器人：**X1**（5-19 ~ 5-21 完成）→ **A1**（6-01 ~ 6-04 持续）→ G1（占位）
> 训练框架：IsaacLab 2.3.0 + IsaacSim 5.1.0 + RSL-RL (PPO)

---

## 目录

1. [项目架构概览](#1-项目架构概览)
2. [参考动作生成阶段](#2-参考动作生成阶段)
3. [X1 训练演进（2026-05-19 ~ 21）](#3-x1-训练演进)
4. [A1 训练演进（2026-06-01 ~ 至今）](#4-a1-训练演进)
5. [Sim2Sim 验证（MuJoCo）](#5-sim2sim-验证mujoco)
6. [Sim2Real 部署框架](#6-sim2real-部署框架)
7. [核心经验教训](#7-核心经验教训)
8. [当前状态与待办](#8-当前状态与待办)

---

## 1. 项目架构概览

### 控制范式：参考动作 + Residual

不直接学绝对关节位置，而是：
```
joint_target[t]  =  ref_motion(phase[t])  +  residual_scale × policy_output[t]
phase[t+1]       =  phase[t] + dt × phase_speed   ← phase_speed 也是 policy 输出
```

- **参考动作 npz**：离线设计的关键帧 + 样条插值（"挥拍模板"）
- **residual**：policy 学的小幅修正（X1 早期 0.4，A1 当前 0.05）
- **phase_speed**：相位速度（0.85~1.15 倍速），自适应不同球速

**好处**：policy 不用从零学挥拍轨迹，只学"如何针对不同来球微调"，极大降低样本复杂度。

### 任务列表

| Task ID | 机器人 | 状态 |
|---------|--------|------|
| `X1-TableTennis` | X1 (5-DoF 上身 + 7-DoF 右臂) | ✅ 已完整迭代到 sim2real 路线图末段 |
| `A1-TableTennis` | A1 (固定底盘 + 升降柱 + 7-DoF 臂) | 🟡 训练中，复用 X1 框架 |
| `Unitree-G1-29dof-TableTennis` | G1 (29-DoF 人形) | 占位 |

### 配置入口（A1 为例）

```
source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/a1/forehand/env_cfg.py
```

单文件覆盖：场景 / 球资产 / 球预设 / 动作 / 命令 / 观测 / 奖励 / 终止 / 域随机化。

---

## 2. 参考动作生成阶段

### 工具链（`scripts/rsl_rl/ref_generate/`）

| 类别 | 脚本 | 作用 |
|------|------|------|
| **生成** | `right_face_sweep.py` / `right_face_fast.py` / `right_sweep.py` | 不同挥拍轨迹的关键帧设计（v50 ~ v85 多代版本） |
| **回放** | `play_reference_motion.py` | 播放 npz，双视角 + 录制视频 |
| **可视化** | `animate_arm_motion.py` | matplotlib 3D 动画（FK），不依赖 IsaacSim |
| **诊断** | `diagnose_hit_geometry.py` | residual=0 下测击球几何（paddle 朝向、过网点） |
| **诊断** | `diagnose_right.py` / `compare_middle_right.py` | 不同挥拍方向对比 |
| **诊断** | `fk_probe.py` | 逐关节 ±0.5 rad 扰动，看 paddle 位移方向 |

### `play_pure_ref.py`（**重要诊断工具**）

绕过 policy（`residual=0`），直接用纯 ref motion 回放，验证参考动作本身是否能击球。

```bash
python scripts/rsl_rl/play_pure_ref.py \
  --task X1-TableTennis \
  --npz path/to/forehand_middle_v74.npz \
  --ball_preset middle \
  --arrive_time 0.55 \
  --hit_phase 0.54 \
  --output_dir logs/pure_ref/diag_v74
```

输出：`paddle_traj.txt`（球拍轨迹+朝向）/ `ball_traj.txt`（球轨迹）/ `joints.txt`（实际 vs 目标）/ `torques.txt`（扭矩饱和检查）/ `rl-video.mp4`。

### 命名规范

- **永不覆盖旧 npz**：版本号递增 `v50`, `v52`, `v74`, `v85` ...
- 命中节点：`MIDDLE` 系列采用 `v74` 版本，`RIGHT` 系列采用 `v50/v52`
- A1 版本带 `_a1` 后缀，与 X1 npz 隔离

### 教训

1. **参考动作必须先单独验证能击球**：用 `play_pure_ref.py` 跑一遍，确认无 policy 介入也能让球过网
2. **关键帧时间必须对齐 hit_phase**：例如 `hit_phase=0.54` 表示挥拍周期 54% 处接触球，npz 设计要让该相位对应正确姿态
3. **paddle_traj.txt 的本地三轴**比关节角更直观——能直接看出"球拍朝向是否对着对手台"

---

## 3. X1 训练演进

X1 是先做的，跑了 **8 个 resume 链式训练**，从 iter 0 到 10000+，把整个 sim2real 路线图走通。每个 run 都有 `README.md` 记录。

### 时间线总表

| Run | iter 范围 | 阶段名 | 关键改动 |
|-----|-----------|--------|---------|
| `2026-05-19_10-01-14` | 0 → 1000 | Stage 1: 模仿基线 | EASY_BALL 固定，模仿权重最大 |
| `2026-05-19_11-22-40` | 1000 → 2000 | 速度解锁 | EASY_BALL vx/vy/vz 加范围；模仿权重大幅下调 |
| `2026-05-20_04-20-08` | 2000 → 4500 | 位置解锁 + 奖励重构 | 引入 `ball_land_placement`；禁用 `phase_ball_alignment`（局部最优陷阱） |
| `2026-05-20_07-26-56` | 4500 → 6000 | 时序对齐 | `ball_arrive_time_est: 0.5205 → 0.55` 让拍面接触时朝上 |
| `2026-05-21_02-19-50` | 6000 → 7500 | TRAIN_BALL + Bell-curve | 单调奖励 → 高斯型最优值 |
| `2026-05-21_07-19-30` | 7500 → 9000 | DR 第一批 + PPO 修复 | 动作延迟 / 力矩 / PD 随机化；`lr 1e-3 → 3e-4` |
| `2026-05-21_09-27-45` | 9000 → 10000 | 移除关节重置 | relaunch 时不再 snap 关节，靠 PD 自然回位 |
| `2026-05-21_11-06-25` | 10000+ | 观测延迟随机化 | obs_delay 随机 [0, 3] 步（0-60ms） |

### Stage 1: 模仿基线（iter 0 → 1000）

**配置**：
- 球：`EASY_BALL` 固定单点 — `x=-0.35, z=1.30, vx=3.5, vz=0.5`
- 奖励权重：`pose_tracking=1.50`, `vel_tracking=0.90`, `joint2_tracking=0.60`（**模仿压倒性主导**）
- DR：完全关闭
- PPO：`lr=1e-3`, `desired_kl=0.01`, `epochs=5`

**目的**：让 policy 先学会跟随参考动作，建立基础挥拍模式。

### Stage 2: 速度解锁（iter 1000 → 2000）

**改动**：
- 球：`vx 3.5 → [3.3, 3.7]`, `vy 0 → ±0.2`, `vz 0.5 → [0.3, 0.7]`
- 模仿权重大幅下调：`pose_tracking 1.50→0.80`, `vel_tracking 0.90→0.50`, `joint2_tracking 0.60→0`

**经验**：
> ⚠️ **模仿权重过高会阻止 policy 对接触/击球类奖励的学习**。模仿建立了基础后必须"放手"，否则 policy 永远在刷模仿分而不真出手。

### Stage 3: 位置解锁 + 奖励重构（iter 2000 → 4500）

**改动**：
- 球位置加随机：`x: -0.35 → [-0.37, -0.33]`, `z: 1.30 → [1.28, 1.32]`
- 显式设 `linear_damping=0.0`（去空气阻力）
- **禁用 `phase_ball_alignment`** —— 这是个**踩过的坑**

**踩坑**：`phase_ball_alignment` 奖励诱发 **phase-freeze 局部最优** —— policy 把 `phase_speed` 压到 0.85 极限，让相位"卡住"对齐球，但永远不挥拍。
**解决**：直接禁用，改用 `ball_land_placement`（落点目标，2D 高斯）做引导。

### Stage 4: 时序对齐（iter 4500 → 6000）

**改动**：仅一行 — `ball_arrive_time_est: 0.5205 → 0.55`

**踩坑 + 经验**：
> ⚠️ **球到达时间估计 50ms 的偏差**就能让 policy 在击球瞬间球拍朝向**朝下**而不是朝上。
> 这次只调一个数，但是**时序精度是击球任务的命门**。后续 sim2real 时也需要类似的人工对齐过程。

### Stage 5: TRAIN_BALL + Bell-curve 奖励（iter 6000 → 7500）

**改动**：
- 新球预设 `TRAIN_BALL`：x[-0.40,-0.30] / y±0.08 / z[1.25,1.35] / vx[3.0,4.0] / vy±0.3 / vz[0.2,0.8]
- **奖励形态从单调改成 Bell-curve（高斯型最优值）**：
  - `racket_approach`: `max_vel` → `optimal_vel=1.0, σ=0.8`
  - `ball_speed_after_hit`: → `optimal_speed=3.5, σ=1.5`
  - `ball_hit_toward_opponent`: → `optimal_vx=-3.0, σ=1.5`
- 球观测噪声：`pos ±0.01, vel ±0.05`

**经验**：
> 单调奖励 ("越快越好") 让 policy 学习暴力发力，关节常打到 limit；Bell-curve **明确告诉 policy 最优值在哪里**，避免无意义压榨。

### Stage 6: DR 第一批 + PPO 修复（iter 7500 → 9000）

**改动**：
- TRAIN_BALL 范围进一步扩大：`vx[2.9, 4.1], vz[0.1, 0.9]`
- **首次开 DR**：动作延迟 [0, 1] 步 / 力矩 ±10% / PD 增益 ±20%
- **PPO 超参修复**（防 `std` 崩溃）：
  - `lr: 1e-3 → 3e-4`
  - `value_loss_coef: 1.0 → 0.5`
  - `epochs: 5 → 3`
  - `desired_kl: 0.01 → 0.008`

**踩坑 + 经验**：
> ⚠️ DR 引入会让环境分布变宽，**原 PPO 超参 (lr=1e-3, epochs=5) 会让 policy 标准差快速崩溃** —— 输出几乎确定，无法适应新分布。
> **必须同步**：lr 减半、KL 阈值收紧、value loss 系数减半、epochs 减少。这套组合是后面所有 DR 训练的标配。

### Stage 7: 移除关节重置（iter 9000 → 10000）

**改动**：删除 `relaunch_ball_if_out` 里的关节 snap 代码 —— 重发球时机器人保持击完球后的姿态，靠 PD 自然回到准备位。

**目的**：仿真便利的 "hard reset" 在真机不存在 —— 真机连续打球时机器人必须自己挥完了"还原"。

**经验**：
> ⚠️ **仿真捷径会害死 sim2real**。任何"非物理上下文重置"都要主动删除（关节 snap、瞬移球到中心、瞬时清零速度等）。

### Stage 8: 观测延迟随机化（iter 10000+）

**改动**：
- `obs_delay` 随机 [0, 3] 步（0-60ms @ 50Hz），匹配真机 mocap → host 链路延迟
- 用 IsaacLab `DelayBuffer` 实现，每个 env reset 时采样
- **Asymmetric actor-critic**：policy 看延迟观测，critic 不延迟（标准 RL 加速训练 trick）
- 新建 `DelayedObsEnv` 包装类作为 env entry_point

**经验**：
> ⚠️ **真机感知链路 40-60ms 必须仿真**。否则 policy 学到的全是"球当下位置"反应式策略，到真机看到的是 60ms 前的位置，时机全错。
> Asymmetric actor-critic（critic 看完整观测）极大加速训练 —— 价值估计准了 policy 才能从延迟观测中学到前馈预测。

---

## 4. A1 训练演进

A1 是 X1 完成后启动的，**复用了 X1 的所有架构**（residual + phase_speed + 同样的奖励组合 + DR_STAGE 框架），换成不同的机械臂（A1 7-DoF + 升降柱锁定）。

### Git 提交时间线

| Commit | 日期 | 说明 |
|--------|------|------|
| `2af420e` | 06-03 06:31 UTC | A1 任务初版（含 staged DR） |
| `b37d3ed` | 06-03 09:52 UTC | **Tune DR 参数** — z 下界 1.00 → 0.90（**埋雷**） |
| `6f2cee0` | 06-03 10:11 UTC | 修 USD 路径，独立 A1 资产 |

### 训练 run 时间线（完整 16 个 run）

| Run | MIDDLE_DR_BALL z | 备注 |
|-----|------------------|------|
| 06-01 ~ 06-02 06:54 (5 个) | 无 `MIDDLE_DR_BALL` | 用其他球预设（TRAIN_BALL / EASY_BALL） |
| **06-02 08:40 ~ 11:00** (4 个) | **`(1.05, 1.15)`** | 100% 合格 ✓ |
| **06-02 14:14** (1 个) | **`(1.00, 1.20)`** | 100% 合格 ✓ |
| 06-03 09:39 (long run, 22.8K iter) | **`(0.90, 1.30)`** ⚠️ | 22% 撞网球（埋雷） |
| 06-04 03:30 / 07:07 / 07:57 | `(0.90, 1.30)` ⚠️ | resume，问题继承 |
| 06-04 (现在) | **`(1.00, 1.30)`** ✓ | 已修复 |

### 踩过的最大的坑：z 下界 0.90 引发 22% 撞网球

**症状**：play 时大部分发球看起来不合理（球贴桌飞、撞网、过不了）。22.8K iter 的长 run reward 高（20+）但 `ball_hit ≈ 0`。

**根因**：6-03 那次 "Tune DR 参数" 把 `z_range` 从 `[1.00, 1.20]` 拉宽到 `[0.90, 1.30]` 想增加多样性。但 `z=0.90 + vz=0.10 + x=0.50 + vx=-2.6` 这个最差组合算下来：
```
t_net = 0.50 / 2.6 = 0.192 s
z_net = 0.90 + 0.10·0.192 - 0.5·9.81·0.192² = 0.74 m  ← 比球网顶 0.9125 矮 17 cm
```
→ **22% 的边界采样组合连球网都过不了**。

**影响**：
- policy 训练样本里 22% 是"无法接到的球"，学到"放弃这种球"的避险行为
- ball_missed_paddle 终止率高居 50-80%
- ball_hit reward 永远 ≈ 0，**policy 收敛到"摆击球姿态但不真出手"的 reward hacking 局部最优**
- 22.8K iter 长 run 的 reward 涨到 20+ 全靠 pose_tracking / racket_face_target 等姿态分

**修复**：z 下界 0.90 → 1.00（一行改动），有效率从 80% 提升到 97.6%。

**经验**：
> ⚠️ **球范围扩张前必须做物理可行性检查**：对每个最差边界组合手算"球是否能过网"，而不是凭直觉认为"放宽边界 = 多样性更好"。
> 一行错误的常数能毁掉一周的训练。

### 真人发球数据驱动（2026-06-04 引入）

发现 sim 与真人的 gap 后引入。

**数据来源**：`data_0602/vrpn_pose_ball_*.txt`（113 段 mocap，120 Hz 采样）

**提取流程**（`x=+0.35` 处即 sim 出生位置的瞬时状态）：
```
mocap (120Hz) → 找过 x=+0.35 的帧 → 中央差分算 (vx,vy,vz) → 109 段有效
       → 每维度的 5/95% 分位 → SERVE_A2_BALL
```

**关键发现**：
| 维度 | sim (MIDDLE_DR) | 真人 5/95% | 真人在 sim 范围占比 |
|------|----------------|-----------|---------------------|
| vx | [-3.4, -2.6] | **[-4.7, -3.2]** | **13%** ❌ |
| vz | [+0.1, +0.3]（全上抛）| **[-1.0, +1.3]**（含下落） | **5%** ❌ |
| z (桌面相对) | +0.24 ~ +0.54 | +0.23 ~ +0.74 | 55% |
| vy | ±0.35 | [-0.5, +0.7]（不对称） | 75% |

**真人 vz 中位 -0.28（球已过抛物线顶点正在下落）**，因为真人击球点在 `x≈+1.30`（球台端线后），球飞到 sim 出生位置 `x=+0.35` 时已经在下落了 —— sim 简化模型把球出生在网附近导致 `vz` 全是上抛，与真实**完全不符**。

### SERVE_STAGE 渐进（2026-06-04 实施）

为渐进迁移到真人发球，引入 `SERVE_STAGE` 环境变量（与 `DR_STAGE` 解耦）：

```
SERVE_STAGE=0  MIDDLE_DR_BALL    球速 3.0 m/s   起点（已修复 z=1.00）
SERVE_STAGE=1  SERVE_A1 (α=0.5)  球速 3.45 m/s  全维度向真人 5/95% 走一半
SERVE_STAGE=2  SERVE_A2 (α=1.0)  球速 3.95 m/s  完整真人 5/95%
SERVE_STAGE=3  SERVE_B1          (待实现) 出生点回移到 x=+1.0
SERVE_STAGE=4  SERVE_B2          (待实现) 出生点 +1.30 + 模拟己方台弹跳
```

`ball_arrive_time_est` 自动随 SERVE_STAGE 同步（球速变化 → 飞行时间变化）：`0.51 → 0.43 → 0.40`。

### 当前 reward hacking 问题（未解决）

A1 所有 run 中 `ball_hit ≈ 0.001` 都没突破，policy 收敛到"摆姿态不出手"局部最优。已尝试：

1. ✓ **修 z=0.90 撞网球问题**（一行）
2. ✓ **拉高击球类 reward weight**：
   - `ball_hit: 1.5 → 2.5`
   - `ball_hit_speed: 1.2 → 2.2`
   - `ball_return: 2.0 → 3.0`
3. 短训 300 iter 数据：`ball_hit 0.0007 → 0.0013 (+86%)`、`ball_missed_paddle 79% → 76%` —— **有正向信号但还远未突破**

**待尝试**（按代价排序）：
- 切回 `SERVE_STAGE=1` 降难度
- 把 `racket_ball_proximity` weight 0.30 → 1.0 加强 dense 信号
- Curriculum：从 `EASY_BALL` 重新开始训
- 直接拉 `ball_hit weight` 到 5.0+

### Action smoothing（2026-06-04 加入）

为 sim2real 准备，在 `ReferenceResidualJointAction` 里加了一阶 IIR 低通（EMA）：
```
smoothed_residual[t] = α · prev + (1-α) · current_action
target = ref_dof + smoothed_residual × scale
```
当前 `α=0.5`（cutoff 5.7 Hz @ 50 Hz policy）。**训练和部署必须用相同 α**，否则等效于真机 PD 增益打折。

---

## 5. Sim2Sim 验证（MuJoCo）

**脚本**：`scripts/sim2sim/x1_play_mujoco.py`（X1 已跑通，A1 待补）

**目的**：把 IsaacSim 训练好的 policy 在 **MuJoCo** 里独立验证 —— 这是真机部署前最重要的"第二次仿真"，能暴露 PhysX 与 MuJoCo 在接触/质量/惯性建模上的差异。

**架构**：
```
[Trained policy .pt/.onnx]   [URDF + 球桌 + 球]
         │                          │
         └──────► MuJoCo 50Hz Loop ◄┘
                       │
            ObservationAssembler (57D)
                       │
                  PolicyRunner (JIT)
                       │
                  PhaseStateMachine
                       │
                  PD 关节目标
```

**复用模块**（来自 `x1_table_tennis_deploy.py`）：
- `MotionLoader` 加载 npz
- `PhaseStateMachine` 相位演化
- `predict_ball_hit_point` / `compute_ball_time_to_arrive` / `compute_ball_bounce_state` 球运动学

**X1 sim2sim 主要发现**：
- IsaacSim PhysX **球-拍接触刚度**比 MuJoCo 高 → sim 里击球速度估计偏快
- MuJoCo 等效 PD damping 比 IsaacSim 大 → 训练 DR 里 `damping_range` 必须放到 (0.5, 2.0) 才能涵盖

---

## 6. Sim2Real 部署框架

### 部署脚本

`scripts/deploy/x1_table_tennis_deploy.py`（650+ 行，独立运行，零 IsaacLab 依赖）

**输入**（57D 观测）：
- 关节状态（位置/速度，7 DoF）
- 球状态（pos relative + vel relative + spin）
- 击球相位 / 球到达时间估计 / 球预测击球点 / 球弹跳状态
- last_action

**输出**（8D action）：
- `action[0:7]` — 7 关节 residual（× 0.05 scaling）
- `action[7]` — phase_speed（映射到 [0.85, 1.15]）

**频率**：50 Hz（与训练 dt 一致）

**模型结构**：MLP `[57 → 512 → 256 → 128 → 8]`，ELU 激活

### 分阶段 DR 路线图（sim2real 准备）

```
DR_STAGE=0  确定性基线（仿真 sanity check）
DR_STAGE=1  + 发球范围（具体范围由 SERVE_STAGE 选）
DR_STAGE=2  + 动作延迟（≈10ms 均值，匹配真机控制环路）
DR_STAGE=3  + 观测延迟 + 观测噪声 + PD 增益 / 力矩随机化（完整 sim2real DR）
```

**用法**：
```bash
DR_STAGE=3 SERVE_STAGE=2 python scripts/rsl_rl/train.py --task A1-TableTennis --headless
```

### Sim2Real 检查清单

部署前必须满足：
1. ✅ 参考动作（`play_pure_ref.py`）能单独击过网（不依赖 policy）
2. ✅ Policy（`play.py`）在 EASY_BALL 稳定击球率 ≥ 70%
3. 🟡 Sim2Sim（MuJoCo）轨迹与 IsaacLab 对齐
4. 🟡 完整 DR（`DR_STAGE=3 SERVE_STAGE=2`）训练稳定
5. 🟡 部署脚本通过单元测试
6. ⏳ 真机硬件检查（关节限位、传感器延迟、PD 增益）

### Action smoothing 一致性约束

训练时启用了 `α=0.5` 的 EMA 滤波（在 `ReferenceResidualJointAction.process_actions` 内），**真机部署必须做同样的滤波**：

```python
# 部署时每个 control step:
smoothed = α * prev_residual + (1-α) * policy_output
target = ref_dof + smoothed * residual_scale
prev_residual = smoothed
```

否则等价于真机 PD 增益打折，挥拍峰值速度和时机全错。

---

## 7. 核心经验教训

### 通用 RL 训练经验

| # | 教训 | 来源 |
|---|------|------|
| 1 | **模仿权重过高阻止后期击球学习** —— 建立基础后必须主动降低 | X1 Stage 2 |
| 2 | **奖励要么 Bell-curve 要么 sparse 事件** —— 单调奖励诱发暴力策略 | X1 Stage 5 |
| 3 | **DR 引入必须同步 PPO 超参降档** —— lr/2, epochs/2, KL 收紧 | X1 Stage 6 |
| 4 | **Asymmetric actor-critic** —— critic 看完整观测加速训练，policy 看 noisy 观测学鲁棒性 | X1 Stage 8 |
| 5 | **Sparse 击球奖励 weight 必须够大** —— 否则被 dense 姿态分淹没 → reward hacking | A1 当前 |

### 击球任务专属

| # | 教训 | 来源 |
|---|------|------|
| 6 | **时序精度是命门** —— `ball_arrive_time_est` 50ms 偏差 = 拍面朝向反向 | X1 Stage 4 |
| 7 | **phase_ball_alignment 局部最优陷阱** —— 让 policy 卡相位不出手 | X1 Stage 3 |
| 8 | **z 下界放宽前必须算"过网约束"** —— 撞网球是无效样本，不是多样性 | A1 b37d3ed |
| 9 | **真人发球数据 vs sim 假设 gap 巨大** —— sim 出生在网边导致 vz 全上抛，真人在 x=+0.35 处大部分已下落 | A1 6-04 |
| 10 | **球速跨度 → 飞行时间错位 25%+** —— 同 hit_phase 配置下击球时机散乱 | A1 SERVE_A2 |

### Sim2Real 专属

| # | 教训 | 来源 |
|---|------|------|
| 11 | **任何"非物理重置"都要删除** —— 关节 snap、瞬移球、清零速度 | X1 Stage 7 |
| 12 | **真机感知链路 40-60ms 延迟必须仿真** —— 否则反应式策略到真机时机全错 | X1 Stage 8 |
| 13 | **Action smoothing 训练/部署必须一致** —— 否则等效 PD 增益打折 | A1 6-04 |
| 14 | **MuJoCo damping 比 PhysX 大** —— DR damping_range 要放到 (0.5, 2.0) | X1 sim2sim |

---

## 8. 当前状态与待办

### 当前最稳定 checkpoint

| 任务 | Run | iter | 状态 |
|------|-----|------|------|
| X1 | `2026-05-21_11-06-25` | 10000+ | ✅ 已完整跑过 sim2real DR 路线图 |
| A1 | `2026-06-03_09-39-52` | 22796 | ⚠️ reward hacking，ball_hit ≈ 0 |
| A1 | `2026-06-04_08-37-50` | 16000+ | 🟡 reward 调高后短训中（300 iter） |

### A1 紧急待办

1. **打破 ball_hit ≈ 0 plateau**（关键瓶颈）：
   - 短期：观察当前 reward 调整后训练 1500+ iter 是否突破
   - 备选：切 `SERVE_STAGE=1` 降难度
   - 备选：进一步加 `racket_ball_proximity weight`
2. **B1/B2 阶段实现**（出生点回移到对方端线，含己方台弹跳）
3. **真人数据相关性**：当前 sim 5 维独立均匀采样，真实是有相关性的 → 改用截断高斯或直接真人轨迹采样

### 中期待办

1. **A1 sim2sim**（MuJoCo）尚未跑通
2. **G1 任务**（29-DoF 人形）从 X1/A1 移植
3. **延迟参数细化**：当前 obs_delay [0, 1] 步偏短，真人测出 40-60ms 应该到 [0, 3] 步
4. **球物理 DR**：质量 / 桌面摩擦 / 恢复系数 —— 之前 PhysX tensor API 兼容性问题暂时禁用

### 长期路线

1. ✅ **A 路径**（同 sim 出生位置匹配真人状态）：A1, A2 已实现
2. ⏳ **B 路径**（出生点后移模拟完整发球）：B1, B2 占位
3. ⏳ **真机部署**：等 A1 训练突破后开始单元测试
4. ⏳ **多桌面/多球类型 DR**：质量 ±10%、摩擦 ±20%、恢复系数 ±5%

---

> 本文档跟踪到 **2026-06-04 09:00 UTC**。
> 维护者：训练阶段切换或踩到新坑请追加 section。
