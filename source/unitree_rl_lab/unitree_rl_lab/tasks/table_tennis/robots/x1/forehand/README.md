# X1 乒乓球机器人 · 前手击球 (Forehand) 训练

X1 机器人（固定底盘 + 升降柱，仅**右臂 7 个关节**可控）学习前手击球：
接住对方发来的球，挥拍把球击过网、落到对方台面。

> 配套给 Claude 的操作手册见 skill `x1-table-tennis-training`；本文件面向人，讲清架构、上手命令与目录。

---

## 架构：参考动作跟踪 + Residual 控制

不是从零学挥拍，而是：

1. **离线参考轨迹**：用关键帧 + 样条插值离线生成挥拍动作 `.npz`（中路 / 左路 / 右路三种方向）。
2. **策略学习 residual**：PPO 学习在参考轨迹上叠加一个小幅度的关节残差（`residual_scale`），
   并学习**相位快慢**（`PhaseSpeedActionCfg`，0.85~1.15 倍速），以适配不同来球。
3. **目标**：靠奖励把球击过网、落到对方台面，同时跟踪参考姿态、保持动作平滑、不超关节/力矩限。

环境随机从 `motion_files` 里采样一条参考动作，并按来球方向匹配（`match_ball_direction`）。

---

## 环境与解释器

统一用 Isaac Sim 解释器，在仓库根目录运行（不经 `unitree_rl_lab.sh` 包装器）：

```bash
cd /root/unitree_rl_lab
/workspace/isaaclab/_isaac_sim/python.sh <脚本> ...
```

直接调脚本时记得自己带 `--headless`。

---

## 上手命令

### 列出任务
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/list_envs.py
```
任务 id：`X1-TableTennis`（本任务）、`A1-TableTennis`、`Unitree-G1-29dof-TableTennis`。

### 训练
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/train.py \
  --task X1-TableTennis --headless --num_envs 4096 --max_iterations 50000
```
恢复训练加 `--resume --load_run "<时间戳目录>" --checkpoint "model_<N>.pt"`。
常用参数：`--num_envs --max_iterations --seed --experiment_name --run_name --logger {tensorboard,wandb}`。

### 回放训练结果（看视频）
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/play.py \
  --task X1-TableTennis --headless --load_run "<时间戳目录>" \
  --checkpoint "model_<N>.pt" --num_envs 1 --video --video_length 400
```
输出：`logs/rsl_rl/<exp>/<run>/videos/play/rl-video-step-0.mp4`、`play_joint_pos.{npz,txt}`。

### 诊断参考动作（pure_ref，绕过策略）
不加载策略、residual=0，单独检验参考轨迹和来球设置好不好——任务短，迭代调参时常用：
```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/play_pure_ref.py \
  --task X1-TableTennis --headless --video --video_length 500 \
  --ball_preset right --arrive_time 0.55 --hit_phase 0.54 \
  --output_dir logs/pure_ref/<新子目录>
```
参数：`--npz` 自定义参考动作、`--ball_preset {middle,left,right,high}`、
`--arrive_time` 覆盖球到达时间估计、`--hit_phase` 覆盖击球时相。
输出（在 `--output_dir`）：视频、`torques.txt`（关节扭矩）、`joints.txt`（实际 vs 目标）、
`paddle_traj.txt`（拍位置/朝向）、`ball_traj.txt`（球轨迹），控制台打印清台率。

> ⚠️ `--output_dir` 永远用新路径，不要覆盖已有目录（`logs/` 已 gitignore，删了不可恢复）。

---

## 配置在哪改

主环境配置（单文件）：`env_cfg.py`（即本目录）

| 想改什么 | 类 |
|----------|----|
| 奖励 / 惩罚（只作用右臂 `RIGHT_ARM_JOINT_NAMES`） | `RewardsCfg` |
| residual 幅度 / 动作延迟 / 相位速度 | `ActionsCfg` |
| 击球时相 / 球到达时间 / 启用哪些 npz | `CommandsCfg` |
| 观测项 / 观测噪声 | `ObservationsCfg` |
| 域随机化（sim2real 关键） | `EventsCfg` |

PPO 超参：`../../agents/rsl_rl_ppo_cfg.py` 的 `TableTennisPPORunnerCfg`
（lr 1e-4、adaptive KL、ELU、隐层 [512,256,128]）。

---

## 参考动作 npz

- 文件在本目录：`forehand_middle.npz`、`forehand_left.npz`、`forehand_right_v<NN>.npz` 等。
- 生成脚本：`create_forehand*.py`（关键帧 + 样条插值，输出 `dof_pos` 形状 `[frames, 7]`，约 100Hz）。
- **版本约定**：迭代版本用 `_vNN` 递增，新版本**另存新文件，不覆盖旧 npz**。
- 当前实际启用哪些看 `CommandsCfg.motion_files`。

---

## 关节定义（右臂 7 DOF）

| 关节 | 含义 |
|------|------|
| yb_1 | shoulder pitch（肩屈伸） |
| yb_2 | shoulder roll（肩外展） |
| yb_3 | shoulder yaw（肩旋转） |
| yb_4 | elbow（肘屈伸） |
| yb_5 | wrist roll |
| yb_6 | wrist pitch |
| yb_7 | wrist yaw |

（限位以 `env_cfg.py` / 机器人资产为准。）

---

## Sim2Real

训练完成后：先在 MuJoCo 做 sim2sim 验证，再部署真机。域随机化按优先级逐步放开
（观测延迟 → 球物理 → PD gains → action delay → 观测噪声 → 球到达时间），每次只加 1~2 项、
训练稳定后再加下一批。具体目标区间见团队的 sim2real DR 路线图。
