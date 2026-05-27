"""生成 X1 正手挥拍参考动作 (3组: 中路/左路/右路)

运行: python create_forehand.py
生成: forehand_middle.npz, forehand_left.npz, forehand_right.npz

关节含义:
  yb_1 = shoulder_pitch  正=前抬, 负=后摆   限位: [-1.05, 3.17]
  yb_2 = shoulder_roll   负=外展            限位: [-3.08, 0.31]
  yb_3 = shoulder_yaw    正=外旋            限位: [-2.78, 2.76]
  yb_4 = elbow           正=屈肘            限位: [-1.91, 1.95]
  yb_5 = wrist_roll      前臂旋转           限位: [-2.79, 2.76]
  yb_6 = wrist_pitch     腕屈伸             限位: [-1.29, 1.51]
  yb_7 = wrist_yaw       腕偏转             限位: [-3.14, 3.14]
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

# ============================================================
#  三组关键帧
#  时间秒, [yb1, yb2, yb3, yb4, yb5, yb6, yb7] rad
# ============================================================

# --- 中路球 v4: 针对 PD 滞后压缩轨迹
#   背景: 之前轨迹幅度过大 (yb_1 0.7→2.3, yb_4 1.3→-0.2), PD 阻尼比 ζ≈0.03 严重欠阻尼,
#         实际关节滞后命令 0.7-1.0 rad, 击球瞬间 paddle 跑偏到 (1.27, -0.31, 1.0)
#         球在 (1.28, 0, 1.03), Y 偏 32cm 完全打不到.
#   策略:
#     - yb_4 全程 hold 1.50 (= URDF 默认值, PD 无需做功, 跟踪误差最小)
#     - yb_1 swing 0.80→1.40 (压缩 60%, 让 PD 跟得上)
#     - yb_2 击球推到 +0.25 (近上限 +0.31, 极力争取 +y)
#     - yb_3, yb_6, yb_7 保持原 wrist flick 模式
#   静态 FK (PD settle 后): paddle ≈ (1.29, -0.25, 1.07), 球 ≈ (1.28, 0, 1.03)
#   预期动态 paddle Y 误差从 -0.32 缩到 -0.10 ~ -0.15 m.
MIDDLE = [
    #  时间       yb1     yb2     yb3     yb4    yb5     yb6     yb7
    #  v34: 修正 v33 face 方向被破坏 + 手臂太低问题.
    #     v33 错误: yb_5=-1.0 把 face 转到 +Y (侧面), yb_4=0.6 让臂太低撞桌.
    #     v34 策略:
    #        - 保持 v32 face 关节核心 (yb_5=0, yb_6=-1, yb_7=1)
    #        - 仅微调 yb_1↑ 抬臂, yb_3 less neg 让肩外送, yb_4 微伸
    #     HIT 静态: paddle (1.282, -0.144, 1.041), face (-0.822, +0.082, +0.564)
    #              face Y=0.08 (几乎纯 -X), 34° 上挑, err 0.145
    #  v50: 解耦位置 (yb_4 早伸 + HOLD) 和速度 (yb_6 snap 中心 t=0.40).
    #     v49 教训: yb_4 在 t=0.20→0.35 完成挥击, peak 速度在 t=0.35, t=0.40 已衰减到 0.66 m/s.
    #         即使加大 yb_6 swing, 整体 swing 时序还是太早.
    #     v50 思路:
    #         (1) yb_4 早伸到 0.05 by t=0.30, 然后 HOLD 到 t=0.45 (paddle X 在 t=0.40 已到位)
    #         (2) yb_6 windup -1.10 @ t=0.30, snap -0.45 @ t=0.45, peak ω 在 t≈0.375 -> 0.40
    #             Δ=0.65, peak ω≈6.5 rad/s, 贡献 ~1.3 m/s 沿 -X+Z (主导 hit 速度)
    #     预期 t=0.40:
    #         paddle 在最大伸出位置 (X≈1.28~1.30), yb_6 ω 仍接近峰值
    #         |v| ≈ 1.0~1.3 m/s (vs v45 0.74, v49 0.66)
    #     代价: yb_4 在 t=0.20-0.30 加速更快 (Δ=0.90 over 0.10s 而非 v45 的 0.15s)
    #  v51: 在 v50 基础上整体往右(+y)平移 + 略微下压 Z, 保持 v50 击球速度.
    #     v50 hit 状态: paddle (1.266, -0.048, 0.996), 球 (1.28, 0, 1.03)
    #     用户反馈: "手臂整体往右, 在v45基础上, 速度也需加上" -> Y 往 0/+y, 保留 v50 速度.
    #     动态扫描 (probe_y_shift_dynamic.py) 结果:
    #         单调 yb_3 -0.15 让 Y 从 -0.048 升到 -0.023 (Δ +0.026m), Z 升 +0.072
    #         加 yb_1 -0.10 把 Z 拉回 1.025 (近 ball Z), Y 保持 -0.022
    #     v51 改动 (在 v50 各帧上叠加常量): yb_1 -= 0.10, yb_3 -= 0.15
    #     预期 hit: paddle (1.228, -0.022, 1.025), |v|≈1.67 m/s, -vx≈1.58
    #              vs v50 paddle (1.266, -0.048, 0.996), |v|=1.68
    #         Y 误差: 0.048 -> 0.022 (减半), Z 误差: 0.034 -> 0.005 (基本消除)
    #         X 误差: 0.014 -> 0.052 (略后移, 但 paddle 仍在球前侧, 面有面积)
    #  v52: 修正 v51 paddle Z=1.025 过高 (face center 高出球面). 用户反馈 "z 太高了".
    #     v51 hit: paddle (1.228, -0.022, 1.025), face_z(loft) +0.42 (~25° 上挑)
    #              face center 因 tip_z 偏移到 ~Z=1.10+ (远高于球 Z=1.03)
    #     动态扫描:
    #         yb_1 -0.20 (vs v51 的 -0.10): 让 paddle Z 从 1.025 -> 0.981, Y 几乎不变
    #         同时 face_z(loft) 从 +0.42 -> +0.33 (~19° 上挑), face center 整体下移
    #     v52 = v50 + (yb_1 -= 0.20) + (yb_3 -= 0.15):
    #         hit: paddle (1.220, -0.020, 0.981), |v|=1.68, -vx=1.62
    #              比 v50 (Z=0.996, Y=-0.048): Y +0.028 右移, Z -0.015 下压, vx +0.06
    #  v53: v52 + 0.10 phase shift (100ms 后挪). 击球率 ~2%, 仍偏早.
    #  v55: v52 + 0.175 phase shift (175ms). 用 sweep_shift.py probe 8 envs × 20s 扫描:
    #     +25  ms -> 0/8  reversals
    #     +100 ms -> 2/9  reversals  (= v53)
    #     +150 ms -> 40/57 reversals
    #     +175 ms -> 54/73 reversals (74% 接触转化率, post |vx|=1.42 m/s)  <-- 选这个
    #     根本原因: ball_arrive_time_est=0.45 用直线飞行估计, 没算弹桌减速.
    #     球弹桌后 vx 从 3.5 降到 1.84, 真实到达 paddle ≈ t=0.55-0.60, 不是 0.45.
    #     额外 PD 滞后 ~30-50ms, 实际最优 shift 落在 +175ms.
    #     v55 关键帧 (= v52 + 0.175):
    #         mid:       0.20 -> 0.375
    #         extend:    0.30 -> 0.475
    #         snap end:  0.45 -> 0.625
    #         follow:    0.60 -> 0.775
    #         return:    0.80 -> 0.975
    #     注意: env_cfg.hit_phase 仍是 0.40, 但实际 peak ω 现在在 phase 0.55. 后续若想
    #         同步, 应把 hit_phase 改成 0.55 并把 swing_timing.hit_phase_end 从 0.50 调到 0.65.
    #  v56: v55 基础上加 +z 速度, 但保留 v55 击球时刻 (t≈0.55) 的 paddle 位置.
    #     失败教训: 直接改 t=0.625 的 yb_1/yb_6 → cubic spline 反向影响 t=0.55 paddle 位置 → 完全错过球.
    #     修复策略: 加 pin keyframe at t=0.55 锁住击球瞬间值 (= v55 在 t=0.55 的插值),
    #         前 (t=0.475) 加深 windup, 后 (t=0.625) 增大 snap end → 击球瞬间位置不变, 但 ω 增加.
    #     v55 在 t=0.55 的 cubic spline 插值值 (击球瞬间):
    #         yb_1 ≈ 1.450, yb_6 ≈ -0.775 (-1.10 与 -0.45 中点)
    #     v56 改动:
    #         t=0.475: yb_6 -1.10 -> -1.20 (windup 深 +0.10)
    #         t=0.55:  NEW pin (yb_1=1.45, yb_6=-0.775) 锁住击球位置
    #         t=0.625: yb_6 -0.45 -> -0.30, yb_1 1.45 -> 1.50 (snap 加幅 + 肩抬, 集中在击球后)
    #     swing Δyb_6: 0.65 -> 0.90 (+38% 峰值 ω), 击球时刻位置不变
    #     yb_1 在 t=0.55→0.625 抬起 0.05 rad / 0.075s = 0.67 rad/s 肩 ω (击球瞬间已开始)
    #
    #  v57: 用 Isaac Lab FK probe 实测后发现 v55/v56 的 pin pose 完全打不到球.
    #     诊断 (scripts/rsl_rl/v56_paddle_ball.py):
    #         v56 在 t=0.55: paddle (0.971, +0.333, +0.936),  ball (1.289, 0, 1.144)
    #         缺口 Δ=(+0.318, -0.333, +0.209), gap = 50.6cm — paddle 离球差大半米.
    #     v57 通过 FK 坐标下降搜索找到能让 paddle 中心到达球位置的关节配置.
    #     v57 PIN @ t=0.55: [1.60, 0.07, -2.05, 0.70, -0.03, -1.045, 1.00]
    #     静态 FK 验证: paddle (1.282, -0.019, 1.152), ball (1.289, 0, 1.144), gap = 2.18cm ✓
    #
    #  v57 致命问题: 静态 FK 的"球目标"用了我手算的解析弹道 (table_z=0.79, restitution=0.905,
    #     friction=0.526), 但 sim 实际物理参数和我假设的不一样, 真实球轨迹比解析高 ~8cm.
    #     dynamic 探针 (v57_dynamic_with_ball.py) 显示:
    #         真实 sim 球 @ t=0.475: (1.281, 0, 1.028)   <- 解析估计 ~(1.15, 0, 1.03)
    #         真实 sim 球 @ t=0.550: (1.376, -0.016, 1.224)  <- 解析估计 (1.29, 0, 1.144)
    #     v57 实际 paddle 在 t=0.55 (PIN 时刻) 才到 (1.300, -0.037, 1.048) - PD 严重欠阻尼,
    #         yb_1 滞后 0.23 rad, paddle Z 比静态 FK 低 12cm. 击球时刻球已经飞高到 1.224, 漏球 18cm.
    #     真实最近接 (v57): t=0.490, gap 3.92cm, paddle 在球弹起初期擦过.
    #
    #  注: 训练奖励函数全部用 sim 真实球位 (ball.data.root_pos_w), 没有这个失误,
    #     仅离线 PIN 设计脚本 (v57_search_hit_pose.py 的 ball_pos_at()) 有此 bug.
    #
    #  v58: 用 sim 真实球轨迹重设计.
    #     PIN 时间 0.55 → 0.475 (匹配真实最近接时刻).
    #     PD-lag 补偿后的静态 FK 目标: real_ball - lag_offset = (1.215, 0.005, 1.149)
    #     v58 PIN @ t=0.475: [1.560, 0.090, -2.100, 0.630, -0.030, -0.975, 1.000]
    #         静态 FK paddle: (1.217, 0.010, 1.147), gap to PD-comp 0.60cm
    #     dynamic 验证 (scripts/rsl_rl/v58_dynamic.py):
    #         t=0.475 paddle (1.274, -0.034, 1.049), ball (1.240, 0.001, 1.040), gap 5.0cm
    #         最近接 t=0.480, gap 4.75cm. 球 X 速度被反弹 (1.240→1.211 倒走), 确认击中.
    #         paddle 击球速度 |v|=1.20 m/s, vz=+0.16 (会把球上抬, 但偏低, 后续可调).
    #     整体 schedule 提前 75ms: windup 0.475→0.40, PIN 0.55→0.475, snap 0.625→0.55,
    #         follow 0.775→0.70, return 0.975→0.90.
    #     env_cfg 同步更新: hit_phase 0.60 → 0.475.
    #  v59 已回退: paddle body frame 假设错误.
    #     v59_paddle_frame_diag.py 假定 Link_yb_paddle body frame 的 (+x,+y,+z) =
    #     STL 视觉的 (face_width, face_normal, handle), 但实际 body frame 轴向未必与 STL 对齐.
    #     基于错误假设的 sweep (yb_5 -0.40 + yb_3 +0.12 + yb_4 -0.05 + yb_2 +0.05)
    #     上线后视觉确认完全脱靶 (paddle 转到偏离球 20-30cm). 回到 V58 baseline.
    #     V58 status: paddle 能击中球但接触点在拍柄/拍面边界 (用户反馈 "打在手柄上没拍出去").
    #     下一步需要先对齐 paddle body frame 真实轴向再做调整, 否则任何 wrist 旋转都是盲调.
    #  v60: 用单关节 sim sweep 测响应矩阵, lstsq 求解平移到 face_center 落球的关节增量.
    #     V58 t=0.475 paddle 原点 (1.274, -0.034, 1.049) 离球 5cm 但 face 在原点延长线 12cm 远,
    #     球击中手柄. V60 推理: face_offset 沿 origin→ball 方向 12cm, 所以 origin 该退 7cm,
    #     落到 (1.321, -0.083, 1.062). 在 hit-window keyframes (t∈{0.300, 0.400, 0.475, 0.550})
    #     上叠加 yb_1 -0.039, yb_2 -0.187, yb_4 +0.110 rad.
    #     sweep 实测验证: paddle 落到 (1.336, -0.078, 1.066), 离 target gap 1.58cm.
    #     球位置受 paddle 干扰移到 (1.281, 0.000, 1.028) (V58 是 1.240,0.001,1.040), 表明
    #     paddle 已经接触并推动球, 不再只是擦过手柄.
    #  v61: V60 还是刚好打在手柄上 + 整体过高. 用户视角 (站机器人背后看 -X 方向):
    #     需要 paddle 朝右 (+Y, 因为机器人面 -X) +4cm, 朝下 -5cm, face_normal 朝向不变,
    #     肘自然跟着低. 在 V60 hit-window keys (t∈{0.300,0.400,0.475,0.550}) 上叠加:
    #         yb_1 -0.234, yb_2 +0.200, yb_3 +0.121, yb_4 +0.067, yb_5 -0.115, yb_6 -0.045, yb_7 -0.000
    #     V61 视频确认击球位置准, 但球只飞到 -0.39 m/s 反弹, 不能过网.
    #
    #  v62: 击球位置准了, 但 ball 飞不回去因为 face_normal 朝向错了.
    #     V61 face_normal world = (+0.14, -0.84, +0.52), 几乎水平朝上+左, 没 -X 分量.
    #     ball 砸 paddle 顶部弹起 (bz=+2.88), 而不是被面击回 (bx 仅 -0.39).
    #     诊断: paddle X 仅领先 ball 18mm, contact normal · 速度 = -0.054 m/s,
    #         paddle 实际是擦过 ball 顶部, 不是把 ball 顶回去.
    #     v62c/v62d sim grid 搜索: yb_5 wrist_roll 是关键, 转 face 朝向直接改善反弹速度.
    #         best: yb_5 -0.40 + yb_4 -0.20 (在 V61 hit-window keys 上叠加)
    #         实测: paddle (+1.235,+0.016,+1.045), ball (+1.207,-0.005,+1.048),
    #               gap 3.5cm (vs V61 6.7cm), bx_min = -1.85 m/s (vs V61 -0.39, +4.7x), bz_max +2.63
    #     joint 限位: yb_4 [-1.911, 1.948], yb_5 [-2.789, 2.761] — 全部安全.
    #
    #  v63: V62 反弹强了但 +Z 不够, 球飞不过网 (z_at_net=1.010, 网顶=0.94, 仅余 7cm).
    #     v63_loft.py probe: yb_4 snap (elbow 屈曲反转) 是最有效 +Z 源.
    #         V62 yb_4 snap@t=0.550 = +0.557 (在 PIN 后还在减小, 即继续伸肘).
    #         V63 改成 +0.80, PIN @0.475 +0.607 → snap @0.550 +0.80 反向屈肘,
    #         前臂上摆给 paddle +Z 速度.
    #     实测: gap 3.9cm, bx=-2.00 (反弹强度保持), bz=+2.39, z_at_net=1.067 (过网 +12.5cm).
    #     更激进 yb_4 snap +1.00/+1.20 PD 滞后变大反而过不了网.
    #
    #  v64: V63 实际 bz 反而比 V62 (无 yb_4 snap) 还低! 真问题在 face_normal Z 朝向, 不在 paddle vz.
    #     诊断 (v64_paddle_vz.py): 击球瞬间 paddle linear v 全部为 (-1.5, -0.4, -0.2),
    #         vz 是负的! 即 paddle 在向下走, 不是向上掀. ball post-vz 完全来自 face_normal Z 反射.
    #     V62 baseline (无 yb_4 snap): bz=+2.89 (实际比 V63 +2.39 高!).
    #     v64_yb5_sweep.py 找 yb_5 snap (wrist_roll, 控制 face_normal 方向):
    #         yb_5=-0.515: bz=+2.89, z_at_net=1.010
    #         yb_5=-0.415 (+0.10): bz=+3.19, z_at_net=0.961
    #         yb_5=-0.365 (+0.15): bz=+3.21, z_at_net=1.120
    #         yb_5=-0.265 (+0.25): bz=+3.09, z_at_net=1.101
    #     V64 = V62 base + yb_5 snap @t=0.550 改成 -0.365 (less negative, face 转更竖):
    #         bx=-1.70, bz=+3.21, z_at_net=1.120. 舍弃 V63 的 yb_4 snap=+0.80.
    #
    #  v65: 用户视觉反馈 V64 仍"球拍平直/从上往下", paddle vz at hit 全部负 (PD swing 物理特性, 改不了符号).
    #     v65b_face.py 诊断: V64 face_n_world Z = +0.74 已经较高, 但只改 snap@0.550 不够,
    #         挥拍前段 (mid/windup/PIN) face 仍偏 horizontal. 球被打到时实际 face 朝向取决于
    #         整个 hit-window 的 cubic spline 插值.
    #     扫 yb_5 hit-window 整体偏移 (mid+windup+PIN+snap 全部加值):
    #         yb_5 hw +0.20: gap=2.9, bz=+3.18, zn=1.298  <-- 选这个
    #     V65 = V64 + yb_5 hit-window 全部 +0.20: gap=2.9cm, bz=+3.18, z_at_net=1.298 (网余 35cm).
    #
    #  v66: V65 face 朝上让 face_n X = -0.04 几乎 0, 球纯垂直反弹, bx=-1.37 飞距弱,
    #     球只落到桌中段 (x_land 1.5s sim 内甚至没落地). 用户视觉确认: 球落到桌前部网前.
    #     v67_pareto.py: yb_5 PIN -0.65 (vs V65 -0.345) 给 face_n=(-0.16,-0.66,+0.73) —
    #         face 同时朝 -X 和 +Z (反弹力分散到 -X 和 +Z 两方向).
    #         意外发现: PIN -X 朝 paddle linear vx 从 -1.47 跃升到 -2.06 (wrist 翻转增益).
    #     v67b_swing_combo.py: PIN-0.65 + yb_4 hw -0.15 (前臂全程更伸):
    #         paddle vx=-2.04, face=(-0.16,-0.68,+0.72), bx=-1.97, bz=+2.46, zn=1.058 ✓,
    #         x_land=-0.62m (球落对方半场 62cm, V65 球 2s 内没落到桌!)
    #     V66 改动 = V65 + yb_5 PIN @0.475 改成 -0.65 + yb_4 hit-window 整体 -0.15.
    #  v67: 用 V68 解析弹道探针发现 V66 球先在己方台 (x=+0.32) 弹再过网 — 不算分.
    #     V66 真实 ball post-vel = (-1.97, -0.03, +1.60) (vz 远低于之前 probe 报的 +2.46,
    #         那是反弹后再升的 max). 解析 z@net = +0.23m, 远低于网顶 0.91m.
    #     V69_face_remap 扫描 yb_6/yb_7 PIN 找到唯一 CLEARS+VALID:
    #         yb_6 PIN -0.30 (was -1.02): vx=-2.85, vz=+2.11, z@net=+1.07m (网余 16cm),
    #         x_first_bounce=-0.33m (对方台), face=(-0.25, -0.72, +0.65).
    #     yb_6 = wrist_pitch, less negative 让 paddle 腕部放松, face_X 从 -0.16 降到 -0.25,
    #         face Y 几乎不变, 反射 vx 翻倍.
    #     V67 = V66 + yb_6 PIN @ t=0.475 改为 -0.30.
    #  v68: 消掉 V67 yb_6 cubic spline overshoot.
    #     V67 在 t=0.400 windup -1.245 → t=0.475 PIN -0.300 (Δ=+0.945 rad / 75ms 大 snap),
    #     spline 在 t≈0.37 先下冲到 -1.365 (超 -1.288 limit 0.077 rad), 实际 PD clamp 在限位上.
    #     v68: windup t=0.400 改 -1.245 → -1.100. spline min 变 -1.232 (留 56 mrad 余量),
    #         swing 幅度 1.065 → 0.932 rad (压 12% peak ω, 但 V67 主要靠 PIN 反射几何, 不靠 ω).
    #
    #  v69 已回退: V67/V68 实测 ref play 都不如 V66 — 用户反馈 "球拍要在击球时从下往上给球
    #     一个力". V67 改的是 face_normal 朝向 (静态反射几何), 不是 paddle linear vz (动态线速度).
    #     V64 注释已确认击球瞬间 paddle vz 是负的 (-0.2), V66 PIN 时刻 yb_1 ω≈1.3 rad/s 仍偏小,
    #     yb_6 大 snap (Δ=0.525) 都发生在 PIN→snap 之后, 击球时刻没用上.
    #     回退到 V66, 后续应该改 yb_1/yb_4 在击球瞬间的 ω 或者把 yb_6 snap 重新分配到 hit 之前.
    #  v70: V66 baseline + 力度小幅提升 (v74 probe E1 变体).
    #     v74_collision_force.py 用真实 phase offset (hit_phase=0.475, ball_arrive_time_est=0.5205)
    #     测得 V66 碰撞瞬间 paddle |v|=1.60, peak |v|=2.11 @ t=0.525 (peak 比 hit 晚 55ms),
    #     ball post |v|=2.50, z@net=-1.21 (远过不了网, 但靠 phase noise 偶尔过).
    #     E1 在 V66 上加大 windup→PIN 的 yb_1/yb_6 Δ:
    #         yb_6 wu  -1.245 → -1.285  (近 limit -1.288, 余 3 mrad)
    #         yb_1 wu  +1.187 → +1.137  (-0.05)
    #         yb_1 PIN +1.287 → +1.337  (+0.05, Δ wu→PIN: 0.10 → 0.20, 2x)
    #     实测: ball post |v|=3.43 (+37%), bv=(-1.66, +0.07, +2.99) — vx vz 都升,
    #         z@net 从 -1.21 升到 +0.52 (大幅靠近网顶 0.91), 仍未确定过网,
    #         但平均能量提升后 phase noise 内"过网得分"概率应明显增加.
    #     不动 PIN 时间 (用户反馈 "时机 ok"), 仅幅度小调.
    #
    #  v71: V70 ref play 实测 "回球后首次落点都在本场" — vx 不够 (-1.66), 弧线没飞过网区.
    #     v75_face_angle.py 试在 V70 上调 yb_6 PIN, 全部超限 (V70 windup -1.285 已贴 limit).
    #     v76_face_pin.py 同步放松 yb_6 windup + 抬 PIN, 找到唯一 CLEARS+VALID 变体 E1:
    #         yb_6 wu  -1.285 → -1.150  (放松 0.135, 给 PIN 上调留 spline buffer)
    #         yb_6 PIN -1.020 → -0.700  (抬 0.320 rad, face 转更竖)
    #     实测: face_n=(-0.02, -0.75, +0.66), bv=(-1.65, +0.08, +3.69), |bv|=4.05,
    #         z@net=+1.07 (过网 +16cm), x_bnc=-0.12 (对方台 12cm 内).
    #     vx 几乎不变 (-1.65 vs V70 -1.66), 但 vz 从 +2.99 → +3.69 (+23%).
    #     球以更高弧线过网, 落对方半场 — 这是 V67 反弹方向 + V70 力度的折中.
    #     V71 = V70 (yb_1 wu -0.05, PIN +0.05) + (yb_6 wu -1.150, PIN -0.700).
    #
    #  v72: V71 vx -1.65 仍偏弱, 用户 "需要更大 X 方向速度, 高度足够".
    #     v77_face_xy.py 在 V71 上单独试 yb_5 PIN, 找到 A4 (yb_5 PIN -0.800):
    #         yb_5 PIN -0.650 → -0.800 (wrist_roll 加深 0.15 rad, 单关节单点修改)
    #     实测: bv=(-2.02, +0.13, +3.58), |bv|=4.11
    #         vx -1.65 → -2.02 (+22%), vz 几乎不变 (+3.58 vs +3.69)
    #         z@net +1.07 → +1.40 (反而升 33cm, 网余 49cm)
    #         x_bnc -0.12 → -0.38 (落点深入对方台 38cm)
    #     反直觉: yb_5 (wrist_roll) 更深没改 face 朝向 (face_n 几乎不变), 但 vx 大幅升 —
    #     原因是 wrist 翻转让肘部更直接对球做 -X 推动 (paddle linear vx 增加, 不只 face 几何).
    #     V72 = V71 + yb_5 PIN -0.650 → -0.800 (仅改 t=0.475 单点).
    #
    #  v73: V72 ref play "有可以过网, 但要更大 vx + 提高过网率".
    #     v79_robustness.py 在 hit_phase ∈ {0.425, 0.450, 0.475, 0.500, 0.525} 上扫多变体,
    #     发现 R2 (yb_6 PIN -0.700 → -0.400) 是单点单调改进:
    #         mean_vx: V72 -1.83 → R2 -2.06 (+13% across 5 phases)
    #         mean_zn: V72 -0.15 → R2 +0.18 (平均网余从负转正)
    #         min_zn:  V72 -2.21 → R2 -1.44 (worst-case 改善 0.77m)
    #         phase=0.475 (中心): V72 z@net=+1.40 → R2 +1.05 (仍过网, 余 14cm)
    #     注: V73 边际提升 vx + 鲁棒区扩大 (其他相位"近 miss"距离缩小), 但根本没解决
    #         peak window 太窄问题 — V72 和 V73 都只有 phase=0.475 单一相位过网.
    #         要真正提升过网率需要 "early PIN hold" (t=0.450 加一帧让 face 提前到位),
    #         留到 V74 测试.
    #     V73 = V72 + yb_6 PIN -0.700 → -0.400 (单点单关节修改).
    #     yb_6 spline min=-1.256 (limit -1.288, 余 32 mrad, 安全).
    #
    #  v74: V73 训练中 ref play 3/4 过网, 但球打网刚好没过, 时机/face 都准 — 只缺 vx.
    #     v81b probe 在 V73 上扫 12 个 vx 推进变体, 单 phase=0.475:
    #         F4 三方温和 (yb_1 wu/PIN ±0.05 + yb_3 PIN -1.850 + yb_5 PIN -0.900):
    #             vx=-2.26 (V73 -2.03, +11%), vz=+3.33 (+11%), z@net=+1.40 (V73 +1.05, +33%),
    #             x_bnc=-0.46 (落对方台 46cm 深), CLR+VAL ✓
    #         三个单点叠加, 互不冲突, 全部远离 limits.
    #     V74 = V73 + (yb_1 wu -0.05, PIN +0.05) + (yb_3 PIN -1.850) + (yb_5 PIN -0.900).
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # ready
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),  # mid
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150,  +1.000]),  # windup (V74: yb_1 -0.05)
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400,  +1.000]),  # PIN    (V74: yb_1+0.05, yb_3-1.85, yb_5-0.900)
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),  # snap
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),  # follow_open
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # 回 ready
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # ready hold
]

# --- 左路球: ball 到 robot +Y 侧 (predicted_y > +0.05 时 motion_id=1)
#   v1: 基于 MIDDLE v74 + yb_3 hit-window 偏移 -0.30 (肩外旋更深, 推 paddle 到 +Y).
#     probe_left_v5.py 实测:
#       5/5 hit, mean bvx=-2.02, gap 1.8~3.4cm across phase sweep [0.425-0.525].
#       paddle @ hit: (1.14, +0.09, 1.08), 球: (1.07, +0.07, 1.08) — Y/Z 对齐精确.
#       面法线: (+0.13, -0.90, -0.43) — 同 MIDDLE/RIGHT, face 朝下但有 -X 分量.
#       与 RIGHT v15 对比: LEFT bvx=-2.02 vs RIGHT bvx=-1.78 (LEFT 更强).
#     ref 动作不需自行过网 — residual policy 训练时学会补偿 (同 MIDDLE v74 模式).
LEFT = [
    #  时间       yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),  # ready
    (0.300, [+1.127, +0.198, -2.204, +0.877, -0.315, -1.045, +1.000]),  # mid
    (0.400, [+1.087, +0.103, -2.279, +0.507, -0.315, -1.150, +1.000]),  # windup
    (0.475, [+1.387, +0.103, -2.150, +0.457, -0.900, -0.400, +1.000]),  # PIN
    (0.550, [+1.437, +0.103, -2.279, +0.407, -0.165, -0.495, +1.000]),  # snap
    (0.700, [+1.450, +0.100, -2.300, +0.850, +0.000, -1.000, +1.000]),  # follow
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),  # 回 ready
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),  # ready hold
]

# --- 右路球: ball 到 robot -Y 侧 (站机器人背后看, 右手方向)
#   v15: v14 PIN plateau + yb_5=-1.10 at PIN (face sweep 最佳 bvx 配置).
#     v14 打到球 (gap 4cm, 5/5 hit) 但 ball 只有 bvx=-1.14 (face 朝 +Y 偏多).
#     yb_5=-1.10 把 face 转向 -X: bvx 从 -1.14 升到 -1.86 (+63%), z_at_net=-0.28.
#     ref 动作不需要自行过网 — residual policy 通过 phase_speed + 角度微调学会回球.
#     MIDDLE v74 也是在 ref play 不能自行过网的基础上训出了 3/4 过网率.
RIGHT = [
    #  时间       yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # hold
    (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),  # windup
    (0.475, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),  # PIN start (yb5=-1.10)
    (0.540, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),  # PIN hold (plateau)
    (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),  # follow
    (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # return hold
]

# ============================================================
FPS = 30
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def generate(name, keyframes, base_y_val=0.0):
    times = np.array([kf[0] for kf in keyframes])
    angles = np.array([kf[1] for kf in keyframes], dtype=np.float32)

    duration = times[-1]
    num_frames = int(duration * FPS) + 1
    t_interp = np.linspace(0, duration, num_frames)

    cs = CubicSpline(times, angles, bc_type="clamped")
    dof = cs(t_interp).astype(np.float32)

    ok = True
    for i in range(7):
        lo, hi = LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo or vmax > hi:
            print(f"  WARNING: joint_yb_{i+1} 超限! "
                  f"[{vmin:.3f}, {vmax:.3f}] vs [{lo:.3f}, {hi:.3f}]")
            ok = False

    base_y = np.full(num_frames, base_y_val, dtype=np.float32)

    out_path = os.path.join(OUTPUT_DIR, f"forehand_{name}.npz")
    np.savez(
        out_path,
        fps=np.float64(FPS),
        upper_body_dof=dof,
        base_y=base_y,
        joint_names=JOINT_NAMES,
    )

    print(f"[{name}] {out_path}")
    print(f"  帧数: {num_frames}, 时长: {duration:.2f}s, FPS: {FPS}")
    if ok:
        print("  关节限位 ✓")
    print(f"  yb2 范围: [{dof[:,1].min():.3f}, {dof[:,1].max():.3f}]")


if __name__ == "__main__":
    generate("middle", MIDDLE)
    generate("left", LEFT)
    generate("right", RIGHT)
