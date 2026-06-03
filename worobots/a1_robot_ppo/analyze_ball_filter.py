"""分析 A1 部署日志中检测到球但未触发挥拍的时间点及原因。

判断"检测到球"：ball_pos 不全为 0（感知系统有数据）。
判断"未触发"：state 保持 IDLE (0)。

触发条件（全部满足才挥拍）:
  1. ball_vx < -1.0        (球朝机器人飞)
  2. ball_x > ROBOT_X + 0.3 = -1.2  (球还在机器人前方)
  3. ball_z > 0.5          (球在桌面以上)
"""

import sys
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    "worobots/a1_robot_ppo/logs/a1_deploy_20260603_160200.csv"

data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
t = data["time_s"].astype(float)
state = data["state"].astype(int)
ball_x = data["ball_x"].astype(float)
ball_y = data["ball_y"].astype(float)
ball_z = data["ball_z"].astype(float)
ball_vx = data["ball_vx"].astype(float)
ball_vy = data["ball_vy"].astype(float)
ball_vz = data["ball_vz"].astype(float)

ROBOT_X = -1.5
VX_THRESH = -1.0
X_OFFSET = 0.3
Z_THRESH = 0.5

# 找到"有球数据但状态为 IDLE"的时间段
ball_detected = ~((ball_x == 0) & (ball_y == 0) & (ball_z == 0))

output_lines = []
output_lines.append(f"日志文件: {csv_path}")
output_lines.append(f"触发条件: vx < {VX_THRESH}, x > {ROBOT_X + X_OFFSET}, z > {Z_THRESH}")
output_lines.append(f"{'='*80}")
output_lines.append("")

# 找连续的"有球但 IDLE"段
in_segment = False
seg_start = 0
segments = []

for i in range(len(t)):
    is_idle_with_ball = (state[i] == 0) and ball_detected[i]
    if is_idle_with_ball and not in_segment:
        seg_start = i
        in_segment = True
    elif not is_idle_with_ball and in_segment:
        segments.append((seg_start, i - 1))
        in_segment = False

if in_segment:
    segments.append((seg_start, len(t) - 1))

# 对于挥拍后立刻的 RECOVERING→IDLE 段有球是正常的（球已飞走），过滤掉
# 只保留纯 IDLE 段中有球数据的片段（前一帧不是 RECOVERING）
filtered_segments = []
for (s, e) in segments:
    # 跳过球刚消失/出现的单帧噪声（至少持续 50ms = 5帧@100Hz）
    duration = t[e] - t[s]
    if duration < 0.05:
        continue
    # 检查这段是否紧跟 RECOVERING（球飞走中）
    if s > 0 and state[s - 1] == 4:
        continue
    filtered_segments.append((s, e))

output_lines.append(f"共发现 {len(filtered_segments)} 段有球但未触发挥拍的时间段:")
output_lines.append("")

for seg_idx, (s, e) in enumerate(filtered_segments):
    t_start = t[s]
    t_end = t[e]
    duration = t_end - t_start

    output_lines.append(f"--- 段 {seg_idx + 1}: t = {t_start:.3f}s ~ {t_end:.3f}s (持续 {duration:.3f}s) ---")

    # 分析这段中每帧不满足条件的原因
    reasons = {"vx_too_slow": 0, "x_too_far_behind": 0, "z_too_low": 0, "all_met": 0}
    sample_frames = []

    for i in range(s, e + 1):
        vx_ok = ball_vx[i] < VX_THRESH
        x_ok = ball_x[i] > (ROBOT_X + X_OFFSET)
        z_ok = ball_z[i] > Z_THRESH

        if vx_ok and x_ok and z_ok:
            reasons["all_met"] += 1
        else:
            if not vx_ok:
                reasons["vx_too_slow"] += 1
            if not x_ok:
                reasons["x_too_far_behind"] += 1
            if not z_ok:
                reasons["z_too_low"] += 1

    total_frames = e - s + 1

    # 打印各原因占比
    output_lines.append(f"  总帧数: {total_frames}")
    output_lines.append(f"  未触发原因统计:")
    if reasons["vx_too_slow"] > 0:
        pct = reasons["vx_too_slow"] / total_frames * 100
        output_lines.append(f"    - vx >= {VX_THRESH} (球未朝机器人飞/速度不够): {reasons['vx_too_slow']} 帧 ({pct:.1f}%)")
    if reasons["x_too_far_behind"] > 0:
        pct = reasons["x_too_far_behind"] / total_frames * 100
        output_lines.append(f"    - x <= {ROBOT_X + X_OFFSET} (球已飞过机器人): {reasons['x_too_far_behind']} 帧 ({pct:.1f}%)")
    if reasons["z_too_low"] > 0:
        pct = reasons["z_too_low"] / total_frames * 100
        output_lines.append(f"    - z <= {Z_THRESH} (球太低/在地面): {reasons['z_too_low']} 帧 ({pct:.1f}%)")
    if reasons["all_met"] > 0:
        pct = reasons["all_met"] / total_frames * 100
        output_lines.append(f"    - 条件全满足但未触发(可能处于非IDLE状态): {reasons['all_met']} 帧 ({pct:.1f}%)")

    # 打印几个代表帧的数值
    sample_indices = [s, (s + e) // 2, e]
    output_lines.append(f"  代表帧数据:")
    for idx in sample_indices:
        output_lines.append(
            f"    t={t[idx]:.3f}s: ball=({ball_x[idx]:.3f}, {ball_y[idx]:.3f}, {ball_z[idx]:.3f}), "
            f"vel=({ball_vx[idx]:.3f}, {ball_vy[idx]:.3f}, {ball_vz[idx]:.3f}), "
            f"vx_ok={ball_vx[idx] < VX_THRESH}, x_ok={ball_x[idx] > ROBOT_X + X_OFFSET}, z_ok={ball_z[idx] > Z_THRESH}"
        )
    output_lines.append("")

# 保存
txt_path = csv_path.replace(".csv", "_ball_filter_analysis.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(output_lines))

print(f"分析结果保存到: {txt_path}")
print()
print("\n".join(output_lines))
