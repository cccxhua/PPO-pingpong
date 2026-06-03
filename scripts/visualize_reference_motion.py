"""生成 X1 参考动作的球拍轨迹动画 (mp4)。

不需要 Isaac Sim，直接从 npz 做正运动学近似 + 3D 可视化。
这里使用上一步 play_reference_motion.py 输出的球拍实际轨迹数据生成动画。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# 从 play_reference_motion.py 的输出中提取的球拍轨迹数据
# (frame_idx, time, yb4, paddle_x, paddle_y, paddle_z)
raw_data = """
0  0.000  1.700  1.356  -0.361  1.174
3  0.100  1.759  1.385  -0.352  1.175
6  0.200  1.657  1.334  -0.366  1.173
9  0.300  1.500  1.255  -0.377  1.167
12  0.400  1.363  1.185  -0.375  1.161
15  0.500  1.200  1.106  -0.361  1.152
18  0.600  0.988  1.010  -0.324  1.141
21  0.700  0.900  0.973  -0.304  1.139
24  0.800  1.091  1.056  -0.345  1.146
27  0.900  1.426  1.217  -0.377  1.164
30  1.000  1.700  1.356  -0.361  1.174
"""

lines = [l.strip().split() for l in raw_data.strip().split('\n')]
data = np.array([[float(v) for v in l] for l in lines])

times = data[:, 1]
paddle_x = data[:, 3]
paddle_y = data[:, 4]
paddle_z = data[:, 5]

# Ball arrival zone
ball_x_range = (1.0, 1.4)
ball_y_range = (-0.15, 0.15)
ball_z_range = (0.9, 1.3)

# Table dimensions (half table on robot side)
table_x = np.array([0.0, 1.37, 1.37, 0.0, 0.0])
table_y = np.array([-0.7625, -0.7625, 0.7625, 0.7625, -0.7625])
table_z = 0.745

# Robot position
robot_x, robot_y, robot_z = 1.5, 0.0, 0.0

fig = plt.figure(figsize=(14, 6))

# Left: 3D trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlabel('X (toward table)')
ax1.set_ylabel('Y (left/right)')
ax1.set_zlabel('Z (height)')
ax1.set_title('Paddle 3D Trajectory (Reference Motion)')

# Draw table surface
ax1.plot(table_x, table_y, [table_z]*5, 'b-', alpha=0.3, linewidth=2)
# Draw net
ax1.plot([0, 0], [-0.9, 0.9], [table_z, table_z], 'g-', alpha=0.3)
ax1.plot([0, 0], [-0.9, 0.9], [table_z+0.15, table_z+0.15], 'g-', alpha=0.3)
# Draw robot base
ax1.scatter([robot_x], [robot_y], [0], c='gray', s=100, marker='s', label='Robot base')

# Draw ball arrival zone (as a box outline)
bx = [ball_x_range[0], ball_x_range[1], ball_x_range[1], ball_x_range[0], ball_x_range[0]]
by = [ball_y_range[0], ball_y_range[0], ball_y_range[1], ball_y_range[1], ball_y_range[0]]
bz_lo = ball_z_range[0]
bz_hi = ball_z_range[1]
ax1.plot(bx, by, [bz_lo]*5, 'r--', alpha=0.4, label='Ball zone (bottom)')
ax1.plot(bx, by, [bz_hi]*5, 'r--', alpha=0.4, label='Ball zone (top)')

# Full trajectory
ax1.plot(paddle_x, paddle_y, paddle_z, 'b-', alpha=0.3, linewidth=1)

# Animate point
point, = ax1.plot([], [], [], 'ro', markersize=10)
trail, = ax1.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
time_text = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes, fontsize=10)

ax1.set_xlim(0.5, 1.6)
ax1.set_ylim(-0.8, 0.5)
ax1.set_zlim(0.7, 1.4)
ax1.view_init(elev=25, azim=-60)

# Right: X-Z side view (time evolution)
ax2 = fig.add_subplot(122)
ax2.set_xlabel('X (toward table <--)')
ax2.set_ylabel('Z (height)')
ax2.set_title('Paddle Side View (X-Z plane)')
ax2.set_xlim(0.8, 1.5)
ax2.set_ylim(0.9, 1.3)
ax2.axhline(y=table_z, color='blue', linestyle='--', alpha=0.3, label='Table surface')
ax2.axhspan(ball_z_range[0], ball_z_range[1], xmin=0, xmax=1, alpha=0.1, color='red', label='Ball Z range')
ax2.axvspan(ball_x_range[0], ball_x_range[1], alpha=0.1, color='green', label='Ball X range')

# Full trajectory in X-Z
ax2.plot(paddle_x, paddle_z, 'b-', alpha=0.3, linewidth=1)
point2, = ax2.plot([], [], 'ro', markersize=10)
trail2, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7)
# Arrow for velocity direction
arrow = None

ax2.legend(loc='upper right', fontsize=8)

n_frames = len(times)

def init():
    point.set_data([], [])
    point.set_3d_properties([])
    trail.set_data([], [])
    trail.set_3d_properties([])
    point2.set_data([], [])
    trail2.set_data([], [])
    time_text.set_text('')
    return point, trail, point2, trail2, time_text

def animate(i):
    idx = i % n_frames

    # 3D
    point.set_data([paddle_x[idx]], [paddle_y[idx]])
    point.set_3d_properties([paddle_z[idx]])
    trail.set_data(paddle_x[:idx+1], paddle_y[:idx+1])
    trail.set_3d_properties(paddle_z[:idx+1])

    # 2D
    point2.set_data([paddle_x[idx]], [paddle_z[idx]])
    trail2.set_data(paddle_x[:idx+1], paddle_z[:idx+1])

    phase = "准备" if idx < 3 else "引拍" if idx < 5 else "击球" if idx < 7 else "随挥" if idx < 9 else "回位"
    time_text.set_text(f't={times[idx]:.2f}s  yb4={data[idx,2]:.2f}  phase: {phase}')

    return point, trail, point2, trail2, time_text

# Create animation: 2 full loops at 10 fps (slow enough to see)
total_anim_frames = n_frames * 2
anim = FuncAnimation(fig, animate, init_func=init, frames=total_anim_frames,
                     interval=200, blit=False)

output_path = '/root/unitree_rl_lab/scripts/reference_motion_trajectory.mp4'
writer = FFMpegWriter(fps=5, metadata=dict(title='X1 Reference Motion'))
anim.save(output_path, writer=writer, dpi=100)
plt.close()

print(f"动画已保存: {output_path}")

# Also save a static plot
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

# X vs time
axes[0].plot(times, paddle_x, 'b-o', markersize=4)
axes[0].axhspan(ball_x_range[0], ball_x_range[1], alpha=0.2, color='red', label='Ball X range')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Paddle X')
axes[0].set_title('Paddle X vs Time\n(smaller X = closer to table)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Y vs time
axes[1].plot(times, paddle_y, 'g-o', markersize=4)
axes[1].axhspan(ball_y_range[0], ball_y_range[1], alpha=0.2, color='red', label='Ball Y range')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Paddle Y')
axes[1].set_title('Paddle Y vs Time\n(0 = center, negative = right)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Z vs time
axes[2].plot(times, paddle_z, 'r-o', markersize=4)
axes[2].axhspan(ball_z_range[0], ball_z_range[1], alpha=0.2, color='red', label='Ball Z range')
axes[2].axhline(y=table_z, color='blue', linestyle='--', alpha=0.5, label='Table height')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Paddle Z')
axes[2].set_title('Paddle Z vs Time\n(height)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
static_path = '/root/unitree_rl_lab/scripts/reference_motion_trajectory.png'
plt.savefig(static_path, dpi=150)
plt.close()
print(f"静态图已保存: {static_path}")
