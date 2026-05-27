"""用正运动学生成 X1 右臂挥拍动作的3D动画视频。

不依赖 Isaac Sim 渲染，用 matplotlib 3D 动画 + 简化的DH模型。
运行: python scripts/animate_arm_motion.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def trans(x, y, z):
    T = np.eye(4)
    T[0,3], T[1,3], T[2,3] = x, y, z
    return T


def forward_kinematics(joints):
    """
    X1 右臂简化正运动学。
    joints: [yb1, yb2, yb3, yb4, yb5, yb6, yb7] in rad

    X1结构 (from USD/URDF):
      - 肩膀相对于 base 的偏移约: x=-0.05, y=-0.2, z=+0.45 (base at x=1.5, z=height)
      - yb1: shoulder_pitch (Y axis) 正=前抬
      - yb2: shoulder_roll (X axis) 负=外展
      - yb3: shoulder_yaw (Z axis)
      - 上臂长度 ~0.28m
      - yb4: elbow_pitch (Y axis) 正=屈肘
      - 前臂长度 ~0.25m
      - yb5: wrist_roll (Z axis)
      - yb6: wrist_pitch (Y axis)
      - yb7: wrist_yaw (Z axis)
      - 球拍偏移 ~0.16m
    """
    yb1, yb2, yb3, yb4, yb5, yb6, yb7 = joints

    # Robot base position (x=1.5, y=0, z depends on lift)
    # joint_lift = -0.09, robot height in cfg
    robot_z = 0.76  # approximate base height from cfg

    # Shoulder offset from base
    shoulder_offset = trans(-0.05, -0.20, 0.45)

    # Chain of transforms
    T = trans(1.5, 0.0, robot_z)  # robot base in world
    T = T @ shoulder_offset       # to shoulder

    points = [T[:3, 3].copy()]  # shoulder

    # yb1: pitch around Y (positive = forward/up)
    T = T @ rot_y(yb1)
    # yb2: roll around X (negative = outward)
    T = T @ rot_x(yb2)
    # yb3: yaw around Z
    T = T @ rot_z(yb3)

    # Upper arm (along -X in robot frame after rotation, approx)
    T = T @ trans(-0.28, 0.0, 0.0)
    points.append(T[:3, 3].copy())  # elbow

    # yb4: elbow pitch (Y axis)
    T = T @ rot_y(yb4)

    # Forearm
    T = T @ trans(-0.25, 0.0, 0.0)
    points.append(T[:3, 3].copy())  # wrist

    # yb5: wrist roll (Z)
    T = T @ rot_z(yb5)
    # yb6: wrist pitch (Y)
    T = T @ rot_y(yb6)
    # yb7: wrist yaw (Z)
    T = T @ rot_z(yb7)

    # Paddle
    T = T @ trans(-0.16, 0.0, 0.0)
    points.append(T[:3, 3].copy())  # paddle center

    return np.array(points)


# Load NPZ
npz_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/x1/forehand/forehand_upper.npz"
)
data = np.load(npz_path, allow_pickle=True)
fps = float(data["fps"])
dof_frames = data["upper_body_dof"]
num_frames = dof_frames.shape[0]
duration = num_frames / fps

print(f"Loaded: {num_frames} frames, {fps} fps, {duration:.2f}s")

# Use actual Isaac Sim paddle positions from our earlier run for calibration
# The FK model is approximate - let's use it for visualization
actual_paddle_data = np.array([
    [1.356, -0.361, 1.174],  # frame 0
    [1.385, -0.352, 1.175],  # frame 3
    [1.334, -0.366, 1.173],  # frame 6
    [1.255, -0.377, 1.167],  # frame 9
    [1.185, -0.375, 1.161],  # frame 12
    [1.106, -0.361, 1.152],  # frame 15
    [1.010, -0.324, 1.141],  # frame 18
    [0.973, -0.304, 1.139],  # frame 21
    [1.056, -0.345, 1.146],  # frame 24
    [1.217, -0.377, 1.164],  # frame 27
    [1.356, -0.361, 1.174],  # frame 30
])

# Compute FK for all frames (for arm visualization)
all_points = []
for i in range(num_frames):
    pts = forward_kinematics(dof_frames[i])
    all_points.append(pts)
all_points = np.array(all_points)

# Calibrate: adjust FK paddle to match actual data
# Use offset correction based on frame 0
fk_paddle_0 = all_points[0, -1]
actual_paddle_0 = actual_paddle_data[0]
offset = actual_paddle_0 - fk_paddle_0

# Apply offset to all FK points
all_points_calibrated = all_points + offset[None, None, :]

# Interpolate actual paddle data to all frames
from scipy.interpolate import interp1d
actual_frames_idx = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
interp_func = interp1d(actual_frames_idx, actual_paddle_data, axis=0, kind='cubic', fill_value='extrapolate')
paddle_all_frames = interp_func(np.arange(num_frames))

# For the arm links, use calibrated FK for shoulder/elbow/wrist, actual data for paddle
# Adjust intermediate points proportionally
for i in range(num_frames):
    # Keep shoulder fixed, interpolate elbow and wrist
    actual_paddle = paddle_all_frames[i]
    fk_shoulder = all_points_calibrated[i, 0]
    # Scale the arm to end at the actual paddle position
    fk_paddle = all_points_calibrated[i, -1]
    if np.linalg.norm(fk_paddle - fk_shoulder) > 0.01:
        scale = np.linalg.norm(actual_paddle - fk_shoulder) / np.linalg.norm(fk_paddle - fk_shoulder)
    else:
        scale = 1.0
    for j in range(1, 4):
        all_points_calibrated[i, j] = fk_shoulder + (all_points_calibrated[i, j] - fk_shoulder) * min(scale, 1.5)
    # Force paddle to actual position
    all_points_calibrated[i, -1] = actual_paddle

# Create animation
fig = plt.figure(figsize=(16, 8))

# View 1: Side view (X-Z)
ax1 = fig.add_subplot(121, projection='3d')
# View 2: Top view (X-Y)
ax2 = fig.add_subplot(122, projection='3d')

# Table
table_x = np.array([-1.37, 1.37, 1.37, -1.37, -1.37])
table_y = np.array([-0.7625, -0.7625, 0.7625, 0.7625, -0.7625])
table_z = 0.745

# Net
net_y = np.linspace(-0.9, 0.9, 10)
net_z_bottom = np.full(10, 0.76)
net_z_top = np.full(10, 0.76 + 0.15)

# Ball arrival zone
ball_zone_corners_x = [1.0, 1.4, 1.4, 1.0, 1.0]
ball_zone_corners_y = [-0.15, -0.15, 0.15, 0.15, -0.15]
ball_zone_z = 1.1

phases = ["Ready", "Backswing", "Backswing", "Backswing", "Hit", "Hit",
          "Hit", "Follow", "Follow", "Follow", "Recovery"]

def setup_ax(ax, title, elev, azim):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_xlim(0.5, 1.8)
    ax.set_ylim(-0.8, 0.5)
    ax.set_zlim(0.6, 1.5)
    ax.view_init(elev=elev, azim=azim)
    # Draw table
    ax.plot(table_x, table_y, [table_z]*5, 'b-', alpha=0.3, linewidth=1)
    # Draw net
    ax.plot([0, 0], [-0.9, 0.9], [table_z, table_z], 'g-', alpha=0.2)
    # Ball zone
    ax.plot(ball_zone_corners_x, ball_zone_corners_y, [ball_zone_z]*5, 'r--', alpha=0.3)
    # Robot base (simplified)
    ax.scatter([1.5], [0], [0.76], c='gray', s=50, marker='s', alpha=0.5)

lines1 = []
lines2 = []
points1 = []
points2 = []
paddle_trail1 = None
paddle_trail2 = None
title_text = None

def init():
    global lines1, lines2, points1, points2, paddle_trail1, paddle_trail2, title_text
    ax1.clear()
    ax2.clear()
    setup_ax(ax1, "Side View", elev=15, azim=-75)
    setup_ax(ax2, "Front View", elev=5, azim=-170)

    lines1 = [ax1.plot([], [], [], 'b-', linewidth=3)[0] for _ in range(3)]
    lines2 = [ax2.plot([], [], [], 'b-', linewidth=3)[0] for _ in range(3)]
    points1 = [ax1.plot([], [], [], 'ko', markersize=6)[0] for _ in range(4)]
    points2 = [ax2.plot([], [], [], 'ko', markersize=6)[0] for _ in range(4)]
    paddle_trail1, = ax1.plot([], [], [], 'r-', alpha=0.4, linewidth=1)
    paddle_trail2, = ax2.plot([], [], [], 'r-', alpha=0.4, linewidth=1)
    title_text = fig.suptitle('', fontsize=14)
    return []

def animate(frame_idx):
    global lines1, lines2, points1, points2, paddle_trail1, paddle_trail2, title_text

    idx = frame_idx % num_frames
    pts = all_points_calibrated[idx]

    ax1.clear()
    ax2.clear()
    setup_ax(ax1, "Side View (X-Z plane)", elev=15, azim=-75)
    setup_ax(ax2, "Front View (looking from table)", elev=5, azim=-170)

    # Draw arm links
    for ax in [ax1, ax2]:
        # Links: shoulder->elbow->wrist->paddle
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=4)
        # Joints
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], c='black', s=60, zorder=5)  # shoulder
        ax.scatter(pts[1, 0], pts[1, 1], pts[1, 2], c='darkblue', s=50, zorder=5)  # elbow
        ax.scatter(pts[2, 0], pts[2, 1], pts[2, 2], c='blue', s=40, zorder=5)  # wrist
        ax.scatter(pts[3, 0], pts[3, 1], pts[3, 2], c='red', s=80, marker='D', zorder=5)  # paddle

        # Trail of paddle
        trail_start = max(0, idx - 10)
        trail_pts = all_points_calibrated[trail_start:idx+1, -1]
        if len(trail_pts) > 1:
            ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2], 'r-', alpha=0.5, linewidth=2)

    t = idx / fps
    phase_idx = min(idx * len(phases) // num_frames, len(phases) - 1)
    phase = phases[phase_idx]
    yb4_val = dof_frames[idx, 3]
    fig.suptitle(f't={t:.2f}s | Phase: {phase} | yb4={yb4_val:.2f} rad | '
                 f'Paddle: ({pts[3,0]:.2f}, {pts[3,1]:.2f}, {pts[3,2]:.2f})', fontsize=12)

    return []

total_frames = num_frames * 3  # 3 loops
anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames,
                     interval=int(1000/fps), blit=False)

output_path = '/root/unitree_rl_lab/scripts/x1_forehand_motion.mp4'
writer = FFMpegWriter(fps=int(fps), metadata=dict(title='X1 Forehand Motion'))
print("Saving animation...")
anim.save(output_path, writer=writer, dpi=100)
plt.close()
print(f"Video saved: {output_path}")
