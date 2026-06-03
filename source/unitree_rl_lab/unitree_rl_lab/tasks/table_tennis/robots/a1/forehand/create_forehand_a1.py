"""Create A1 forehand reference motions (middle).

Based on A1's init_state as ready pose, swing designed within A1 joint limits.
init_state: [1.56, -0.12, -1.70, 1.50, 2.03, 0.00, -0.39]

A1 limits: yb1[-1.04,3.14], yb2[-3.14,0.26], yb3[-2.758,2.758],
           yb4[-1.92,1.92], yb5[-2.758,2.758], yb6[-1.57,1.57], yb7[-2.758,2.758]
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

FPS = 30
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])

A1_LIMITS = np.array([
    [-1.04,  3.14],   # yb1 shoulder_pitch
    [-3.14,  0.26],   # yb2 shoulder_roll
    [-2.758, 2.758],  # yb3 shoulder_yaw
    [-1.92,  1.92],   # yb4 elbow
    [-2.758, 2.758],  # yb5 wrist_roll
    [-1.57,  1.57],   # yb6 wrist_pitch
    [-2.758, 2.758],  # yb7 wrist_yaw
])

# v2: 加大挥拍幅度，匹配X1运动趋势
# X1关键变化: yb1 +0.45, yb4 -1.0(伸肘), yb5 -0.9(wrist snap), yb6 +0.75(wrist pitch)
# A1 init: [1.56, -0.12, -1.70, 1.50, 2.03, 0.00, -0.39]
MIDDLE = [
    #  时间    yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.000, [+1.560, -0.120, -1.700, +1.500, +2.030, +0.000, -0.390]),  # ready (init_state)
    (0.300, [+1.400, -0.200, -1.700, +1.650, +2.400, +0.200, -0.390]),  # backswing (yb4↑弯肘, yb5↑蓄力)
    (0.400, [+1.350, -0.200, -1.650, +1.700, +2.500, +0.300, -0.390]),  # windup peak
    (0.475, [+2.000, -0.050, -1.750, +0.500, +1.100, -0.700, -0.390]),  # HIT (yb1+0.45, yb4-1.0, yb5-0.9, yb6-0.7)
    (0.550, [+2.100, -0.030, -1.800, +0.400, +1.000, -0.900, -0.390]),  # follow through
    (0.700, [+1.900, -0.080, -1.750, +0.900, +1.500, -0.400, -0.390]),  # recovery
    (0.900, [+1.560, -0.120, -1.700, +1.500, +2.030, +0.000, -0.390]),  # return
    (1.000, [+1.560, -0.120, -1.700, +1.500, +2.030, +0.000, -0.390]),  # ready hold
]


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
        lo, hi = A1_LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo or vmax > hi:
            print(f"  WARNING: joint_yb_{i+1} exceeds limits! "
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
    print(f"[{name}] {out_path} - {num_frames} frames")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate("middle", MIDDLE)
