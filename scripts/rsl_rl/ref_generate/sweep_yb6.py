"""Sweep yb_6 PIN values to find direct-clear face angle.

Generates temp npz, runs play_pure_ref for 150 steps each, reports clearing stats.
"""
import subprocess
import os
import sys
import numpy as np
from scipy.interpolate import CubicSpline

JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])
FPS = 30
LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])

BASE_KEYFRAMES = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.607, -0.315, -1.100,  +1.000]),
    (0.475, [+1.737, +0.103, -1.850, +0.257, -1.500, -0.300,  +1.000]),  # PIN - yb_6 will be varied
    (0.550, [+1.787, +0.103, -1.979, +0.207, -0.165, -0.445,  +1.000]),
    (0.700, [+1.750, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def generate_npz(keyframes, out_path):
    times = np.array([kf[0] for kf in keyframes])
    angles = np.array([kf[1] for kf in keyframes], dtype=np.float32)
    duration = times[-1]
    num_frames = int(duration * FPS) + 1
    t_interp = np.linspace(0, duration, num_frames)
    cs = CubicSpline(times, angles, bc_type="clamped")
    dof = cs(t_interp).astype(np.float32)

    for i in range(7):
        lo, hi = LIMITS[i]
        if dof[:, i].min() < lo or dof[:, i].max() > hi:
            return False

    base_y = np.full(num_frames, 0.0, dtype=np.float32)
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof, base_y=base_y, joint_names=JOINT_NAMES)
    return True


def test_variant(npz_path, arrive_time=0.50, steps=150):
    cmd = [
        "/isaac-sim/python.sh", "scripts/rsl_rl/play_pure_ref.py",
        "--task", "X1-TableTennis",
        "--ball_preset", "middle",
        "--npz", npz_path,
        "--video_length", str(steps),
        "--output_dir", "/tmp/sweep_tmp",
        "--arrive_time", str(arrive_time),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    trials = 0
    clears = 0
    directs = 0
    for line in output.split("\n"):
        if "[trial" in line:
            trials += 1
            if "cleared=True" in line:
                clears += 1
            if "direct=True" in line:
                directs += 1
    return trials, clears, directs


if __name__ == "__main__":
    os.chdir("/root/unitree_rl_lab")

    # Sweep yb_6 at PIN (index 3 in keyframes, joint index 5)
    yb6_values = [-0.800, -0.600, -0.400, -0.200, 0.000, +0.200, +0.400]

    print(f"{'yb6_PIN':>10} {'trials':>6} {'clears':>6} {'direct':>6} {'clr%':>6} {'dir%':>6}")
    print("-" * 52)

    for yb6 in yb6_values:
        kf = [list(row) for row in BASE_KEYFRAMES]
        kf[3] = (0.475, [+1.737, +0.103, -1.850, +0.257, -1.500, yb6, +1.000])
        kf_tuples = [(t, v) for t, v in kf]

        npz_path = f"/tmp/sweep_yb6_{yb6:+.3f}.npz"
        if not generate_npz(kf_tuples, npz_path):
            print(f"{yb6:>10.3f}  LIMIT VIOLATION - skipped")
            continue

        trials, clears, directs = test_variant(npz_path)
        clr_pct = clears / trials * 100 if trials > 0 else 0
        dir_pct = directs / trials * 100 if trials > 0 else 0
        print(f"{yb6:>10.3f} {trials:>6} {clears:>6} {directs:>6} {clr_pct:>5.1f}% {dir_pct:>5.1f}%")
