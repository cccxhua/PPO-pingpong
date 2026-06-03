"""Generate a sweep motion for yb_2: from min to max limit, ALL other joints at 0."""
import os
import numpy as np

FPS = 30
DURATION = 2.0
NUM_FRAMES = int(DURATION * FPS) + 1

JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])

YB2_MIN = -3.081
YB2_MAX = +0.314

# All joints at 0, only yb_2 sweeps
dof = np.zeros((NUM_FRAMES, 7), dtype=np.float32)
dof[:, 1] = np.linspace(YB2_MIN, YB2_MAX, NUM_FRAMES).astype(np.float32)

base_y = np.zeros(NUM_FRAMES, dtype=np.float32)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_yb2.npz")
np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof, base_y=base_y, joint_names=JOINT_NAMES)
print(f"Generated: {out_path}")
print(f"  yb_2 sweep: {YB2_MIN} -> {YB2_MAX}, all others = 0")
