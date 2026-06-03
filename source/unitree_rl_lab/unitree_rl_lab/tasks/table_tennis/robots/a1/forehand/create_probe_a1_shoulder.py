"""Shoulder wide-sweep probe for A1 (yb2 especially -- never tested before).

Across probes 2-5 the paddle pose barely changed under yb1,yb3,yb4,yb5,yb6,yb7:
the arm can't translate the paddle (radial singularity) and the wrist joints
(yb6,yb7) spin about ~the face normal so they can't reorient the face either.
The face normal (Link_yb_paddle local +Y) is stuck at ~(0.64,-0.74,0.15), pointing
too sideways (-Y); the reward wants it ~(+0.99,+0.07,-0.11) toward target.

GAP: yb2 (shoulder_roll) was held at Pc in EVERY probe. It is the prime candidate
to yaw the whole arm and remove the -Y, and it has the longest lever for +X
translation. This probe widely sweeps yb2, then yb1, then yb3 (rest held at Pc),
logging both paddle position and face normal.

Output: probe_a1_shoulder.npz
Run: play_pure_ref --npz probe_a1_shoulder.npz --ball_preset middle --video --video_length 400
"""

import os
import numpy as np

FPS = 30
NUM_FRAMES = 105
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])
A1_LIMITS = np.array([
    [-1.04, 3.14], [-3.14, 0.26], [-2.758, 2.758], [-1.92, 1.92],
    [-2.758, 2.758], [-1.57, 1.57], [-2.758, 2.758],
])
Pc = np.array([1.769, -0.762, -1.600, 1.138, 0.484, -0.418, 1.043], dtype=np.float32)

START = 31
# (joint_index, frame_start, frame_end, val_start, val_end)
SWEEPS = [
    (1, 31, 54, -2.50, +0.20),   # yb2 shoulder_roll (NEVER tested) - full range
    (0, 57, 78, +3.00, -1.00),   # yb1 - wide
    (2, 81, 102, -2.70, +1.00),  # yb3 - wide
]


def generate():
    dof = np.tile(Pc, (NUM_FRAMES, 1)).astype(np.float32)
    for j, fs, fe, v0, v1 in SWEEPS:
        for f in range(fs, fe + 1):
            if 0 <= f < NUM_FRAMES:
                a = (f - fs) / (fe - fs)
                dof[f, j] = v0 + a * (v1 - v0)
    ok = True
    for i in range(7):
        lo, hi = A1_LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo - 1e-4 or vmax > hi + 1e-4:
            print(f"  WARNING joint_yb_{i+1}: [{vmin:.3f},{vmax:.3f}] vs [{lo:.3f},{hi:.3f}]")
            ok = False
    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, "probe_a1_shoulder.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[probe-shoulder] {out_path} - {NUM_FRAMES} frames")
    for j, fs, fe, v0, v1 in SWEEPS:
        print(f"  yb{j+1} sweep f{fs}..f{fe} (t={fs-START}..{fe-START}): {v0:+.2f} -> {v1:+.2f}")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate()
