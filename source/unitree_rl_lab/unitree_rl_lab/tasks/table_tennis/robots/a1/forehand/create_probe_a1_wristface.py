"""Wrist-face orientation sweep for A1.

Decisive finding so far: the arm (yb1-4) barely TRANSLATES the paddle (radial
singularity / compact-wrist geometry), but the BALL is returned by REFLECTING off
the paddle face. The codebase defines the face normal as Link_yb_paddle local +Y
(observations.racket_normal / rewards.racket_face_toward_target). At contact pose
Pc the normal points (0.65,-0.74,0.20) -- too sideways (-Y); the reward wants it
pointing from the racket toward target world (+0.7,0,0.9) ~ (+0.99,+0.07,-0.11).

This probe holds the arm (yb1,yb2,yb3,yb4) at Pc and SWEEPS the wrist joints
yb6 (pitch) then yb7 (yaw) across wide ranges, holding the others at Pc. The
analysis reads upper_body_dof from this npz (frame->commanded angle) and the
logged face normal (paddle_traj Y columns) to find the wrist angles whose face
normal best aligns with the target direction. Those become the contact pose.

Output: probe_a1_wristface.npz
Run: play_pure_ref --npz probe_a1_wristface.npz --ball_preset middle --video --video_length 400
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
# (joint_index, frame_start, frame_end, val_start, val_end) -- linear sweeps, others at Pc
SWEEPS = [
    (5, 33, 64, -1.45, +1.45),   # yb6 wrist_pitch across its range
    (6, 67, 102, -1.20, +2.60),  # yb7 wrist_yaw across a wide range
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
    out_path = os.path.join(OUTPUT_DIR, "probe_a1_wristface.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[probe-wristface] {out_path} - {NUM_FRAMES} frames")
    for j, fs, fe, v0, v1 in SWEEPS:
        print(f"  yb{j+1} sweep frames f{fs}..f{fe} (t={fs-START}..{fe-START}): {v0:+.2f} -> {v1:+.2f}")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate()
