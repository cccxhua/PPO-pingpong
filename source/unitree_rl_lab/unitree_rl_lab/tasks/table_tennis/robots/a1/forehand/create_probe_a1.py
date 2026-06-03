"""FK probe clip for A1 paddle at the contact pose Pc.

Why: the A1 asset is USD-only (no URDF/pxr), so we cannot compute the paddle
forward kinematics offline. We could not predict which joints move the paddle in
world +X (toward the opponent) nor which joint values point the paddle FACE in
+X -- e.g. yb1 turned out to lift the paddle UP, not forward, and the face was
found pointing DOWN at contact. This probe measures the contact-point Jacobian
empirically.

How: hold ALL joints at the contact pose Pc for the whole clip, EXCEPT bump ONE
joint at a time inside a dedicated frame window (smooth raised-cosine, returns to
Pc between windows). play_pure_ref logs paddle world pos + face normal per step,
so reading the paddle pose at each bump's peak vs the Pc baseline gives that
joint's effect on (px,py,pz) and on the face normal (nx,ny,nz).

Playback note: play_pure_ref starts at phase ~0.2985 (frame ~31) and plays to
frame 104, so every probe window must live in frames 31..104.

Output: probe_a1.npz (run via: play_pure_ref --npz probe_a1.npz --ball_preset middle)
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

Pc = np.array([1.769, -0.762, -1.600, 1.138, 0.484, -0.418, 1.043], dtype=np.float32)

# (joint_index, peak_frame, half_width_frames, amplitude)
# windows are separated so the paddle returns to the Pc baseline between bumps
PROBE = [
    (0, 40,  4, +0.30),   # yb1 shoulder_pitch
    (2, 51,  4, +0.30),   # yb3 shoulder_yaw
    (3, 62,  4, -0.30),   # yb4 elbow (extend direction)
    (4, 73,  4, +0.40),   # yb5 wrist_roll
    (5, 84,  4, +0.40),   # yb6 wrist_pitch
    (6, 95,  4, +0.40),   # yb7 wrist_yaw
]


def generate():
    dof = np.tile(Pc, (NUM_FRAMES, 1)).astype(np.float32)
    for j, fc, hw, amp in PROBE:
        for f in range(fc - hw, fc + hw + 1):
            if 0 <= f < NUM_FRAMES:
                w = 0.5 * (1.0 - np.cos(np.pi * (f - (fc - hw)) / hw))  # 0->1->0 over [fc-hw, fc+hw]
                dof[f, j] = Pc[j] + amp * w
    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, "probe_a1.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[probe] {out_path} - {NUM_FRAMES} frames")
    print("  windows (joint peak@frame amp): " +
          ", ".join(f"yb{j+1}@f{fc}={amp:+.2f}" for j, fc, hw, amp in PROBE))


if __name__ == "__main__":
    generate()
