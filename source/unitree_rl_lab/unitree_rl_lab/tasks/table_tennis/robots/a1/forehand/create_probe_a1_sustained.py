"""Sustained-hold FK probe for A1 paddle at contact pose Pc.

Why: the ±4-frame pulse probe (create_probe_a1.py) under-measures the static
position Jacobian because the damped joints (esp. stiffness=120 wrists) never
reach steady state during a brief pulse. This version holds ONE joint at a LARGE
offset for many frames so the paddle settles -> true static dPos/dOrient per rad.

Layout (playback starts at frame ~31 and loops the whole clip, so every window
lives in frames 31..104 and is played in order). Each joint: raised-cosine ramp
up (RAMP frames), HOLD at Pc+amp (HOLD frames, measure the last few), ramp down,
then a baseline settle gap before the next joint.

Output: probe_a1_sustained.npz
Run: play_pure_ref --npz probe_a1_sustained.npz --ball_preset middle --video --video_length 400
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

A1_LIMITS = np.array([
    [-1.04, 3.14], [-3.14, 0.26], [-2.758, 2.758], [-1.92, 1.92],
    [-2.758, 2.758], [-1.57, 1.57], [-2.758, 2.758],
])

RAMP = 3   # frames to ramp in/out (raised cosine)
HOLD = 7   # frames held at the offset (measure the LAST 3)

# (joint_index, hold_start_frame, amplitude)  -- spaced ~12 frames apart, all in 31..104
PROBE = [
    (0, 37, +0.60),   # yb1 shoulder_pitch
    (2, 49, +0.60),   # yb3 shoulder_yaw
    (3, 61, -0.60),   # yb4 elbow (extend)
    (4, 73, +0.60),   # yb5 wrist_roll
    (5, 85, +0.60),   # yb6 wrist_pitch
    (6, 97, +0.60),   # yb7 wrist_yaw
]


def generate():
    dof = np.tile(Pc, (NUM_FRAMES, 1)).astype(np.float32)
    for j, hs, amp in PROBE:
        # ramp up
        for i in range(RAMP):
            f = hs - RAMP + i
            if 0 <= f < NUM_FRAMES:
                w = 0.5 * (1.0 - np.cos(np.pi * i / RAMP))
                dof[f, j] = Pc[j] + amp * w
        # hold
        for i in range(HOLD):
            f = hs + i
            if 0 <= f < NUM_FRAMES:
                dof[f, j] = Pc[j] + amp
        # ramp down
        for i in range(RAMP):
            f = hs + HOLD + i
            if 0 <= f < NUM_FRAMES:
                w = 0.5 * (1.0 + np.cos(np.pi * i / RAMP))
                dof[f, j] = Pc[j] + amp * w

    ok = True
    for i in range(7):
        lo, hi = A1_LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo or vmax > hi:
            print(f"  WARNING: joint_yb_{i+1} exceeds limits! [{vmin:.3f},{vmax:.3f}] vs [{lo:.3f},{hi:.3f}]")
            ok = False

    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, "probe_a1_sustained.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[probe-sustained] {out_path} - {NUM_FRAMES} frames, RAMP={RAMP} HOLD={HOLD}")
    print("  windows (joint hold_start amp; measure last 3 hold frames):")
    for j, hs, amp in PROBE:
        meas = list(range(hs + HOLD - 3, hs + HOLD))  # last 3 hold frames -> t = f-31
        print(f"    yb{j+1}: hold f{hs}..f{hs+HOLD-1} amp{amp:+.2f}  measure frames {meas} (t={[f-31 for f in meas]})")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate()
