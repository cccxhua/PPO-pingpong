"""Retract-to-midrange FK probe for A1.

Why: probe2/probe3 proved the contact pose Pc sits on the arm's +X reach
boundary (radial singularity) -- every joint translates the paddle <0.09 m/rad,
so no swing built on Pc can give the ball +X speed. This probe RETRACTS the arm
into the dexterous mid-workspace (bend elbow yb4, lower shoulder yb1 to keep
height) to a new baseline Pc2, then bumps each joint at Pc2 with sustained holds.
If a strong joint (yb1/yb3, 28 Nm, long lever) now shows a MUCH larger +X
translation, that confirms Pc2 is off the singularity and is a viable contact
region; we then design the forward swing so contact happens while extending from
~Pc2 toward Pc (extension velocity = +X paddle speed).

The script PRINTS Pc2 (commanded) and the exact timesteps to read (t = frame-31,
because play_pure_ref starts playback at frame ~31 and loops the clip).

Output: probe_a1_retract.npz
Run: play_pure_ref --npz probe_a1_retract.npz --ball_preset middle --video --video_length 400
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

# Retracted mid-workspace pose: bend elbow (yb4 +0.60 = flex/pull back), lower
# shoulder pitch (yb1 -0.30) to keep paddle height while pulling toward base.
RETRACT = np.array([-0.30, 0.0, 0.0, +0.60, 0.0, 0.0, 0.0], dtype=np.float32)
Pc2 = (Pc + RETRACT).astype(np.float32)

RAMP = 2
HOLD = 6  # measure last 3

# bumps applied ON TOP of Pc2: (joint_index, amplitude)
BUMPS = [
    (0, +0.40),   # yb1 shoulder_pitch
    (2, +0.50),   # yb3 shoulder_yaw
    (3, -0.40),   # yb4 elbow EXTEND (the forward-swing direction)
    (4, +0.50),   # yb5 wrist_roll
    (5, +0.50),   # yb6 wrist_pitch
    (6, +0.50),   # yb7 wrist_yaw
]

START = 31  # first played frame


def rc_up(i, n):
    return 0.5 * (1.0 - np.cos(np.pi * i / n))


def generate():
    dof = np.tile(Pc, (NUM_FRAMES, 1)).astype(np.float32)
    sched = []  # (label, joint, measure_frames)

    f = START
    # ramp Pc -> Pc2
    for i in range(RAMP + 1):
        if f < NUM_FRAMES:
            dof[f] = Pc + RETRACT * rc_up(i, RAMP)
        f += 1
    # hold Pc2 baseline
    base_meas = []
    for i in range(HOLD):
        if f < NUM_FRAMES:
            dof[f] = Pc2
            if i >= HOLD - 3:
                base_meas.append(f)
        f += 1
    sched.append(("BASE_Pc2", None, base_meas))

    # each bump
    for j, amp in BUMPS:
        # ramp up from Pc2
        for i in range(1, RAMP + 1):
            if f < NUM_FRAMES:
                dof[f] = Pc2.copy()
                dof[f, j] = Pc2[j] + amp * rc_up(i, RAMP)
            f += 1
        meas = []
        for i in range(HOLD):
            if f < NUM_FRAMES:
                dof[f] = Pc2.copy()
                dof[f, j] = Pc2[j] + amp
                if i >= HOLD - 3:
                    meas.append(f)
            f += 1
        sched.append((f"yb{j+1} amp{amp:+.2f}", j, meas))
        # ramp down to Pc2
        for i in range(1, RAMP + 1):
            if f < NUM_FRAMES:
                dof[f] = Pc2.copy()
                dof[f, j] = Pc2[j] + amp * (1.0 - rc_up(i, RAMP))
            f += 1

    # limit check
    ok = True
    for i in range(7):
        lo, hi = A1_LIMITS[i]
        vmin, vmax = dof[:, i].min(), dof[:, i].max()
        if vmin < lo or vmax > hi:
            print(f"  WARNING joint_yb_{i+1}: [{vmin:.3f},{vmax:.3f}] vs [{lo:.3f},{hi:.3f}]")
            ok = False

    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, "probe_a1_retract.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[probe-retract] {out_path} - {NUM_FRAMES} frames, last played frame f{f-1}")
    print(f"  Pc  = {np.round(Pc,3).tolist()}")
    print(f"  Pc2 = {np.round(Pc2,3).tolist()}  (retract: yb1{RETRACT[0]:+.2f}, yb4{RETRACT[3]:+.2f})")
    print("  READ MAP (t = frame-31):")
    for label, j, frames in sched:
        ts = [fr - START for fr in frames]
        print(f"    {label:14s}: frames {frames}  t={ts}")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate()
