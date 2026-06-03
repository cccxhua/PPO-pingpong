"""Create A1 forehand reference motion v2 (middle) — a STRONGER scoring swing.

Motivation: the in-use forehand_middle_a1.npz is a weak swing — only yb_4/5/6
move (ROM 0.6-0.9 rad) and the shoulder (yb_1) is locked, so the paddle's
linear speed at contact is too low to return the ball over the net.

This v2 keeps the SAME phase structure as the in-use clip:
  - 105 frames @ 30 fps (indices 0..104, duration 104/30 s)
  - HIT at frame 50  (== hit_phase 0.475, matches env_cfg)
so the existing ball-alignment config (hit_phase=0.475, ball_arrive_time_est)
still lines up.

It is anchored at the in-use clip's REST pose [1.769, ...] (NOT the script's
[1.56, ...]) so the validated interception geometry is preserved. We only:
  (1) amplify the proven yb_4/yb_5/yb_6 snap (down/up/up), and
  (2) add shoulder yb_1 forward drive + small yb_3 yaw to push the paddle in +X.

A1 limits: yb1[-1.04,3.14], yb2[-3.14,0.26], yb3[-2.758,2.758],
           yb4[-1.92,1.92], yb5[-2.758,2.758], yb6[-1.57,1.57], yb7[-2.758,2.758]

Output: forehand_middle_a1_v2.npz  (does NOT overwrite the original).
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

FPS = 30
NUM_FRAMES = 105                      # match in-use clip
DURATION = (NUM_FRAMES - 1) / FPS     # 104/30 = 3.4667 s
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

REST = [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]  # in-use clip frame 0

# v2-I (v7-BASED, LATER + STRONGER). The user picked v7 (logs/pure_ref/
# middle_a1_v2_7) as the best motion so far, but said the CONTACT SPEED is too
# low and the swing should move a FEW FRAMES LATER. v7's played window is npz
# frames 31..104 (phase-aligned init starts at frame 31); reconstructed from its
# joints.txt TARGET columns, v7's swing is weak: yb1 only rises +0.08 (1.769 ->
# 1.850 peak at ~f69) and yb4 extends slowly (1.36 -> 1.00, ~ -0.47 rad/s). The
# orientation joints that make the face point +X are HELD: yb2 const -0.762,
# yb3 PINNED -1.600 (probe6: moving it flips the face to -Y = no return),
# yb7 const 1.043, and the wrists reach yb5~0.484 / yb6~-0.418 at contact.
#
# This version KEEPS v7's held-orientation strategy but:
#   (1) shifts the contact/peak ~4 frames LATER (v7 peak f64/f69 -> f68 here) so
#       the high-velocity part of the swing lands on the ball, and
#   (2) greatly INCREASES contact speed via the power joints only: yb1 (shoulder
#       pitch, 28 Nm, long lever) driven to 2.15 (~5x v7's rise) and yb4 (elbow)
#       extended to 0.77 (~2.5x v7's speed), concentrated into f54->f68 so peak
#       +X paddle velocity spans the contact window.
# yb5/yb6 are moved to their contact-hold values by ~f54 and HELD through contact
# so the face stays pointed +X *while the paddle is moving*.
# play_pure_ref logs joints.txt + ball_traj.txt, so one run pins the true contact
# frame and confirms the moving-contact face normal.
# REVERTED TO v7. The user rejected the faster/aggressive variants ("太慢"/"不行")
# and confirmed v7 (logs/pure_ref/middle_a1_v2_7) is the good motion. The npz on
# disk (forehand_middle_a1_v2.npz) is the EXACT v7 reference, reconstructed
# frame-by-frame from v7's joints.txt commanded columns (played window = frames
# 31..104, phase-aligned init starts at frame 31). The keyframes below are v7's
# landmark poses so re-running this script reproduces v7 closely (spline approx);
# for the byte-exact clip use the log-reconstructed npz already on disk.
#
# v7 character: orientation joints HELD (yb2 const -0.762, yb3 PINNED -1.600 ->
# face +X, yb7 const 1.043; wrists reach yb5~0.485 / yb6~-0.418 at contact). The
# power stroke is GENTLE: yb1 only 1.79->1.85 (peak ~f69), yb4 1.61->1.00.
Pc = [+1.769, -0.762, -1.600, +1.138, +0.484, -0.418, +1.043]  # contact pose (face -> +X)
MIDDLE_V2 = [
    #  time(s)  yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.0000, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f0  ready (== frozen post-swing hold)
    (1.0333, [+1.786, -0.762, -1.600, +1.613, +0.031, -1.084, +1.043]),  # f31 windup / play start (yb4 high, yb6 deep)
    (1.5333, [+1.761, -0.762, -1.600, +1.359, +0.294, -0.697, +1.043]),  # f46 unwinding toward contact
    (1.8667, [+1.770, -0.762, -1.600, +1.136, +0.485, -0.417, +1.043]),  # f56 contact approach (face set)
    (2.1333, [+1.830, -0.762, -1.600, +1.043, +0.484, -0.418, +1.043]),  # f64 contact
    (2.3000, [+1.850, -0.762, -1.600, +1.001, +0.484, -0.417, +1.043]),  # f69 PEAK (yb1 max, yb4 min)
    (2.4667, [+1.833, -0.762, -1.600, +1.048, +0.454, -0.464, +1.043]),  # f74 follow
    (2.7000, [+1.780, -0.762, -1.600, +1.209, +0.372, -0.587, +1.043]),  # f81 return
    (3.0333, [+1.750, -0.762, -1.600, +1.380, +0.269, -0.739, +1.043]),  # f91 return
    (3.4333, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f103 ready
    (DURATION, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f104 ready (== frame 0)
]


def generate(name, keyframes):
    times = np.array([kf[0] for kf in keyframes])
    angles = np.array([kf[1] for kf in keyframes], dtype=np.float32)

    t_interp = np.linspace(0, DURATION, NUM_FRAMES)
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

    # peak angular speed near hit (frame 50)
    vel = np.diff(dof, axis=0) * FPS
    pf = int(np.argmax(np.abs(vel).max(1)))
    print(f"  hit frame 50 dof = {np.round(dof[50], 3).tolist()}")
    print(f"  peak |vel| at frame {pf}: {np.round(vel[pf], 2).tolist()} rad/s")

    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, f"forehand_{name}.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[{name}] {out_path} - {NUM_FRAMES} frames, hit@frame50 (phase {50/(NUM_FRAMES-1):.3f})")
    if ok:
        print("  Joint limits OK")


if __name__ == "__main__":
    generate("middle_a1_v2", MIDDLE_V2)
