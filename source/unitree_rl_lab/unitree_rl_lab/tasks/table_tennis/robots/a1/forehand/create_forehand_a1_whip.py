"""A1 forehand WHIP exploration -- sequential proximal->distal "whip" swing.

Goal (user): compress the swing, drive a kinematic-chain WHIP where the shoulder
(yb1) leads and momentum is passed down to the wrist (yb5/yb6) so the paddle TIP
reaches peak +X speed AT the ball-contact window, returning the ball harder.

Measured baseline (v7): paddle peak +X speed only ~0.30 m/s and it peaks BEFORE
the ball arrives; by the contact window (steps 26-33 ~= npz frames 57-64) the
paddle has slowed to ~0.12 m/s. Face normal (local +Y) is best (faceX~0.88,
faceY~+0.02) around the mid-wrist values (yb5~0.27, yb6~-0.73) but drifts to
faceY~-0.34 (ball pushed -Y) as the wrist opens further.

Whip strategy here:
  - LOAD (f31-46): cock everything -- yb1 LOW (loaded shoulder), yb4 HIGH (flexed
    elbow), yb5 LOW / yb6 DEEP (cocked wrist). yb3 PINNED -1.600, yb2/-0.762, yb7/1.043.
  - FIRE sequentially so distal joints peak LAST, near the contact window (~f60):
      yb1 (shoulder) accelerates first (f46->f54),
      yb4 (elbow) extends next   (f54->f59),
      yb5/yb6 (wrist) SNAP last   (f58->f62)  -> tip peak speed at contact.
  - FOLLOW then a clean monotonic return to ready.

Output: forehand_middle_a1_whip.npz  (does NOT touch v7's forehand_middle_a1_v2.npz)
Run: python scripts/rsl_rl/play_pure_ref.py --npz <abs>/forehand_middle_a1_whip.npz \
        --ball_preset middle --video --video_length 400 --output_dir logs/pure_ref/whip1
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

FPS = 30
NUM_FRAMES = 105
DURATION = (NUM_FRAMES - 1) / FPS
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JOINT_NAMES = np.array([
    "joint_yb_1", "joint_yb_2", "joint_yb_3",
    "joint_yb_4", "joint_yb_5", "joint_yb_6", "joint_yb_7",
])
A1_LIMITS = np.array([
    [-1.04, 3.14], [-3.14, 0.26], [-2.758, 2.758], [-1.92, 1.92],
    [-2.758, 2.758], [-1.57, 1.57], [-2.758, 2.758],
])
READY = [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]

# WHIP keyframes -- BEST = v5 (restored). Velocity ladder (cmd yb4 peak -> actual
# elbow vel AT contact step40 -> ball return vx):
#   v3 -1.65 -> 0.30 -> 0.48 m/s ; v5 -3.70 -> 1.62 -> 0.87 m/s ; v6 -10.55 -> 1.02
#   -> 0.81 m/s (over-aggressive: peak velocity landed AFTER contact, step62).
# => v5 is the practical optimum. The hard ceiling is TORQUE/INERTIA, not yb4's
#   20 rad/s velocity limit: with only ~0.7 s / ~1 rad of runway before the ball,
#   28 Nm spins the elbow to ~1.6 rad/s by contact -> paddle ~0.9 m/s. The ball
#   needs ~3-4 m/s to clear the net (x=0, z>0.94) from the contact point (x=-1.36),
#   so pure-reference cannot clear with these actuators; v5 is the best seed.
#
# v5 = on-ball geometry (wrist CLOSED yb5=0.206/yb6=-0.827, yb2/3/7 locked, yb1
# lift) + a smooth SUSTAINED elbow extension whose velocity peaks at contact
# (step ~37-40 == npz frame ~53). mapping: npz_frame = 31.2 + 0.5955*step.
WHIP = [
    #  time(s)  yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.0000, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f0  ready / post-swing hold
    (1.3333, [+1.420, -0.762, -1.600, +1.750, +0.206, -0.827, +1.043]),  # f40 LOAD (step~15): yb1 LOW, ELBOW COCKED HIGH, wrist closed
    (1.6000, [+1.720, -0.762, -1.600, +1.350, +0.206, -0.827, +1.043]),  # f48 drive (step~28): elbow extending
    (1.7667, [+1.980, -0.762, -1.600, +0.850, +0.206, -0.827, +1.043]),  # f53 CONTACT (step~37): elbow extending HARD through ball, wrist closed
    (1.9333, [+2.180, -0.762, -1.600, +0.350, +0.206, -0.827, +1.043]),  # f58 FOLLOW (step~45): keep elbow torque saturated past contact
    (2.2667, [+2.050, -0.762, -1.600, +1.100, +0.206, -0.827, +1.043]),  # f68 unwind (step~62)
    (2.7333, [+1.800, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f82 return (step~85)
    (3.2000, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f96 ready
    (DURATION, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f104 ready
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
        if vmin < lo - 1e-4 or vmax > hi + 1e-4:
            print(f"  WARNING joint_yb_{i+1}: [{vmin:.3f},{vmax:.3f}] vs [{lo:.3f},{hi:.3f}]")
            ok = False

    vel = np.diff(dof, axis=0) * FPS
    # report per-joint peak velocity and its frame, to verify the proximal->distal sequence
    for i in range(7):
        pf = int(np.argmax(np.abs(vel[:, i])))
        print(f"  yb{i+1} peak |vel| {vel[pf, i]:+.2f} rad/s at frame {pf}")
    pf = int(np.argmax(np.abs(vel).max(1)))
    print(f"  overall peak |vel| at frame {pf}: {np.round(vel[pf], 2).tolist()} rad/s")

    base_y = np.zeros(NUM_FRAMES, dtype=np.float32)
    out_path = os.path.join(OUTPUT_DIR, f"forehand_{name}.npz")
    np.savez(out_path, fps=np.float64(FPS), upper_body_dof=dof,
             base_y=base_y, joint_names=JOINT_NAMES)
    print(f"[{name}] {out_path} - {NUM_FRAMES} frames")
    if ok:
        print("  Joint limits OK")


# WHIP_FLAT -- same swing as v5 but the CONTACT-segment wrist is re-aimed per the
# frame-56 face probe to flatten the paddle face from lofted (+Z) toward horizontal
# +X. Probe sensitivities at contact: yb7 dX=+0.71 / dZ=-0.63 per rad, yb5 dZ=-0.62
# per rad. So raise yb7 (1.043 -> ~1.45) and yb5 (0.206 -> ~0.55) ONLY in the drive/
# contact/follow keyframes (f48/f53/f58), ramped in from the f40 load and back out by
# f68, leaving load + return identical to v5. Target: faceX up, faceZ down (flat +X).
WHIP_FLAT = [
    #  time(s)  yb1     yb2     yb3     yb4     yb5     yb6     yb7
    (0.0000, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f0  ready
    (1.3333, [+1.420, -0.762, -1.600, +1.750, +0.206, -0.827, +1.043]),  # f40 LOAD (wrist still v5)
    (1.6000, [+1.500, -0.762, -1.600, +1.350, +0.400, -0.827, +1.250]),  # f48 drive: wrist OPENING (flatten), yb1 -0.22 lower
    (1.7667, [+1.760, -0.762, -1.600, +0.850, +0.550, -0.827, +1.450]),  # f53 CONTACT: face flat +X, yb1 -0.22 lower (~11cm down)
    (1.9333, [+1.960, -0.762, -1.600, +0.350, +0.550, -0.827, +1.450]),  # f58 FOLLOW: hold flat face, yb1 -0.22 lower
    (2.2667, [+1.830, -0.762, -1.600, +1.100, +0.300, -0.827, +1.150]),  # f68 unwind (wrist easing back), yb1 -0.22 lower
    (2.7333, [+1.800, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f82 return (v5)
    (3.2000, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f96 ready
    (DURATION, [+1.769, -0.762, -1.600, +1.445, +0.206, -0.827, +1.043]),  # f104 ready
]


if __name__ == "__main__":
    generate("middle_a1_whip", WHIP)
    generate("middle_a1_whip_flat", WHIP_FLAT)
