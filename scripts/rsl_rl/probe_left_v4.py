"""LEFT v4: Multi-joint sweep to reach ball at Y=+0.07, Z=1.08.

Strategy: Since yb_2 is limited (+0.314), use yb_3 (shoulder_yaw) to rotate
the arm swing plane toward +Y. yb_3 has wide limits [-2.78, +2.76].

MIDDLE v74 yb_3 at hit window ≈ -1.85 to -1.98. Making yb_3 less negative
(toward -1.5 to -1.6) should rotate the arm outward, pushing paddle toward +Y.

Also probe: different yb_4 (elbow) values can change the reach and Z.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_left_v4.py --task X1-TableTennis
"""

import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="X1-TableTennis")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if "--headless" not in sys.argv:
    args_cli.headless = True
    sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import isaaclab_tasks, unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

BALL_POS = np.array([-0.35, +0.03, 1.3])
BALL_VEL = np.array([3.5, +0.10, 0.5])
DURATION = 1.0
HIT_PHASE = 0.475
BALL_ARRIVE_TIME_EST = 0.55

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])

# MIDDLE v74 baseline
MIDDLE_BASE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
]


def make_variant(yb3_shift=0.0, yb2_shift=0.0, yb1_shift=0.0, yb4_shift=0.0):
    """Apply shifts to hit-window frames only (t=0.300-0.700)."""
    keys = []
    for t, a in MIDDLE_BASE:
        na = list(a)
        if 0.25 <= t <= 0.75:
            na[0] += yb1_shift
            na[1] = min(na[1] + yb2_shift, 0.310)
            na[2] += yb3_shift
            na[3] += yb4_shift
        keys.append((t, na))
    return keys


VARIANTS = {
    "BASE": ("MIDDLE as-is", make_variant()),
    "A": ("yb3+0.15", make_variant(yb3_shift=+0.15)),
    "B": ("yb3+0.30", make_variant(yb3_shift=+0.30)),
    "C": ("yb3+0.45", make_variant(yb3_shift=+0.45)),
    "D": ("yb3+0.30 yb2+0.10", make_variant(yb3_shift=+0.30, yb2_shift=+0.10)),
    "E": ("yb3+0.30 yb1-0.10", make_variant(yb3_shift=+0.30, yb1_shift=-0.10)),
    "F": ("yb3+0.45 yb1-0.10", make_variant(yb3_shift=+0.45, yb1_shift=-0.10)),
    "G": ("yb3+0.45 yb2+0.10", make_variant(yb3_shift=+0.45, yb2_shift=+0.10)),
    "H": ("yb3+0.45 yb2+0.10 yb1-0.10", make_variant(yb3_shift=+0.45, yb2_shift=+0.10, yb1_shift=-0.10)),
    "I": ("yb3+0.60", make_variant(yb3_shift=+0.60)),
    "J": ("yb3+0.60 yb2+0.10", make_variant(yb3_shift=+0.60, yb2_shift=+0.10)),
}


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    viol = []
    for i in range(7):
        lo, hi = LIMITS[i]
        mn, mx = y[:, i].min(), y[:, i].max()
        if mn < lo - 0.01 or mx > hi + 0.01:
            viol.append(f"yb{i+1}")
    return viol


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    device = env.unwrapped.device
    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_ids = [robot.find_joints(f"joint_yb_{i}")[0][0] for i in range(1, 8)]
    env_origin = scene.env_origins[0].cpu().numpy()
    sim_dt = float(env.unwrapped.sim.get_physics_dt())

    def run_motion(keyframes):
        times = np.array([k[0] for k in keyframes])
        angs = np.array([k[1] for k in keyframes], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        initial_phase = (HIT_PHASE - BALL_ARRIVE_TIME_EST / DURATION) % 1.0
        q0 = spline(initial_phase)
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device) + scene.env_origins[0]
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
        ball_state[0, 10:13] = torch.zeros(3, device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        n_steps = int(1.2 / sim_dt)
        min_gap, min_t = 1e9, -1
        hit = False
        pv_hit, pp_hit, bp_hit = np.zeros(3), np.zeros(3), np.zeros(3)
        bv_post = np.zeros(3)
        face_hit = np.zeros(3)

        for step in range(n_steps):
            t = step * sim_dt
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            ft = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                ft[0, jid] = float(target[k])
            robot.set_joint_position_target(ft, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy() - env_origin
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            if gap < min_gap:
                min_gap, min_t = gap, t
                pp_hit, bp_hit, pv_hit = p.copy(), bp.copy(), pv.copy()
                pq = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()
                rot = Rotation.from_quat([pq[1], pq[2], pq[3], pq[0]])
                face_hit = rot.apply([1, 0, 0])

            if not hit and bv[0] < -0.5 and t > 0.3:
                hit = True
                bv_post = bv.copy()

        if not hit:
            bv_post = bv.copy()
        return min_gap, min_t, hit, pp_hit, bp_hit, pv_hit, face_hit, bv_post

    print(f"\n{'='*120}")
    print(f"  LEFT v4: yb_3 (shoulder_yaw) sweep to push paddle toward +Y")
    print(f"  Ball target @ t=0.55: X≈1.11, Y≈+0.07, Z≈1.08")
    print(f"{'='*120}")
    print(f"\n{'ID':<5} {'desc':<28} {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'pp_x':>6} {'pp_y':>6} {'pp_z':>6} | "
          f"{'Δy':>5} {'Δz':>5} | "
          f"{'pvx':>6} {'pvz':>6} | "
          f"{'bvx':>6} {'bvz':>6} | lim")
    print("-" * 130)

    for vid in ["BASE", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        desc, keys = VARIANTS[vid]
        lim = check_limits(keys)
        lim_str = ",".join(lim) if lim else "ok"
        mg, mt, hit, pp, bp, pv, fn, bv = run_motion(keys)
        hit_s = "Y" if hit else "N"
        dy = pp[1] - bp[1]
        dz = pp[2] - bp[2]
        print(f" {vid:<4} {desc:<28} {mg:>5.3f} {mt:>5.3f} {hit_s:>3} | "
              f"{pp[0]:>+6.3f} {pp[1]:>+6.3f} {pp[2]:>+6.3f} | "
              f"{dy:>+5.3f} {dz:>+5.3f} | "
              f"{pv[0]:>+6.2f} {pv[2]:>+6.2f} | "
              f"{bv[0]:>+6.2f} {bv[2]:>+6.2f} | {lim_str}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
