"""RIGHT v17: upward sweep strategy.

Key insight: with yb_2=-0.25, forward swing (yb_1 increase) = paddle moves DOWN.
Solution: reverse the swing — pre-extend forward, then sweep BACKWARD (yb_1 decrease)
which moves paddle UPWARD. Ball hits paddle while it's moving up.

Also: yb_3 sweep from less-negative to more-negative gives backward/upward tip motion.

Structure:
- Extend paddle ahead of ball (phase 0.400)
- Sweep backward/upward through ball arrival (phase 0.500-0.550)
- Follow through (phase 0.650)

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_v17.py --task X1-TableTennis
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
import isaaclab_tasks, unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

BALL_POS = np.array([-0.35, -0.03, 1.3])
BALL_VEL = np.array([3.5, -0.10, 0.5])
BALL_ARRIVE_TIME_EST = 0.55
DURATION = 1.0
HIT_PHASE = 0.475
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

# Strategy: at phase≈0.45 (ball arrival), paddle should be:
# 1. At X≈1.10, Y≈-0.06, Z≈1.03 (where ball is)
# 2. Moving UPWARD (+pvz) — achieved by yb_1 DECREASING
# 3. Face normal has -X component (to reflect ball toward opponent)
#
# Design: extend forward early, then retract/lift as ball arrives.
# The "hit" happens while paddle is in backward (upward) phase.

VARIANTS = {
    "A": {
        "desc": "Pre-extend@0.40 yb1=1.55, retract@0.52 yb1=1.35 (Δ=-0.20/120ms)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),  # extend (v14 PIN pos)
            (0.520, [+1.350, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),  # retract (upward)
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "B": {
        "desc": "A + bigger retraction: yb1 1.55→1.20 (Δ=-0.35, more upward)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),
            (0.520, [+1.200, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),  # big retract
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "C": {
        "desc": "A + yb_3 forward sweep at same time (compound: up+forward)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.900, +0.200, -0.700, -0.500, +0.800]),  # yb3 more neg
            (0.520, [+1.350, -0.250, -1.650, +0.200, -0.700, -0.500, +0.800]),  # yb3 forward sweep
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "D": {
        "desc": "B + yb_3 sweep -1.90→-1.60 (backward yb1 + forward yb3 = up+forward)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.900, +0.200, -0.700, -0.500, +0.800]),
            (0.520, [+1.200, -0.250, -1.600, +0.200, -0.700, -0.500, +0.800]),  # yb1 down + yb3 fwd
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "E": {
        "desc": "B + yb_6 snap -0.50→+0.20 during retract (wrist lifts paddle)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),
            (0.520, [+1.200, -0.250, -1.800, +0.200, -0.700, +0.200, +0.800]),  # yb6 snap up
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.500, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "F": {
        "desc": "D + E combined (all upward drivers: yb1 down + yb3 fwd + yb6 up)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.400, [+1.550, -0.250, -1.900, +0.200, -0.700, -0.500, +0.800]),
            (0.520, [+1.200, -0.250, -1.600, +0.200, -0.700, +0.200, +0.800]),
            (0.650, [+1.400, -0.250, -1.950, +0.300, -0.300, -0.500, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "G": {
        "desc": "v14 baseline (PIN plateau, for comparison)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),
            (0.475, [+1.500, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),
            (0.540, [+1.500, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),
            (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "H": {
        "desc": "v14 + yb_5=-1.10 at PIN (best face from earlier sweep)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),
            (0.475, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),
            (0.540, [+1.500, -0.250, -1.800, +0.200, -1.100, -0.500, +0.800]),
            (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
}

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def analyze_trajectory(x0, z0, vx, vz):
    if vx >= 0:
        return None, None, False, False
    t_net = x0 / (-vx)
    z_at_net = z0 + vz * t_net - 0.5 * G * t_net * t_net
    a, b, c = 0.5 * G, -vz, TABLE_Z - z0
    disc = b*b - 4*a*c
    if disc < 0:
        return z_at_net, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2*a)
    x_bounce = x0 + vx * t_bounce
    clears = z_at_net > NET_Z and t_net < t_bounce
    valid = x_bounce < NET_X
    return z_at_net, x_bounce, clears, valid


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, 1.0, 1001)
    y = cs(t_dense)
    viol = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo - 0.01 or y[:, i].max() > hi + 0.01:
            viol.append(f"yb{i+1}=[{y[:, i].min():+.3f},{y[:, i].max():+.3f}]")
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

    def run_variant(keyframes):
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
        bv_post, pv_hit = np.zeros(3), np.zeros(3)
        bp_post = np.zeros(3)

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

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy()
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))
            if gap < min_gap:
                min_gap, min_t = gap, t
            if not hit and bv[0] < -0.5 and t > 0.3:
                hit = True
                bv_post = bv.copy()
                bp_post = (bp - env_origin).copy()
                pv_hit = pv.copy()

        if not hit:
            bv_post = bv.copy()
            bp_post = (bp - env_origin).copy()
        zn, xb, clr, val = analyze_trajectory(bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return min_gap, min_t, hit, bv_post, pv_hit, zn, xb, clr and val

    print(f"\n{'='*110}")
    print(f"  RIGHT v17: UPWARD SWEEP (hit_phase={HIT_PHASE}, ball_arrive={BALL_ARRIVE_TIME_EST})")
    print(f"{'='*110}")
    print(f"\n{'ID':<3} {'gap':>5} {'t':>5} {'HIT':>3} | {'pvx':>6} {'pvy':>6} {'pvz':>6} | "
          f"{'bvx':>6} {'bvy':>6} {'bvz':>6} | {'zn':>6} {'xb':>6} {'CLR':>3} | desc")
    print("-" * 115)

    for vid in sorted(VARIANTS.keys()):
        var = VARIANTS[vid]
        lim = check_limits(var["keys"])
        if lim:
            print(f" {vid:<2} LIMIT: {lim}")
            continue
        try:
            mg, mt, hit, bv, pv, zn, xb, clr = run_variant(var["keys"])
        except Exception as e:
            print(f" {vid:<2} ERROR: {e}")
            continue
        zns = f"{zn:+.2f}" if zn is not None else "  -  "
        xbs = f"{xb:+.2f}" if xb is not None else "  -  "
        print(f" {vid:<2} {mg:>5.3f} {mt:>5.3f} {'Y' if hit else 'N':>3} | "
              f"{pv[0]:>+6.2f} {pv[1]:>+6.2f} {pv[2]:>+6.2f} | "
              f"{bv[0]:>+6.2f} {bv[1]:>+6.2f} {bv[2]:>+6.2f} | "
              f"{zns:>6} {xbs:>6} {'Y' if clr else 'N':>3} | {var['desc'][:45]}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
