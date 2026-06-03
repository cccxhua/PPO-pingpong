"""Probe RIGHT v15c: fix spline overshoot by adding anchor keyframe.

Problem: CubicSpline(clamped) overshoots when yb_6 transitions from -0.800→-0.950→-0.300.
Fix: add hold keyframe at t=0.300 to anchor the spline flat before windup.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_v15c.py --task X1-TableTennis
"""

import argparse
import sys

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

import isaaclab_tasks  # noqa
import unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

BALL_POS = np.array([-0.35, -0.03, 1.3])
BALL_VEL = np.array([3.5, -0.10, 0.5])
BALL_ARRIVE_TIME_EST = 0.55
DURATION = 1.0
HIT_PHASE = 0.475
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

# v15c: add "hold" at t=0.300 to anchor spline, prevent yb_6 undershoot.
# Structure: ready(0.0) → hold(0.30) → windup(0.42) → swing(0.50) → follow(0.63) → return(1.0)
# The hold-to-windup interval is 120ms (gradual change)
# The windup-to-swing interval is 80ms (explosive snap)

VARIANTS = {
    "A": {
        "desc": "Base: hold anchor + yb_6 snap -0.85→-0.30, yb_5=-1.10",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # anchor
            (0.420, [+1.300, -0.250, -2.150, +0.350, -0.300, -0.850, +0.800]),  # windup (minimal yb6 change)
            (0.500, [+1.650, -0.250, -1.650, +0.050, -1.100, -0.300, +0.800]),  # swing
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),  # follow
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "B": {
        "desc": "A + bigger yb_1 sweep (1.20→1.75 = push forward)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.400, -0.300, -0.850, +0.800]),  # pull back more
            (0.500, [+1.750, -0.250, -1.600, +0.000, -1.100, -0.300, +0.800]),  # push forward more
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "C": {
        "desc": "A + yb_6 snap to +0.10 (max loft snap Δ=0.95)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.300, -0.250, -2.150, +0.350, -0.300, -0.850, +0.800]),
            (0.500, [+1.650, -0.250, -1.650, +0.050, -1.100, +0.100, +0.800]),  # yb6=+0.10
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "D": {
        "desc": "B + yb_6 snap to +0.10 (strong shoulder + wrist loft)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.400, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, -0.250, -1.600, +0.000, -1.100, +0.100, +0.800]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "E": {
        "desc": "D + yb_4 elbow snap 0.50→-0.10 (extend arm)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.500, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, -0.250, -1.600, -0.100, -1.100, +0.100, +0.800]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "F": {
        "desc": "D + yb_7 at swing =+1.20 (more face toward -X)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.400, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, -0.250, -1.600, +0.000, -1.100, +0.100, +1.200]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "G": {
        "desc": "D + yb_2=0.00 at swing (less abduction, like MIDDLE)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.400, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, +0.000, -1.600, +0.000, -1.100, +0.100, +0.800]),
            (0.630, [+1.450, -0.150, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "H": {
        "desc": "D + yb_3=-1.400 at swing (more horizontal sweep)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.200, +0.400, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, -0.250, -1.400, +0.000, -1.100, +0.100, +0.800]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "I": {
        "desc": "All-in: B shoulder + C wrist + elbow snap + yb_7=1.0",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.150, -0.250, -2.200, +0.500, -0.300, -0.850, +0.800]),
            (0.500, [+1.800, -0.250, -1.500, -0.100, -1.100, +0.100, +1.000]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
            (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
        ],
    },
    "J": {
        "desc": "D but yb_5=-0.900 at swing (MIDDLE exact value)",
        "keys": [
            (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.300, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),
            (0.420, [+1.200, -0.250, -2.150, +0.400, -0.300, -0.850, +0.800]),
            (0.500, [+1.750, -0.250, -1.600, +0.000, -0.900, +0.100, +0.800]),
            (0.630, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.700, +0.800]),
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
    a = 0.5 * G
    b = -vz
    c = TABLE_Z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        return z_at_net, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2 * a)
    x_bounce = x0 + vx * t_bounce
    clears_net = z_at_net > NET_Z and t_net < t_bounce
    valid = x_bounce < NET_X
    return z_at_net, x_bounce, clears_net, valid


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    violations = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo - 0.01 or y[:, i].max() > hi + 0.01:
            violations.append(f"yb_{i+1}=[{y[:, i].min():+.3f},{y[:, i].max():+.3f}] vs [{lo:.3f},{hi:.3f}]")
    return violations


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
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]
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
        min_gap = 1e9
        min_gap_t = -1
        hit_detected = False
        ball_vel_post = np.zeros(3)
        ball_pos_post = np.zeros(3)
        paddle_vel_at_hit = np.zeros(3)

        for step in range(n_steps):
            t = step * sim_dt
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy()
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            if gap < min_gap:
                min_gap = gap
                min_gap_t = t

            if not hit_detected and bv[0] < -0.5 and t > 0.3:
                hit_detected = True
                ball_vel_post = bv.copy()
                ball_pos_post = bp.copy() - env_origin
                paddle_vel_at_hit = pv.copy()

        if not hit_detected:
            ball_vel_post = bv.copy()
            ball_pos_post = bp.copy() - env_origin

        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            ball_pos_post[0], ball_pos_post[2], ball_vel_post[0], ball_vel_post[2])

        return dict(
            min_gap=min_gap, min_gap_t=min_gap_t, hit=hit_detected,
            bvx=ball_vel_post[0], bvy=ball_vel_post[1], bvz=ball_vel_post[2],
            pvx=paddle_vel_at_hit[0], pvy=paddle_vel_at_hit[1], pvz=paddle_vel_at_hit[2],
            z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid,
        )

    print(f"\n{'='*120}")
    print(f"  RIGHT v15c ACTIVE SWING w/ anchor (hit_phase={HIT_PHASE}, ball_arrive={BALL_ARRIVE_TIME_EST})")
    print(f"{'='*120}")
    print(f"\n{'ID':<4} {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'pvx':>6} {'pvy':>6} {'pvz':>6} | {'bvx':>6} {'bvy':>6} {'bvz':>6} | "
          f"{'zn':>6} {'xb':>6} {'CLR':>3} | desc")
    print("-" * 120)

    for var_id in sorted(VARIANTS.keys()):
        var = VARIANTS[var_id]
        keys = var["keys"]
        desc = var["desc"]

        lim_viol = check_limits(keys)
        if lim_viol:
            print(f" {var_id:<3} LIMIT VIOLATION: {lim_viol}")
            continue

        try:
            r = run_variant(keys)
        except Exception as e:
            print(f" {var_id:<3} ERROR: {e}")
            continue

        zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
        xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
        clr = "Y" if r['clears'] and r['valid'] else "N"
        hit = "Y" if r['hit'] else "N"

        print(f" {var_id:<3} {r['min_gap']:>5.3f} {r['min_gap_t']:>5.3f} {hit:>3} | "
              f"{r['pvx']:>+6.2f} {r['pvy']:>+6.2f} {r['pvz']:>+6.2f} | "
              f"{r['bvx']:>+6.2f} {r['bvy']:>+6.2f} {r['bvz']:>+6.2f} | "
              f"{zn:>6} {xb:>6} {clr:>3} | {desc[:50]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
