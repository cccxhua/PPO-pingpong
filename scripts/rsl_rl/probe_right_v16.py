"""Probe RIGHT v16: Use MIDDLE v74 keyframes (proven to clear net) with yb_2 shift.

Key insight: MIDDLE v74 achieves bvx=-2.26, bvz=+3.33, z_at_net=+1.40, CLR+VALID.
The RIGHT ball is at y≈-0.08, only 8cm away from MIDDLE's y≈0.
Instead of fighting arm geometry, just shift yb_2 slightly negative to reach the ball.

Sweep yb_2 shift from 0 to -0.30 to find minimum shift that reaches y=-0.08.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_v16.py --task X1-TableTennis
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

# MIDDLE v74 keyframes (proven CLR+VALID)
MIDDLE_V74 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
]

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def make_shifted(yb2_delta):
    """Shift all yb_2 values by delta."""
    shifted = []
    for t, joints in MIDDLE_V74:
        new_joints = joints.copy()
        new_joints[1] += yb2_delta
        shifted.append((t, new_joints))
    return shifted


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
        paddle_pos_at_hit = np.zeros(3)
        ball_pos_at_hit = np.zeros(3)

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
                paddle_pos_at_hit = (p - env_origin).copy()
                ball_pos_at_hit = (bp - env_origin).copy()

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
            paddle_pos=paddle_pos_at_hit, ball_pos=ball_pos_at_hit,
            z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid,
        )

    print(f"\n{'='*120}")
    print(f"  RIGHT v16: MIDDLE v74 + yb_2 shift (hit_phase={HIT_PHASE}, ball_arrive={BALL_ARRIVE_TIME_EST})")
    print(f"  Ball: pos={BALL_POS}, vel={BALL_VEL}, predicted_y@robot={BALL_POS[1]+BALL_VEL[1]*(1.85/BALL_VEL[0]):.3f}")
    print(f"{'='*120}")
    print(f"\n{'Δyb2':>6} | {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'pvx':>6} {'pvz':>6} | {'bvx':>6} {'bvy':>6} {'bvz':>6} | "
          f"{'zn':>6} {'xb':>6} {'CLR':>3} | {'pad_y':>6} {'ball_y':>6}")
    print("-" * 110)

    # Sweep yb_2 shift from 0 (MIDDLE original) to -0.35
    shifts = [0.00, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35]

    for delta in shifts:
        keys = make_shifted(delta)
        lim_viol = check_limits(keys)
        if lim_viol:
            print(f"{delta:>+6.2f} | LIMIT VIOLATION: {lim_viol}")
            continue

        try:
            r = run_variant(keys)
        except Exception as e:
            print(f"{delta:>+6.2f} | ERROR: {e}")
            continue

        zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
        xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
        clr = "Y" if r['clears'] and r['valid'] else "N"
        hit = "Y" if r['hit'] else "N"

        print(f"{delta:>+6.2f} | {r['min_gap']:>5.3f} {r['min_gap_t']:>5.3f} {hit:>3} | "
              f"{r['pvx']:>+6.2f} {r['pvz']:>+6.2f} | "
              f"{r['bvx']:>+6.2f} {r['bvy']:>+6.2f} {r['bvz']:>+6.2f} | "
              f"{zn:>6} {xb:>6} {clr:>3} | "
              f"{r['paddle_pos'][1]:>+6.3f} {r['ball_pos'][1]:>+6.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
