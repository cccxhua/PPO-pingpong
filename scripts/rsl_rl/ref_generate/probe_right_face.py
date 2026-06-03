"""Probe RIGHT face normal: sweep yb_5/yb_6 at PIN to find clears-net configuration.

Based on MIDDLE's v67-v72 tuning experience:
  - yb_5 (wrist_roll) rotates the face normal direction
  - yb_6 (wrist_pitch) tilts face up/down and provides snap speed

Sweeps yb_5 PIN ∈ [-1.1, -0.5] and yb_6 PIN ∈ [-0.8, +0.2] at hit_phase=0.475.
Reports: ball post-hit velocity, z_at_net, x_bounce, clears+valid.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_face.py --task X1-TableTennis
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

# Baseline RIGHT keyframes v2: face toward opponent (yb_6 base=-1.0) + smooth arm
RIGHT_BASE = [
    (0.000, [+1.000, -0.880, -2.100, +1.000, +0.000, -1.000, +1.400]),  # ready
    (0.200, [+1.050, -0.880, -2.050, +0.750, -0.200, -1.050, +1.400]),  # mid
    (0.300, [+1.100, -0.880, -2.000, +0.550, -0.350, -1.150, +1.400]),  # windup
    (0.375, [+1.300, -0.880, -1.850, +0.450, -0.900, -0.400, +1.400]),  # PIN (idx=3)
    (0.450, [+1.350, -0.880, -2.000, +0.550, -0.300, -0.500, +1.400]),  # snap
    (0.600, [+1.200, -0.880, -2.100, +0.800, +0.000, -1.000, +1.400]),  # follow
    (0.800, [+1.000, -0.880, -2.100, +1.000, +0.000, -1.000, +1.400]),  # return
    (1.000, [+1.000, -0.880, -2.100, +1.000, +0.000, -1.000, +1.400]),  # hold
]
PIN_IDX = 3  # index of PIN keyframe

BALL_POS = np.array([-0.35, -0.03, 1.3])
BALL_VEL = np.array([3.5, -0.10, 0.5])

BALL_ARRIVE_TIME_EST = 0.5205
HIT_PHASE = 0.475
DURATION = 1.0
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

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
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo - 0.01 or y[:, i].max() > hi + 0.01:
            return False
    return True


def make_variant(yb5_pin, yb6_pin):
    """Create keyframes variant with modified yb_5 and yb_6 at PIN."""
    keys = []
    for t, vals in RIGHT_BASE:
        keys.append((t, list(vals)))
    keys[PIN_IDX][1][4] = yb5_pin  # yb_5
    keys[PIN_IDX][1][5] = yb6_pin  # yb_6
    return keys


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

    print(f"\nenv_origin = {env_origin}")
    print(f"Sweeping yb_5/yb_6 at PIN (idx={PIN_IDX}, t={RIGHT_BASE[PIN_IDX][0]})")
    print(f"Baseline PIN: yb_5={RIGHT_BASE[PIN_IDX][1][4]}, yb_6={RIGHT_BASE[PIN_IDX][1][5]}")
    print(f"Ball: pos={BALL_POS}, vel={BALL_VEL}")

    def run_variant(keys):
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        initial_phase = (HIT_PHASE - BALL_ARRIVE_TIME_EST / DURATION) % 1.0
        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(initial_phase)
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device)
        ball_state[0, 0:3] += scene.env_origins[0]
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
        ball_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        sim_dt = float(env.unwrapped.sim.get_physics_dt())
        n_steps = int(1.5 / sim_dt)

        min_gap = 1e9
        hit_detected = False
        ball_vel_post = np.zeros(3)
        ball_pos_post = np.zeros(3)

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
            bp = ball.data.root_pos_w[0].cpu().numpy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            if gap < min_gap:
                min_gap = gap

            if not hit_detected and bv[0] < -0.5 and t > 0.3:
                hit_detected = True
                ball_vel_post = bv.copy()
                ball_pos_post = bp.copy() - env_origin

        if not hit_detected:
            ball_vel_post = bv.copy()
            ball_pos_post = bp.copy() - env_origin

        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            ball_pos_post[0], ball_pos_post[2], ball_vel_post[0], ball_vel_post[2])

        return dict(
            min_gap=min_gap, hit_detected=hit_detected,
            ball_vel_post=ball_vel_post,
            z_at_net=z_at_net, x_bounce=x_bounce,
            clears=clears, valid=valid,
        )

    # Sweep ranges (based on MIDDLE's v65-v72 experience)
    yb5_values = [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5]
    yb6_values = [-0.8, -0.6, -0.4, -0.2, 0.0, +0.2]

    print(f"\n{'yb5':>6} {'yb6':>6} {'gap':>5} {'HIT':>4} "
          f"{'bvx':>6} {'bvy':>6} {'bvz':>6} "
          f"{'zn':>6} {'xb':>6} {'CLR':>4} {'lim':>4}")
    print("=" * 85)

    best = None
    for yb5 in yb5_values:
        for yb6 in yb6_values:
            keys = make_variant(yb5, yb6)
            lim_ok = check_limits(keys)
            if not lim_ok:
                print(f"{yb5:>+6.2f} {yb6:>+6.2f}   --- LIMIT VIOLATION ---")
                continue

            try:
                r = run_variant(keys)
            except Exception as e:
                print(f"{yb5:>+6.2f} {yb6:>+6.2f}   ERROR: {e}")
                continue

            bv = r['ball_vel_post']
            zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
            xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
            clr = "Y" if r['clears'] and r['valid'] else "N"
            hit = "Y" if r['hit_detected'] else "N"

            print(f"{yb5:>+6.2f} {yb6:>+6.2f} {r['min_gap']:>5.3f} {hit:>4} "
                  f"{bv[0]:>+6.2f} {bv[1]:>+6.2f} {bv[2]:>+6.2f} "
                  f"{zn:>6} {xb:>6} {clr:>4}   {'✓' if lim_ok else '✗'}")

            if r['clears'] and r['valid'] and r['hit_detected']:
                if best is None or abs(r['ball_vel_post'][0]) > abs(best['bvx']):
                    best = dict(yb5=yb5, yb6=yb6, bvx=bv[0], bvz=bv[2],
                                z_at_net=r['z_at_net'], x_bounce=r['x_bounce'])

    print(f"\n=== Best CLEARS+VALID ===")
    if best:
        print(f"  yb_5 PIN = {best['yb5']:+.2f}, yb_6 PIN = {best['yb6']:+.2f}")
        print(f"  ball vx={best['bvx']:+.2f}, vz={best['bvz']:+.2f}")
        print(f"  z@net={best['z_at_net']:+.2f}, x_bounce={best['x_bounce']:+.2f}")
    else:
        print("  No variant clears net! Need to expand sweep range or adjust other joints.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
