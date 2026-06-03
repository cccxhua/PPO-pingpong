"""Sweep yb_5/yb_6/yb_7 at RIGHT PIN to find face orientation that clears net.

Ball hits paddle (v14 confirmed 5/5), but post-hit vx is weak (-1.14) because face
points mostly +Y. Need to rotate face toward -X for proper return.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/sweep_right_face.py --task X1-TableTennis
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


def make_keyframes(yb5_pin, yb6_pin, yb7_pin):
    return [
        (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # hold
        (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),  # windup
        (0.475, [+1.500, -0.250, -1.800, +0.200, yb5_pin, yb6_pin, yb7_pin]),  # PIN start
        (0.540, [+1.500, -0.250, -1.800, +0.200, yb5_pin, yb6_pin, yb7_pin]),  # PIN hold
        (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),  # follow
        (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # return
    ]


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

    def run_trial(keyframes):
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
            min_gap=min_gap, hit=hit_detected,
            bvx=ball_vel_post[0], bvy=ball_vel_post[1], bvz=ball_vel_post[2],
            z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid,
        )

    # Sweep yb_5, yb_6, yb_7 at PIN
    # Baseline: yb_5=-0.700, yb_6=-0.500, yb_7=+0.800
    # MIDDLE reference: yb_5=-0.900, yb_6=-0.400, yb_7=+1.000
    print(f"\n{'='*100}")
    print(f"  SWEEP: yb_5/yb_6/yb_7 at RIGHT PIN (hit_phase={HIT_PHASE}, ball_arrive={BALL_ARRIVE_TIME_EST})")
    print(f"{'='*100}")
    print(f"\n{'yb5':>6} {'yb6':>6} {'yb7':>6} | {'gap':>5} {'HIT':>3} | "
          f"{'bvx':>6} {'bvy':>6} {'bvz':>6} | {'zn':>6} {'xb':>6} {'CLR':>3}")
    print("-" * 90)

    configs = [
        # Baseline
        (-0.700, -0.500, +0.800),
        # Adjust yb_5 (wrist_roll) — more negative = more like MIDDLE
        (-0.900, -0.500, +0.800),
        (-1.100, -0.500, +0.800),
        (-0.500, -0.500, +0.800),
        # Adjust yb_6 (wrist_pitch) — less negative = face tilts up
        (-0.700, -0.300, +0.800),
        (-0.700, -0.100, +0.800),
        (-0.700, +0.000, +0.800),
        (-0.700, -0.700, +0.800),
        # Adjust yb_7 (wrist_yaw) — more positive = like MIDDLE
        (-0.700, -0.500, +1.000),
        (-0.700, -0.500, +1.200),
        (-0.700, -0.500, +0.500),
        # Combined: closer to MIDDLE
        (-0.900, -0.400, +1.000),
        (-0.900, -0.300, +1.000),
        (-1.000, -0.400, +0.800),
        (-1.000, -0.300, +1.000),
        # More aggressive face rotation
        (-0.900, -0.200, +1.000),
        (-0.900, -0.100, +1.000),
        (-1.100, -0.300, +1.000),
        (-0.700, -0.200, +1.200),
        (-0.500, -0.200, +1.200),
    ]

    results = []
    for yb5, yb6, yb7 in configs:
        kf = make_keyframes(yb5, yb6, yb7)
        try:
            r = run_trial(kf)
            results.append((yb5, yb6, yb7, r))
        except Exception as e:
            print(f"{yb5:>+6.2f} {yb6:>+6.2f} {yb7:>+6.2f} | ERROR: {e}")
            continue

        zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
        xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
        clr = "Y" if r['clears'] and r['valid'] else "N"
        hit = "Y" if r['hit'] else "N"
        print(f"{yb5:>+6.2f} {yb6:>+6.2f} {yb7:>+6.2f} | {r['min_gap']:>5.3f} {hit:>3} | "
              f"{r['bvx']:>+6.2f} {r['bvy']:>+6.2f} {r['bvz']:>+6.2f} | {zn:>6} {xb:>6} {clr:>3}")

    # Summary: best configs
    print(f"\n=== BEST CONFIGS (sorted by -bvx, filtering hit=True) ===")
    hit_results = [(y5, y6, y7, r) for y5, y6, y7, r in results if r['hit']]
    hit_results.sort(key=lambda x: x[3]['bvx'])  # most negative bvx first
    for y5, y6, y7, r in hit_results[:5]:
        zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
        xb = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else "  -  "
        clr = "Y" if r['clears'] and r['valid'] else "N"
        print(f"  yb5={y5:+.2f} yb6={y6:+.2f} yb7={y7:+.2f}: bv=({r['bvx']:+.2f},{r['bvy']:+.2f},{r['bvz']:+.2f}) zn={zn} xb={xb} CLR={clr}")

    # Configs that clear net
    clr_results = [(y5, y6, y7, r) for y5, y6, y7, r in results if r['clears'] and r['valid']]
    if clr_results:
        print(f"\n=== CLEARS NET + VALID ({len(clr_results)} configs) ===")
        for y5, y6, y7, r in clr_results:
            print(f"  yb5={y5:+.2f} yb6={y6:+.2f} yb7={y7:+.2f}: bv=({r['bvx']:+.2f},{r['bvy']:+.2f},{r['bvz']:+.2f}) "
                  f"zn={r['z_at_net']:+.2f} xb={r['x_bounce']:+.2f}")
    else:
        print(f"\n  *** NO CONFIG CLEARS NET — need stronger -vx or more +vz")
        print(f"  *** Best z_at_net among hits:")
        hit_by_zn = sorted(hit_results, key=lambda x: x[3]['z_at_net'] if x[3]['z_at_net'] is not None else -99, reverse=True)
        for y5, y6, y7, r in hit_by_zn[:5]:
            zn = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else "  -  "
            print(f"    yb5={y5:+.2f} yb6={y6:+.2f} yb7={y7:+.2f}: bvx={r['bvx']:+.2f} bvz={r['bvz']:+.2f} zn={zn}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
