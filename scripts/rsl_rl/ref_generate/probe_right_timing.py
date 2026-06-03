"""Quick probe: sweep ball_arrive_time_est with MIDDLE keyframes + yb2 shift.

The core question: at what timing does MIDDLE actually hit this ball?
Sweep ball_arrive_time_est from 0.40 to 0.65 to find the optimal.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_right_timing.py --task X1-TableTennis
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
DURATION = 1.0
HIT_PHASE = 0.475
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

# MIDDLE v74 with yb2 shift -0.15 (to reach ball at y=-0.08)
KEYS = [
    (0.000, [+1.000, +0.150, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.048, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, -0.047, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, -0.047, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, -0.047, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, -0.050, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.150, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.150, -2.000, +1.400, +0.000, -1.000, +1.000]),
]


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

    times_kf = np.array([k[0] for k in KEYS])
    angs_kf = np.array([k[1] for k in KEYS], dtype=np.float64)
    spline = CubicSpline(times_kf, angs_kf, bc_type="clamped")

    def run(bat):
        initial_phase = (HIT_PHASE - bat / DURATION) % 1.0
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

        n_steps = int(1.5 / sim_dt)
        min_gap, min_t = 1e9, -1
        hit, bv_post, bp_post, pv_hit = False, np.zeros(3), np.zeros(3), np.zeros(3)

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
            if not hit and bv[0] < -0.5 and t > 0.2:
                hit = True
                bv_post, bp_post, pv_hit = bv.copy(), (bp - env_origin).copy(), pv.copy()

        if not hit:
            bv_post = bv.copy()
            bp_post = (bp - env_origin).copy()
        zn, xb, clr, val = analyze_trajectory(bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return min_gap, min_t, hit, bv_post, pv_hit, zn, xb, clr and val

    print(f"\n{'bat':>5} | {'gap':>5} {'t':>5} {'HIT':>3} | {'pvx':>6} {'pvz':>6} | {'bvx':>6} {'bvz':>6} | {'zn':>6} {'xb':>6} {'CLR':>3}")
    print("-" * 80)

    for bat in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65]:
        mg, mt, hit, bv, pv, zn, xb, clr = run(bat)
        zns = f"{zn:+.2f}" if zn is not None else "  -  "
        xbs = f"{xb:+.2f}" if xb is not None else "  -  "
        print(f"{bat:>5.2f} | {mg:>5.3f} {mt:>5.3f} {'Y' if hit else 'N':>3} | "
              f"{pv[0]:>+6.2f} {pv[2]:>+6.2f} | {bv[0]:>+6.2f} {bv[2]:>+6.2f} | {zns:>6} {xbs:>6} {'Y' if clr else 'N':>3}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
