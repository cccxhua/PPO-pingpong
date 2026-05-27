"""V59: raise paddle Z so face center meets ball at X-crossing time.

Diagnosis: V58 had PIN at t=0.475 but actual paddle-ball X-crossing happens at t≈0.510
in dynamic execution. At X-crossing, ball Z is 1.135 but paddle Z stuck at 1.052 (yb_1
PD lag prevents climbing). Ball passes 8cm above paddle origin → hits handle region.

Fix: raise yb_1 across schedule so paddle Z lifts faster. Try several yb_1 increments,
report dynamic paddle pose at X-crossing.
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


# Base V58 keyframes; apply yb_1 lift per variant.
V58_BASE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def lift_yb1(keys, lift):
    """Add `lift` rad to yb_1 in active swing keyframes (0.40, 0.475, 0.55)."""
    out = []
    for t, vals in keys:
        v = list(vals)
        if 0.39 < t < 0.56:
            v[0] += lift
        out.append((t, v))
    return out


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
    env_origin = scene.env_origins[0]

    def run_traj(keys):
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(0.0)
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        robot.set_joint_position_target(full, env_ids=ids)
        for _ in range(150):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())

        ball_root_state = ball.data.default_root_state.clone()
        ball_root_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
        ball_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_root_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
        ball_root_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
        ball.write_root_state_to_sim(ball_root_state, env_ids=ids)
        scene.write_data_to_sim()

        sim_dt = float(env.unwrapped.sim.get_physics_dt())
        log = []
        n_steps = int(0.70 / sim_dt)
        for step in range(n_steps):
            t = step * sim_dt
            target = spline(min(t, 1.0))
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            p = robot.data.body_pos_w[0, paddle_idx]
            b = ball.data.root_pos_w[0]
            log.append((t, (p - env_origin).cpu().numpy(), (b - env_origin).cpu().numpy()))
        return log

    print(f"\n{'lift':>6}  {'closest':<14}  {'paddle@close':<22}  {'ball@close':<22}  {'ΔZ@close':<8}  {'X-cross':<10}")
    for lift in [0.00, 0.10, 0.20, 0.30, 0.40]:
        keys = lift_yb1(V58_BASE, lift)
        log = run_traj(keys)

        # closest approach
        gaps = [(t, np.linalg.norm(p - b), p, b) for (t, p, b) in log if 0.30 < t < 0.65]
        i_min = min(range(len(gaps)), key=lambda i: gaps[i][1])
        t_min, gap_min, p_min, b_min = gaps[i_min]

        # X-crossing time (paddle X = ball X)
        x_cross_t = None
        x_cross_p = None
        x_cross_b = None
        for i in range(1, len(log)):
            t1, p1, b1 = log[i - 1]
            t2, p2, b2 = log[i]
            if t2 < 0.40 or t2 > 0.60:
                continue
            d1 = p1[0] - b1[0]
            d2 = p2[0] - b2[0]
            if d1 * d2 < 0:  # sign change
                alpha = abs(d1) / (abs(d1) + abs(d2))
                x_cross_t = t1 + alpha * (t2 - t1)
                x_cross_p = p1 + alpha * (p2 - p1)
                x_cross_b = b1 + alpha * (b2 - b1)
                break

        if x_cross_t:
            xc_str = f"t={x_cross_t:.3f}"
            xz_str = f"{x_cross_p[2] - x_cross_b[2]:+.3f}"
            xz_info = f"  paddle@cross=({x_cross_p[0]:+.3f},{x_cross_p[1]:+.3f},{x_cross_p[2]:+.3f}) ball@cross=({x_cross_b[0]:+.3f},{x_cross_b[1]:+.3f},{x_cross_b[2]:+.3f}) ΔZ={xz_str}"
        else:
            xc_str = "no-cross"
            xz_info = ""

        print(f"{lift:+.2f}  t={t_min:.3f} g={gap_min*100:.1f}cm  ({p_min[0]:+.3f},{p_min[1]:+.3f},{p_min[2]:+.3f})  ({b_min[0]:+.3f},{b_min[1]:+.3f},{b_min[2]:+.3f})  {p_min[2]-b_min[2]:+.3f}    {xc_str}")
        print(xz_info)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
