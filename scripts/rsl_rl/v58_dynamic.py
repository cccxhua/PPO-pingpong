"""V58 dynamic probe: drive new v58 keyframes, log paddle vs real sim ball.

V58 design (PIN moved from t=0.55 to t=0.475):
  - PIN pose found by v58_search_hit_pose.py: targets real sim ball at t=0.475.
  - Windup (-75ms) and snap (+75ms) follow v57 pattern.
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


# V58: PIN moved from 0.55 → 0.475 (matching real sim ball arrival time).
V58 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # ready
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),  # mid
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),  # windup (-75ms)
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),  # PIN — V58 hit pose
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),  # snap (+75ms)
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),  # follow
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # return
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),  # hold
]


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

    times = np.array([k[0] for k in V58])
    angs = np.array([k[1] for k in V58], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    # Init: settle at t=0
    full = robot.data.default_joint_pos[0:1].clone()
    q0 = spline(0.0)
    for k, jid in enumerate(yb_joint_ids):
        full[0, jid] = float(q0[k])
    v0 = torch.zeros_like(full)
    ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(full, v0, env_ids=ids)
    robot.set_joint_position_target(full, env_ids=ids)
    for _ in range(200):
        robot.set_joint_position_target(full, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(env.unwrapped.sim.get_physics_dt())

    # Launch ball
    ball_root_state = ball.data.default_root_state.clone()
    ball_root_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
    ball_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_root_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
    ball_root_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
    ball.write_root_state_to_sim(ball_root_state, env_ids=ids)
    scene.write_data_to_sim()

    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    log = []
    n_steps = int(0.85 / sim_dt)
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

    print(f"\n=== V58 dynamic — paddle vs REAL sim ball ===")
    print(f"{'t':>6}  {'paddle':<28}  {'ball':<28}  {'gap':<7}  {'ΔZ':>6}")
    for tt in np.arange(0.30, 0.71, 0.025):
        idx = min(range(len(log)), key=lambda i: abs(log[i][0] - tt))
        t_log, p, b = log[idx]
        gap = np.linalg.norm(p - b)
        marker = " ★" if abs(tt - 0.475) < 0.005 else ""
        print(f"{tt:>6.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})       ({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})       {gap*100:>5.1f}cm  {p[2]-b[2]:>+.3f}{marker}")

    # Closest approach
    gaps = []
    for entry in log:
        t, p, b = entry
        if 0.30 < t < 0.80:
            gaps.append((t, np.linalg.norm(p - b), p, b))
    if gaps:
        i_min = min(range(len(gaps)), key=lambda i: gaps[i][1])
        t_min, gap_min, p_min, b_min = gaps[i_min]
        d = p_min - b_min
        print(f"\n=== V58 closest approach ===")
        print(f"  t={t_min:.3f}, gap={gap_min*100:.2f}cm")
        print(f"  paddle = ({p_min[0]:+.3f}, {p_min[1]:+.3f}, {p_min[2]:+.3f})")
        print(f"  ball   = ({b_min[0]:+.3f}, {b_min[1]:+.3f}, {b_min[2]:+.3f})")
        print(f"  Δ      = ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})")

    # Paddle velocity at hit
    if len(log) > 4:
        i_hit = min(range(len(log)), key=lambda i: abs(log[i][0] - 0.475))
        if 1 < i_hit < len(log) - 1:
            t_pre = log[i_hit - 1][0]
            t_post = log[i_hit + 1][0]
            p_pre = log[i_hit - 1][1]
            p_post = log[i_hit + 1][1]
            v = (p_post - p_pre) / (t_post - t_pre)
            print(f"\n=== Paddle velocity @ t≈0.475 (FD) ===")
            print(f"  v = ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}) m/s, |v|={np.linalg.norm(v):.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
