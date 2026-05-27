"""V57 dynamic probe v2: drive trajectory AND read ball position from sim.
Resolves whether paddle is above or below ball during actual play.
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


V57 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.375, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000, +1.000]),
    (0.475, [+1.450, +0.070, -2.050, +0.750, +0.000, -1.250, +1.000]),
    (0.550, [+1.600, +0.070, -2.050, +0.700, -0.030, -1.045, +1.000]),
    (0.625, [+1.750, +0.070, -2.050, +0.650, +0.000, -0.500, +1.000]),
    (0.775, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.975, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
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

    # Build spline
    times = np.array([k[0] for k in V57])
    angs = np.array([k[1] for k in V57], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    # 1) Initialize: settle at t=0 ready pose, set ball to launch state
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

    # Launch ball: pos (-0.35, 0, 1.3), vel (3.5, 0, 0.5) — EASY_BALL
    # ball is RigidObject, set its root state in world frame
    ball_root_state = ball.data.default_root_state.clone()
    # default_root_state shape: (n_envs, 13). [0:3]=pos, [3:7]=quat, [7:10]=lin_vel, [10:13]=ang_vel
    ball_root_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
    ball_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_root_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
    ball_root_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
    ball.write_root_state_to_sim(ball_root_state, env_ids=ids)
    # Disable relaunch interval event by stepping just sim, not the env
    scene.write_data_to_sim()

    # 2) Drive trajectory + log paddle and ball positions
    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    log = []
    n_sim_steps = int(0.85 / sim_dt)  # full episode through follow phase
    for step in range(n_sim_steps):
        t = step * sim_dt
        target = spline(min(t, 1.0))
        full_target = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full_target[0, jid] = float(target[k])
        robot.set_joint_position_target(full_target, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)

        actual_q = np.array([robot.data.joint_pos[0, jid].item() for jid in yb_joint_ids])
        p = robot.data.body_pos_w[0, paddle_idx]
        b = ball.data.root_pos_w[0]
        p_local = (p - env_origin).cpu().numpy()
        b_local = (b - env_origin).cpu().numpy()
        log.append((t, p_local, b_local, actual_q, target.copy()))

    # 3) Print paddle vs ball through hit window — using REAL ball pos
    print(f"\n=== Dynamic v57 — REAL ball pos from sim ===")
    print(f"{'t':>6}  {'paddle':<28}  {'ball (real)':<28}  {'gap':<8}  {'paddle_Z':>9}  {'ball_Z':>8}  {'paddle-ball Z':>12}")
    for tt in np.arange(0.30, 0.71, 0.025):
        idx = min(range(len(log)), key=lambda i: abs(log[i][0] - tt))
        t_log, p, b, _, _ = log[idx]
        gap = np.linalg.norm(p - b)
        marker = " ★" if abs(tt - 0.55) < 0.005 else ""
        relation = ("paddle ABOVE" if p[2] > b[2] else "paddle below")
        print(f"{tt:>6.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})       ({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})       {gap*100:>5.1f}cm  {p[2]:>+.3f}    {b[2]:>+.3f}    {p[2]-b[2]:>+.3f}  {relation}{marker}")

    # 4) Closest approach
    gaps = []
    for entry in log:
        t, p, b, _, _ = entry
        if 0.30 < t < 0.80:
            gaps.append((t, np.linalg.norm(p - b), p, b))
    if gaps:
        i_min = min(range(len(gaps)), key=lambda i: gaps[i][1])
        t_min, gap_min, p_min, b_min = gaps[i_min]
        d = p_min - b_min
        print(f"\n=== DYNAMIC closest approach ===")
        print(f"  t={t_min:.3f}, gap={gap_min*100:.2f}cm")
        print(f"  paddle = ({p_min[0]:+.3f},{p_min[1]:+.3f},{p_min[2]:+.3f})")
        print(f"  ball   = ({b_min[0]:+.3f},{b_min[1]:+.3f},{b_min[2]:+.3f})")
        print(f"  Δ(paddle-ball) = ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")
        if d[2] > 0:
            print(f"  paddle is {d[2]*100:.1f}cm ABOVE ball  ← matches user observation if paddle is up")
        else:
            print(f"  paddle is {-d[2]*100:.1f}cm BELOW ball  ← user said opposite, suggests perception or trajectory mismatch")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
