"""V57 dynamic probe: drive the spline trajectory through PD and log
actual paddle vs ball position over time. Reveals the PD tracking lag.
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


def ball_pos(t):
    g = 9.81; x0, z0 = -0.35, 1.3; vx0, vz0 = 3.5, 0.5; tz = 0.79
    fric, rest = 0.526, 0.905
    a = 0.5 * g; b = -vz0; c = tz - z0
    t_b = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if t < t_b:
        return np.array([x0 + vx0 * t, 0, z0 + vz0 * t - 0.5 * g * t * t])
    tau = t - t_b
    x_b = x0 + vx0 * t_b
    vx_b = vx0 * fric
    vz_b = -(vz0 - g * t_b) * rest
    return np.array([x_b + vx_b * tau, 0, tz + vz_b * tau - 0.5 * g * tau * tau])


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]
    env_origin = scene.env_origins[0]

    # Build spline
    times = np.array([k[0] for k in V57])
    angs = np.array([k[1] for k in V57], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    # 1) Initialize: settle at t=0 ready pose
    full = robot.data.default_joint_pos[0:1].clone()
    q0 = spline(0.0)
    for k, jid in enumerate(yb_joint_ids):
        full[0, jid] = float(q0[k])
    v0 = torch.zeros_like(full)
    ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(full, v0, env_ids=ids)
    robot.set_joint_position_target(full, env_ids=ids)
    for _ in range(300):
        robot.set_joint_position_target(full, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(env.unwrapped.sim.get_physics_dt())

    # 2) Drive the trajectory in real-time (phase_speed = 1.0)
    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    # decimation: PD targets get updated every `decimation` sim steps. From env_cfg: decimation=4
    # In play mode, ref motion is sampled at FPS=30 then interpolated. Drive PD at sim_dt.
    log = []  # (t, paddle_pos, target_q, actual_q)
    n_sim_steps = int(1.0 / sim_dt)
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
        p_local = (p - env_origin).cpu().numpy()
        log.append((t, p_local, target.copy(), actual_q))

    # 3) Print paddle vs ball through hit window
    print(f"\n=== Dynamic v57 paddle (driven by PD, phase_speed=1.0) ===")
    print(f"{'t':>6}  {'paddle (actual)':<28}  {'ball':<28}  {'gap':<7}  {'paddle Z':>8}  {'ball Z':>7}  {'ΔZ':>6}")
    for tt in np.arange(0.40, 0.71, 0.025):
        # find closest log entry
        idx = min(range(len(log)), key=lambda i: abs(log[i][0] - tt))
        t_log, p, target_q, actual_q = log[idx]
        b = ball_pos(tt)
        gap = np.linalg.norm(p - b)
        marker = " ★" if abs(tt - 0.55) < 0.005 else ""
        print(f"{tt:>6.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})       ({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})       {gap*100:>5.1f}cm  {p[2]:>+.3f}    {b[2]:>+.3f}  {p[2]-b[2]:>+.3f}{marker}")

    # 4) Tracking error per joint at t=0.55
    print(f"\n=== Joint tracking error at t=0.55 (PD lag) ===")
    idx = min(range(len(log)), key=lambda i: abs(log[i][0] - 0.55))
    t_log, p_log, target_q, actual_q = log[idx]
    print(f"  log t={t_log:.3f}")
    for k, jn in enumerate(yb_joint_names):
        err = actual_q[k] - target_q[k]
        print(f"  {jn}: target={target_q[k]:+.3f}  actual={actual_q[k]:+.3f}  err={err:+.3f} rad")

    # 5) Find closest approach in dynamic trajectory
    gaps = []
    for entry in log:
        t, p, _, _ = entry
        if 0.30 < t < 0.80:
            b = ball_pos(t)
            gaps.append((t, np.linalg.norm(p - b), p, b))
    if gaps:
        i_min = min(range(len(gaps)), key=lambda i: gaps[i][1])
        t_min, gap_min, p_min, b_min = gaps[i_min]
        d = p_min - b_min
        print(f"\n=== Dynamic closest approach ===")
        print(f"  t={t_min:.3f}, gap={gap_min*100:.2f}cm")
        print(f"  paddle = ({p_min[0]:+.3f},{p_min[1]:+.3f},{p_min[2]:+.3f})")
        print(f"  ball   = ({b_min[0]:+.3f},{b_min[1]:+.3f},{b_min[2]:+.3f})")
        print(f"  Δ(paddle-ball) = ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
