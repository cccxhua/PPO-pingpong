"""V56 paddle vs ball trajectory probe.

Plays v56 keyframes through Isaac Lab FK, logs paddle position vs phase,
computes analytic ball trajectory, finds closest approach + the joint deltas
needed to close the gap.

usage: python scripts/rsl_rl/v56_paddle_ball.py --task X1-TableTennis-Forehand
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="X1-TableTennis-Forehand")
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


# ============================================================
# Reference motion variants
# ============================================================
V55 = [
    (0.00,  [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
    (0.375, [   1.30,   0.30,  -2.00,   0.50,   0.00,  -1.00,   1.00]),
    (0.475, [   1.45,   0.30,  -2.00,   0.05,   0.00,  -1.10,   1.00]),
    (0.625, [   1.45,   0.30,  -2.00,   0.05,   0.00,  -0.45,   1.00]),
    (0.775, [   1.45,   0.30,  -2.00,   0.85,   0.00,  -1.00,   1.00]),
    (0.975, [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
    (1.00,  [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
]
V56 = [
    (0.00,  [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
    (0.375, [   1.30,   0.30,  -2.00,   0.50,   0.00,  -1.00,   1.00]),
    (0.475, [   1.45,   0.30,  -2.00,   0.05,   0.00,  -1.20,   1.00]),
    (0.55,  [   1.45,   0.30,  -2.00,   0.05,   0.00,  -0.775, 1.00]),  # PIN
    (0.625, [   1.50,   0.30,  -2.00,   0.05,   0.00,  -0.30,  1.00]),
    (0.775, [   1.45,   0.30,  -2.00,   0.85,   0.00,  -1.00,   1.00]),
    (0.975, [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
    (1.00,  [   1.00,   0.30,  -2.00,   1.40,   0.00,  -1.00,   1.00]),
]


def build_spline(kfs):
    times = np.array([k[0] for k in kfs])
    angs = np.array([k[1] for k in kfs], dtype=np.float64)
    return CubicSpline(times, angs, bc_type="clamped")


# ============================================================
# Analytic ball trajectory (EASY_BALL: x=-0.35, z=1.3, vx=3.5, vz=0.5)
# ============================================================
def ball_traj(t_grid):
    g = 9.81
    x0, z0 = -0.35, 1.3
    vx0, vz0 = 3.5, 0.5
    table_z = 0.79
    fric, rest = 0.526, 0.905

    a = 0.5 * g
    b = -vz0
    c = table_z - z0
    t_b = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    pos = np.zeros((len(t_grid), 3))
    vel = np.zeros((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        if t < t_b:
            pos[i] = [x0 + vx0 * t, 0, z0 + vz0 * t - 0.5 * g * t * t]
            vel[i] = [vx0, 0, vz0 - g * t]
        else:
            tau = t - t_b
            x_b = x0 + vx0 * t_b
            vx_b = vx0 * fric
            vz_b = -(vz0 - g * t_b) * rest
            pos[i] = [x_b + vx_b * tau, 0, table_z + vz_b * tau - 0.5 * g * tau * tau]
            vel[i] = [vx_b, 0, vz_b - g * tau]
    return pos, vel, t_b


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]
    env_origin = scene.env_origins[0]

    def paddle_at(joint_vals):
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(joint_vals[k])
        v = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v, env_ids=ids)
        robot.set_joint_position_target(full, env_ids=ids)
        for _ in range(400):  # settle PD
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())
        p = robot.data.body_pos_w[0, paddle_idx]
        local = (p - env_origin).cpu().numpy()
        return local

    # Sweep paddle position over phase t for both motions
    t_grid = np.array([0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.70])
    print("\n=== Static FK (PD-settled) for v55 and v56 paddle position vs phase ===")
    print(f"{'t':>6}  {'v55 paddle':<28}  {'v56 paddle':<28}  {'Δ(v56-v55)':<22}")
    paddle_v55 = {}
    paddle_v56 = {}
    sp55 = build_spline(V55)
    sp56 = build_spline(V56)
    for t in t_grid:
        q55 = sp55(t)
        q56 = sp56(t)
        p55 = paddle_at(q55)
        p56 = paddle_at(q56)
        paddle_v55[t] = p55
        paddle_v56[t] = p56
        d = p56 - p55
        print(f"{t:>6.3f}  ({p55[0]:+.3f},{p55[1]:+.3f},{p55[2]:+.3f})       "
              f"({p56[0]:+.3f},{p56[1]:+.3f},{p56[2]:+.3f})       "
              f"({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")

    # Ball trajectory
    bp, bv, t_b = ball_traj(t_grid)
    print(f"\n=== Ball (bounce @ t={t_b:.3f}, post-bounce vx={3.5*0.526:.2f}) ===")
    for i, t in enumerate(t_grid):
        print(f"  t={t:.3f}  ball=({bp[i,0]:+.3f},{bp[i,1]:+.3f},{bp[i,2]:+.3f})  "
              f"vel=({bv[i,0]:+.2f},{bv[i,1]:+.2f},{bv[i,2]:+.2f})")

    # Closest approach
    print("\n=== Closest approach (v56 paddle vs ball) ===")
    gaps = []
    for i, t in enumerate(t_grid):
        p = paddle_v56[t]
        gap = np.linalg.norm(p - bp[i])
        gaps.append(gap)
        print(f"  t={t:.3f}  gap={gap*100:.1f}cm  paddle-ball=({p[0]-bp[i,0]:+.3f},{p[1]-bp[i,1]:+.3f},{p[2]-bp[i,2]:+.3f})")
    i_min = int(np.argmin(gaps))
    t_min = t_grid[i_min]
    p_min = paddle_v56[t_min]
    b_min = bp[i_min]
    delta = b_min - p_min
    print(f"\n  ➜ closest @ t={t_min:.3f}, gap={gaps[i_min]*100:.1f}cm")
    print(f"     paddle: ({p_min[0]:+.3f},{p_min[1]:+.3f},{p_min[2]:+.3f})")
    print(f"     ball  : ({b_min[0]:+.3f},{b_min[1]:+.3f},{b_min[2]:+.3f})")
    print(f"     need to move paddle by Δ=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})")

    # Per-joint Jacobian at hit pose (t=0.55) — tells us how to nudge paddle to ball
    print("\n=== Jacobian at v56 hit pose (t=0.55), ±0.10 rad each joint ===")
    q_hit = sp56(0.55)
    p_base = paddle_at(q_hit)
    print(f"  baseline paddle: ({p_base[0]:+.3f},{p_base[1]:+.3f},{p_base[2]:+.3f})")
    print(f"  {'joint':>10}  {'+0.10':>32}  {'-0.10':>32}")
    for k, jn in enumerate(yb_joint_names):
        deltas = []
        for sign in [+1, -1]:
            qp = q_hit.copy()
            qp[k] += sign * 0.10
            pp = paddle_at(qp)
            deltas.append(pp - p_base)
        d_pos = deltas[0]
        d_neg = deltas[1]
        print(f"  {jn:>10}  ({d_pos[0]:+.3f},{d_pos[1]:+.3f},{d_pos[2]:+.3f})  "
              f"({d_neg[0]:+.3f},{d_neg[1]:+.3f},{d_neg[2]:+.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
