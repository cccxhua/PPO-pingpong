"""V57 refinement: continue from coarse search, also evaluate full v57 motion
to verify paddle sweeps from below through ball (gives +z velocity).
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

LIMITS = {
    "joint_yb_1": (-1.05, 3.17),
    "joint_yb_2": (-3.08, 0.31),
    "joint_yb_3": (-2.78, 2.76),
    "joint_yb_4": (-1.91, 1.95),
    "joint_yb_5": (-2.79, 2.76),
    "joint_yb_6": (-1.29, 1.51),
    "joint_yb_7": (-3.14, 3.14),
}


def ball_pos_at(t):
    g = 9.81
    x0, z0 = -0.35, 1.3
    vx0, vz0 = 3.5, 0.5
    table_z = 0.79
    fric, rest = 0.526, 0.905
    a = 0.5 * g; b = -vz0; c = table_z - z0
    t_b = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if t < t_b:
        return np.array([x0 + vx0 * t, 0, z0 + vz0 * t - 0.5 * g * t * t])
    tau = t - t_b
    x_b = x0 + vx0 * t_b
    vx_b = vx0 * fric
    vz_b = -(vz0 - g * t_b) * rest
    return np.array([x_b + vx_b * tau, 0, table_z + vz_b * tau - 0.5 * g * tau * tau])


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

    def paddle_at(joint_vals, settle_steps=100):
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(joint_vals[k])
        v = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v, env_ids=ids)
        robot.set_joint_position_target(full, env_ids=ids)
        for _ in range(settle_steps):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())
        p = robot.data.body_pos_w[0, paddle_idx]
        return (p - env_origin).cpu().numpy()

    HIT_TIME = 0.55
    ball_target = ball_pos_at(HIT_TIME)
    print(f"\n=== Target: paddle should be at ball pos at t={HIT_TIME} ===")
    print(f"  ball = ({ball_target[0]:+.3f},{ball_target[1]:+.3f},{ball_target[2]:+.3f})")

    # Start from previous search FINAL
    best_q = np.array([1.700, -0.050, -2.050, 0.600, 0.000, -0.925, 1.000])
    p0 = paddle_at(best_q)
    best_gap = np.linalg.norm(p0 - ball_target)
    print(f"  start q = {best_q.tolist()}")
    print(f"  start paddle = ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f})  gap={best_gap*100:.1f}cm")

    # Refine: 5 outer rounds, step from 0.05 → 0.01
    print(f"\n=== Refine ===")
    for outer, step in enumerate([0.05, 0.03, 0.02, 0.01, 0.01]):
        improved = False
        for k, jn in enumerate(yb_joint_names):
            lo, hi = LIMITS[jn]
            for sign in [+1, -1]:
                q_try = best_q.copy()
                q_try[k] = np.clip(q_try[k] + sign * step, lo + 0.01, hi - 0.01)
                p = paddle_at(q_try)
                gap = np.linalg.norm(p - ball_target)
                if gap < best_gap - 0.001:
                    best_gap = gap
                    best_q = q_try.copy()
                    improved = True
                    print(f"  iter {outer} step={step}: {jn} {sign:+d} -> q[{k}]={best_q[k]:+.3f}  gap={gap*100:.2f}cm  paddle=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        if not improved:
            print(f"  iter {outer} step={step}: no improvement")

    print(f"\n=== Best static pin pose ===")
    p_final = paddle_at(best_q)
    print(f"  q = {[f'{v:+.4f}' for v in best_q]}")
    print(f"  paddle = ({p_final[0]:+.4f},{p_final[1]:+.4f},{p_final[2]:+.4f})  gap={best_gap*100:.2f}cm")

    # ============================================================
    # Now evaluate paddle Z velocity through hit window for the proposed v57
    # ============================================================
    yb_hit = best_q  # use refined hit pose
    # Design windup/snap symmetric around pin, lifting paddle through hit:
    #   yb_1 lower at windup, higher at snap (paddle rises through hit → +z velocity)
    #   yb_6 windup deep, snap forward (wrist whip)
    V57 = [
        (0.00,  [1.00,  0.30, -2.00, 1.40, 0.00, -1.00,  1.00]),
        (0.375, [1.40, (yb_hit[1]+0.30)/2, (yb_hit[2]-2.00)/2, (yb_hit[3]+1.40)/2, 0.00, -1.00,  1.00]),
        (0.475, [yb_hit[0]-0.15, yb_hit[1], yb_hit[2], yb_hit[3]+0.05, 0.00, -1.25, 1.00]),  # windup: yb_1 -0.15 (paddle low), yb_6 deep -1.25
        (0.55,  list(yb_hit)),  # PIN
        (0.625, [yb_hit[0]+0.15, yb_hit[1], yb_hit[2], yb_hit[3]-0.05, 0.00, -0.50, 1.00]),  # snap: yb_1 +0.15 (paddle high → +z vel), yb_6 -0.50
        (0.775, [1.45,  0.10, -2.00, 0.85, 0.00, -1.00,  1.00]),
        (0.975, [1.00,  0.30, -2.00, 1.40, 0.00, -1.00,  1.00]),
        (1.00,  [1.00,  0.30, -2.00, 1.40, 0.00, -1.00,  1.00]),
    ]
    print(f"\n=== V57 keyframes ===")
    for t, vals in V57:
        print(f"  ({t:.3f}, [{', '.join(f'{v:+.3f}' for v in vals)}])")

    # Build spline and check paddle position+velocity at hit
    times = np.array([k[0] for k in V57])
    angs = np.array([k[1] for k in V57], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    print(f"\n=== V57 paddle trajectory (FK on spline interp) ===")
    print(f"  {'t':>6}  {'paddle':<28}  {'ball':<28}  {'gap':<8}")
    for tt in np.linspace(0.45, 0.65, 9):
        q = spline(tt)
        p = paddle_at(q)
        b = ball_pos_at(tt)
        gap = np.linalg.norm(p - b)
        marker = " ★" if abs(tt - HIT_TIME) < 0.01 else ""
        print(f"  {tt:>6.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})       ({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})       {gap*100:>5.1f}cm{marker}")

    # Estimate paddle velocity at t=0.55 by FD on spline (joint velocity → paddle velocity via Jacobian implicit)
    dt = 0.01
    p_pre = paddle_at(spline(0.55 - dt))
    p_post = paddle_at(spline(0.55 + dt))
    paddle_vel = (p_post - p_pre) / (2 * dt)
    print(f"\n=== Paddle velocity at t=0.55 (FD ±10ms) ===")
    print(f"  vel = ({paddle_vel[0]:+.2f}, {paddle_vel[1]:+.2f}, {paddle_vel[2]:+.2f}) m/s, |v|={np.linalg.norm(paddle_vel):.2f}")
    print(f"  +Z velocity: {paddle_vel[2]:+.2f} m/s ({'good — ball will rise' if paddle_vel[2] > 0 else 'BAD — ball will sink'})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
