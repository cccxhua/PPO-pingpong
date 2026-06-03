"""V57 hit-pose search.

Predict ball-paddle collision point analytically, then iteratively search
joint configurations that put paddle on ball at the chosen hit time.

Usage:
    python scripts/rsl_rl/v57_search_hit_pose.py --task X1-TableTennis
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
    """Analytic ball position at absolute time t since launch.
    EASY_BALL: (-0.35, 0, 1.3) launched with vx=3.5, vz=0.5.
    """
    g = 9.81
    x0, z0 = -0.35, 1.3
    vx0, vz0 = 3.5, 0.5
    table_z = 0.79
    fric, rest = 0.526, 0.905

    a = 0.5 * g
    b = -vz0
    c = table_z - z0
    t_b = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    if t < t_b:
        return np.array([x0 + vx0 * t, 0, z0 + vz0 * t - 0.5 * g * t * t]), t_b
    tau = t - t_b
    x_b = x0 + vx0 * t_b
    vx_b = vx0 * fric
    vz_b = -(vz0 - g * t_b) * rest
    return np.array([x_b + vx_b * tau, 0, table_z + vz_b * tau - 0.5 * g * tau * tau]), t_b


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

    def paddle_at(joint_vals, settle_steps=150):
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

    # ============================================================
    # Step 1: Predict ball position at hit time
    # ============================================================
    HIT_TIME = 0.55  # phase t at peak swing (matching create_forehand.py v56 design)
    ball_target, t_bounce = ball_pos_at(HIT_TIME)
    print(f"\n=== Ball trajectory ===")
    print(f"  bounce @ t={t_bounce:.3f}")
    for tt in np.linspace(0.40, 0.70, 7):
        bp, _ = ball_pos_at(tt)
        print(f"  t={tt:.3f}  ball=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f})")
    print(f"\n  TARGET (t={HIT_TIME}): paddle should be at ball pos = ({ball_target[0]:+.3f},{ball_target[1]:+.3f},{ball_target[2]:+.3f})")

    # ============================================================
    # Step 2: Coordinate-descent search starting from v55 hit pose
    # ============================================================
    # v55 hit values (interpolated at t=0.55)
    q = np.array([1.45, 0.30, -2.00, 0.05, 0.00, -0.775, 1.00])

    print(f"\n=== Coordinate-descent search ===")
    print(f"  start q = {q.tolist()}")
    p0 = paddle_at(q)
    print(f"  start paddle = ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f})  gap={np.linalg.norm(p0-ball_target)*100:.1f}cm")

    best_q = q.copy()
    best_gap = np.linalg.norm(p0 - ball_target)
    # Try several broad combos. yb_2 controls Y heavily, yb_1 Z, yb_4 X.
    # Also want to keep paddle face direction usable — keep yb_3,5,6,7 close to v55.
    candidates = [
        # (label, [yb1, yb2, yb3, yb4, yb5, yb6, yb7])
        ("base v55",                 [1.45,  0.30, -2.00, 0.05, 0.00, -0.775, 1.00]),
        # 闭合 Y gap (yb_2 减小) + X gap (yb_4 增大) + Z gap (yb_1 增大)
        ("yb2-0.3, yb4+0.3, yb1+0.2",[1.65,  0.00, -2.00, 0.35, 0.00, -0.775, 1.00]),
        ("yb2-0.5, yb4+0.4, yb1+0.3",[1.75, -0.20, -2.00, 0.45, 0.00, -0.775, 1.00]),
        ("yb2-0.7, yb4+0.5, yb1+0.4",[1.85, -0.40, -2.00, 0.55, 0.00, -0.775, 1.00]),
        ("yb2-1.0, yb4+0.5, yb1+0.4",[1.85, -0.70, -2.00, 0.55, 0.00, -0.775, 1.00]),
        ("yb2-1.2, yb4+0.6, yb1+0.5",[1.95, -0.90, -2.00, 0.65, 0.00, -0.775, 1.00]),
        # 调整 yb_3 (shoulder yaw) 也影响 paddle 位置: 试更小负值 (less external rotation)
        ("yb2-0.5, yb4+0.4, yb3-1.5",[1.75, -0.20, -1.50, 0.45, 0.00, -0.775, 1.00]),
        ("yb2-0.5, yb4+0.4, yb3-1.7",[1.75, -0.20, -1.70, 0.45, 0.00, -0.775, 1.00]),
        ("yb2-0.5, yb4+0.4, yb3-2.3",[1.75, -0.20, -2.30, 0.45, 0.00, -0.775, 1.00]),
    ]
    print(f"\n  {'label':<35}  {'paddle':<28}  {'gap':<6}  {'Δxyz':<22}")
    for label, q_try in candidates:
        q_arr = np.array(q_try)
        # clip to limits
        for k, jn in enumerate(yb_joint_names):
            lo, hi = LIMITS[jn]
            q_arr[k] = np.clip(q_arr[k], lo + 0.01, hi - 0.01)
        p = paddle_at(q_arr)
        gap = np.linalg.norm(p - ball_target)
        d = p - ball_target
        marker = " ★" if gap < best_gap else ""
        print(f"  {label:<35}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})       {gap*100:>5.1f}cm  ({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}){marker}")
        if gap < best_gap:
            best_gap = gap
            best_q = q_arr.copy()

    print(f"\n  best so far: gap={best_gap*100:.1f}cm  q={best_q.tolist()}")

    # ============================================================
    # Step 3: Refine — coordinate descent on best candidate
    # ============================================================
    print(f"\n=== Refine: coordinate descent (steps of ±0.05 rad) ===")
    step_init = 0.05
    for outer in range(3):  # capped at 3 to fit timeout
        improved = False
        for k, jn in enumerate(yb_joint_names):
            lo, hi = LIMITS[jn]
            for sign in [+1, -1]:
                q_try = best_q.copy()
                q_try[k] = np.clip(q_try[k] + sign * step_init, lo + 0.01, hi - 0.01)
                p = paddle_at(q_try)
                gap = np.linalg.norm(p - ball_target)
                if gap < best_gap - 0.001:
                    best_gap = gap
                    best_q = q_try.copy()
                    improved = True
                    print(f"  iter {outer}: {jn} {sign:+d} -> q[{k}]={best_q[k]:+.3f}  gap={gap*100:.1f}cm  paddle=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        if not improved:
            step_init *= 0.5
            if step_init < 0.01:
                break

    print(f"\n=== FINAL ===")
    p_final = paddle_at(best_q)
    gap_final = np.linalg.norm(p_final - ball_target)
    print(f"  q       = {[f'{v:+.3f}' for v in best_q]}")
    print(f"  paddle  = ({p_final[0]:+.3f},{p_final[1]:+.3f},{p_final[2]:+.3f})")
    print(f"  ball    = ({ball_target[0]:+.3f},{ball_target[1]:+.3f},{ball_target[2]:+.3f})")
    print(f"  gap     = {gap_final*100:.1f}cm")
    print(f"\n  >>> v57 PIN keyframe at t=0.55:")
    print(f"      ({best_q[0]:+.3f}, {best_q[1]:+.3f}, {best_q[2]:+.3f}, {best_q[3]:+.3f}, {best_q[4]:+.3f}, {best_q[5]:+.3f}, {best_q[6]:+.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
