"""V58 hit-pose search — uses REAL sim ball trajectory at t=0.475.

Key insight (from v57_dynamic_with_ball.py probe):
  - Real sim ball @ t=0.475: (+1.281, 0, +1.028)
  - Real sim ball @ t=0.490: (+1.322, 0, +1.057) — closest approach at gap 3.92cm
  - PD-lag offset (dynamic - static):  (+0.066, -0.005, -0.121)

Goal: put STATIC FK paddle at (1.215, 0.005, 1.149)
  → after PD lag, dynamic paddle should land at (1.281, 0, 1.028) ≈ real ball.
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

    def paddle_at(joint_vals, settle_steps=120):
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

    # PD-lag-compensated target = real ball (1.281, 0, 1.028) - PD_offset (+0.066, -0.005, -0.121)
    target = np.array([1.215, 0.005, 1.149])
    real_ball = np.array([1.281, 0.0, 1.028])
    print(f"\n=== V58 hit-pose search ===")
    print(f"  real ball @ t=0.475 = ({real_ball[0]:+.3f}, {real_ball[1]:+.3f}, {real_ball[2]:+.3f})")
    print(f"  static FK target    = ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})  (= ball - PD_lag_offset)")

    # Start from v57 PIN
    best_q = np.array([1.600, 0.070, -2.050, 0.700, -0.030, -1.045, 1.000])
    p0 = paddle_at(best_q)
    best_gap = np.linalg.norm(p0 - target)
    print(f"\n  v57 PIN q = {best_q.tolist()}")
    print(f"  paddle = ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})  gap={best_gap*100:.2f}cm")

    print(f"\n=== Coordinate descent ===")
    for outer, step in enumerate([0.05, 0.03, 0.02, 0.01, 0.01]):
        improved = False
        for k, jn in enumerate(yb_joint_names):
            lo, hi = LIMITS[jn]
            for sign in [+1, -1]:
                q_try = best_q.copy()
                q_try[k] = np.clip(q_try[k] + sign * step, lo + 0.01, hi - 0.01)
                p = paddle_at(q_try)
                gap = np.linalg.norm(p - target)
                if gap < best_gap - 0.001:
                    best_gap = gap
                    best_q = q_try.copy()
                    improved = True
                    print(f"  iter {outer} step={step}: {jn} {sign:+d} -> q[{k}]={best_q[k]:+.3f}  gap={gap*100:.2f}cm  paddle=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        if not improved:
            print(f"  iter {outer} step={step}: no improvement")

    p_final = paddle_at(best_q)
    print(f"\n=== V58 PIN pose @ t=0.475 ===")
    print(f"  q       = [{', '.join(f'{v:+.3f}' for v in best_q)}]")
    print(f"  static paddle = ({p_final[0]:+.3f}, {p_final[1]:+.3f}, {p_final[2]:+.3f})")
    print(f"  static gap to PD-comp target  = {best_gap*100:.2f}cm")
    print(f"  predicted dynamic paddle     ≈ ({p_final[0]+0.066:+.3f}, {p_final[1]-0.005:+.3f}, {p_final[2]-0.121:+.3f})")
    print(f"  predicted dynamic gap to ball = {np.linalg.norm(p_final + np.array([0.066,-0.005,-0.121]) - real_ball)*100:.2f}cm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
