"""Probe paddle and ball positions during RIGHT_EXP motion to diagnose contact issues."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="X1-TableTennis")
parser.add_argument("--npz", type=str, required=True)
parser.add_argument("--arrive_time", type=float, default=0.55)
parser.add_argument("--ball_y", type=float, default=0.20)
parser.add_argument("--ball_vy", type=float, default=0.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    npz_abs = os.path.abspath(args_cli.npz)
    env_cfg.commands.motion.motion_files = [npz_abs]
    env_cfg.commands.motion.match_ball_direction = False
    env_cfg.commands.motion.hit_phase = 0.0
    env_cfg.commands.motion.hit_phase_noise = 0.0
    env_cfg.commands.motion.ball_arrive_time_est = args_cli.arrive_time
    env_cfg.terminations.ball_on_own_table = None
    env_cfg.terminations.ball_missed_paddle = None

    # Override ball launch params
    ball_params = dict(
        x_range=(-0.35, -0.35), y_range=(args_cli.ball_y, args_cli.ball_y),
        z_range=(1.3, 1.3), vx_range=(3.5, 3.5),
        vy_range=(args_cli.ball_vy, args_cli.ball_vy), vz_range=(0.5, 0.5),
    )
    env_cfg.events.reset_ball.params.update({"ball_cfg": env_cfg.events.reset_ball.params["ball_cfg"], **ball_params})
    env_cfg.events.relaunch_ball.params.update({"ball_cfg": env_cfg.events.relaunch_ball.params["ball_cfg"], **ball_params})

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    zero_action = torch.zeros((args_cli.num_envs, action_dim), device=device)

    scene = env.unwrapped.scene
    ball = scene["ball"]
    robot = scene["robot"]

    # Find paddle body index
    paddle_body_names = robot.data.body_names
    paddle_idx = None
    for i, name in enumerate(paddle_body_names):
        if "paddle" in name.lower() or "yb_7" in name.lower():
            paddle_idx = i
    if paddle_idx is None:
        # Try finding the last link of right arm
        for i, name in enumerate(paddle_body_names):
            if "yb_7" in name or "wrist_yaw" in name:
                paddle_idx = i
    print(f"[INFO] Body names: {paddle_body_names}")
    print(f"[INFO] Paddle body index: {paddle_idx}")

    dt = 0.02  # step_dt = decimation(4) * sim_dt(0.005)
    duration = 1.0
    total_steps = int(duration / dt)  # 50 steps per cycle

    print(f"\n{'step':>4} {'phase':>6} {'pad_x':>7} {'pad_y':>7} {'pad_z':>7} | {'ball_x':>7} {'ball_y':>7} {'ball_z':>7} | {'gap':>6}")
    print("-" * 80)

    for step in range(total_steps * 2):  # Run 2 full cycles
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_action)

        phase = (step % total_steps) / total_steps

        # Paddle position (body origin in world frame, relative to env origin)
        if paddle_idx is not None:
            pad_pos = (robot.data.body_pos_w[0, paddle_idx] - scene.env_origins[0]).cpu().numpy()
        else:
            pad_pos = np.array([0, 0, 0])

        # Ball position
        ball_pos = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()

        gap = float(np.linalg.norm(pad_pos - ball_pos))

        # Print every step in hit window, every 5 outside
        if 0.25 <= phase <= 0.65 or step % 10 == 0:
            print(f"{step:4d} {phase:6.3f} {pad_pos[0]:7.3f} {pad_pos[1]:7.3f} {pad_pos[2]:7.3f} | "
                  f"{ball_pos[0]:7.3f} {ball_pos[1]:7.3f} {ball_pos[2]:7.3f} | {gap:6.3f}",
                  flush=True)

        # Also detect minimum gap
        if step == 0:
            min_gap = gap
            min_gap_step = step
        elif gap < min_gap:
            min_gap = gap
            min_gap_step = step

    print(f"\n[RESULT] Minimum gap: {min_gap:.4f}m at step {min_gap_step} (phase {(min_gap_step % total_steps)/total_steps:.3f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
