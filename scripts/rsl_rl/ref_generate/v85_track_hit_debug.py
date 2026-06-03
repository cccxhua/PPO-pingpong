"""V85: 排查 track_ball_hit 为何不设 ball_was_hit.

V84 trace 显示 ball vx 从 +3.04 → -2.97 (撞拍发生), 但 cmd.ball_was_hit 全程 False.
V85 在每 env.step 后:
  - 直接读 contact sensor 的 force history
  - 计算 paddle-ball dist
  - 模拟 track_ball_hit 的逻辑, 看为什么 (force>0.1) & (dist<0.25) 没满足
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    scene = env.unwrapped.scene
    ball = scene["ball"]
    robot = scene["robot"]
    contact_sensor = scene.sensors["contact_forces"]
    cmd = env.unwrapped.command_manager.get_term("motion")

    device = env.unwrapped.device
    action_dim = env.unwrapped.action_manager.total_action_dim
    zeros = torch.zeros(args_cli.num_envs, action_dim, device=device)

    # find paddle body in sensor and robot
    sensor_paddle_idx = contact_sensor.body_names.index("Link_yb_paddle")
    robot_paddle_idx = robot.body_names.index("Link_yb_paddle")
    print(f"sensor_paddle_idx = {sensor_paddle_idx} (in {len(contact_sensor.body_names)} sensor bodies)")
    print(f"robot_paddle_idx  = {robot_paddle_idx} (in {len(robot.body_names)} robot bodies)")
    print(f"history_length    = {contact_sensor.cfg.history_length}")

    print(f"\n{'step':>4} {'ball_x':>7} {'ball_z':>6} {'ball_vx':>7} "
          f"{'pad_x':>6} {'pad_z':>6} {'dist':>6} "
          f"{'fmag':>6} {'fmag_max':>8} "
          f"{'was_hit':>7}")

    for step in range(60):
        env.step(zeros)
        bp = ball.data.root_pos_w[0].cpu().numpy()
        bp_local = bp - scene.env_origins[0].cpu().numpy()
        bv = ball.data.root_lin_vel_w[0].cpu().numpy()
        pp = robot.data.body_pos_w[0, robot_paddle_idx].cpu().numpy()
        pp_local = pp - scene.env_origins[0].cpu().numpy()
        dist = float(np.linalg.norm(pp - bp))

        nf = contact_sensor.data.net_forces_w_history  # [num_envs, history, num_bodies, 3]
        # latest frame
        f_latest = nf[0, 0, sensor_paddle_idx, :]
        fmag_latest = float(torch.norm(f_latest).item())
        # max over history
        f_hist = nf[0, :, sensor_paddle_idx, :]
        fmag_max = float(torch.norm(f_hist, dim=-1).max().item())

        was_hit = bool(cmd.ball_was_hit[0].item())

        # only print interesting frames
        if step < 5 or fmag_max > 0.05 or dist < 0.3 or step % 5 == 0 or (15 <= step <= 30):
            print(f"{step:>4} {bp_local[0]:>+7.3f} {bp_local[2]:>+6.3f} {bv[0]:>+7.2f} "
                  f"{pp_local[0]:>+6.3f} {pp_local[2]:>+6.3f} {dist:>6.3f} "
                  f"{fmag_latest:>6.2f} {fmag_max:>8.2f} "
                  f"{str(was_hit):>7}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
