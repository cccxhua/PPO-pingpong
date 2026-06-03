"""V84: 通过 env.step() 完整 pipeline 跑 zero-action ref motion, 实时检查
ball_return reward 是否真的为 0.

V83 用 sim.step 直接驱动, 跳过了 event_manager 和 reward_manager.
V84 用 env.step(zeros) 通过完整 manager pipeline, 看 reward_manager 实际算出的:
  - ball_return reward
  - ball_land_opponent reward
  - ball_hit reward
  - ball_was_hit flag
  - ball position, velocity (env-relative)

如果通过完整 pipeline ball_return 也是 0, bug 在 reward 计算或 event order.
如果通过完整 pipeline ball_return 是非零, bug 在训练 policy noise / phase_speed.
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
    cmd = env.unwrapped.command_manager.get_term("motion")
    rew_mgr = env.unwrapped.reward_manager

    device = env.unwrapped.device
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"action_dim = {action_dim}")
    zeros = torch.zeros(args_cli.num_envs, action_dim, device=device)

    print(f"\nstep_dt = {env.unwrapped.step_dt}, decimation = {env.unwrapped.cfg.decimation}, "
          f"sim_dt = {env.unwrapped.sim.get_physics_dt()}")
    print(f"hit_phase = {cmd.cfg.hit_phase}, hit_phase_noise = {cmd.cfg.hit_phase_noise}")

    # 探针 reward terms 的索引
    rew_term_names = [t for t in rew_mgr.active_terms]
    print(f"reward terms: {rew_term_names}")

    # Header
    print(f"\n{'step':>4} {'phase':>5} {'ball_x':>7} {'ball_z':>6} {'vx':>6} {'vz':>6} "
          f"{'was_hit':>7} {'r_hit':>6} {'r_speed':>7} {'r_return':>8} {'r_land':>6} {'done':>5}")

    n_steps = 100
    cum_return = 0.0
    cum_land = 0.0
    cum_hit = 0.0
    return_fired = 0
    for step in range(n_steps):
        obs, rew, term, trunc, info = env.step(zeros)
        bp = ball.data.root_pos_w[0].cpu().numpy() - scene.env_origins[0].cpu().numpy()
        bv = ball.data.root_lin_vel_w[0].cpu().numpy()
        was_hit = bool(cmd.ball_was_hit[0].item())
        phase = float(cmd.phase[0].item())

        # 抓 reward_manager 内每个 term 当前 step 的 reward
        # rew_mgr._step_reward[env_id, term_idx]
        step_rewards = rew_mgr._step_reward[0]  # [num_terms]
        rew_dict = {n: float(step_rewards[i].item()) for i, n in enumerate(rew_term_names)}
        r_hit = rew_dict.get("ball_hit", 0.0)
        r_speed = rew_dict.get("ball_hit_speed", 0.0)
        r_return = rew_dict.get("ball_return", 0.0)
        r_land = rew_dict.get("ball_land_opponent", 0.0)

        cum_return += r_return
        cum_land += r_land
        cum_hit += r_hit
        if r_return > 0:
            return_fired += 1

        done = bool((term[0] | trunc[0]).item())
        # Print key transitions: when ball passes net or paddle hits
        crossed = bp[0] < 0.0 and bv[0] < -0.5
        important = (
            step < 5 or  # initial
            (step % 5 == 0) or  # periodic
            crossed or
            was_hit or
            done
        )
        if important:
            print(f"{step:>4} {phase:>5.2f} {bp[0]:>+7.3f} {bp[2]:>+6.3f} {bv[0]:>+6.2f} {bv[2]:>+6.2f} "
                  f"{str(was_hit):>7} {r_hit:>6.3f} {r_speed:>7.3f} {r_return:>8.3f} {r_land:>6.3f} {str(done):>5}")
        if done:
            print(f"  *** EPISODE TERMINATED at step {step}, term={term.item()}, trunc={trunc.item()}")
            break

    print(f"\n=== summary ===")
    print(f"  cum r_hit:    {cum_hit:.3f}")
    print(f"  cum r_return: {cum_return:.3f}  (fired in {return_fired} steps)")
    print(f"  cum r_land:   {cum_land:.3f}")
    print(f"  final ball_was_hit = {bool(cmd.ball_was_hit[0].item())}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
