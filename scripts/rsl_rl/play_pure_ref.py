"""Play pure reference motion (zero residual) — diagnostic script.

Bypasses the policy entirely: feeds zero actions to the env so the robot
tracks `ref_dof` exactly with phase_speed=1.0. Use this to inspect what
the reference motion *itself* produces — independent of any trained policy.

Usage:
    python scripts/rsl_rl/play_pure_ref.py --task X1-TableTennis --video --video_length 400

    # play a custom npz (overrides cfg's motion_files, disables ball terminations,
    # starts phase at 0 so the whole clip plays once):
    python scripts/rsl_rl/play_pure_ref.py --task X1-TableTennis \\
        --npz logs/pure_ref/realpingpong.npz \\
        --video --video_length 2800
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play pure reference motion (no policy).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="logs/pure_ref")
parser.add_argument(
    "--npz",
    type=str,
    default=None,
    help="Path to a single motion npz to play instead of the cfg's default motion_files. "
         "Disables ball-related terminations and starts phase at 0 so the full clip plays once.",
)
parser.add_argument(
    "--ball_preset",
    type=str,
    default=None,
    choices=["middle", "left", "right"],
    help="Override ball launch params for testing specific motions. "
         "Keeps hit_phase from cfg (unlike --npz which sets hit_phase=0).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if "--headless" not in sys.argv:
        args_cli.headless = True
        sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.dict import print_dict

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

    if args_cli.npz is not None:
        npz_abs = os.path.abspath(args_cli.npz)
        assert os.path.isfile(npz_abs), f"--npz file not found: {npz_abs}"
        env_cfg.commands.motion.motion_files = [npz_abs]
        env_cfg.commands.motion.match_ball_direction = False
        if args_cli.ball_preset is None:
            env_cfg.commands.motion.hit_phase = 0.0
            env_cfg.commands.motion.hit_phase_noise = 0.0
            env_cfg.terminations.ball_on_own_table = None
            env_cfg.terminations.ball_missed_paddle = None
        print(f"[INFO] --npz override: motion_files = [{npz_abs}]")

    if args_cli.ball_preset is not None:
        BALL_PRESETS = {
            "middle": dict(x_range=(-0.35, -0.35), y_range=(0.0, 0.0),
                           z_range=(1.3, 1.3), vx_range=(3.5, 3.5),
                           vy_range=(0.0, 0.0), vz_range=(0.5, 0.5)),
            "left":   dict(x_range=(-0.35, -0.35), y_range=(+0.3, +0.3),
                           z_range=(1.3, 1.3), vx_range=(3.5, 3.5),
                           vy_range=(+0.4, +0.4), vz_range=(0.5, 0.5)),
            "right":  dict(x_range=(-0.35, -0.35), y_range=(-0.03, -0.03),
                           z_range=(1.3, 1.3), vx_range=(3.5, 3.5),
                           vy_range=(-0.10, -0.10), vz_range=(0.5, 0.5)),
        }
        bp = BALL_PRESETS[args_cli.ball_preset]
        env_cfg.events.reset_ball.params.update({"ball_cfg": env_cfg.events.reset_ball.params["ball_cfg"], **bp})
        env_cfg.events.relaunch_ball.params.update({"ball_cfg": env_cfg.events.relaunch_ball.params["ball_cfg"], **bp})
        env_cfg.terminations.ball_on_own_table = None
        env_cfg.terminations.ball_missed_paddle = None
        print(f"[INFO] --ball_preset={args_cli.ball_preset}: {bp}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        out_dir = os.path.abspath(args_cli.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": out_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording pure-reference video to:", out_dir)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    zero_action = torch.zeros((args_cli.num_envs, action_dim), device=device)

    print(f"[INFO] action_dim = {action_dim}, feeding zeros (residual=0, phase_speed=1.0).")

    scene = env.unwrapped.scene
    ball = scene["ball"]
    NET_X = 0.0
    NET_Z = 0.84
    TELEPORT_THRESH = 0.30  # ball position jump > 0.3m means it was relaunched

    num_trials = 0
    num_clears = 0
    cleared_this_trial = False
    prev_bx = None
    prev_bz = None
    prev_pos = None

    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_action)

        ball_pos_local = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
        bx, bz = float(ball_pos_local[0]), float(ball_pos_local[2])

        teleported = (
            prev_pos is not None
            and float(((ball_pos_local - prev_pos) ** 2).sum() ** 0.5) > TELEPORT_THRESH
        )
        if teleported:
            num_trials += 1
            if cleared_this_trial:
                num_clears += 1
            rate = num_clears / num_trials * 100.0
            print(f"[trial {num_trials:4d}] cleared={cleared_this_trial}  "
                  f"rate = {num_clears}/{num_trials} = {rate:.1f}%")
            cleared_this_trial = False
            prev_bx = None
            prev_bz = None

        if prev_bx is not None and prev_bx > NET_X >= bx and not cleared_this_trial:
            f = prev_bx / (prev_bx - bx) if prev_bx != bx else 0.0
            z_at_net = prev_bz + f * (bz - prev_bz)
            if z_at_net > NET_Z:
                cleared_this_trial = True

        prev_bx, prev_bz = bx, bz
        prev_pos = ball_pos_local.copy()

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
