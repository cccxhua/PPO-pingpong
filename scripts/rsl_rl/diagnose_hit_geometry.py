"""Diagnose paddle-ball geometry at hit moment (residual = 0).

Runs the pure reference motion (no policy) for several ball cycles and
records, for each timestep where the motion phase is in [0.30, 0.55],
the world positions of the paddle and ball plus their distance.

Outputs the per-cycle minimum distance and the timestep where it occurred,
so you can see what the *reference motion alone* achieves.

Usage:
    python scripts/rsl_rl/diagnose_hit_geometry.py --task X1-TableTennis
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose paddle-ball geometry.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_cycles", type=int, default=4, help="ball cycles to record")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if "--headless" not in sys.argv:
    args_cli.headless = True
    sys.argv.append("--headless")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

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

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]
    device = env.unwrapped.device
    zero_action = torch.zeros((args_cli.num_envs, action_dim), device=device)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    racket_body_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    cmd = env.unwrapped.command_manager.get_term("motion")
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]

    step_dt = env.unwrapped.step_dt
    print(f"[INFO] step_dt = {step_dt:.4f}s, action_dim = {action_dim}")
    print(f"[INFO] paddle body idx = {racket_body_idx}")
    print(f"[INFO] running {args_cli.num_cycles} ball cycles, recording phase ∈ [0.30, 0.55]")
    print()

    cycle_records = []  # list of (cycle_id, [(t, phase, ball_pos, paddle_pos, dist), ...])
    cur_cycle = []
    cur_cycle_id = 0
    prev_was_hit = False
    sim_t = 0.0

    max_cycles = args_cli.num_cycles
    max_steps = int(15.0 * max_cycles / step_dt)  # safety cap

    for step in range(max_steps):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_action)
        sim_t += step_dt

        phase = cmd.phase[0].item()
        ball_pos = ball.data.root_pos_w[0, :3].cpu().numpy()
        paddle_pos = robot.data.body_pos_w[0, racket_body_idx].cpu().numpy()
        env_origin = scene.env_origins[0].cpu().numpy()
        ball_local = ball_pos - env_origin
        paddle_local = paddle_pos - env_origin
        dist = float(((ball_pos - paddle_pos) ** 2).sum() ** 0.5)
        joint_actual = [robot.data.joint_pos[0, jid].item() for jid in yb_joint_ids]
        joint_target = cmd.ref_dof[0].cpu().numpy().tolist()

        in_window = 0.30 <= phase <= 0.55
        if in_window:
            cur_cycle.append((sim_t, phase, ball_local.copy(), paddle_local.copy(), dist, joint_actual, joint_target))
            prev_was_hit = True
        else:
            if prev_was_hit and cur_cycle:
                cycle_records.append((cur_cycle_id, cur_cycle))
                cur_cycle_id += 1
                cur_cycle = []
                if cur_cycle_id >= max_cycles:
                    break
            prev_was_hit = False

    if cur_cycle:
        cycle_records.append((cur_cycle_id, cur_cycle))

    print(f"[RESULT] recorded {len(cycle_records)} cycles\n")
    print(" === per-cycle minimum-distance frame === ")
    print(f" {'cyc':>3}  {'t':>6}  {'phase':>5}  {'ball (x,y,z) local':>25}  {'paddle (x,y,z) local':>25}  {'d_min':>6}  {'dxyz':>22}")
    for cid, frames in cycle_records:
        best = min(frames, key=lambda f: f[4])
        t, ph, b, p, d, ja, jt = best
        dxyz = b - p
        print(f" {cid:>3d}  {t:>6.3f}  {ph:>5.3f}  ({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  {d:>6.3f}  ({dxyz[0]:+.3f},{dxyz[1]:+.3f},{dxyz[2]:+.3f})")
        print(f"      joint actual:  yb_1={ja[0]:+.3f} yb_2={ja[1]:+.3f} yb_3={ja[2]:+.3f} yb_4={ja[3]:+.3f} yb_5={ja[4]:+.3f} yb_6={ja[5]:+.3f} yb_7={ja[6]:+.3f}")
        print(f"      joint target:  yb_1={jt[0]:+.3f} yb_2={jt[1]:+.3f} yb_3={jt[2]:+.3f} yb_4={jt[3]:+.3f} yb_5={jt[4]:+.3f} yb_6={jt[5]:+.3f} yb_7={jt[6]:+.3f}")

    print("\n === full first cycle === ")
    if cycle_records:
        _, frames = cycle_records[0]
        print(f" {'t':>6}  {'phase':>5}  {'ball.x':>7}  {'ball.z':>7}  {'pad.x':>7}  {'pad.z':>7}  {'dist':>6}  {'dx':>6}  {'dz':>6}")
        for t, ph, b, p, d, ja, jt in frames:
            print(f" {t:>6.3f}  {ph:>5.3f}  {b[0]:>7.3f}  {b[2]:>7.3f}  {p[0]:>7.3f}  {p[2]:>7.3f}  {d:>6.3f}  {b[0]-p[0]:>+6.3f}  {b[2]-p[2]:>+6.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
