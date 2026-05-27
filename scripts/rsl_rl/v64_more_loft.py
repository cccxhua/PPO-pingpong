"""V64: 在 V63 基础上微调 paddle 面朝向更朝上 (+Z), 让球过网余量更大.

V63: bx=-2.00, bz=+2.39, z_at_net=1.067 (过网仅 +12.5cm sim 测), 实际视频球过不了网.
策略: hit window 全部 yb_6 (wrist_pitch) 略增加, 让 paddle 头略上扬.
  - V63 yb_6: mid=-1.045, windup=-1.245, PIN=-1.020, snap=-0.495
  - 整体 yb_6 加 +0.10 / +0.15 / +0.20 看 face_normal Z 变化和球过网情况

也试 yb_5 微调 (face 围 handle 翻更竖) 备选.
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


V63 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.027, -0.515, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.657, -0.515, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.607, -0.545, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.800, -0.515, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

FACE_NORMAL_BODY = np.array([0.0, 0.0, -1.0])


def apply_delta(keys, deltas, t_lo=0.29, t_hi=0.56):
    out = []
    for t, vals in keys:
        v = list(vals)
        if t_lo < t < t_hi:
            for j, d in deltas.items():
                v[j] += d
        out.append((t, v))
    return out


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    ball = scene["ball"]
    device = env.unwrapped.device

    paddle_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    yb_joint_names = [f"joint_yb_{i}" for i in range(1, 8)]
    yb_joint_ids = [robot.find_joints(n)[0][0] for n in yb_joint_names]

    def run(keys):
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")
        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(0.0)
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(env.unwrapped.sim.get_physics_dt())
        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
        ball_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        sim_dt = float(env.unwrapped.sim.get_physics_dt())
        log = []
        for step in range(int(1.5 / sim_dt)):
            t = step * sim_dt
            target = spline(min(t, 1.0))
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
            q = robot.data.body_quat_w[0, paddle_idx].clone()
            R = quat_to_R(q).cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, R, bp, bv))
        # closest paddle-ball moment
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[3]))
        t_hit = hit[0]
        post = [r for r in log if r[0] > t_hit + 0.01]
        bx_min = min(r[4][0] for r in post)
        bz_max = max(r[4][2] for r in post)
        z_peak = max(r[3][2] for r in post)
        # ball z at x=0 (net plane) — first crossing
        z_at_net = None
        for i in range(1, len(post)):
            x_prev, x_now = post[i-1][3][0], post[i][3][0]
            if x_prev > 0 and x_now <= 0:
                f = x_prev / (x_prev - x_now)
                z_at_net = post[i-1][3][2] + f * (post[i][3][2] - post[i-1][3][2])
                break
        return hit, bx_min, bz_max, z_peak, z_at_net

    print(f"\n{'variant':<55}  {'face_n (xyz)':<24}  {'gap':>5} {'bx':>6} {'bz':>6} {'z_pk':>6} {'z_at_net':>9}")
    probes = [
        ("V63 baseline", {}),
        # yb_2 snap (shoulder_roll 外展, 给 ball +Z, V62c 测过 1.129)
        ("yb_2 snap only +0.18", {1: +0.18}),  # NOTE: this adds to ALL hit-window keys
        # 真的只改 snap (单 keyframe): need different approach
        # yb_1 整 +Z: 上抬 shoulder
        ("yb_1 hit-window +0.10 (整体上抬)", {0: +0.10}),
        ("yb_1 hit-window +0.15", {0: +0.15}),
        ("yb_1 hit-window +0.20", {0: +0.20}),
        # yb_4 整体减 (更伸肘, paddle 更前 + 更高一点)
        ("yb_4 hit-window -0.10", {3: -0.10}),
        # yb_2 整体加 (注意限位 +0.314)
        ("yb_2 hit-window +0.10 (限位+0.31)", {1: +0.10}),
        ("yb_2 hit-window +0.15", {1: +0.15}),
        # 组合: yb_1 +0.10 + yb_2 +0.10
        ("yb_1 +0.10, yb_2 +0.10", {0: +0.10, 1: +0.10}),
        ("yb_1 +0.15, yb_2 +0.10", {0: +0.15, 1: +0.10}),
    ]

    for label, deltas in probes:
        keys = apply_delta(V63, deltas) if deltas else V63
        (t_, p, R, bp, bv), bx, bz, zp, zn = run(keys)
        n = R @ FACE_NORMAL_BODY
        gap = np.linalg.norm(p - bp) * 100
        zn_str = f"{zn:.3f}" if zn is not None else "miss"
        print(f"{label:<55}  ({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  {gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zp:>+6.3f} {zn_str:>9}")

    print(f"\n  目标: face_normal Z 分量 ↑, gap < 5cm, bx < -1.5, z_at_net > 1.0 (网顶 0.94)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
