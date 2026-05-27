"""V66: V65 face 朝上 OK, z_at_net=1.298 网余 35cm, 但 bx=-1.37 反弹弱, 球飞距不够.
需要增大 paddle |vx| 同时保留 face Z 角度.

策略 (V65 = V64 + yb_5 hit-window +0.20):
  1. yb_1 hit-window 整体 +0.10/+0.20 (paddle 抬高 + 加速)
  2. yb_4 hit-window 整体 -0.10/-0.20 (前臂伸更直, paddle 更前 + 速度大)
  3. yb_1 windup -0.20 + snap +0.20 (扩大 swing 幅度, 提高峰值 ω)
  4. yb_4 windup deeper + snap extend (前臂大幅 swing)
  5. 组合
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


# V65 (current create_forehand.py)
V65 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.027, -0.315, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.657, -0.315, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.607, -0.345, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.557, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def edit_kf(keys, t, j, new_val):
    out = []
    for kt, vals in keys:
        v = list(vals)
        if abs(kt - t) < 1e-6:
            v[j] = new_val
        out.append((kt, v))
    return out


def edits(keys, mods):
    out = keys
    for t, j, v in mods:
        out = edit_kf(out, t, j, v)
    return out


def apply_delta(keys, deltas, t_lo=0.29, t_hi=0.56):
    out = []
    for t, vals in keys:
        v = list(vals)
        if t_lo < t < t_hi:
            for j, d in deltas.items():
                v[j] += d
        out.append((t, v))
    return out


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
            v = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, v, bp, bv))
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[3]))
        post = [r for r in log if r[0] > hit[0] + 0.01]
        bx = min(r[4][0] for r in post)
        bz = max(r[4][2] for r in post)
        zn = None
        x_table = None  # ball lands on table (z=0.79) after net (x<0)
        for i in range(1, len(post)):
            xp, xn = post[i-1][3][0], post[i][3][0]
            if xp > 0 and xn <= 0:
                f = xp / (xp - xn)
                zn = post[i-1][3][2] + f * (post[i][3][2] - post[i-1][3][2])
            zp_, zn_ = post[i-1][3][2], post[i][3][2]
            if zp_ > 0.79 and zn_ <= 0.79 and post[i-1][3][0] < 0:
                f = (zp_ - 0.79) / (zp_ - zn_)
                x_table = post[i-1][3][0] + f * (post[i][3][0] - post[i-1][3][0])
                break
        return hit, bx, bz, zn, x_table

    print(f"\n{'variant':<60} {'paddle vx':<8} {'gap':>5} {'bx':>6} {'bz':>6} {'zn':>6} {'x_land':>7}")

    probes = [
        ("V65 baseline", V65),
        # 路径 1: yb_1 hit-window 整体 +0.10/+0.20 (paddle 抬 + 加速)
        ("yb_1 hw +0.10 (paddle 抬高)", apply_delta(V65, {0: +0.10})),
        ("yb_1 hw +0.20", apply_delta(V65, {0: +0.20})),
        # 路径 2: yb_4 hit-window 整体 -0.10/-0.20 (前臂伸更直 + paddle 更前)
        ("yb_4 hw -0.10", apply_delta(V65, {3: -0.10})),
        ("yb_4 hw -0.20", apply_delta(V65, {3: -0.20})),
        ("yb_4 hw -0.30", apply_delta(V65, {3: -0.30})),
        # 路径 3: yb_1 windup 减 + snap 加 (扩大 swing)
        ("yb_1 windup -0.10 + snap +0.10",
            edits(V65, [(0.400, 0, +1.087), (0.550, 0, +1.537)])),
        ("yb_1 windup -0.20 + snap +0.20",
            edits(V65, [(0.400, 0, +0.987), (0.550, 0, +1.637)])),
        # 路径 4: yb_4 windup curl 深 + snap extend
        ("yb_4 windup +0.30 + snap -0.20",
            edits(V65, [(0.400, 3, +0.957), (0.550, 3, +0.357)])),
        ("yb_4 windup +0.50 + snap -0.30",
            edits(V65, [(0.400, 3, +1.157), (0.550, 3, +0.257)])),
        # 路径 5: 综合 paddle 整体抬高 + 前臂伸 + 大 swing
        ("yb_1 hw +0.10 + yb_4 hw -0.10",
            apply_delta(V65, {0: +0.10, 3: -0.10})),
        ("yb_1 hw +0.10 + yb_4 hw -0.20",
            apply_delta(V65, {0: +0.10, 3: -0.20})),
        ("yb_1 hw +0.20 + yb_4 hw -0.20",
            apply_delta(V65, {0: +0.20, 3: -0.20})),
        # 路径 6: PIN 提前 (paddle 在 PIN 时更靠前)
        ("yb_1 PIN +0.10 (1.387)", edit_kf(V65, 0.475, 0, +1.387)),
        ("yb_4 PIN -0.10 (0.507)", edit_kf(V65, 0.475, 3, +0.507)),
        # combo: PIN advance + snap power
        ("yb_1 PIN+0.10 + snap+0.10",
            edits(V65, [(0.475, 0, +1.387), (0.550, 0, +1.537)])),
    ]

    rows = []
    for label, keys in probes:
        (t_, p, v, bp, bv), bx, bz, zn, xt = run(keys)
        gap = np.linalg.norm(p - bp) * 100
        zn_s = f"{zn:.3f}" if zn else "miss"
        xt_s = f"{xt:+.2f}" if xt else "n/a"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        print(f"{label:<60} {v[0]:+.2f}    {gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zn_s:>6} {xt_s:>7} {clears}")
        rows.append((label, v[0], bx, bz, zn, gap, xt))

    print(f"\n=== Top 8 by bx (most negative = hardest hit) with zn > 0.94 and gap < 5 ===")
    valid = [r for r in rows if r[4] is not None and r[4] > 0.94 and r[5] < 5.0]
    valid.sort(key=lambda r: r[2])
    for label, vx, bx, bz, zn, gap, xt in valid[:8]:
        zn_s = f"{zn:.3f}"
        xt_s = f"{xt:+.2f}" if xt else "n/a"
        print(f"  bx={bx:+.2f}  bz={bz:+.2f}  zn={zn_s}  x_land={xt_s}  gap={gap:.1f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
