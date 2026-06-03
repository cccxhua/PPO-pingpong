"""V65b: 直接测 face_normal world Z 分量 + bz + zn.

用户反馈 "球拍平直/从上往下": face_normal world Z 不够大.
V64 paddle vz at hit = -0.21 改不了 (PD swing 物理特性).
关键是 face 朝向. yb_5 已 +0.15, 但视觉上 face 还不够朝上.
更激进 yb_5 + yb_6 反向探针.

paddle face_normal_body = (0, 0, -1) (paddle 沿 -Z body axis).
face_n_world = -R[:,2]. 要让 face_n_z 接近 +1.0 (朝正上), face 倾斜更大.
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


V64 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.027, -0.515, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.657, -0.515, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.607, -0.545, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.557, -0.365, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

FACE_BODY = np.array([0.0, 0.0, -1.0])


def edit_kf(keys, t, j, new_val):
    out = []
    for kt, vals in keys:
        v = list(vals)
        if abs(kt - t) < 1e-6:
            v[j] = new_val
        out.append((kt, v))
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


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


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
            q = robot.data.body_quat_w[0, paddle_idx].cpu().numpy().copy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, q, bp, bv))
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[3]))
        R = quat_to_R(hit[2])
        face_n = R @ FACE_BODY
        post = [r for r in log if r[0] > hit[0] + 0.01]
        bx = min(r[4][0] for r in post)
        bz = max(r[4][2] for r in post)
        zn = None
        for i in range(1, len(post)):
            xp, xn = post[i-1][3][0], post[i][3][0]
            if xp > 0 and xn <= 0:
                f = xp / (xp - xn)
                zn = post[i-1][3][2] + f * (post[i][3][2] - post[i-1][3][2])
                break
        gap = np.linalg.norm(hit[1] - hit[3]) * 100
        return face_n, gap, bx, bz, zn

    print(f"\n{'variant':<55}  {'face_n (xyz)':<22}  {'gap':>5} {'bx':>6} {'bz':>6} {'zn':>6}")

    probes = [
        ("V64 baseline", V64),
        # 反方向: yb_6 snap MORE negative (反向 wrist tilt)
        ("yb_6 snap -0.60 (was -0.495)", edit_kf(V64, 0.550, 5, -0.60)),
        ("yb_6 snap -0.80", edit_kf(V64, 0.550, 5, -0.80)),
        ("yb_6 snap -1.00", edit_kf(V64, 0.550, 5, -1.00)),
        # yb_6 hit-window 全部 - (整体 wrist 角度)
        ("yb_6 hit-window -0.20", apply_delta(V64, {5: -0.20})),
        ("yb_6 hit-window +0.20", apply_delta(V64, {5: +0.20})),
        # yb_5 更激进 less negative
        ("yb_5 snap -0.20", edit_kf(V64, 0.550, 4, -0.20)),
        ("yb_5 snap -0.10", edit_kf(V64, 0.550, 4, -0.10)),
        ("yb_5 snap 0.00", edit_kf(V64, 0.550, 4, 0.00)),
        # yb_5 反方向更负
        ("yb_5 snap -0.50", edit_kf(V64, 0.550, 4, -0.50)),
        ("yb_5 snap -0.65", edit_kf(V64, 0.550, 4, -0.65)),
        # yb_5 hit-window 整体调
        ("yb_5 hit-window +0.10 (less rolled all)", apply_delta(V64, {4: +0.10})),
        ("yb_5 hit-window +0.15", apply_delta(V64, {4: +0.15})),
        ("yb_5 hit-window +0.20", apply_delta(V64, {4: +0.20})),
        # yb_2 hit-window 上限 (shoulder roll 外展)
        ("yb_2 hit-window +0.10", apply_delta(V64, {1: +0.10})),
        # 组合: yb_5 hit-window +0.15 + yb_2 +0.10
        ("yb_5 hw+0.15 + yb_2 hw+0.10", apply_delta(V64, {4: +0.15, 1: +0.10})),
        # yb_7 (wrist_yaw) 改 paddle yaw
        ("yb_7 snap +1.30", edit_kf(V64, 0.550, 6, +1.30)),
        ("yb_7 snap +0.70", edit_kf(V64, 0.550, 6, +0.70)),
        ("yb_7 hit-window +0.30", apply_delta(V64, {6: +0.30})),
        ("yb_7 hit-window -0.30", apply_delta(V64, {6: -0.30})),
    ]

    rows = []
    for label, keys in probes:
        face_n, gap, bx, bz, zn = run(keys)
        zn_s = f"{zn:.3f}" if zn else "miss"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        print(f"{label:<55}  ({face_n[0]:+.2f},{face_n[1]:+.2f},{face_n[2]:+.2f})  "
              f"{gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zn_s:>6} {clears}")
        rows.append((label, face_n[2], bz, zn, gap, bx))

    print(f"\n=== face_n Z 排序 (越正越上仰) ===")
    rows.sort(key=lambda r: -r[1])
    for label, fz, bz, zn, gap, bx in rows[:8]:
        zn_s = f"{zn:.3f}" if zn else "miss"
        print(f"  fz={fz:+.2f}  bz={bz:+.2f}  zn={zn_s}  gap={gap:.1f}  bx={bx:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
