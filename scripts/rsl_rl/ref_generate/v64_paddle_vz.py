"""V64: 用户要求 ball post-vz 提升. V63 加 yb_4 snap +0.80 反而让 bz 从 +2.89 降到 +2.39!
回退 V62 base, 找真能提升 paddle vz at hit 的关节变化.

打印 paddle vel at hit moment, 找 +Z 速度最大的变体.
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


# V62 base = V61 + (yb_4 -0.20, yb_5 -0.40), 没有 yb_4 snap 修改
V62 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.027, -0.515, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.657, -0.515, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.607, -0.545, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.557, -0.515, -0.495,  +1.000]),
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
        bx_min = min(r[4][0] for r in post)
        bz_max = max(r[4][2] for r in post)
        z_peak = max(r[3][2] for r in post)
        z_at_net = None
        for i in range(1, len(post)):
            x_prev, x_now = post[i-1][3][0], post[i][3][0]
            if x_prev > 0 and x_now <= 0:
                f = x_prev / (x_prev - x_now)
                z_at_net = post[i-1][3][2] + f * (post[i][3][2] - post[i-1][3][2])
                break
        return hit, bx_min, bz_max, z_peak, z_at_net

    print(f"\n{'variant':<55}  {'paddle v':<22} {'gap':>5} {'bx':>6} {'bz':>6} {'z_pk':>6} {'z_at_net':>9}")
    probes = [
        ("V62 baseline (no yb_4 snap mod)", V62),
        # 直接调 yb_1 snap 提高肩抬, 看 paddle vz at hit
        ("yb_1 snap +1.55", edit_kf(V62, 0.550, 0, +1.55)),
        ("yb_1 snap +1.60", edit_kf(V62, 0.550, 0, +1.60)),
        ("yb_1 snap +1.65", edit_kf(V62, 0.550, 0, +1.65)),
        ("yb_1 snap +1.70", edit_kf(V62, 0.550, 0, +1.70)),
        ("yb_1 snap +1.80", edit_kf(V62, 0.550, 0, +1.80)),
        # yb_1 hit-window 整体上抬 (PIN 也抬, 改击球位置)
        ("yb_1 hit-window +0.05 (PIN 改)", apply_delta(V62, {0: +0.05})),
        # PIN 不改, 改 windup 更低 (让 PIN 时 yb_1 在加速中)
        ("yb_1 windup -0.10 (1.187→1.087)", edit_kf(V62, 0.400, 0, +1.087)),
        ("yb_1 windup -0.15 + snap +1.55",
            edit_kf(edit_kf(V62, 0.400, 0, +1.037), 0.550, 0, +1.55)),
        # 早 windup: 加新 keyframe at t=0.350 (early windup)
        # skip — 复杂
        # 加大 face_normal +Z 同时维持 -X: yb_5 调
        ("yb_5 snap +0.10 (less rolled)", edit_kf(V62, 0.550, 4, -0.415)),
        # yb_6 snap (wrist 上掀)
        ("yb_6 snap +0.10 (was -0.495)", edit_kf(V62, 0.550, 5, +0.10)),
        ("yb_6 snap +0.30", edit_kf(V62, 0.550, 5, +0.30)),
        # combo: V62 + yb_1 snap +1.60 + yb_6 snap +0.10
        ("yb_1 snap +1.60 + yb_6 snap +0.10",
            edit_kf(edit_kf(V62, 0.550, 0, +1.60), 0.550, 5, +0.10)),
    ]

    for label, keys in probes:
        (t_, p, v, bp, bv), bx, bz, zp, zn = run(keys)
        gap = np.linalg.norm(p - bp) * 100
        zn_str = f"{zn:.3f}" if zn is not None else "miss"
        print(f"{label:<55}  ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})  {gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zp:>+6.3f} {zn_str:>9}")

    print(f"\n  paddle v 是 hit 时刻 paddle linear velocity, 关注 vz 分量 (越正越能给 ball +Z)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
