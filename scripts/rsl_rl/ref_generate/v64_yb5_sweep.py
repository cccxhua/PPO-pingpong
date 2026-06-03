"""V64 yb_5 sweep: 找最大 ball post-vz.

V64_paddle_vz 关键发现:
  - paddle vz at hit 全部为负 (-0.17~-0.28), 不是 +Z 速度来源
  - bz 来自 face_normal Z 分量 (paddle 反射方向)
  - yb_5 snap +0.10 给 bz +3.19 (best so far, vs V62 +2.89)
  - 即 yb_5 less negative = face 更竖 = ball 更上抛

V62 base (yb_5 snap = -0.515, NO yb_4 snap +0.80 because那让 bz 反降)
本脚本: yb_5 snap 从 -0.515 扫到 -0.215, 步长 0.05
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


# V62 base = no yb_4 snap +0.80 (V63 misdirection)
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
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, bp, bv))
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[2]))
        post = [r for r in log if r[0] > hit[0] + 0.01]
        bx_min = min(r[3][0] for r in post)
        bz_max = max(r[3][2] for r in post)
        z_peak = max(r[2][2] for r in post)
        z_at_net = None
        for i in range(1, len(post)):
            x_prev, x_now = post[i-1][2][0], post[i][2][0]
            if x_prev > 0 and x_now <= 0:
                f = x_prev / (x_prev - x_now)
                z_at_net = post[i-1][2][2] + f * (post[i][2][2] - post[i-1][2][2])
                break
        return hit, bx_min, bz_max, z_peak, z_at_net

    print(f"\n{'variant':<55}  {'gap':>5} {'bx':>6} {'bz':>6} {'z_pk':>6} {'z_at_net':>9}")
    probes = [
        ("V62 baseline (yb_5 snap=-0.515)", V62),
        ("yb_5 snap -0.465 (+0.05)", edit_kf(V62, 0.550, 4, -0.465)),
        ("yb_5 snap -0.415 (+0.10)", edit_kf(V62, 0.550, 4, -0.415)),
        ("yb_5 snap -0.365 (+0.15)", edit_kf(V62, 0.550, 4, -0.365)),
        ("yb_5 snap -0.315 (+0.20)", edit_kf(V62, 0.550, 4, -0.315)),
        ("yb_5 snap -0.265 (+0.25)", edit_kf(V62, 0.550, 4, -0.265)),
        ("yb_5 snap -0.215 (+0.30)", edit_kf(V62, 0.550, 4, -0.215)),
        # combos with yb_5 +0.15: 加点 yb_4 -0.05 看能否保 z_at_net
        ("yb_5 +0.15 + yb_4 snap -0.05", edit_kf(edit_kf(V62, 0.550, 4, -0.365), 0.550, 3, +0.507)),
        ("yb_5 +0.15 + yb_4 snap +0.05", edit_kf(edit_kf(V62, 0.550, 4, -0.365), 0.550, 3, +0.607)),
        ("yb_5 +0.20 + yb_4 snap -0.10", edit_kf(edit_kf(V62, 0.550, 4, -0.315), 0.550, 3, +0.457)),
        # combo: yb_5 +0.10 + yb_6 snap +0.30 (yb_6 helps z_at_net, yb_5 helps bz)
        ("yb_5 +0.10 + yb_6 snap +0.30", edit_kf(edit_kf(V62, 0.550, 4, -0.415), 0.550, 5, -0.195)),
        ("yb_5 +0.15 + yb_6 snap +0.30", edit_kf(edit_kf(V62, 0.550, 4, -0.365), 0.550, 5, -0.195)),
    ]

    best = None
    for label, keys in probes:
        (t_, p, bp, bv), bx, bz, zp, zn = run(keys)
        gap = np.linalg.norm(p - bp) * 100
        zn_str = f"{zn:.3f}" if zn is not None else "miss"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        print(f"{label:<55}  {gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zp:>+6.3f} {zn_str:>9} {clears}")
        # 评分: bz 越大越好, 但要求 zn > 0.94 (过网) and gap < 5
        if zn is not None and zn > 0.94 and gap < 5.0:
            score = bz
            if best is None or score > best[0]:
                best = (score, label, bx, bz, zn, gap)

    if best:
        print(f"\n  最佳 (bz max, 过网, gap<5cm): {best[1]}")
        print(f"    bx={best[2]:+.2f}, bz={best[3]:+.2f}, z_at_net={best[4]:.3f}, gap={best[5]:.1f}cm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
