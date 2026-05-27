"""V63: 在 V62 (yb_4 -0.20, yb_5 -0.40) 基础上加 +Z loft.

V62: bx=-1.85, bz=+2.63, gap=3.5cm — 反弹够但 +Z 不够过网.
策略: 让 paddle 在 hit 时刻有更多 +Z 速度
  - yb_1 snap (shoulder_pitch +Z): t=0.55 加大 yb_1 (越大 = 越往上抬)
  - yb_6 snap (wrist_pitch +Z): t=0.55 加大 yb_6 (越正 = 腕向上掀)
  - 同时保持 V62 的 PIN @ 0.475 不变, 不破坏击球位置

V62 hit window keys (V61 + yb_4 -0.20, yb_5 -0.40):
  (0.300, [+1.127, +0.198, -1.904, +1.027, -0.515, -1.045,  +1.000])
  (0.400, [+1.187, +0.103, -1.979, +0.657, -0.515, -1.245,  +1.000])
  (0.475, [+1.287, +0.103, -1.979, +0.607, -0.545, -1.020,  +1.000])  PIN
  (0.550, [+1.437, +0.103, -1.979, +0.557, -0.515, -0.495,  +1.000])  snap
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
        # 跟踪到 ball 落地或飞远
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
        # find hit moment (closest paddle-ball)
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[2]))
        post = [r for r in log if r[0] > hit[0] + 0.01]  # after hit
        bx_min = min(r[3][0] for r in post) if post else 0
        bz_max = max(r[3][2] for r in post) if post else 0
        # ball trajectory after hit: peak Z and Z at x=0 (net cross)
        z_peak = max(r[2][2] for r in post) if post else 0
        # find when ball.x crosses 0 (net plane), report ball z
        z_at_net = None
        for i in range(1, len(post)):
            x_prev, x_now = post[i-1][2][0], post[i][2][0]
            if x_prev > 0 and x_now <= 0:
                # interpolate
                f = x_prev / (x_prev - x_now)
                z_at_net = post[i-1][2][2] + f * (post[i][2][2] - post[i-1][2][2])
                break
        # ball x at z=0.79 (table level, second bounce)
        x_at_table = None
        for i in range(1, len(post)):
            z_prev, z_now = post[i-1][2][2], post[i][2][2]
            if z_prev > 0.79 and z_now <= 0.79 and post[i-1][2][0] < 1.0:
                f = (z_prev - 0.79) / (z_prev - z_now)
                x_at_table = post[i-1][2][0] + f * (post[i][2][0] - post[i-1][2][0])
                break
        return hit, bx_min, bz_max, z_peak, z_at_net, x_at_table, log

    print(f"\n{'variant':<60}  {'gap':>5} {'bx':>6} {'bz':>6} {'z_peak':>7} {'z_at_net':>9} {'x_at_table':>11}")
    probes = [
        ("V62 baseline", V62),
        # yb_4 snap (elbow 屈 → 前臂上摆 → +Z)
        ("yb_4 snap +0.80", edit_kf(V62, 0.550, 3, +0.80)),
        ("yb_4 snap +1.00", edit_kf(V62, 0.550, 3, +1.00)),
        ("yb_4 snap +1.20", edit_kf(V62, 0.550, 3, +1.20)),
        # combo: yb_4 snap + yb_1 snap (双 +Z 源)
        ("yb_4 snap +0.80 + yb_1 snap +1.55",
            edit_kf(edit_kf(V62, 0.550, 3, +0.80), 0.550, 0, +1.55)),
        ("yb_4 snap +1.00 + yb_1 snap +1.55",
            edit_kf(edit_kf(V62, 0.550, 3, +1.00), 0.550, 0, +1.55)),
        ("yb_4 snap +1.00 + yb_1 snap +1.60",
            edit_kf(edit_kf(V62, 0.550, 3, +1.00), 0.550, 0, +1.60)),
        # yb_2 snap (略外展, 加 +Z) — 注意 yb_2 限位 +0.314
        ("yb_2 snap +0.20 (limit 0.31)", edit_kf(V62, 0.550, 1, +0.30)),
        ("yb_2 snap +0.20 + yb_4 snap +0.80",
            edit_kf(edit_kf(V62, 0.550, 1, +0.30), 0.550, 3, +0.80)),
        # 更激进: yb_4 snap + yb_2 snap + yb_1 snap
        ("yb_4 snap +1.00 + yb_2 snap +0.20 + yb_1 snap +1.55",
            edit_kf(edit_kf(edit_kf(V62, 0.550, 3, +1.00), 0.550, 1, +0.30), 0.550, 0, +1.55)),
    ]

    for label, keys in probes:
        hit, bx, bz, zp, zn, xt, log = run(keys)
        gap = np.linalg.norm(hit[1] - hit[2]) * 100
        zn_str = f"{zn:.3f}" if zn is not None else "miss"
        xt_str = f"{xt:+.3f}" if xt is not None else "n/a"
        print(f"{label:<60}  {gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zp:>+7.3f}  {zn_str:>9} {xt_str:>11}")

    print(f"\n  z_at_net: ball z 在过网平面 (x=0). 网顶高 = 0.94m. > 0.94 才能过网.")
    print(f"  x_at_table: ball 落到 table z=0.79 时的 x. < 0 = 落在对方半场.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
