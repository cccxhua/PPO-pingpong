"""V65: 用户要求 "球拍从下往上给球力" — paddle vz at hit 必须 > 0.

V64 paddle vz = -0.17 m/s (向下走), 球拍在向下/平移挥, 不是上挥.
诊断: PD 跟踪让击球瞬间关节角速度跟当前 windup→PIN 之间梯度走.
  当前 V64 hit window:
    t=0.400 (windup): yb_1=1.187, yb_4=0.657, yb_6=-1.245
    t=0.475 (PIN):    yb_1=1.287, yb_4=0.607, yb_6=-1.020
    t=0.550 (snap):   yb_1=1.437, yb_4=0.557, yb_6=-0.495
  PIN 处 yb_1 ω = +1.33 rad/s (正向, 抬肩), yb_4 ω = -0.67 rad/s (伸肘),
  yb_6 ω = +3 rad/s (腕掀). 但合成 paddle linear vz 还是 -0.17.

V65 策略 (要 paddle vz at hit > 0):
  路径 A: yb_1 windup LOWER (e.g., 0.987 vs 1.187), PIN 不变 → yb_1 ω 在 PIN 翻倍
    肩抬加速度大, 应该把 paddle 上推
  路径 B: yb_4 windup CURL DEEPER (e.g., 1.157 vs 0.657), PIN 不变 → yb_4 在 PIN 处快速伸肘
    前臂从弯到伸的角速度大, paddle 上挥
  路径 C: yb_6 windup CURL DEEPER, snap 也大幅伸 → wrist 上掀
  路径 D: 组合 A + B (双源)

注意: PIN 不动, 只改 windup. 这样击球位置不变.
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


# V64 (current create_forehand.py)
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


def edit_kf(keys, t, j, new_val):
    out = []
    for kt, vals in keys:
        v = list(vals)
        if abs(kt - t) < 1e-6:
            v[j] = new_val
        out.append((kt, v))
    return out


def edits(keys, mods):
    """mods: list of (t, j, new_val)"""
    out = keys
    for t, j, v in mods:
        out = edit_kf(out, t, j, v)
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
        z_at_net = None
        for i in range(1, len(post)):
            x_prev, x_now = post[i-1][3][0], post[i][3][0]
            if x_prev > 0 and x_now <= 0:
                f = x_prev / (x_prev - x_now)
                z_at_net = post[i-1][3][2] + f * (post[i][3][2] - post[i-1][3][2])
                break
        return hit, bx_min, bz_max, z_at_net

    print(f"\n{'variant':<55} {'paddle v at hit':<22}  {'gap':>5} {'bx':>6} {'bz':>6} {'z_net':>6}")
    probes = [
        ("V64 baseline", V64),
        # 路径 A: yb_1 windup LOWER (PIN 不变, hit 时刻 yb_1 ω 大)
        ("A1: yb_1 windup -0.10 (1.087)", edit_kf(V64, 0.400, 0, +1.087)),
        ("A2: yb_1 windup -0.20 (0.987)", edit_kf(V64, 0.400, 0, +0.987)),
        ("A3: yb_1 windup -0.30 (0.887)", edit_kf(V64, 0.400, 0, +0.887)),
        # 路径 B: yb_4 windup CURL DEEPER (前臂弯, 上挥力)
        ("B1: yb_4 windup +0.20 (0.857)", edit_kf(V64, 0.400, 3, +0.857)),
        ("B2: yb_4 windup +0.40 (1.057)", edit_kf(V64, 0.400, 3, +1.057)),
        ("B3: yb_4 windup +0.60 (1.257)", edit_kf(V64, 0.400, 3, +1.257)),
        ("B4: yb_4 windup +0.80 (1.457)", edit_kf(V64, 0.400, 3, +1.457)),
        # 路径 C: yb_6 windup curl 更深 + snap 不变 (wrist whip)
        ("C1: yb_6 windup -1.245 (no chg, ref)", V64),  # 已是 V64 本身
        ("C2: yb_6 windup -1.30 (-0.06)", edit_kf(V64, 0.400, 5, -1.30)),
        # 路径 D: 组合 A2 + B2
        ("D1: A2+B2 (yb_1 0.987, yb_4 1.057)",
            edits(V64, [(0.400, 0, +0.987), (0.400, 3, +1.057)])),
        ("D2: A1+B2", edits(V64, [(0.400, 0, +1.087), (0.400, 3, +1.057)])),
        ("D3: A1+B3", edits(V64, [(0.400, 0, +1.087), (0.400, 3, +1.257)])),
        # 也试同时改 PIN @ 0.475 略低 (paddle 在 PIN 时 LOW, snap 后 HIGH)
        ("E1: PIN yb_1 -0.05 (1.237)", edit_kf(V64, 0.475, 0, +1.237)),
        ("E2: PIN yb_1 -0.10 (1.187)", edit_kf(V64, 0.475, 0, +1.187)),
        # 还有一个: 提前 mid (t=0.300) 也压低
        ("F1: mid+windup yb_4 deeper",
            edits(V64, [(0.300, 3, +1.227), (0.400, 3, +1.057)])),
    ]

    rows = []
    for label, keys in probes:
        (t_, p, v, bp, bv), bx, bz, zn = run(keys)
        gap = np.linalg.norm(p - bp) * 100
        zn_str = f"{zn:.3f}" if zn is not None else "miss"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        vz_mark = "↑" if v[2] > 0 else "↓"
        print(f"{label:<55} ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}){vz_mark}  "
              f"{gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zn_str:>6} {clears}")
        rows.append((label, v[2], bx, bz, zn, gap))

    # 最佳: 要求 paddle vz > 0 OR 显著大于 V64 的 -0.17
    print(f"\n=== paddle vz 排序 (越正越好, 真上挥) ===")
    rows.sort(key=lambda r: -r[1])
    for label, vz, bx, bz, zn, gap in rows[:6]:
        zn_s = f"{zn:.3f}" if zn else "miss"
        print(f"  vz={vz:+.2f}  bz={bz:+.2f}  zn={zn_s}  gap={gap:.1f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
