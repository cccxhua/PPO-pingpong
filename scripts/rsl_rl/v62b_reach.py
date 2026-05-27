"""V62b: 用户猜测正确 — paddle X 没真正够到球, 只擦肩.
V61 closest moment: paddle X=1.329, ball X=1.311, paddle 才领先 ball 18mm,
而 paddle 此时正在以 -1.27 m/s 退出 → ball 仅得 -0.39 X 速度.

策略: 让 paddle 在 hit window 整体推到更前 (+X), 这样 ball 真正撞上 paddle 面.
- 减小 yb_4 (elbow 更伸): 主要 +X 影响
- 增大 yb_1 (shoulder 前送): 也有 +X
- 同时为了保持 V61 PIN 位置准确, 只改变 mid/windup/PIN/snap 几个 hit-window keyframe

把 paddle 推前 ~5-10cm, 使 ball X=1.311 时 paddle X 在 1.36-1.41, 然后 ball 撞过来.
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


V61 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.227, -0.115, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.857, -0.115, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.807, -0.145, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.757, -0.115, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

HIT_T = 0.475


def apply_delta(keys, deltas, t_lo=0.29, t_hi=0.56):
    """对 hit-window 的 keyframe 加 delta."""
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
        for step in range(int(0.65 / sim_dt)):
            t = step * sim_dt
            target = spline(min(t, 1.0))
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            if 0.40 < t < 0.60:
                p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
                v = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
                bp = ball.data.root_pos_w[0].cpu().numpy().copy()
                bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
                log.append((t, p, v, bp, bv))
        # closest paddle-ball moment
        hit = min(log, key=lambda r: np.linalg.norm(r[1] - r[3]))
        bx_min = min(r[4][0] for r in log)
        bz_max = max(r[4][2] for r in log)
        return hit, bx_min, bz_max, log

    # baseline
    (t0, p0, v0_, bp0, bv0), bx0, bz0, log0 = run(V61)
    print(f"\nV61 baseline:")
    print(f"  hit @ t={t0:.3f}, paddle ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f}) ball ({bp0[0]:+.3f},{bp0[1]:+.3f},{bp0[2]:+.3f})")
    print(f"  paddle-ball Δ = ({(p0[0]-bp0[0])*1000:+.0f},{(p0[1]-bp0[1])*1000:+.0f},{(p0[2]-bp0[2])*1000:+.0f}) mm")
    print(f"  paddle vel = ({v0_[0]:+.2f},{v0_[1]:+.2f},{v0_[2]:+.2f}), ball post: bx_min={bx0:+.2f}, bz_max={bz0:+.2f}")

    # =========================================================
    # 探索: 让 paddle X 提前到位 (整个 hit window 推前)
    #   - yb_4 -delta: elbow 更伸 (主要 +X)
    #   - yb_1 +delta: shoulder 前送 (+X +Z)
    # =========================================================
    print(f"\n=== 探索 paddle X reach: 减 yb_4 / 加 yb_1 ===")
    print(f"{'variant':<40}  {'paddle X@hit':>10} {'paddle-ball Δx':>14} {'paddle vx':>10} {'ball bx_min':>11} {'ball bz_max':>11}")

    probes = [
        ("V61 baseline", {}),
        # 单调 yb_4 减 (elbow 更伸 → +X)
        ("yb_4 -0.05", {3: -0.05}),
        ("yb_4 -0.10", {3: -0.10}),
        ("yb_4 -0.15", {3: -0.15}),
        ("yb_4 -0.20", {3: -0.20}),
        # 单调 yb_1 加 (shoulder 前送 → +X +Z)
        ("yb_1 +0.05", {0: +0.05}),
        ("yb_1 +0.10", {0: +0.10}),
        ("yb_1 +0.15", {0: +0.15}),
        # 组合: 推前 + 略压 (减 yb_1 反推 Z)
        ("yb_4 -0.10, yb_1 +0.10", {3: -0.10, 0: +0.10}),
        ("yb_4 -0.15, yb_1 +0.10", {3: -0.15, 0: +0.10}),
        ("yb_4 -0.20, yb_1 +0.10", {3: -0.20, 0: +0.10}),
        # 更激进
        ("yb_4 -0.20, yb_1 +0.15", {3: -0.20, 0: +0.15}),
        # 推前 + 加大 wrist snap (yb_6 加大幅度)
        ("yb_4 -0.10 + yb_6 deeper windup -1.20→-1.40", {3: -0.10, 5: 0.00}),
    ]

    best_label = None
    best_bx = 0.0
    for label, deltas in probes:
        keys = apply_delta(V61, deltas) if deltas else V61
        (t, p, v, bp, bv), bx, bz, log = run(keys)
        dx = (p[0] - bp[0]) * 1000
        print(f"{label:<40}  {p[0]:>+10.3f} {dx:>+12.0f}mm   {v[0]:>+10.2f} {bx:>+11.2f} {bz:>+11.2f}")
        if bx < best_bx:
            best_bx = bx
            best_label = label

    print(f"\n  最大反向速度: {best_label}, ball bx = {best_bx:+.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
