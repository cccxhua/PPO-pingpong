"""V62c: 调 paddle face 朝向, 让 face_normal 有强 -X 分量, 这样 ball 被面"打回".

V61 baseline: face_normal world = (+0.14, -0.84, +0.52), ball post bx = -0.39.
理想 face_normal: (-0.7~-0.9, 0, +0.3~+0.5) → 强 -X (反弹) + 略 +Z (loft 过网).

7-DOF arm 中改 face 朝向但不改 paddle 位置的关节是 yb_5/yb_6/yb_7 (wrist 三轴).
之外 yb_3 (shoulder_yaw) 也主要影响朝向. 这里探索这 4 个关节的 hit-window 调整,
单关节 + 组合, 看 ball post-vx 改善.
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
FACE_NORMAL_BODY = np.array([0.0, 0.0, -1.0])  # PHASE A: -z_local


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
                q = robot.data.body_quat_w[0, paddle_idx].clone()
                R = quat_to_R(q).cpu().numpy()
                bp = ball.data.root_pos_w[0].cpu().numpy().copy()
                bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
                log.append((t, p, v, R, bp, bv))
        hit = min(log, key=lambda r: np.linalg.norm(r[1] - r[4]))
        bx_min = min(r[5][0] for r in log)
        bz_max = max(r[5][2] for r in log)
        return hit, bx_min, bz_max, log

    (t0, p0, v0_, R0, bp0, bv0), bx0, bz0, log0 = run(V61)
    n0 = R0 @ FACE_NORMAL_BODY
    print(f"\nV61 baseline:")
    print(f"  hit @ t={t0:.3f}, paddle ({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:+.3f}) ball ({bp0[0]:+.3f},{bp0[1]:+.3f},{bp0[2]:+.3f})")
    print(f"  face_normal world = ({n0[0]:+.3f},{n0[1]:+.3f},{n0[2]:+.3f})")
    print(f"  paddle vel = ({v0_[0]:+.2f},{v0_[1]:+.2f},{v0_[2]:+.2f}), ball post: bx_min={bx0:+.2f}, bz_max={bz0:+.2f}")

    # =========================================================
    # 探索 wrist + shoulder_yaw 朝向调整
    # 目标: face_normal X 分量更负
    # =========================================================
    print(f"\n=== 朝向探索 ===")
    print(f"{'variant':<48}  {'face_n (xyz)':<24}  {'paddle pos':<24}  {'gap cm':<6} {'bx_min':>7} {'bz_max':>7}")

    probes = [
        ("V61 baseline", {}),
        # yb_5 (wrist_roll) sweep — 围绕 handle 翻转 face
        ("yb_5 -0.30", {4: -0.30}),
        ("yb_5 -0.50", {4: -0.50}),
        ("yb_5 -0.70", {4: -0.70}),
        ("yb_5 +0.30", {4: +0.30}),
        ("yb_5 +0.50", {4: +0.50}),
        # yb_6 (wrist_pitch)  sweep — 倾 face
        ("yb_6 -0.30", {5: -0.30}),
        ("yb_6 +0.30", {5: +0.30}),
        # yb_3 (shoulder_yaw) sweep — 转肩 → 改 face
        ("yb_3 +0.30", {2: +0.30}),
        ("yb_3 +0.50", {2: +0.50}),
        ("yb_3 -0.30", {2: -0.30}),
        # yb_7 (wrist_yaw) sweep
        ("yb_7 +0.50", {6: +0.50}),
        ("yb_7 -0.50", {6: -0.50}),
        # combo: 先把 face 转向 -X (yb_5/yb_3) 再 yb_4 推前
        ("yb_5 -0.50, yb_3 +0.30, yb_4 -0.10", {4: -0.50, 2: +0.30, 3: -0.10}),
        ("yb_5 -0.70, yb_3 +0.50, yb_4 -0.10", {4: -0.70, 2: +0.50, 3: -0.10}),
        ("yb_5 -0.50, yb_4 -0.20", {4: -0.50, 3: -0.20}),
    ]

    for label, deltas in probes:
        keys = apply_delta(V61, deltas) if deltas else V61
        (t, p, v, R, bp, bv), bx, bz, log = run(keys)
        n = R @ FACE_NORMAL_BODY
        gap = np.linalg.norm(p - bp) * 100
        print(f"{label:<48}  ({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  "
              f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  {gap:>4.1f}  {bx:>+7.2f} {bz:>+7.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
