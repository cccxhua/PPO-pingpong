"""V62d: 精细搜索 yb_5 wrist_roll + yb_4 elbow extension 组合.

V62c 找到方向: yb_5 -0.30~-0.50 把 face_normal 转向更 -X, ball 反向速度 -1.6 ~ -2.1 m/s.
yb_5 -0.50 + yb_4 -0.20 = -2.12 m/s (BEST so far).

精细 grid: yb_5 ∈ {-0.30, -0.40, -0.50, -0.60} × yb_4 ∈ {0, -0.10, -0.20, -0.30}
+ 加上 yb_1 微调 (Z 方向控制).

目标: 最大反向 bx, 同时 bz_max ∈ [+1.5, +3.0] (够过网但不过分高), gap < 5cm.
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
                bp = ball.data.root_pos_w[0].cpu().numpy().copy()
                bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
                log.append((t, p, bp, bv))
        # closest moment
        hit = min(log, key=lambda r: np.linalg.norm(r[1] - r[2]))
        bx_min = min(r[3][0] for r in log)
        bz_max = max(r[3][2] for r in log)
        return hit, bx_min, bz_max, log

    print(f"\n{'yb_5':>5} {'yb_4':>5} {'yb_1':>5}    "
          f"{'paddle pos':<24}  {'ball pos':<24}  {'gap cm':<6} {'bx_min':>7} {'bz_max':>7}")

    rows = []
    yb5_grid = [-0.20, -0.30, -0.40, -0.50, -0.60]
    yb4_grid = [0.0, -0.10, -0.20, -0.30]
    yb1_grid = [0.0, +0.10]
    for yb5 in yb5_grid:
        for yb4 in yb4_grid:
            for yb1 in yb1_grid:
                deltas = {}
                if yb5 != 0: deltas[4] = yb5
                if yb4 != 0: deltas[3] = yb4
                if yb1 != 0: deltas[0] = yb1
                keys = apply_delta(V61, deltas) if deltas else V61
                (t, p, bp, bv), bx, bz, log = run(keys)
                gap = np.linalg.norm(p - bp) * 100
                rows.append((yb5, yb4, yb1, p, bp, gap, bx, bz))
                print(f"{yb5:>+.2f} {yb4:>+.2f} {yb1:>+.2f}    "
                      f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  ({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f})  "
                      f"{gap:>4.1f}  {bx:>+7.2f} {bz:>+7.2f}")

    print(f"\n=== Top 5 反向速度 (bx_min 最负, gap < 8cm) ===")
    valid = [r for r in rows if r[5] < 8.0]
    valid.sort(key=lambda r: r[6])
    for r in valid[:5]:
        yb5, yb4, yb1, p, bp, gap, bx, bz = r
        print(f"  yb_5{yb5:+.2f} yb_4{yb4:+.2f} yb_1{yb1:+.2f}: "
              f"bx={bx:+.2f}, bz={bz:+.2f}, gap={gap:.1f}cm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
