"""V61: 把 V60 paddle 在 t=0.475 平移 (ΔY=+0.06, ΔZ=-0.10) — 朝右(+Y, 用户站机器人背后视角)
+ 朝下(-Z), 同时 face_normal 朝向不变. 手肘 Z 自然跟着降.

7-DOF 减去 (3 位置变化 + 3 face_normal 不变) = 6 约束, 还剩 1 个 redundant DOF.
直接最小二乘加正则求解, sim 实测验证.

state vector (7 维): [paddle_x, paddle_y, paddle_z, face_normal_x, face_normal_y, face_normal_z, elbow_z]
target Δ:           [0,        +0.06,    -0.10,    0,             0,             0,             auto (信息项)]

face_normal = R[:, 2] (PHASE A 验证 yb_7 wrist_yaw 绕 face_normal).
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


# V60 baseline (current keyframes in create_forehand.py)
V60 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.361, -0.002, -2.025, +1.160, +0.000, -1.000,  +1.000]),
    (0.400, [+1.421, -0.097, -2.100, +0.790, +0.000, -1.200,  +1.000]),
    (0.475, [+1.521, -0.097, -2.100, +0.740, -0.030, -0.975,  +1.000]),
    (0.550, [+1.671, -0.097, -2.100, +0.690, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

HIT_T = 0.475
ELBOW_LINK = "link_yb_4"
PADDLE_LINK = "Link_yb_paddle"


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)


def apply_delta(keys, deltas, t_lo=0.29, t_hi=0.56):
    """deltas: dict {joint_idx: delta_rad} on keyframes within (t_lo, t_hi)."""
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

    paddle_idx = robot.find_bodies(PADDLE_LINK)[0][0]
    elbow_idx = robot.find_bodies(ELBOW_LINK)[0][0]
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
        snapshot = None
        for step in range(int(0.55 / sim_dt)):
            t = step * sim_dt
            target = spline(min(t, 1.0))
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            if abs(t - HIT_T) < sim_dt * 0.6:
                paddle_p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
                paddle_q = robot.data.body_quat_w[0, paddle_idx].clone()
                R = quat_to_R(paddle_q).cpu().numpy()
                elbow_p = robot.data.body_pos_w[0, elbow_idx].cpu().numpy().copy()
                # state: [paddle_xyz, face_normal_xyz, elbow_z]
                # face_normal = R[:, 2] (PHASE A 验证 yb_7 绕 face_normal)
                snapshot = np.concatenate([
                    paddle_p,                # 3 paddle position
                    R[:, 2],                 # 3 face normal direction in world
                    [elbow_p[2]],            # 1 elbow z (info only)
                ])  # total 7
        return snapshot, paddle_p, elbow_p, R

    # =========================================================
    # 1) baseline V60
    # =========================================================
    s0, p0, e0, R0 = run(V60)
    print(f"\nV60 baseline @ t={HIT_T}:")
    print(f"  paddle origin world = ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")
    print(f"  elbow  origin world = ({e0[0]:+.3f}, {e0[1]:+.3f}, {e0[2]:+.3f})")
    print(f"  paddle R col2 (face_normal) world = ({R0[0,2]:+.3f}, {R0[1,2]:+.3f}, {R0[2,2]:+.3f})")

    # =========================================================
    # 2) probe each joint with +0.10 rad on hit-window keyframes
    # =========================================================
    print(f"\n=== Single-joint probe (+0.10 rad on hit-window keys) ===")
    h = 0.10
    J = np.zeros((7, 7))  # rows = state dims (paddle xyz, normal xyz, elbow_z); cols = joints
    for j in range(7):
        keys = apply_delta(V60, {j: h})
        s, p, e, R = run(keys)
        ds = (s - s0) / h
        J[:, j] = ds
        print(f"  yb_{j+1}: Δpaddle=({ds[0]*100:+.1f},{ds[1]*100:+.1f},{ds[2]*100:+.1f})cm/rad, "
              f"Δface_n=({ds[3]:+.2f},{ds[4]:+.2f},{ds[5]:+.2f}), Δelbow_z={ds[6]*1000:+.0f}mm")

    # =========================================================
    # 3) solve: paddle 朝右(+Y) +0.06, 朝下 -0.10, face_normal 不变, elbow 自由
    # =========================================================
    target = np.zeros(7)
    target[0] = 0.0     # paddle X 不变
    target[1] = +0.04   # paddle Y +4cm (用户视角"右" = +Y, 因为机器人面 -X)
    target[2] = -0.05   # paddle Z -5cm (向下)
    # face_normal 锁住 (target[3:6] = 0)
    # elbow_z 不约束 (target[6] = 0, weight = 0)
    W = np.diag([
        50.0, 50.0, 50.0,         # paddle xyz (硬约束)
        20.0, 20.0, 20.0,         # face_normal (锁朝向)
        0.0,                      # elbow_z 自由
    ])
    A = W @ J
    b = W @ target
    reg = 0.05  # 较强正则避免极大 joint 移动 (linear approx 在大步长失效)
    AtA = A.T @ A + reg * np.eye(7)
    Atb = A.T @ b
    dq = np.linalg.solve(AtA, Atb)
    print(f"\n  推荐 Δq (rad):")
    for j in range(7):
        print(f"    yb_{j+1}: {dq[j]:+.4f}")

    # =========================================================
    # 4) verify
    # =========================================================
    keys = apply_delta(V60, {j: float(dq[j]) for j in range(7)})
    s, p, e, R = run(keys)
    print(f"\n=== Verification ===")
    target_pos = p0 + np.array([0.0, +0.04, -0.05])
    print(f"  paddle: V60 ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f}) -> V61 ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    print(f"          target paddle = ({target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f})")
    print(f"          err = ({(p[0]-target_pos[0])*1000:+.0f}, {(p[1]-target_pos[1])*1000:+.0f}, {(p[2]-target_pos[2])*1000:+.0f}) mm")
    print(f"  elbow:  V60 z={e0[2]:+.3f}  ->  V61 z={e[2]:+.3f}   Δz = {(e[2]-e0[2])*1000:+.0f}mm")
    err_face = np.linalg.norm(R[:, 2] - R0[:, 2])
    print(f"  face_normal axis err: {err_face:.3f}  (0 = no rotation)")

    # absolute keyframes for hit window
    print(f"\n=== V61 keyframes (hit window only, t in (0.29, 0.56)) ===")
    print(f"  V61 = V60 + Δ on yb_1..yb_7")
    for t, vals in V60:
        if 0.29 < t < 0.56:
            new = [vals[k] + float(dq[k]) for k in range(7)]
            print(f"    ({t:.3f}, [" + ", ".join(f"{v:+.3f}" for v in new) + "]),")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
