"""V60 sweep: 在 V58 基础上调 yb_1/yb_2/yb_4 把 paddle 原点平移到目标
(1.321, -0.083, 1.062), 让 face center 接到球, 而不是手柄.

V58 t=0.475 paddle 原点 = (1.274, -0.034, 1.049), ball = (1.240, +0.001, +1.040).
V60 推理: face 在 origin→ball 延长线上 ~12cm 处, 所以 origin 该退 7cm 沿 ball→origin 方向,
落点 (1.321, -0.083, 1.062), Δ = (+47, -49, +13) mm.

关节方向 (世界):
  +X: 减 yb_4 (elbow 更伸) 或 增 yb_1 (shoulder_pitch 抬前)
  -Y: 减 yb_2 (shoulder_roll 更内收)
  +Z: 增 yb_1 (略抬)

V58 t=0.475 关节: [1.560, 0.090, -2.100, 0.630, -0.030, -0.975, 1.000]

逐关节小步长扫描, 报告每组实际 paddle 原点在 t=0.475 的位置.
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


V58 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

TARGET_PADDLE_ORIGIN = np.array([1.321, -0.083, 1.062])
HIT_T = 0.475


def apply(keys, deltas, t_lo=0.29, t_hi=0.56):
    """deltas: dict {joint_idx: delta_rad}, 在 (t_lo, t_hi) 时间窗口的 keyframe 上加."""
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
        paddle_at_hit = None
        ball_at_hit = None
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
                paddle_at_hit = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
                ball_at_hit = ball.data.root_pos_w[0].cpu().numpy().copy()
        return paddle_at_hit, ball_at_hit

    p0, b0 = run(V58)
    delta_target = TARGET_PADDLE_ORIGIN - p0
    print(f"\nV58 baseline @ t={HIT_T}: paddle = ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")
    print(f"  ball = ({b0[0]:+.3f}, {b0[1]:+.3f}, {b0[2]:+.3f})")
    print(f"  target paddle = {TARGET_PADDLE_ORIGIN.tolist()}")
    print(f"  needed Δ = ({delta_target[0]*1000:+.0f}, {delta_target[1]*1000:+.0f}, {delta_target[2]*1000:+.0f}) mm")

    # V58 actual movement direction per joint:
    # yb_1 +0.10: ?  yb_2 -0.10: ?  yb_4 -0.10: ?
    # 先单独探每个关节对 (X, Y, Z) 的影响, 然后线性组合
    print(f"\n{'variant':<40}  {'paddle@hit (xyz)':<24}  {'Δ from V58 (mm)':<24}  {'gap to target'}")

    probes = [
        ("V58 baseline", {}),
        ("yb_1 +0.05", {0: +0.05}),
        ("yb_1 +0.10", {0: +0.10}),
        ("yb_1 +0.20", {0: +0.20}),
        ("yb_2 -0.05", {1: -0.05}),
        ("yb_2 -0.10", {1: -0.10}),
        ("yb_2 -0.15", {1: -0.15}),
        ("yb_3 -0.05", {2: -0.05}),
        ("yb_3 -0.10", {2: -0.10}),
        ("yb_3 +0.05", {2: +0.05}),
        ("yb_3 +0.10", {2: +0.10}),
        ("yb_4 -0.05", {3: -0.05}),
        ("yb_4 -0.10", {3: -0.10}),
        ("yb_4 -0.15", {3: -0.15}),
        ("yb_4 +0.05", {3: +0.05}),
    ]

    results = {}
    for label, deltas in probes:
        keys = apply(V58, deltas) if deltas else V58
        p, b = run(keys)
        if p is None:
            print(f"  {label}: failed (no hit moment captured)")
            continue
        dp = p - p0
        gap = np.linalg.norm(p - TARGET_PADDLE_ORIGIN)
        results[label] = (p, dp, gap)
        print(f"  {label:<40}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
              f"({dp[0]*1000:+5.0f},{dp[1]*1000:+5.0f},{dp[2]*1000:+5.0f})  "
              f"{gap*100:.2f}cm")

    # 用单关节响应做线性组合, 估计达到 target 需要的 multi-joint deltas
    print(f"\n=== 单关节响应矩阵 (Δpaddle per +0.1 rad on joint) ===")
    response = {}  # joint -> Δ paddle per +0.1 rad
    for j in [0, 1, 2, 3]:
        # 找 +0.10 / +0.05 哪个有, 取最大正向差分
        best = None
        for label, deltas in probes:
            if deltas == {j: +0.10} or deltas == {j: -0.10}:
                p, dp, _ = results[label]
                sign = +1 if list(deltas.values())[0] > 0 else -1
                response[j] = sign * dp / 0.1  # per +0.1 rad
                print(f"  yb_{j+1} per +0.1 rad: ({response[j][0]*1000:+.0f}, "
                      f"{response[j][1]*1000:+.0f}, {response[j][2]*1000:+.0f}) mm")
                break

    if len(response) >= 3:
        # 用 yb_1, yb_2, yb_4 (跳过 yb_3 yaw, 它主要改朝向不平移) 求最小二乘
        cols = []
        joints_used = [0, 1, 3]
        for j in joints_used:
            cols.append(response[j])
        A = np.column_stack(cols)  # 3x3
        b_vec = delta_target  # 3
        x, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
        # response[j] = sign * dp / 0.1 = sensitivity in m/rad. So A @ Δq = b 直接给 Δq in rad.
        # 早先版本误把 x 当 0.1-rad-unit 又乘 0.1, 应用值少 10 倍, paddle 几乎没动.
        print(f"\n  线性最小二乘解 (Δq in rad): {x}")
        deltas_solve = {joints_used[i]: float(x[i]) for i in range(3)}
        print(f"  推荐 Δ: yb_1 += {deltas_solve[0]:+.4f}, yb_2 += {deltas_solve[1]:+.4f}, yb_4 += {deltas_solve[3]:+.4f}")

        # 实测验证
        keys = apply(V58, deltas_solve)
        p, b = run(keys)
        gap = np.linalg.norm(p - TARGET_PADDLE_ORIGIN)
        print(f"\n  实测组合: paddle = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        print(f"           ball   = ({b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f})")
        print(f"           gap to target = {gap*100:.2f}cm")
        print(f"           paddle - ball = ({(p[0]-b[0])*1000:+.0f}, {(p[1]-b[1])*1000:+.0f}, {(p[2]-b[2])*1000:+.0f}) mm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
