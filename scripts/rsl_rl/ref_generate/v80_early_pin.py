"""V80: Early PIN 方案 — 在 V73 基础上加 t=0.450 中间帧让 face 提前到位.

V79 诊断: V72/V73 都只有 phase=0.475 单一相位过网, 偏离 25ms 就 face 几何错.
V80 思路: t=0.450 加帧 (face 关节已到 PIN 值), 让 face 几何从 0.450 到 0.475 都正确.

设计候选:
  U1: t=0.450 加帧 = V73 PIN 值 (所有 7 关节 hold) — 最激进
  U2: t=0.450 加帧, 仅 face (yb_5 -0.800, yb_6 -0.400), 其他关节 spline 自然插值
  U3: t=0.460 加帧 (温和, 10ms 提前)
  U4: t=0.500 加帧 = PIN 值 (PIN 后保持, 应付 late phase)
  U5: U1 + U4 双向延伸 (0.450, 0.475, 0.500 三点同 PIN)
  U6: 整体 PIN 提前到 t=0.450 (compensate PD lag)
  U7: U2 加 yb_1 早抬 (让肩 ω 也提前峰值)

风险: spline 两个相邻相同值会在中间 overshoot. 需要 limit check 验证.
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


# V73 = 当前 npz
V73 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.137, +0.103, -1.979, +0.507, -0.315, -1.150,  +1.000]),
    (0.475, [+1.337, +0.103, -1.979, +0.457, -0.800, -0.400,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

PIN_V73 = [+1.337, +0.103, -1.979, +0.457, -0.800, -0.400,  +1.000]

BALL_ARRIVE_TIME_EST = 0.5205
DURATION = 1.0

NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.79
G = 9.81
FACE_BODY = np.array([0.0, 0.0, -1.0])

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def edit_kf(keys, t, j, new_val):
    out = []
    for kt, vals in keys:
        v = list(vals)
        if abs(kt - t) < 1e-6:
            v[j] = new_val
        out.append((kt, v))
    return out


def add_kf(keys, t, vals):
    out = list(keys) + [(t, list(vals))]
    out.sort(key=lambda kv: kv[0])
    return out


def edits(keys, mods):
    out = keys
    for t, j, v in mods:
        out = edit_kf(out, t, j, v)
    return out


def shift_pin(keys, t_old, t_new):
    """Move keyframe from t_old to t_new (preserve values)."""
    return [(t_new if abs(kt - t_old) < 1e-6 else kt, v) for kt, v in keys]


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def analyze_trajectory(x0, z0, vx, vz):
    if vx >= 0:
        return None, None, False, False
    t_net = x0 / (-vx)
    z_at_net = z0 + vz * t_net - 0.5 * G * t_net * t_net
    a = 0.5 * G; b = -vz; c = TABLE_Z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        return z_at_net, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2 * a)
    x_bounce = x0 + vx * t_bounce
    clears_net = z_at_net > NET_Z and t_net < t_bounce
    valid = x_bounce < NET_X
    return z_at_net, x_bounce, clears_net, valid


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    bad = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo or y[:, i].max() > hi:
            bad.append(f"yb_{i+1}=[{y[:,i].min():+.3f},{y[:,i].max():+.3f}]")
    return bad


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

    def run(keys, hit_phase):
        initial_phase = (hit_phase - BALL_ARRIVE_TIME_EST) % 1.0
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")
        full = robot.data.default_joint_pos[0:1].clone()
        q0 = spline(initial_phase)
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
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy().copy()
            q = robot.data.body_quat_w[0, paddle_idx].cpu().numpy().copy()
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, q, pv, bp, bv))

        hw = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.70]
        if not hw:
            return None
        hi, hr = min(hw, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][4]))
        gap = float(np.linalg.norm(hr[1] - hr[4]))
        post_idx = min(hi + 2, len(log) - 1)
        bp_post = log[post_idx][4]
        bv_post = log[post_idx][5]
        face_n = quat_to_R(hr[2]) @ FACE_BODY
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return dict(gap=gap, face_n=face_n, ball_vel_post=bv_post,
                    z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid)

    # ===== U 系列: early PIN 方案 =====

    # U1: t=0.450 加 PIN 值 (所有 7 关节 = V73 PIN)
    U1 = add_kf(V73, 0.450, PIN_V73)

    # U2: t=0.450 加帧, 仅 face (yb_5/yb_6 到 PIN), 其他关节用 0.400→0.475 中点
    # 0.400 yb_1=1.137, yb_4=0.507; 0.475 yb_1=1.337, yb_4=0.457
    # 中点: yb_1=1.237, yb_4=0.482
    U2_vals = [+1.237, +0.103, -1.979, +0.482, -0.800, -0.400, +1.000]
    U2 = add_kf(V73, 0.450, U2_vals)

    # U3: t=0.460 加帧 = PIN, 较温和 (15ms 提前 vs 25ms)
    U3 = add_kf(V73, 0.460, PIN_V73)

    # U4: t=0.500 加 PIN 值 (PIN 后保持, 应付 late phase)
    U4 = add_kf(V73, 0.500, PIN_V73)

    # U5: U1 + U4 (0.450, 0.475, 0.500 三点同 PIN)
    U5 = add_kf(add_kf(V73, 0.450, PIN_V73), 0.500, PIN_V73)

    # U6: 整体 PIN 时刻提前到 t=0.450 (移动 0.475 帧)
    U6 = shift_pin(V73, 0.475, 0.450)
    # 注意: 这破坏 hit_phase 对齐, 因为 env 里 hit_phase=0.475 = 球到达时刻
    # 这相当于让 motion peak 早于球到达, paddle 已减速时碰球. 大概率变差, 但测一下确认.

    # U7: U2 + yb_1 早抬 (yb_1 PIN 值在 0.450 到位, 让肩 ω 提前)
    U7_vals = [+1.337, +0.103, -1.979, +0.482, -0.800, -0.400, +1.000]
    U7 = add_kf(V73, 0.450, U7_vals)

    # U8: U2 仅 face, 但 yb_5 用更深的值让 PIN→snap 之后 face 还稳定
    # 即: face 在 0.450 到位 (-0.800), 然后 PIN @ 0.475 仍 -0.800, 之后 0.550 snap 到 -0.165 (V73)
    # 不变 — 这就是 U2

    # U9: t=0.450 + t=0.475 PIN (face hold) + t=0.475 之后允许 yb_5 提早收回 (snap 提前)
    U9_face = [+1.237, +0.103, -1.979, +0.482, -0.800, -0.400, +1.000]
    U9 = edits(add_kf(V73, 0.450, U9_face),
               [(0.550, 4, +0.000)])  # snap yb_5 -0.165 → 0 (early recover)

    # U10: 同 U2 但 yb_6 PIN 值用 -0.500 (face X 弱一点, 容许更宽 PIN window)
    U10_vals = [+1.237, +0.103, -1.979, +0.482, -0.800, -0.500, +1.000]
    U10 = edits(add_kf(V73, 0.450, U10_vals), [(0.475, 5, -0.500)])

    probes = [
        ("V73 baseline", V73),
        ("U1: t=0.450 加全 PIN", U1),
        ("U2: t=0.450 face only (yb_5/yb_6 PIN)", U2),
        ("U3: t=0.460 加全 PIN (温和)", U3),
        ("U4: t=0.500 加 PIN (late hold)", U4),
        ("U5: U1 + U4 双向 hold", U5),
        ("U6: PIN 整体移到 t=0.450", U6),
        ("U7: U2 + yb_1 早抬", U7),
        ("U9: U2 + yb_5 snap 早回", U9),
        ("U10: U2 但 yb_6 PIN -0.500", U10),
    ]

    phases = [0.425, 0.450, 0.475, 0.500, 0.525]

    print(f"\n{'variant':<48} {'CLR/N':<6} ", end="")
    for p in phases:
        print(f"{f'p={p:.3f}':<8} ", end="")
    print(f"{'mean_zn':>8} {'min_zn':>8} {'mean_vx':>9}")
    print("-" * 175)

    rows = []
    for label, keys in probes:
        bad = check_limits(keys)
        if bad:
            print(f"{label:<48} LIMITS: {bad}")
            continue
        results = []
        for hp in phases:
            try:
                r = run(keys, hp)
                results.append(r)
            except Exception:
                results.append(None)
        clears = sum(1 for r in results if r is not None and r["clears"] and r["valid"])
        zn_list = [r["z_at_net"] for r in results if r is not None and r["z_at_net"] is not None]
        vx_list = [r["ball_vel_post"][0] for r in results if r is not None]
        mean_zn = np.mean(zn_list) if zn_list else float("nan")
        min_zn = min(zn_list) if zn_list else float("nan")
        mean_vx = np.mean(vx_list) if vx_list else float("nan")

        print(f"{label:<48} {clears}/{len(phases):<4} ", end="")
        for r in results:
            if r is None:
                print(f"{'ERR':<8} ", end="")
            else:
                zn = r["z_at_net"]
                marker = "*" if r["clears"] and r["valid"] else " "
                print(f"{marker}{zn:>+5.2f}  ", end="")
        print(f"{mean_zn:>+8.2f} {min_zn:>+8.2f} {mean_vx:>+9.2f}")
        rows.append((label, clears, mean_zn, min_zn, mean_vx, results))

    print("\n=== Sorted by clearance count, then min_zn ===")
    rows.sort(key=lambda r: (-r[1], -r[3]))
    for label, clears, mean_zn, min_zn, mean_vx, _ in rows:
        print(f"  CLEARS {clears}/{len(phases)}  mean_zn={mean_zn:+.2f}  min_zn={min_zn:+.2f}  "
              f"mean_vx={mean_vx:+.2f}  {label}")

    print("\n=== 候选排序 (优先过网数, 次 vx) ===")
    rows.sort(key=lambda r: (-r[1], r[4]))
    for label, clears, mean_zn, min_zn, mean_vx, _ in rows[:5]:
        print(f"  TOP: CLEARS {clears}, vx={mean_vx:+.2f}  zn[mean/min]={mean_zn:+.2f}/{min_zn:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
