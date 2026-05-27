"""V81: V73 baseline (mean_vx -2.06), 用户要更大 vx, 高度足够.

V73 = V72 + (yb_6 PIN -0.700 → -0.400)
   keys: 0.000 ready, 0.300 mid, 0.400 wu, 0.475 PIN, 0.550 snap, 0.700 follow, 0.900/1.000 ready

不破 yb_6 limit (V73 wu -1.150, PIN -0.400, spline min -1.256, 余 32 mrad).
不动 yb_6 (避免限位风险).

测试方向:
  A: yb_5 PIN 更深 (-0.900, -1.000) — V72→V73 -0.800 是 v77 单点 max, 没试过更深
  B: yb_5 hit-window mid+wu 整体加深 — V73 仅 PIN 单点
  C: yb_3 less negative (肩外送, paddle linear vx 推力增加)
  D: yb_1 windup 加深 + PIN 抬高 (大 ω)
  E: yb_4 (elbow) — windup 屈, PIN 伸, 给 paddle vx 推力
  F: 组合
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


def edits(keys, mods):
    out = keys
    for t, j, v in mods:
        out = edit_kf(out, t, j, v)
    return out


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

    # ===== A: yb_5 PIN 更深 (V73 -0.800 是 v77 单点 max, 但 v77 是 V71 base 不是 V73 base) =====
    A1 = edits(V73, [(0.475, 4, -0.900)])
    A2 = edits(V73, [(0.475, 4, -1.000)])
    A3 = edits(V73, [(0.475, 4, -1.100)])
    A4 = edits(V73, [(0.475, 4, -0.700)])  # 反向 sanity check

    # ===== B: yb_5 hit-window mid+wu 整体加深 =====
    B1 = edits(V73, [(0.300, 4, -0.465), (0.400, 4, -0.465)])
    B2 = edits(V73, [(0.300, 4, -0.515), (0.400, 4, -0.515)])
    B3 = edits(V73, [(0.300, 4, -0.465), (0.400, 4, -0.465), (0.475, 4, -0.900)])
    B4 = edits(V73, [(0.300, 4, -0.515), (0.400, 4, -0.515), (0.475, 4, -1.000)])

    # ===== C: yb_3 less negative (肩外送) =====
    C1 = edits(V73, [(0.475, 2, -1.850)])
    C2 = edits(V73, [(0.475, 2, -1.700)])
    C3 = edits(V73, [(0.300, 2, -1.804), (0.400, 2, -1.879), (0.475, 2, -1.879)])  # hw 整体 +0.10
    C4 = edits(V73, [(0.300, 2, -1.704), (0.400, 2, -1.779), (0.475, 2, -1.779)])  # hw 整体 +0.20

    # ===== D: yb_1 (shoulder pitch) windup 深 + PIN 高 =====
    D1 = edits(V73, [(0.400, 0, +1.087), (0.475, 0, +1.387)])  # wu -0.05 PIN +0.05
    D2 = edits(V73, [(0.400, 0, +1.037), (0.475, 0, +1.437)])  # wu -0.10 PIN +0.10
    D3 = edits(V73, [(0.400, 0, +0.987), (0.475, 0, +1.487)])  # wu -0.15 PIN +0.15

    # ===== E: yb_4 (elbow) windup 屈深 + PIN 伸 =====
    # V73 yb_4: 0.300=0.877, 0.400 wu=0.507, 0.475 PIN=0.457, 0.550=0.407
    # 加深 windup (更屈, 上摆) → PIN 伸更猛 → paddle 推力增加
    E1 = edits(V73, [(0.400, 3, +0.607), (0.475, 3, +0.357)])  # wu +0.10 屈, PIN -0.10 伸
    E2 = edits(V73, [(0.400, 3, +0.707), (0.475, 3, +0.307)])  # 更深
    E3 = edits(V73, [(0.400, 3, +0.607)])  # 仅 wu 加深

    # ===== F: 大火力组合 =====
    F1 = edits(V73, [(0.475, 4, -0.900), (0.475, 2, -1.850)])  # yb_5 深 + yb_3 外送
    F2 = edits(V73, [(0.475, 4, -0.900), (0.400, 0, +1.087), (0.475, 0, +1.387)])  # yb_5 深 + yb_1 加大
    F3 = edits(V73, [(0.475, 4, -0.900), (0.400, 0, +1.037), (0.475, 0, +1.437)])  # yb_5 深 + yb_1 大幅
    F4 = edits(V73, [(0.300, 4, -0.465), (0.400, 4, -0.465), (0.475, 4, -0.900),
                     (0.475, 2, -1.850)])  # yb_5 hw 全深 + yb_3
    F5 = edits(V73, [(0.475, 4, -0.900), (0.475, 2, -1.850),
                     (0.400, 0, +1.087), (0.475, 0, +1.387)])  # 三方齐发

    probes = [
        ("V73 baseline", V73),
        ("A1: yb_5 PIN -0.900", A1),
        ("A2: yb_5 PIN -1.000", A2),
        ("A3: yb_5 PIN -1.100", A3),
        ("A4: yb_5 PIN -0.700 (sanity)", A4),
        ("B1: yb_5 mid/wu -0.465", B1),
        ("B2: yb_5 mid/wu -0.515", B2),
        ("B3: B1 + yb_5 PIN -0.900", B3),
        ("B4: B2 + yb_5 PIN -1.000", B4),
        ("C1: yb_3 PIN -1.850", C1),
        ("C2: yb_3 PIN -1.700", C2),
        ("C3: yb_3 hw +0.10", C3),
        ("C4: yb_3 hw +0.20", C4),
        ("D1: yb_1 wu-0.05 PIN+0.05", D1),
        ("D2: yb_1 wu-0.10 PIN+0.10", D2),
        ("D3: yb_1 wu-0.15 PIN+0.15", D3),
        ("E1: yb_4 wu+0.10 PIN-0.10", E1),
        ("E2: yb_4 wu+0.20 PIN-0.15", E2),
        ("E3: yb_4 wu+0.10 only", E3),
        ("F1: yb_5-0.9 + yb_3-1.85", F1),
        ("F2: yb_5-0.9 + yb_1 mid", F2),
        ("F3: yb_5-0.9 + yb_1 大", F3),
        ("F4: yb_5 hw-0.9 + yb_3", F4),
        ("F5: yb_5+yb_3+yb_1 三方齐发", F5),
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

    print("\n=== Sorted by mean_vx (核心: 越负越好) ===")
    rows.sort(key=lambda r: r[4])
    for label, clears, mean_zn, min_zn, mean_vx, _ in rows:
        print(f"  vx={mean_vx:+.2f}  CLEARS {clears}/{len(phases)}  "
              f"zn[mean/min]={mean_zn:+.2f}/{min_zn:+.2f}  {label}")

    print("\n=== Sorted by CLEARS, then mean_vx ===")
    rows.sort(key=lambda r: (-r[1], r[4]))
    for label, clears, mean_zn, min_zn, mean_vx, _ in rows[:8]:
        print(f"  TOP: CLEARS {clears}/{len(phases)}  vx={mean_vx:+.2f}  "
              f"zn[mean/min]={mean_zn:+.2f}/{min_zn:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
