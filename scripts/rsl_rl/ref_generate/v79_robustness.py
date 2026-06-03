"""V79: V72 单次测好 (z@net=+1.40), 但用户反馈 "有可以过网的" — 实际过网率仍偏低.
原因: env hit_phase_noise=0.05 + ball relaunch 时序变动 → 真实命中相位不固定.

策略: 不追 vx, 而是让 V72 在 phase ±0.05 抖动下更鲁棒.
对每个候选, 扫 hit_phase ∈ [0.425, 0.475, 0.525] (env noise 边界 + 中心), 测:
  - clearance_rate: CLEARS+VALID 的相位数 / 总相位数
  - mean_z@net: 平均网余 (越大越鲁棒)
  - min_z@net: 最差相位的网余 (worst-case)
  - mean_vx: 平均 -X 速度

候选思路:
  1. 加大整体能量 (yb_1 windup 加深, yb_6 windup 加深) - 提高 paddle peak |v|
  2. 加宽 peak 窗口 (添加 t=0.45 / t=0.50 中间帧, 让 PIN 持续更长)
  3. yb_5 hit-window 整体加深 (V72 仅 PIN 改, 其他时刻 yb_5 仍 -0.315)
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


# V72 = 当前 npz
V72 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.137, +0.103, -1.979, +0.507, -0.315, -1.150,  +1.000]),
    (0.475, [+1.337, +0.103, -1.979, +0.457, -0.800, -0.700,  +1.000]),
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


def add_kf(keys, t, vals):
    """Insert a new keyframe."""
    out = list(keys) + [(t, list(vals))]
    out.sort(key=lambda kv: kv[0])
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
        """Run with custom hit_phase to simulate phase noise."""
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

    # ===== A 类: 加大 paddle peak |v| (windup 深, ω 大, vx 升 + 能量足) =====
    P1 = edits(V72, [(0.400, 0, +1.087), (0.400, 5, -1.200)])  # yb_1 wu -0.05 + yb_6 wu -0.05
    P2 = edits(V72, [(0.400, 0, +1.037), (0.400, 5, -1.250)])  # 双关节再深
    P3 = edits(V72, [(0.400, 5, -1.250)])  # 仅 yb_6 wu 加深 (留 38 mrad buffer)
    P3b = edits(V72, [(0.400, 5, -1.270)])  # 18 mrad buffer (尽量极限)

    # ===== B 类: 改善对齐 — yb_3 less negative (肩外送, paddle linear vx 增加) =====
    Q1 = edits(V72, [(0.475, 2, -1.850)])  # yb_3 PIN less neg
    Q2 = edits(V72, [(0.475, 2, -1.700)])
    Q3 = edits(V72, [(0.300, 2, -1.804), (0.400, 2, -1.879), (0.475, 2, -1.879)])  # hw 整体 +0.10

    # ===== C 类: 加宽 peak 窗口 (中间帧让 paddle 高速持续更久) =====
    P4 = add_kf(V72, 0.450, [+1.250, +0.103, -1.979, +0.470, -0.700, -0.850,  +1.000])
    P5 = add_kf(V72, 0.500, [+1.380, +0.103, -1.979, +0.440, -0.700, -0.600,  +1.000])
    P6 = add_kf(add_kf(V72, 0.450, [+1.250, +0.103, -1.979, +0.470, -0.700, -0.850, +1.000]),
                0.500, [+1.380, +0.103, -1.979, +0.440, -0.700, -0.600, +1.000])

    # ===== D 类: yb_5 hit-window 整体加深 (V72 仅 PIN 单点) =====
    P7 = edits(V72, [(0.300, 4, -0.465), (0.400, 4, -0.465)])
    P8 = edits(V72, [(0.300, 4, -0.465), (0.400, 4, -0.465), (0.550, 4, -0.300)])

    # ===== E 类: yb_5 -0.800 + yb_6 PIN 抬 (face X 增 + 已优化 wrist_roll) =====
    # v76/v77 没在 yb_5=-0.800 基础上试 yb_6 PIN raise
    R1 = edits(V72, [(0.475, 5, -0.500)])  # yb_6 PIN -0.700 → -0.500 (face_x 增)
    R2 = edits(V72, [(0.475, 5, -0.400)])
    R3 = edits(V72, [(0.475, 5, -0.500), (0.400, 5, -1.050)])  # PIN 抬时 wu 同步放松
    R4 = edits(V72, [(0.475, 5, -0.400), (0.400, 5, -1.000)])

    # ===== F 类: 综合大火力 — A + E (windup 深 + face X 增) =====
    S1 = edits(V72, [(0.400, 0, +1.087), (0.400, 5, -1.200), (0.475, 5, -0.500)])
    S2 = edits(V72, [(0.400, 0, +1.087), (0.400, 5, -1.250), (0.475, 5, -0.500)])
    S3 = edits(V72, [(0.400, 0, +1.037), (0.400, 5, -1.250), (0.475, 5, -0.500), (0.475, 2, -1.850)])

    # ===== G 类: 综合 + yb_5 hw =====
    T1 = edits(V72, [(0.400, 0, +1.087), (0.400, 5, -1.200),
                     (0.300, 4, -0.465), (0.400, 4, -0.465)])
    T2 = edits(V72, [(0.400, 0, +1.087), (0.400, 5, -1.200), (0.475, 5, -0.500),
                     (0.300, 4, -0.465), (0.400, 4, -0.465)])

    probes = [
        ("V72 baseline", V72),
        ("P1: yb_1+yb_6 wu 中等加深", P1),
        ("P2: yb_1+yb_6 wu 大幅加深", P2),
        ("P3: yb_6 wu -1.250", P3),
        ("P3b: yb_6 wu -1.270 (limit-18)", P3b),
        ("Q1: yb_3 PIN -1.850", Q1),
        ("Q2: yb_3 PIN -1.700", Q2),
        ("Q3: yb_3 hw +0.10", Q3),
        ("P4: + t=0.450 中间帧", P4),
        ("P5: + t=0.500 中间帧", P5),
        ("P6: + t=0.450/0.500 双帧", P6),
        ("P7: yb_5 hw mid/wu 深", P7),
        ("P8: yb_5 hw mid/wu/snap 深", P8),
        ("R1: yb_6 PIN -0.500", R1),
        ("R2: yb_6 PIN -0.400", R2),
        ("R3: yb_6 wu-1.05 + PIN-0.5", R3),
        ("R4: yb_6 wu-1.00 + PIN-0.4", R4),
        ("S1: yb_1/yb_6 wu + yb_6 PIN-0.5", S1),
        ("S2: yb_1+yb_6 wu 深 + PIN-0.5", S2),
        ("S3: 大火力 + yb_3", S3),
        ("T1: A + yb_5 hw", T1),
        ("T2: A + yb_6 PIN-0.5 + yb_5 hw", T2),
    ]

    # 扫 hit_phase ∈ {0.425, 0.450, 0.475, 0.500, 0.525} (env noise [0.425, 0.525])
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
                if r is None:
                    results.append(None)
                else:
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

    print("\n=== 提示 ===")
    print("  CLEARS 数量越多越鲁棒. min_zn 最差相位仍能过网最重要 (>0.91 才算过网).")
    print("  P 方案中 mean_vx 最大 = paddle 峰值能量最高 (但需配合 face 几何).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
