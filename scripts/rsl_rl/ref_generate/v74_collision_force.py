"""V74: 测出 ball-paddle 碰撞瞬间精确状态, 然后测试各 force-up 变体.

用户当前 env_cfg 设置:
  hit_phase = 0.475
  ball_arrive_time_est = 0.5205
  → initial_phase = (0.475 - 0.5205) mod 1 = 0.9545
  → 在 sim_t 时刻, motion phase = (0.9545 + sim_t) mod 1
  → motion PIN (phase 0.475) 出现在 sim_t = 0.5205

probe 应用同样的 phase offset, 让 spline(phase) 而非 spline(sim_t),
从而模拟真实环境的 motion 走时. 这样 probe 测到的 hit_t 是真实 sim 时刻,
paddle 在那一刻的姿态就是 ball 真正接触 paddle 的姿态.

力度策略:
  A. 把 PIN 时间提前到对齐 hit_t (peak ω 落在球到达瞬间)
  B. 加大 windup→PIN 的 Δ (yb_6/yb_1), 直接拉高 peak |v|
  C. 提前 windup, 给 PD 更长 settle time, 减少 lag
  D. 组合
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


# V66 当前 keyframes (匹配 create_forehand.py MIDDLE)
V66 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.507, -0.315, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

# 真实 env_cfg 设置
HIT_PHASE_CFG = 0.475
BALL_ARRIVE_TIME_EST = 0.5205
INITIAL_PHASE = (HIT_PHASE_CFG - BALL_ARRIVE_TIME_EST) % 1.0
DURATION = 1.0  # motion 总时长 (npz 1.0s)

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


def shift_t(keys, t_old, t_new):
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

    def run(keys):
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")
        full = robot.data.default_joint_pos[0:1].clone()
        # warmup 从 INITIAL_PHASE 开始 (匹配真实 env reset 后状态)
        q0 = spline(INITIAL_PHASE)
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
            # 应用真实 phase offset, 模拟 env_cfg 的 _phase_aligned_init
            phase = (INITIAL_PHASE + t / DURATION) % 1.0
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

        # 找碰撞瞬间 (paddle-ball 距离最小, hit window 加宽到 [0.40, 0.70] 因为 motion 整体后移)
        hw = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.70]
        hi, hr = min(hw, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][4]))
        gap = float(np.linalg.norm(hr[1] - hr[4]))

        # paddle peak |v| 在哪一帧
        idx_peak = max(range(len(log)), key=lambda i: np.linalg.norm(log[i][3]))
        t_peak = log[idx_peak][0]
        v_peak = float(np.linalg.norm(log[idx_peak][3]))

        post_idx = min(hi + 2, len(log) - 1)
        bp_post = log[post_idx][4]
        bv_post = log[post_idx][5]
        face_n = quat_to_R(hr[2]) @ FACE_BODY
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            bp_post[0], bp_post[2], bv_post[0], bv_post[2])

        return dict(
            hit_t=hr[0], gap=gap,
            paddle_pos=hr[1], paddle_vel=hr[3], face_n=face_n,
            ball_pos_pre=hr[4], ball_vel_pre=hr[5],
            ball_pos_post=bp_post, ball_vel_post=bv_post,
            peak_t=t_peak, peak_v=v_peak,
            z_at_net=z_at_net, x_bounce=x_bounce,
            clears=clears, valid=valid,
        )

    # ========= 变体集 (锁定 PIN 时间, 仅调幅度让 paddle |v| 稍大) =========
    # 用户反馈: 时机 ok, 力度需要"稍微大一些" — 不动 timing, 只加大 windup→PIN→snap Δ.

    # B 系列: yb_6 (wrist_pitch) 加深 windup → 加大 PIN→snap 的 ω
    B1 = edits(V66, [(0.400, 5, -1.285)])  # windup -1.245 → -1.285 (limit -1.288)
    B2 = edits(V66, [(0.400, 5, -1.285), (0.475, 5, -1.080)])  # 同时 PIN 略深, Δ_PIN→snap +0.060

    # C 系列: yb_1 (shoulder_pitch) 加大 windup→PIN+snap 幅度
    C1 = edits(V66, [(0.400, 0, +1.137), (0.475, 0, +1.337)])  # windup -0.05, PIN +0.05 (Δ +0.10)
    C2 = edits(V66, [(0.400, 0, +1.087), (0.475, 0, +1.387), (0.550, 0, +1.500)])  # 更大幅度

    # D 系列: yb_4 (elbow) windup 加深 (更深屈, PIN→snap 伸得更猛)
    D1 = edits(V66, [(0.400, 3, +0.700), (0.550, 3, +0.300)])  # 幅度 +0.20 vs V66

    # E 系列: 多关节微调组合 (yb_6 + yb_1)
    E1 = edits(V66, [(0.400, 5, -1.285), (0.400, 0, +1.137), (0.475, 0, +1.337)])
    # E2 在 E1 基础上 yb_1 snap 也抬一点
    E2 = edits(V66, [(0.400, 5, -1.285), (0.400, 0, +1.087), (0.475, 0, +1.387), (0.550, 0, +1.500)])

    # F 系列: yb_5 (wrist_roll) PIN/snap 调整, 改变 face_normal 让能量传递更直
    # V66 PIN -0.65 → -0.55 (face X 分量略增), 不大动以免破坏 face geometry
    F1 = edits(V66, [(0.475, 4, -0.550), (0.550, 4, -0.100)])

    # G: 综合 — yb_6 加深 + yb_1 加幅 + yb_4 微调
    G1 = edits(V66, [(0.400, 5, -1.285), (0.400, 0, +1.137), (0.475, 0, +1.337),
                     (0.400, 3, +0.600), (0.550, 3, +0.350)])

    probes = [
        ("V66 baseline", V66),
        ("B1: yb_6 windup -1.245→-1.285", B1),
        ("B2: yb_6 wu -1.285 + PIN -1.080", B2),
        ("C1: yb_1 wu -0.05 PIN +0.05 (Δ+0.10)", C1),
        ("C2: yb_1 wu -0.10 PIN +0.10 snap +0.06", C2),
        ("D1: yb_4 wu +0.20 snap -0.10 (大伸)", D1),
        ("E1: B1 + C1 (yb_6 + yb_1 中等)", E1),
        ("E2: B1 + C2 (yb_6 + yb_1 大幅)", E2),
        ("F1: yb_5 PIN/snap 微调 face", F1),
        ("G1: yb_6 + yb_1 + yb_4 综合", G1),
    ]

    print(f"\n{'variant':<48} {'hit_t':>6} {'gap':>5} {'pdl_v(x,y,z)':<22} "
          f"{'|pdl_v|':>7} {'pk_t':>6} {'lag':>6} "
          f"{'ball_v_post':<22} {'|bv|':>5} {'z@net':>6} flags")
    print("-" * 175)

    rows = []
    for label, keys in probes:
        bad = check_limits(keys)
        if bad:
            print(f"{label:<48} LIMITS: {bad}")
            continue
        try:
            r = run(keys)
        except Exception as e:
            print(f"{label:<48} ERROR: {e}")
            continue
        pv = r["paddle_vel"]
        bv = r["ball_vel_post"]
        bvm = float(np.linalg.norm(bv))
        pvm = float(np.linalg.norm(pv))
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        lag_ms = (r["peak_t"] - r["hit_t"]) * 1000
        print(f"{label:<48} {r['hit_t']:>6.3f} {r['gap']*100:>5.1f} "
              f"({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f}) "
              f"{pvm:>7.2f} {r['peak_t']:>6.3f} {lag_ms:>+5.0f}ms "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) "
              f"{bvm:>5.2f} {zn_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by ball post |v| (核心力度指标) ===")
    rows.sort(key=lambda r: -np.linalg.norm(r[1]["ball_vel_post"]))
    for label, r in rows:
        bv = r["ball_vel_post"]; pv = r["paddle_vel"]
        bvm = np.linalg.norm(bv)
        pvm = np.linalg.norm(pv)
        lag = (r["peak_t"] - r["hit_t"]) * 1000
        print(f"  |bv|={bvm:.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"|pv|={pvm:.2f}  pv_z={pv[2]:+.2f}  lag={lag:+.0f}ms  "
              f"z@net={r['z_at_net']:+.2f}  {label}")

    print("\n=== Variants with CLEARS net AND VALID ===")
    valid = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid:
        print("  (none)")
    for label, r in valid:
        bv = r["ball_vel_post"]
        print(f"  z@net={r['z_at_net']:+.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  {label}")

    # V66 baseline 详细诊断
    print("\n=== V66 详细碰撞诊断 ===")
    base = next(r for label, r in rows if "baseline" in label)
    print(f"  hit_t = {base['hit_t']:.4f}s  (paddle-ball 距离最小)")
    print(f"  gap   = {base['gap']*100:.2f} cm (碰撞瞬间间距)")
    print(f"  paddle pos @ hit:  ({base['paddle_pos'][0]:+.3f}, {base['paddle_pos'][1]:+.3f}, {base['paddle_pos'][2]:+.3f})")
    print(f"  paddle vel @ hit:  ({base['paddle_vel'][0]:+.3f}, {base['paddle_vel'][1]:+.3f}, {base['paddle_vel'][2]:+.3f})  |v|={np.linalg.norm(base['paddle_vel']):.2f}")
    print(f"  paddle peak |v| = {base['peak_v']:.2f} @ t={base['peak_t']:.3f}  (lag {(base['peak_t']-base['hit_t'])*1000:+.0f}ms)")
    print(f"  face_n @ hit:      ({base['face_n'][0]:+.3f}, {base['face_n'][1]:+.3f}, {base['face_n'][2]:+.3f})")
    print(f"  ball pos pre-hit:  ({base['ball_pos_pre'][0]:+.3f}, {base['ball_pos_pre'][1]:+.3f}, {base['ball_pos_pre'][2]:+.3f})")
    print(f"  ball vel pre-hit:  ({base['ball_vel_pre'][0]:+.3f}, {base['ball_vel_pre'][1]:+.3f}, {base['ball_vel_pre'][2]:+.3f})")
    print(f"  ball vel post-hit: ({base['ball_vel_post'][0]:+.3f}, {base['ball_vel_post'][1]:+.3f}, {base['ball_vel_post'][2]:+.3f})  |bv|={np.linalg.norm(base['ball_vel_post']):.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
