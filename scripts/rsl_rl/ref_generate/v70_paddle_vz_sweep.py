"""V70: 扫 yb_6 snap timing (A) + yb_1 hit ω (B) 组合.

关键指标 (v68/v69 都没测):
    paddle_vz @ hit  - 击球瞬间 paddle 是否在向上走
    ball_post_vz     - 击球后球的初始 vz (从 hit_idx+2 帧取, 不是 max-over-window)

Baseline V66:
    yb_6 hit-window: -1.245 (windup) → -1.020 (PIN) → -0.495 (snap)
        Δ_pre = 0.225 rad / 75ms = 3 rad/s   (击球瞬间 yb_6 ω)
        Δ_post = 0.525 rad / 75ms = 7 rad/s  (snap 主体, 但发生在 hit 之后)
    yb_1 hit-window: 1.187 (windup) → 1.287 (PIN) → 1.437 (snap)
        Δ_pre = 0.10 rad / 75ms = 1.33 rad/s  (击球时刻肩 ω, 偏小)

A series: yb_6 snap timing — 把 snap 主体提前到 hit 之前
B series: yb_1 hit ω    — 降 windup, 让 windup→PIN 段更陡
C series: A+B 组合
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
    """Run cubic spline over full duration, check if joint commands stay in limits."""
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    bad = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo or y[:, i].max() > hi:
            bad.append(f"yb_{i+1}=[{y[:,i].min():+.3f},{y[:,i].max():+.3f}]vs[{lo:+.3f},{hi:+.3f}]")
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
            q = robot.data.body_quat_w[0, paddle_idx].cpu().numpy().copy()
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, q, pv, bp, bv))

        hw = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.60]
        hi, hr = min(hw, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][4]))
        post_idx = min(hi + 2, len(log) - 1)
        bp_post = log[post_idx][4]
        bv_post = log[post_idx][5]
        face_n = quat_to_R(hr[2]) @ FACE_BODY
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return dict(paddle_pos=hr[1], paddle_vel=hr[3],
                    ball_pos=bp_post, ball_vel=bv_post,
                    face_n=face_n, z_at_net=z_at_net, x_bounce=x_bounce,
                    clears=clears, valid=valid, hit_t=hr[0])

    print(f"\n{'variant':<55} {'paddle_v (xyz)':<22} {'ball post-v':<22} "
          f"{'face_n':<22} {'z@net':>6} {'x_bnc':>6} flags")
    print("-" * 145)

    # === A: yb_6 snap timing — 把 Δ 从 PIN→snap 移到 windup→PIN ===
    # baseline V66: windup=-1.245, PIN=-1.020, snap=-0.495. Δ_pre=0.225, Δ_post=0.525
    # 想要: Δ_pre 大 (peak ω 在 hit), Δ_post 小
    A1 = edits(V66, [(0.475, 5, -0.80), (0.550, 5, -0.60)])   # Δ_pre=0.445, Δ_post=0.20
    A2 = edits(V66, [(0.475, 5, -0.70), (0.550, 5, -0.50)])   # Δ_pre=0.545, Δ_post=0.20
    A3 = edits(V66, [(0.475, 5, -0.60), (0.550, 5, -0.45)])   # Δ_pre=0.645, Δ_post=0.15
    A4 = edits(V66, [(0.475, 5, -0.50), (0.550, 5, -0.40)])   # Δ_pre=0.745, Δ_post=0.10
    # 加 windup 更深以扩 pre Δ (但要避开 -1.288 limit)
    A5 = edits(V66, [(0.400, 5, -1.260), (0.475, 5, -0.70), (0.550, 5, -0.50)])  # 撞上限
    # === B: yb_1 hit ω 增加 (降 windup, 抬 PIN 让 windup→PIN 段陡) ===
    # baseline yb_1 hit-window: 1.187 → 1.287 → 1.437. Δ_pre=0.10
    B1 = edits(V66, [(0.400, 0, +1.100), (0.475, 0, +1.350)])  # Δ_pre=0.25 (vs 0.10)
    B2 = edits(V66, [(0.400, 0, +1.050), (0.475, 0, +1.350)])  # Δ_pre=0.30
    B3 = edits(V66, [(0.400, 0, +1.000), (0.475, 0, +1.350)])  # Δ_pre=0.35
    B4 = edits(V66, [(0.400, 0, +1.050), (0.475, 0, +1.400)])  # Δ_pre=0.35, PIN 高
    # === C: A+B 组合 ===
    C1 = edits(V66, [(0.400, 0, +1.050), (0.475, 0, +1.350),
                     (0.475, 5, -0.70), (0.550, 5, -0.50)])
    C2 = edits(V66, [(0.400, 0, +1.000), (0.475, 0, +1.350),
                     (0.475, 5, -0.60), (0.550, 5, -0.45)])
    C3 = edits(V66, [(0.400, 0, +1.050), (0.475, 0, +1.400),
                     (0.475, 5, -0.50), (0.550, 5, -0.40)])

    probes = [
        ("V66 baseline", V66),
        ("A1: yb_6 PIN -0.80, snap -0.60", A1),
        ("A2: yb_6 PIN -0.70, snap -0.50", A2),
        ("A3: yb_6 PIN -0.60, snap -0.45", A3),
        ("A4: yb_6 PIN -0.50, snap -0.40", A4),
        ("B1: yb_1 wu 1.10, PIN 1.35", B1),
        ("B2: yb_1 wu 1.05, PIN 1.35", B2),
        ("B3: yb_1 wu 1.00, PIN 1.35", B3),
        ("B4: yb_1 wu 1.05, PIN 1.40", B4),
        ("C1: B2 + A2 (yb_1 wu 1.05, PIN 1.35; yb_6 PIN -0.70)", C1),
        ("C2: B3 + A3 (yb_1 wu 1.00, PIN 1.35; yb_6 PIN -0.60)", C2),
        ("C3: B2 + A4 (yb_1 wu 1.05, PIN 1.40; yb_6 PIN -0.50)", C3),
    ]

    rows = []
    for label, keys in probes:
        bad = check_limits(keys)
        if bad:
            print(f"{label:<55} LIMITS: {bad}")
            continue
        try:
            r = run(keys)
        except Exception as e:
            print(f"{label:<55} ERROR: {e}")
            continue
        pv = r["paddle_vel"]
        bv = r["ball_vel"]
        fn = r["face_n"]
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<55} ({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f}) "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) "
              f"({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f}) "
              f"{zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by paddle vz @ hit (most +Z = best 'from below up') ===")
    rows.sort(key=lambda r: -r[1]["paddle_vel"][2])
    for label, r in rows[:8]:
        pv = r["paddle_vel"]; bv = r["ball_vel"]
        xb = r["x_bounce"]; xb_s = "n/a" if xb is None else f"{xb:+.2f}"
        print(f"  paddle_vz={pv[2]:+.2f}  ball_vz={bv[2]:+.2f}  ball_vx={bv[0]:+.2f}  "
              f"z@net={r['z_at_net']:+.2f}  x_bnc={xb_s}  {label}")

    print("\n=== Variants with CLEARS net AND VALID first bounce ===")
    valid_rows = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid_rows:
        print("  (none)")
    for label, r in valid_rows:
        pv = r["paddle_vel"]; bv = r["ball_vel"]
        print(f"  paddle_vz={pv[2]:+.2f}  ball_vz={bv[2]:+.2f}  ball_vx={bv[0]:+.2f}  "
              f"z@net={r['z_at_net']:+.2f}  x_bnc={r['x_bounce']:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
