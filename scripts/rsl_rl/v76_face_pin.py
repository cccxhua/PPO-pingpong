"""V76: yb_6 windup 放松 + PIN 抬高, 让 face_n X 分量从 +0.04 跳到 -0.20+.

V75 教训:
  V70 yb_6 windup=-1.285 已贴 limit (-1.288), 任何 PIN 上调都让 spline overshoot 限位.
  必须先放松 windup, 才能把 PIN 抬上去.

V67 历史 (create_forehand.py 注释):
  yb_6 PIN -1.020 → -0.300, ball vx -1.36 → -2.85 (CLEARS net, z@net +1.07m).
  V67 windup 仍是 -1.245, spline overshoot 到 -1.365 (超 limit 0.077 rad), 不安全.

V76 = V66 base + 安全 yb_6 配置:
  yb_6 windup -1.245 → -1.100 / -1.000 (留 spline 下冲 buffer)
  yb_6 PIN    -1.020 → -0.500 / -0.300 (抬 face)
  + 保留 V70 yb_1 改进 (wu -0.05, PIN +0.05)

预期: ball vx 从 -1.66 → -2.0 ~ -2.5, 过网概率显著提升.
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


# 基线: V66 + V70 yb_1 改动 (drop yb_6 windup change to give yb_6 room)
BASE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.137, +0.103, -1.979, +0.507, -0.315, -1.245,  +1.000]),  # V70 yb_1 wu
    (0.475, [+1.337, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),  # V70 yb_1 PIN
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

V70 = [  # 当前 npz (含 yb_6 windup -1.285)
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
    (0.400, [+1.137, +0.103, -1.979, +0.507, -0.315, -1.285,  +1.000]),
    (0.475, [+1.337, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

HIT_PHASE_CFG = 0.475
BALL_ARRIVE_TIME_EST = 0.5205
INITIAL_PHASE = (HIT_PHASE_CFG - BALL_ARRIVE_TIME_EST) % 1.0
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

    def run(keys):
        times = np.array([k[0] for k in keys])
        angs = np.array([k[1] for k in keys], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")
        full = robot.data.default_joint_pos[0:1].clone()
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

        hw = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.70]
        hi, hr = min(hw, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][4]))
        gap = float(np.linalg.norm(hr[1] - hr[4]))
        post_idx = min(hi + 2, len(log) - 1)
        bp_post = log[post_idx][4]
        bv_post = log[post_idx][5]
        face_n = quat_to_R(hr[2]) @ FACE_BODY
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return dict(hit_t=hr[0], gap=gap, paddle_pos=hr[1], paddle_vel=hr[3],
                    face_n=face_n, ball_vel_post=bv_post, bp_post=bp_post,
                    z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid)

    # ===== E: yb_6 windup 放松 + PIN 抬高 (face X 增大) =====
    # 在 BASE (V66 + yb_1 改动) 上, yb_6 PIN 抬高同时 windup 放松防止 spline 越限.
    E1 = edits(BASE, [(0.400, 5, -1.150), (0.475, 5, -0.700)])  # 中等
    E2 = edits(BASE, [(0.400, 5, -1.100), (0.475, 5, -0.500)])  # 较大
    E3 = edits(BASE, [(0.400, 5, -1.000), (0.475, 5, -0.300)])  # V67 值
    E4 = edits(BASE, [(0.400, 5, -1.100), (0.475, 5, -0.500), (0.550, 5, -0.200)])  # snap 同步
    E5 = edits(BASE, [(0.400, 5, -1.000), (0.475, 5, -0.300), (0.550, 5, +0.000)])  # snap 抬到 0
    E6 = edits(BASE, [(0.400, 5, -1.150), (0.475, 5, -0.700), (0.550, 5, -0.200)])

    # ===== F: E + yb_1 PIN 再抬 (combo) =====
    F1 = edits(E2, [(0.475, 0, +1.387), (0.550, 0, +1.500)])  # E2 + yb_1 PIN +0.05 snap +0.06
    F2 = edits(E3, [(0.475, 0, +1.387)])  # E3 + yb_1 PIN +0.05

    probes = [
        ("V70 (current npz)", V70),
        ("BASE (V66 + yb_1 only)", BASE),
        ("E1: yb_6 wu -1.150 PIN -0.700", E1),
        ("E2: yb_6 wu -1.100 PIN -0.500", E2),
        ("E3: yb_6 wu -1.000 PIN -0.300 (V67)", E3),
        ("E4: E2 + snap -0.200", E4),
        ("E5: E3 + snap +0.000", E5),
        ("E6: yb_6 wu -1.150 PIN -0.700 snap -0.200", E6),
        ("F1: E2 + yb_1 PIN+snap 加幅", F1),
        ("F2: E3 + yb_1 PIN +0.05", F2),
    ]

    print(f"\n{'variant':<48} {'gap':>5} "
          f"{'face_n(x,y,z)':<22} "
          f"{'ball_v(x,y,z)':<22} {'|bv|':>5} {'z@net':>6} {'x_bnc':>6} flags")
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
        bv = r["ball_vel_post"]
        fn = r["face_n"]
        bvm = float(np.linalg.norm(bv))
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<48} {r['gap']*100:>5.1f} "
              f"({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f}) "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) "
              f"{bvm:>5.2f} {zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by ball post |vx| (核心: 过网需要 vx ≥ 2.0) ===")
    rows.sort(key=lambda r: r[1]["ball_vel_post"][0])
    for label, r in rows:
        bv = r["ball_vel_post"]; fn = r["face_n"]
        xb = r["x_bounce"]; xb_s = "n/a" if xb is None else f"{xb:+.2f}"
        print(f"  vx={bv[0]:+.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"face_x={fn[0]:+.2f}  z@net={r['z_at_net']:+.2f}  x_bnc={xb_s}  {label}")

    print("\n=== CLEARS net AND VALID (落对方台) ===")
    valid = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid:
        print("  (none)")
    for label, r in valid:
        bv = r["ball_vel_post"]
        print(f"  z@net={r['z_at_net']:+.2f}  x_bnc={r['x_bounce']:+.2f}  "
              f"bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
