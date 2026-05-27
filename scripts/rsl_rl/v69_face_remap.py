"""V69: 解决 V66 face Y 过大 (-0.68) X 不够 (-0.16) 问题.

V66 face_n = (-0.16, -0.68, +0.72): 法向几乎在 YZ 平面.
ball 入射 ≈ (+1.84, 0, +2.5) (post-bounce on robot table).
反射 vx_post = -1.97 (不够), vz_post = +1.60 (太低).

要让 face 更朝 -X: 换 yb_6 (wrist_pitch) 和 yb_7 (wrist_yaw).
yb_5 单调没用 (它沿 paddle 长轴 roll, 不改 face 主方向).

paddle face_normal_body = (0,0,-1).
face_n_world = -R[:,2], R 由 yb_5/yb_6/yb_7 wrist 链决定.

策略 (扫描):
  1) yb_7 PIN 调整 (wrist_yaw, 让 paddle 整体绕竖直轴转, 把 face 从 -Y 转到 -X)
  2) yb_6 PIN 调整 (wrist_pitch, 控制 face 上下倾角)
  3) yb_3 PIN 调整 (shoulder_yaw, 改大臂朝向)
  4) 综合
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
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, q, bp, bv))

        # 找 hit (paddle-ball 距离最小)
        hw = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.60]
        hi, hr = min(hw, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][3]))
        # post-hit 取 hit+2 帧
        post_idx = min(hi + 2, len(log) - 1)
        bp_post = log[post_idx][3]
        bv_post = log[post_idx][4]
        face_n = quat_to_R(hr[2]) @ FACE_BODY
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return dict(paddle_pos=hr[1], ball_pos=bp_post, ball_vel=bv_post,
                    face_n=face_n, z_at_net=z_at_net, x_bounce=x_bounce,
                    clears=clears, valid=valid)

    print(f"\n{'variant':<55} {'ball post-v':<22} {'face_n':<22} "
          f"{'z@net':>6} {'x_bnc':>6} flags")
    print("-" * 130)

    probes = [
        ("V66 baseline", V66),
        # yb_7 (wrist_yaw) PIN sweep — 把 face 从 -Y 转到 -X
        ("V66 + yb_7 PIN +0.50 (was +1.00)", edit_kf(V66, 0.475, 6, +0.50)),
        ("V66 + yb_7 PIN +0.30",             edit_kf(V66, 0.475, 6, +0.30)),
        ("V66 + yb_7 PIN +0.10",             edit_kf(V66, 0.475, 6, +0.10)),
        ("V66 + yb_7 PIN -0.20",             edit_kf(V66, 0.475, 6, -0.20)),
        ("V66 + yb_7 PIN +1.50 (反向)",      edit_kf(V66, 0.475, 6, +1.50)),
        ("V66 + yb_7 PIN +2.00",             edit_kf(V66, 0.475, 6, +2.00)),
        # yb_7 hit-window 整体调整
        ("V66 + yb_7 hw -0.50",  apply_delta(V66, {6: -0.50})),
        ("V66 + yb_7 hw -0.70",  apply_delta(V66, {6: -0.70})),
        ("V66 + yb_7 hw +0.50",  apply_delta(V66, {6: +0.50})),
        # yb_6 (wrist_pitch) PIN sweep — 控制 face 上下倾
        ("V66 + yb_6 PIN -0.60 (was -1.02)", edit_kf(V66, 0.475, 5, -0.60)),
        ("V66 + yb_6 PIN -0.30",             edit_kf(V66, 0.475, 5, -0.30)),
        ("V66 + yb_6 PIN +0.00",             edit_kf(V66, 0.475, 5, +0.00)),
        ("V66 + yb_6 PIN +0.30",             edit_kf(V66, 0.475, 5, +0.30)),
        ("V66 + yb_6 PIN -1.20",             edit_kf(V66, 0.475, 5, -1.20)),
        # yb_3 (shoulder_yaw) PIN — 大臂朝向
        ("V66 + yb_3 PIN -1.50 (was -1.98)", edit_kf(V66, 0.475, 2, -1.50)),
        ("V66 + yb_3 PIN -2.40",             edit_kf(V66, 0.475, 2, -2.40)),
        # 组合: yb_7 + yb_6
        ("V66 + yb_7 PIN +0.30 + yb_6 PIN -0.60",
            edits(V66, [(0.475, 6, +0.30), (0.475, 5, -0.60)])),
        ("V66 + yb_7 PIN +0.10 + yb_6 PIN -0.30",
            edits(V66, [(0.475, 6, +0.10), (0.475, 5, -0.30)])),
        ("V66 + yb_7 hw -0.50 + yb_6 hw +0.30",
            apply_delta(V66, {6: -0.50, 5: +0.30})),
        ("V66 + yb_7 hw -0.70 + yb_6 hw +0.30",
            apply_delta(V66, {6: -0.70, 5: +0.30})),
        # 极端
        ("V66 + yb_7 hw -1.00", apply_delta(V66, {6: -1.00})),
    ]

    rows = []
    for label, keys in probes:
        try:
            r = run(keys)
        except Exception as e:
            print(f"{label:<55} ERROR: {e}")
            continue
        v = r["ball_vel"]
        fn = r["face_n"]
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<55} ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})  "
              f"({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f}) "
              f"{zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Variants with both CLEARS net AND VALID first bounce ===")
    valid_rows = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid_rows:
        print("  (none — face direction still wrong)")
    for label, r in valid_rows:
        v = r["ball_vel"]
        fn = r["face_n"]
        print(f"  z@net={r['z_at_net']:+.2f}  x_bnc={r['x_bounce']:+.2f}  "
              f"vx={v[0]:+.2f} vz={v[2]:+.2f}  "
              f"fn=({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f})  {label}")

    print("\n=== Top 8 by face_n X (most -X = best for backhit) ===")
    rows.sort(key=lambda r: r[1]["face_n"][0])
    for label, r in rows[:8]:
        v = r["ball_vel"]; fn = r["face_n"]
        xb = r["x_bounce"]
        xb_s = "n/a" if xb is None else f"{xb:+.2f}"
        print(f"  fn=({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f})  "
              f"vx={v[0]:+.2f} vz={v[2]:+.2f}  z@net={r['z_at_net']:+.2f}  "
              f"x_bnc={xb_s}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
