"""V81b: V73 base, 时机 ok, 球打网刚好没过 — 只缺 vx (+几 cm 弧线).

砍 phase 到单点 0.475 (用户确认时机准), 砍变体到 ~12 个核心.
不动 yb_6 (避限位).
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

HIT_PHASE = 0.475
BALL_ARRIVE_TIME_EST = 0.5205
INITIAL_PHASE = (HIT_PHASE - BALL_ARRIVE_TIME_EST) % 1.0
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
        return dict(gap=gap, face_n=face_n, ball_vel_post=bv_post, paddle_vel=hr[3],
                    z_at_net=z_at_net, x_bounce=x_bounce, clears=clears, valid=valid)

    # 12 个核心变体, 围绕大力度
    A1 = edits(V73, [(0.475, 4, -0.900)])  # yb_5 PIN 更深
    A2 = edits(V73, [(0.475, 4, -1.000)])
    B1 = edits(V73, [(0.300, 4, -0.465), (0.400, 4, -0.465), (0.475, 4, -0.900)])  # yb_5 hw 整体深
    C1 = edits(V73, [(0.475, 2, -1.850)])  # yb_3 less neg (肩外送)
    C2 = edits(V73, [(0.475, 2, -1.700)])
    D1 = edits(V73, [(0.400, 0, +1.087), (0.475, 0, +1.387)])  # yb_1 wu-0.05 PIN+0.05
    D2 = edits(V73, [(0.400, 0, +1.037), (0.475, 0, +1.437)])  # yb_1 wu-0.10 PIN+0.10
    E1 = edits(V73, [(0.400, 3, +0.607), (0.475, 3, +0.357)])  # yb_4 wu屈+0.10 PIN伸-0.10
    F1 = edits(V73, [(0.475, 4, -0.900), (0.475, 2, -1.850)])  # yb_5深 + yb_3外送
    F2 = edits(V73, [(0.475, 4, -0.900), (0.400, 0, +1.037), (0.475, 0, +1.437)])  # yb_5+yb_1
    F3 = edits(V73, [(0.475, 4, -1.000), (0.475, 2, -1.700),
                     (0.400, 0, +1.037), (0.475, 0, +1.437)])  # 三方大火力
    F4 = edits(V73, [(0.475, 4, -0.900), (0.475, 2, -1.850),
                     (0.400, 0, +1.087), (0.475, 0, +1.387)])  # 三方温和

    probes = [
        ("V73 baseline", V73),
        ("A1: yb_5 PIN -0.900", A1),
        ("A2: yb_5 PIN -1.000", A2),
        ("B1: yb_5 hw -0.465 + PIN -0.900", B1),
        ("C1: yb_3 PIN -1.850", C1),
        ("C2: yb_3 PIN -1.700", C2),
        ("D1: yb_1 wu-0.05 PIN+0.05", D1),
        ("D2: yb_1 wu-0.10 PIN+0.10", D2),
        ("E1: yb_4 wu+0.10 PIN-0.10", E1),
        ("F1: yb_5-0.9 + yb_3-1.85", F1),
        ("F2: yb_5-0.9 + yb_1 大", F2),
        ("F3: 三方大火力", F3),
        ("F4: 三方温和", F4),
    ]

    print(f"\n{'variant':<40} {'gap':>4} "
          f"{'pv_x':>5} {'face_x':>6} "
          f"{'bv_x':>6} {'bv_z':>6} {'|bv|':>5} "
          f"{'z@net':>6} {'x_bnc':>6} flags")
    print("-" * 130)

    rows = []
    for label, keys in probes:
        bad = check_limits(keys)
        if bad:
            print(f"{label:<40} LIMITS: {bad}")
            continue
        try:
            r = run(keys)
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")
            continue
        if r is None:
            print(f"{label:<40} no hit window")
            continue
        bv = r["ball_vel_post"]; fn = r["face_n"]; pv = r["paddle_vel"]
        bvm = float(np.linalg.norm(bv))
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLR " if r['clears'] else "    ") + ("VAL" if r['valid'] else "OWN")
        print(f"{label:<40} {r['gap']*100:>4.1f} "
              f"{pv[0]:>+5.2f} {fn[0]:>+6.2f} "
              f"{bv[0]:>+6.2f} {bv[2]:>+6.2f} {bvm:>5.2f} "
              f"{zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by ball post |vx| (越负越好) ===")
    rows.sort(key=lambda r: r[1]["ball_vel_post"][0])
    for label, r in rows:
        bv = r["ball_vel_post"]; fn = r["face_n"]
        flag = ("CLR " if r['clears'] else "    ") + ("VAL" if r['valid'] else "OWN")
        zn = r['z_at_net'] if r['z_at_net'] is not None else float('nan')
        xb = r['x_bounce'] if r['x_bounce'] is not None else float('nan')
        print(f"  vx={bv[0]:+.2f}  vz={bv[2]:+.2f}  face_x={fn[0]:+.2f}  "
              f"z@net={zn:+.2f}  x_bnc={xb:+.2f}  {flag}  {label}")

    print("\n=== CLEARS+VALID 中 |vx| 最大 ===")
    valid = [r for r in rows if r[1]['clears'] and r[1]['valid']]
    valid.sort(key=lambda r: r[1]["ball_vel_post"][0])
    if not valid:
        print("  (none)")
    for label, r in valid:
        bv = r["ball_vel_post"]
        print(f"  vx={bv[0]:+.2f}  z@net={r['z_at_net']:+.2f}  x_bnc={r['x_bounce']:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
