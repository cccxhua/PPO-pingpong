"""V71: 第二轮 — D yb_4 elbow snap, E paddle hit-Z 抬高, F 联合.

V70 finding: V66 paddle_vz 已 +0.25 (架构上限), 主要瓶颈是 ball_vx 不够 (~-2.0).
    要过网需要 vx≤-3.0 或 vz≥+2.8. paddle_vz 改进很难 (PD lag).
    A2 给最高 ball_vx=-2.35, 但 ball_vz 降. 还差 0.65 m/s vx 或 1.6 m/s vz.

新策略:
    D: yb_4 hit-window 反向 snap (hit 时刻 elbow 屈曲增加, 而非 V66 的 extending)
        yb_4 + 是 屈肘. V66 hit-window: 0.507 → 0.457 → 0.407 (elbow extending).
        反向: hit 时刻 yb_4 增加 → elbow 屈曲, paddle 沿前臂方向被甩起来
    E: yb_1 PIN 抬高 (paddle hit 时刻 z0 更高 → 更多 trajectory 余量)
    F: 把 paddle 更靠 -Z 击球后再向上 — i.e. yb_2 (shoulder_roll) snap 让 paddle 上挑
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

    print(f"\n{'variant':<60} {'paddle_v':<22} {'ball post-v':<22} "
          f"{'paddle_z':>7} {'z@net':>6} {'x_bnc':>6} flags")
    print("-" * 140)

    # === D: yb_4 elbow 反向 snap ===
    # V66 hit window: 0.507 → 0.457 → 0.407 (extending). Δ_pre=-0.05, Δ_post=-0.05
    # 反向 V1: 让 hit 时刻 yb_4 增大 (屈曲 snap)
    D1 = edits(V66, [(0.400, 3, +0.700), (0.475, 3, +0.900)])  # windup 浅, PIN 屈曲
    D2 = edits(V66, [(0.400, 3, +0.700), (0.475, 3, +1.100), (0.550, 3, +0.900)])  # 大屈曲 snap
    D3 = edits(V66, [(0.400, 3, +0.500), (0.475, 3, +0.800), (0.550, 3, +0.600)])  # 中等
    D4 = edits(V66, [(0.400, 3, +0.300), (0.475, 3, +0.700), (0.550, 3, +0.500)])  # 大幅 windup→PIN

    # === E: yb_1 PIN 抬高 (paddle hit 时刻 z0 更高) ===
    E1 = edits(V66, [(0.475, 0, +1.500)])  # PIN +0.213
    E2 = edits(V66, [(0.475, 0, +1.600), (0.550, 0, +1.700)])

    # === F: yb_2 shoulder_roll 反向 (从 +0.103 → 负, 让 paddle 上挑) ===
    # 注意: yb_2 limit [-3.081, +0.314], 当前 0.103 离上限近. 改 negative direction.
    F1 = edits(V66, [(0.475, 1, -0.150), (0.550, 1, -0.300)])
    F2 = edits(V66, [(0.475, 1, +0.250), (0.550, 1, +0.300)])  # 反向: 朝 +Y 上挑

    # === G: D + A2 联合 (D yb_4 snap + A2 yb_6 PIN -0.70) ===
    G1 = edits(V66, [(0.400, 3, +0.500), (0.475, 3, +0.800), (0.550, 3, +0.600),
                     (0.475, 5, -0.700), (0.550, 5, -0.500)])
    G2 = edits(V66, [(0.400, 3, +0.700), (0.475, 3, +1.000), (0.550, 3, +0.700),
                     (0.475, 5, -0.700), (0.550, 5, -0.500)])

    # === H: 整体 swing 提前 (mid 帧加速) ===
    H1 = edits(V66, [(0.300, 5, -0.800)])  # mid yb_6 浅 → windup (0.30→0.40) Δ 加大
    H2 = edits(V66, [(0.300, 5, -0.600)])  # 更浅
    H3 = edits(V66, [(0.300, 5, -0.500), (0.475, 5, -0.700), (0.550, 5, -0.500)])  # 联合提前+ PIN 浅

    probes = [
        ("V66 baseline", V66),
        ("D1: yb_4 wu+0.70 PIN+0.90", D1),
        ("D2: yb_4 wu+0.70 PIN+1.10 snap+0.90", D2),
        ("D3: yb_4 wu+0.50 PIN+0.80 snap+0.60", D3),
        ("D4: yb_4 wu+0.30 PIN+0.70 snap+0.50", D4),
        ("E1: yb_1 PIN +1.50", E1),
        ("E2: yb_1 PIN +1.60 snap +1.70", E2),
        ("F1: yb_2 PIN -0.15 snap -0.30", F1),
        ("F2: yb_2 PIN +0.25 snap +0.30", F2),
        ("G1: D3 + A2 (yb_4 snap + yb_6 PIN -0.70)", G1),
        ("G2: D2-mid + A2", G2),
        ("H1: yb_6 mid -0.80 (early acc)", H1),
        ("H2: yb_6 mid -0.60", H2),
        ("H3: yb_6 mid -0.50 + PIN -0.70 + snap -0.50", H3),
    ]

    rows = []
    for label, keys in probes:
        bad = check_limits(keys)
        if bad:
            print(f"{label:<60} LIMITS: {bad}")
            continue
        try:
            r = run(keys)
        except Exception as e:
            print(f"{label:<60} ERROR: {e}")
            continue
        pv = r["paddle_vel"]
        bv = r["ball_vel"]
        pz = r["paddle_pos"][2]
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<60} ({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f}) "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) "
              f"{pz:>+7.3f} {zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by ball post |v| (overall hit strength) ===")
    rows.sort(key=lambda r: -np.linalg.norm(r[1]["ball_vel"]))
    for label, r in rows[:8]:
        bv = r["ball_vel"]; pv = r["paddle_vel"]
        bvm = np.linalg.norm(bv)
        xb = r["x_bounce"]; xb_s = "n/a" if xb is None else f"{xb:+.2f}"
        print(f"  |bv|={bvm:.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"pv_z={pv[2]:+.2f}  z@net={r['z_at_net']:+.2f}  x_bnc={xb_s}  {label}")

    print("\n=== Sorted by z@net (closest to clearing 0.91) ===")
    rows.sort(key=lambda r: -(r[1]["z_at_net"] or -99))
    for label, r in rows[:8]:
        bv = r["ball_vel"]; pv = r["paddle_vel"]
        xb = r["x_bounce"]; xb_s = "n/a" if xb is None else f"{xb:+.2f}"
        print(f"  z@net={r['z_at_net']:+.2f}  x_bnc={xb_s}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"pv_z={pv[2]:+.2f}  {label}")

    print("\n=== Variants with CLEARS net AND VALID first bounce ===")
    valid_rows = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid_rows:
        print("  (none)")
    for label, r in valid_rows:
        bv = r["ball_vel"]
        print(f"  z@net={r['z_at_net']:+.2f}  x_bnc={r['x_bounce']:+.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
