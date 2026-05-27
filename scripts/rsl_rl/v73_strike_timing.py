"""V73: 拉长 strike phase 让 PD 在 hit 时刻处于 peak ω, 不改方向只改时序.

V72 框架错了 — 改 windup pose 破坏 hit 几何, ball 飞偏.
真正瓶颈是 PD lag: 命令 t=0.475 PIN, 实际关节 t=0.520 才到位, 球已经过了.

V73 思路: 同样 windup→PIN→snap pose, 但把 strike phase 从 75ms 拉长.
    V66: windup t=0.400, PIN t=0.475, snap t=0.550 (strike 跨 150ms, hit 段 75ms)
    V73_a: windup t=0.300, PIN t=0.475, snap t=0.550 (strike 跨 250ms, hit 段 175ms)
    V73_b: windup t=0.250, PIN t=0.475, snap t=0.625 (更平缓, 加大 follow-through)
    V73_c: windup t=0.200, PIN t=0.475, snap t=0.550 (strike 跨 350ms)

PD 在 175ms 时间内能更接近命令值. 同时 yb_6 大 snap (Δ=0.525 from PIN to snap)
本来就在 hit 之后, 这部分 PD lag 让它实际在 t=0.55-0.60 才达到. 拉长 strike 后,
yb_6 peak ω 应该在 t≈0.50 — 接近 hit 时刻而不是更晚.
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


# V66 关节关键值 (windup/PIN/snap, 不改 pose 只改时间)
WINDUP = [+1.187, +0.103, -1.979, +0.507, -0.315, -1.245,  +1.000]
PIN    = [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]
SNAP   = [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]
READY  = [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]
MID    = [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]
FOLLOW = [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]


def make_keys(t_mid, t_windup, t_pin, t_snap, t_follow=0.700, t_return=0.900):
    return [
        (0.000, READY),
        (t_mid, MID),
        (t_windup, WINDUP),
        (t_pin, PIN),
        (t_snap, SNAP),
        (t_follow, FOLLOW),
        (t_return, READY),
        (1.000, READY),
    ]


V66 = make_keys(0.300, 0.400, 0.475, 0.550)

NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.79
G = 9.81
FACE_BODY = np.array([0.0, 0.0, -1.0])
LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


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

        # paddle |v| 时序: 找 peak |v| 在哪一帧
        idx_peak = max(range(len(log)), key=lambda i: np.linalg.norm(log[i][3]))
        t_peak = log[idx_peak][0]
        v_peak = np.linalg.norm(log[idx_peak][3])

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
                    clears=clears, valid=valid, hit_t=hr[0],
                    peak_t=t_peak, peak_v=v_peak)

    probes = [
        # (label, t_mid, t_windup, t_pin, t_snap)
        ("V66 baseline (wu 0.400, PIN 0.475)", make_keys(0.300, 0.400, 0.475, 0.550)),
        ("V73_a wu 0.350 (strike 125ms)",       make_keys(0.250, 0.350, 0.475, 0.550)),
        ("V73_b wu 0.300 (strike 175ms)",       make_keys(0.200, 0.300, 0.475, 0.550)),
        ("V73_c wu 0.250 (strike 225ms)",       make_keys(0.150, 0.250, 0.475, 0.550)),
        ("V73_d wu 0.200 (strike 275ms)",       make_keys(0.100, 0.200, 0.475, 0.550)),
        # 同时 snap 也外推 (让 PIN→snap 段也变长, peak ω 不会太靠 hit 之前)
        ("V73_e wu 0.300 + snap 0.625",          make_keys(0.200, 0.300, 0.475, 0.625, 0.800)),
        ("V73_f wu 0.250 + snap 0.625",          make_keys(0.150, 0.250, 0.475, 0.625, 0.800)),
        ("V73_g wu 0.200 + snap 0.700",          make_keys(0.100, 0.200, 0.475, 0.700, 0.850)),
    ]

    print(f"\n{'variant':<48} {'paddle_z @ wu/hit/post':<28} "
          f"{'paddle_v @ hit':<22} {'peak_t':>7} {'peak|v|':>7} "
          f"{'ball post-v':<22} {'z@net':>6} flags")
    print("-" * 165)

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
        bv = r["ball_vel"]
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        ht = r["hit_t"]
        pt = r["peak_t"]
        pkv = r["peak_v"]
        # 没 windup/post 数据, 简化为 hit 时 paddle z 和峰值时刻 lag
        lag_ms = (pt - ht) * 1000
        print(f"{label:<48} hit_t={ht:.3f} pk_t={pt:.3f} (lag {lag_ms:+.0f}ms) "
              f"({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f}) "
              f"|pk|={pkv:.2f} "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) "
              f"{zn_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by peak time alignment with hit (lag closest to 0) ===")
    rows.sort(key=lambda r: abs(r[1]["peak_t"] - r[1]["hit_t"]))
    for label, r in rows:
        bv = r["ball_vel"]; pv = r["paddle_vel"]
        lag_ms = (r["peak_t"] - r["hit_t"]) * 1000
        print(f"  lag={lag_ms:+.0f}ms  pk|v|={r['peak_v']:.2f}  "
              f"hit_pv=({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f})  "
              f"bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"z@net={r['z_at_net']:+.2f}  {label}")

    print("\n=== Sorted by ball post |v| ===")
    rows.sort(key=lambda r: -np.linalg.norm(r[1]["ball_vel"]))
    for label, r in rows[:6]:
        bv = r["ball_vel"]
        print(f"  |bv|={np.linalg.norm(bv):.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"z@net={r['z_at_net']:+.2f}  {label}")

    print("\n=== CLEARS net AND VALID ===")
    valid = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid:
        print("  (none)")
    for label, r in valid:
        bv = r["ball_vel"]
        print(f"  z@net={r['z_at_net']:+.2f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
