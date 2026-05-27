"""V72: 重新设计 — 'low-to-high' topspin forehand.

V70/V71 共扫 25+ variant, 全部 OWN bounce, z@net 最高 +0.49 < 网顶 0.91.
根因: 当前架构是"top-down"挥拍, paddle hit 时 z=1.029 (近 ball z=1.03),
    paddle_vz=+0.25 (太小). 整个挥拍是横扫不是上挑.

V72 思路:
    windup (t=0.40): paddle z 拉到最低 (0.85-0.90)
        - yb_1 (shoulder_pitch) 降 (1.187 → 0.7-0.8 可控范围)
        - yb_4 (elbow) 屈 (0.507 → 1.0-1.5 让前臂收回, paddle 拉下来)
    PIN (t=0.475): paddle 回到 ball z=1.03 + vz 强正
        - yb_1 大 swing 升 (0.7 → 1.3)
        - yb_4 大 swing 伸 (1.2 → 0.5)
    snap (t=0.55): paddle 高 (z=1.15+)
        - yb_1 → 1.5
        - yb_4 → 0.3

Δyb_1 windup→PIN = 0.6 rad / 75ms = 8 rad/s 肩 ω (vs V66 1.33)
Δyb_4 windup→PIN = -0.7 rad / 75ms = -9.3 rad/s 肘 ω (vs V66 -0.67)

PD lag 严重, 但用 forward FK 看命令 paddle z trajectory 是否走"low-to-high".
然后实测 sim 看 PD 跟踪后是否还有 +Z 速度.
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

    def run(keys, log_paddle_z=False):
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

        # paddle_z trajectory at t=0.30, 0.40, 0.475, 0.55
        z_traj = []
        for t_target in [0.30, 0.40, 0.475, 0.55]:
            idx = min(range(len(log)), key=lambda i: abs(log[i][0] - t_target))
            z_traj.append((t_target, log[idx][1][2]))

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
                    z_traj=z_traj)

    # === V72: low-to-high 设计 ===
    # paddle_z 顺序: 高 → 低 (windup) → 中 (PIN) → 高 (snap)
    # 调整 yb_1, yb_4 是主要杠杆; yb_6 保留 V66 做二次驱动
    V72_a = [  # 温和版: yb_1/yb_4 中等 swing
        (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (0.300, [+0.900, +0.198, -1.904, +1.000, -0.315, -1.045,  +1.000]),  # mid: 肩降 + 肘屈
        (0.400, [+0.700, +0.103, -1.979, +1.300, -0.315, -1.245,  +1.000]),  # windup: paddle LOW
        (0.475, [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),  # PIN: V66 hit pose
        (0.550, [+1.500, +0.103, -1.979, +0.300, -0.165, -0.495,  +1.000]),  # snap: high follow
        (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
        (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    ]
    V72_b = [  # 中等版: yb_1 更深 windup
        (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (0.300, [+0.800, +0.198, -1.904, +1.100, -0.315, -1.045,  +1.000]),
        (0.400, [+0.500, +0.103, -1.979, +1.500, -0.315, -1.245,  +1.000]),  # paddle 极低
        (0.475, [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),
        (0.550, [+1.500, +0.103, -1.979, +0.300, -0.165, -0.495,  +1.000]),
        (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
        (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    ]
    V72_c = [  # 仅改 yb_1: 看 shoulder pitch 单独效果
        (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (0.300, [+0.900, +0.198, -1.904, +0.877, -0.315, -1.045,  +1.000]),
        (0.400, [+0.700, +0.103, -1.979, +0.507, -0.315, -1.245,  +1.000]),
        (0.475, [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),
        (0.550, [+1.500, +0.103, -1.979, +0.407, -0.165, -0.495,  +1.000]),
        (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
        (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    ]
    V72_d = [  # 仅改 yb_4: 看 elbow flex 单独效果
        (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (0.300, [+1.127, +0.198, -1.904, +1.000, -0.315, -1.045,  +1.000]),
        (0.400, [+1.187, +0.103, -1.979, +1.300, -0.315, -1.245,  +1.000]),
        (0.475, [+1.287, +0.103, -1.979, +0.457, -0.650, -1.020,  +1.000]),
        (0.550, [+1.437, +0.103, -1.979, +0.300, -0.165, -0.495,  +1.000]),
        (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
        (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    ]
    V72_e = [  # 极端版: 最大 low-to-high swing
        (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (0.300, [+0.700, +0.198, -1.904, +1.300, -0.315, -1.045,  +1.000]),
        (0.400, [+0.300, +0.103, -1.979, +1.700, -0.315, -1.245,  +1.000]),  # 极低 windup
        (0.475, [+1.300, +0.103, -1.979, +0.500, -0.650, -1.020,  +1.000]),
        (0.550, [+1.600, +0.103, -1.979, +0.200, -0.165, -0.495,  +1.000]),  # 极高 snap
        (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
        (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
        (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    ]

    probes = [
        ("V66 baseline", V66),
        ("V72_a: 温和 low-to-high (yb_1 wu 0.7, yb_4 wu 1.3)", V72_a),
        ("V72_b: 中等 (yb_1 wu 0.5, yb_4 wu 1.5)", V72_b),
        ("V72_c: 仅 yb_1 (wu 0.7)", V72_c),
        ("V72_d: 仅 yb_4 (wu 1.3)", V72_d),
        ("V72_e: 极端 (yb_1 wu 0.3, yb_4 wu 1.7)", V72_e),
    ]

    print(f"\n{'variant':<55} {'paddle_z @ t=0.30/0.40/0.475/0.55':<40} "
          f"{'paddle_v @ hit':<22} {'ball post-v':<22} {'z@net':>6} flags")
    print("-" * 165)

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
        zts = r["z_traj"]
        z_str = " ".join(f"{z:.3f}" for _, z in zts)
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<55} {z_str:<40} ({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f}) "
              f"({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f}) {zn_s} {flag}")
        rows.append((label, r))

    print("\n=== Sorted by paddle vz @ hit ===")
    rows.sort(key=lambda r: -r[1]["paddle_vel"][2])
    for label, r in rows:
        pv = r["paddle_vel"]; bv = r["ball_vel"]
        zts = r["z_traj"]
        zwu = zts[1][1]; zhit = zts[2][1]; zsnap = zts[3][1]
        rise = zhit - zwu  # paddle rise from windup to hit
        print(f"  pv_z={pv[2]:+.3f}  bv=({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})  "
              f"pdl_rise(wu→hit)={rise:+.3f}  pdl(wu/hit/snap)={zwu:.3f}/{zhit:.3f}/{zsnap:.3f}  "
              f"z@net={r['z_at_net']:+.2f}  {label}")

    print("\n=== Variants with CLEARS net AND VALID ===")
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
