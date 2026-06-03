"""V68: 真正诊断 V66 是否先在己方台弹再过网.

之前 probe 的 z_at_net 找的是 ball x>0→x<0 的第一次穿越, 但球可能先在
robot 台 (x>0) 上弹一下再过网. 用户视觉确认: V66 球先落己方台.

V66 解析:
    hit @ (1.27, 1.05), post-v = (-1.97, ?, +2.46) (max over post-hit log)
    t_net = 1.27 / 1.97 = 0.645s
    z_at_net = 1.05 + 2.46*0.645 - 4.905*0.645² = 0.60m  <-- 远低于网顶 0.91m
    -> 球先在 robot 台 (x>0) 弹一次, 反弹后才过网.

本探针:
  1) 拿 hit 后第 1 帧的 ball pos+vel 作"初始抛物线参数"
  2) 解析积分: 首次 z<=0.79 时 x 在哪 (= 首次落点)
  3) 解析积分: x 第一次 = 0 时 z 是多少 (= 直接过网高度, 不算反弹)
  4) 判断: 首次落点 x>0 (己方台/桌外) → 无效, x<0 (对方台) → 有效
  5) 扫 yb_5 PIN, yb_5 hit-window, 看哪些组合给 valid + 过网

关键诊断指标:
  bx0, bz0 = ball post-hit initial velocity (frame after hit)
  x_first_bounce = ball first time z=0.79 location (x value)
  z_at_net_direct = ball z when crossing x=0 BEFORE any table bounce
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


# V66 (current create_forehand.py)
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

NET_X = 0.0
NET_Z = 0.9125  # table top 0.76 + net height 0.1525 = 0.9125
TABLE_Z = 0.79  # ball center when touching table top (radius ~0.02)
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
    """从初始 ball pos+vel 解析抛物线, 返回:
        z_net_direct   - 球到 x=0 时 z (无反弹, < TABLE_Z 即穿桌子)
        x_first_bounce - 球首次 z=TABLE_Z 时 x
        clears_net     - z_net_direct > NET_Z and not yet bounced
        valid_hit      - first bounce on opponent side (x < 0)
    """
    # Time to reach x=0 (assume vx < 0)
    if vx >= 0:
        return None, None, False, False
    t_net = x0 / (-vx)
    z_at_net_direct = z0 + vz * t_net - 0.5 * G * t_net * t_net

    # Time to reach z=TABLE_Z (going down).
    # z = z0 + vz*t - 0.5*g*t² = TABLE_Z
    # 0.5*g*t² - vz*t + (TABLE_Z - z0) = 0
    a = 0.5 * G
    b = -vz
    c = TABLE_Z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        # ball never reaches table (very unlikely for physical hit)
        return z_at_net_direct, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2 * a)  # take later root (going down)
    x_first_bounce = x0 + vx * t_bounce

    # ball reaches net BEFORE it would touch table?
    clears_net = z_at_net_direct > NET_Z and t_net < t_bounce
    valid_hit = x_first_bounce < NET_X  # first bounce on opponent side
    return z_at_net_direct, x_first_bounce, clears_net, valid_hit


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

        # 找 hit 时刻 (paddle-ball 距离最小)
        hit_window = [(i, r) for i, r in enumerate(log) if 0.40 < r[0] < 0.60]
        hit_idx, hit_row = min(hit_window, key=lambda ir: np.linalg.norm(ir[1][1] - ir[1][3]))
        # 取 hit 后 1-2 帧的 ball vel 作初始 post-hit (避免取 hit 瞬间的过渡值)
        post_idx = min(hit_idx + 2, len(log) - 1)
        ball_post_pos = log[post_idx][3]
        ball_post_vel = log[post_idx][4]

        # face_n at hit
        R = quat_to_R(hit_row[2])
        face_n = R @ FACE_BODY

        # 解析弹道
        z_at_net, x_bounce, clears, valid = analyze_trajectory(
            ball_post_pos[0], ball_post_pos[2],
            ball_post_vel[0], ball_post_vel[2],
        )

        return {
            "hit_t": hit_row[0],
            "paddle_pos": hit_row[1],
            "ball_post_pos": ball_post_pos,
            "ball_post_vel": ball_post_vel,
            "face_n": face_n,
            "z_at_net": z_at_net,
            "x_bounce": x_bounce,
            "clears": clears,
            "valid": valid,
        }

    print(f"\n{'variant':<55} {'ball post-v (xyz)':<23} {'face_n':<22} "
          f"{'z@net':>6} {'x_bnc':>6} {'flags'}")
    print("-" * 130)

    probes = [
        ("V66 baseline", V66),
        # 加大 yb_5 hit-window (face 更朝上 + 更朝 -X)
        ("V66 + yb_5 hw +0.10",  apply_delta(V66, {4: +0.10})),
        ("V66 + yb_5 hw +0.20",  apply_delta(V66, {4: +0.20})),
        ("V66 + yb_5 hw +0.30",  apply_delta(V66, {4: +0.30})),
        # yb_5 PIN 单独调 (击球瞬间 face 朝向)
        ("V66 + yb_5 PIN -0.45", edit_kf(V66, 0.475, 4, -0.45)),
        ("V66 + yb_5 PIN -0.55", edit_kf(V66, 0.475, 4, -0.55)),
        ("V66 + yb_5 PIN -0.55 + snap -0.30",
            edits(V66, [(0.475, 4, -0.55), (0.550, 4, -0.30)])),
        ("V66 + yb_5 PIN -0.45 + snap -0.20",
            edits(V66, [(0.475, 4, -0.45), (0.550, 4, -0.20)])),
        ("V66 + yb_5 PIN -0.45 + snap -0.10",
            edits(V66, [(0.475, 4, -0.45), (0.550, 4, -0.10)])),
        # 加 yb_1 hw 抬肩 (paddle 更高 -> z_h 大)
        ("V66 + yb_1 hw +0.10",  apply_delta(V66, {0: +0.10})),
        ("V66 + yb_1 hw +0.20",  apply_delta(V66, {0: +0.20})),
        # yb_1 抬 + yb_5 less PIN (face 不那么 -X 但更 +Z)
        ("V66 + yb_1 hw +0.10 + yb_5 PIN -0.45",
            edits(apply_delta(V66, {0: +0.10}), [(0.475, 4, -0.45)])),
        ("V66 + yb_1 hw +0.20 + yb_5 PIN -0.45",
            edits(apply_delta(V66, {0: +0.20}), [(0.475, 4, -0.45)])),
        # yb_4 收回 (前臂少伸, paddle 更靠近身体, hit 时 z 更高)
        ("V66 + yb_4 hw +0.10",  apply_delta(V66, {3: +0.10})),
        ("V66 + yb_4 hw +0.20",  apply_delta(V66, {3: +0.20})),
        # 综合: yb_1+ yb_4+ yb_5 PIN-
        ("V66 + yb_1 hw +0.15 + yb_4 hw +0.15 + yb_5 PIN -0.50",
            edits(apply_delta(V66, {0: +0.15, 3: +0.15}), [(0.475, 4, -0.50)])),
    ]

    rows = []
    for label, keys in probes:
        r = run(keys)
        v = r["ball_post_vel"]
        fn = r["face_n"]
        zn_s = f"{r['z_at_net']:>+5.2f}" if r['z_at_net'] is not None else "  n/a"
        xb_s = f"{r['x_bounce']:>+5.2f}" if r['x_bounce'] is not None else "  n/a"
        flag = ("CLEARS " if r['clears'] else "       ") + ("VALID" if r['valid'] else "OWN  ")
        print(f"{label:<55} ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})  "
              f"({fn[0]:+.2f},{fn[1]:+.2f},{fn[2]:+.2f}) "
              f"{zn_s} {xb_s} {flag}")
        rows.append((label, r))

    print("\n=== Variants with both CLEARS net AND VALID first bounce (x<0) ===")
    valid_rows = [r for r in rows if r[1]["clears"] and r[1]["valid"]]
    if not valid_rows:
        print("  (none — need bigger changes)")
    for label, r in valid_rows:
        v = r["ball_post_vel"]
        print(f"  z@net={r['z_at_net']:+.2f}  x_bounce={r['x_bounce']:+.2f}  "
              f"vx={v[0]:+.2f} vz={v[2]:+.2f}  fn={r['face_n']}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
