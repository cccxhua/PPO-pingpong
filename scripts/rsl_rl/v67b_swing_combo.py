"""V67b: V67 找到 yb_5 PIN 调整能给 face 带 -X 分量, x_land 到 -0.53.
现在叠加 yb_1/yb_4 swing 加速度看能否打更远 (x_land < -1.0).

Best from V67: yb_5 PIN -0.55 + snap -0.50: bx=-1.79, x_land=-0.50, gap=3.1 ✓
              yb_5 PIN -0.65: bx=-1.92, x_land=-0.53, gap=3.6 ✓

V67b base = V65 + yb_5 PIN -0.55 + snap -0.50 (好 zn 余量, gap OK).
叠加 swing 增强:
  - yb_1 hw +0.05/+0.10 (paddle 整体抬, swing 强)
  - yb_1 windup -0.10 (大 swing 幅度)
  - yb_4 hw -0.10 (前臂略伸更直)
  - yb_4 windup +0.20 (前臂 swing 幅度大)
  - 也试 yb_5 PIN -0.65 base
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


V65 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +1.027, -0.315, -1.045,  +1.000]),
    (0.400, [+1.187, +0.103, -1.979, +0.657, -0.315, -1.245,  +1.000]),
    (0.475, [+1.287, +0.103, -1.979, +0.607, -0.345, -1.020,  +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.557, -0.165, -0.495,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]

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


# Bases
B_PIN_55_SNAP_50 = edits(V65, [(0.475, 4, -0.55), (0.550, 4, -0.50)])  # mid -X face
B_PIN_65 = edit_kf(V65, 0.475, 4, -0.65)  # most -X face


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

    def run(keys, T_max=2.0):
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
        for step in range(int(T_max / sim_dt)):
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
            v = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy().copy()
            q = robot.data.body_quat_w[0, paddle_idx].cpu().numpy().copy()
            bp = ball.data.root_pos_w[0].cpu().numpy().copy()
            bv = ball.data.root_lin_vel_w[0].cpu().numpy().copy()
            log.append((t, p, v, q, bp, bv))
        hit_window = [r for r in log if 0.40 < r[0] < 0.60]
        hit = min(hit_window, key=lambda r: np.linalg.norm(r[1] - r[4]))
        face_n = quat_to_R(hit[3]) @ FACE_BODY
        post = [r for r in log if r[0] > hit[0] + 0.01]
        bx = min(r[5][0] for r in post)
        bz = max(r[5][2] for r in post)
        zn = None
        x_table = None
        for i in range(1, len(post)):
            xp, xn = post[i-1][4][0], post[i][4][0]
            if xp > 0 and xn <= 0 and zn is None:
                f = xp / (xp - xn)
                zn = post[i-1][4][2] + f * (post[i][4][2] - post[i-1][4][2])
            zp_, zn_ = post[i-1][4][2], post[i][4][2]
            if zp_ > 0.79 and zn_ <= 0.79 and post[i-1][4][0] < 0:
                f = (zp_ - 0.79) / (zp_ - zn_)
                x_table = post[i-1][4][0] + f * (post[i][4][0] - post[i-1][4][0])
                break
        return hit[1], hit[2], face_n, bx, bz, zn, x_table

    print(f"\n{'variant':<60} {'p_v':<22} {'face_n':<22}{'gap':>5} {'bx':>6} {'bz':>6} {'zn':>6} {'x_land':>7}")

    probes = [
        # baselines
        ("V65 baseline (no PIN tweak)", V65),
        ("base: PIN-0.55 snap-0.50 (V67 best balance)", B_PIN_55_SNAP_50),
        ("base: PIN-0.65 (V67 max bx)", B_PIN_65),
        # On B_PIN_55_SNAP_50: add swing power
        ("PIN-0.55+sn-0.50 + yb_1 hw +0.05",
            apply_delta(B_PIN_55_SNAP_50, {0: +0.05})),
        ("PIN-0.55+sn-0.50 + yb_1 hw +0.10",
            apply_delta(B_PIN_55_SNAP_50, {0: +0.10})),
        ("PIN-0.55+sn-0.50 + yb_4 hw -0.10",
            apply_delta(B_PIN_55_SNAP_50, {3: -0.10})),
        ("PIN-0.55+sn-0.50 + yb_4 hw -0.15",
            apply_delta(B_PIN_55_SNAP_50, {3: -0.15})),
        ("PIN-0.55+sn-0.50 + yb_1 windup -0.10",
            edit_kf(B_PIN_55_SNAP_50, 0.400, 0, +1.087)),
        ("PIN-0.55+sn-0.50 + yb_4 windup +0.20",
            edit_kf(B_PIN_55_SNAP_50, 0.400, 3, +0.857)),
        ("PIN-0.55+sn-0.50 + yb_4 windup +0.40",
            edit_kf(B_PIN_55_SNAP_50, 0.400, 3, +1.057)),
        # On B_PIN_65: similar
        ("PIN-0.65 + yb_1 hw +0.05",
            apply_delta(B_PIN_65, {0: +0.05})),
        ("PIN-0.65 + yb_4 hw -0.10",
            apply_delta(B_PIN_65, {3: -0.10})),
        ("PIN-0.65 + yb_4 hw -0.15",
            apply_delta(B_PIN_65, {3: -0.15})),
        ("PIN-0.65 + yb_4 windup +0.20",
            edit_kf(B_PIN_65, 0.400, 3, +0.857)),
        # Combo
        ("PIN-0.55+sn-0.50 + yb_4 hw -0.10 + yb_1 hw +0.05",
            apply_delta(B_PIN_55_SNAP_50, {0: +0.05, 3: -0.10})),
        ("PIN-0.65 + yb_4 hw -0.10 + yb_1 hw +0.05",
            apply_delta(B_PIN_65, {0: +0.05, 3: -0.10})),
        # Even more aggressive PIN
        ("yb_5 PIN -0.75", edit_kf(V65, 0.475, 4, -0.75)),
        ("yb_5 PIN -0.85", edit_kf(V65, 0.475, 4, -0.85)),
    ]

    rows = []
    for label, keys in probes:
        p, v, face_n, bx, bz, zn, xt = run(keys)
        gap = np.linalg.norm(p - np.array([1.27, 0, 1.05])) * 0  # placeholder, use actual
        # recompute gap properly: just use paddle pos vs ball pos at hit, but we don't have ball pos
        # actually run returns hit[1] which is paddle pos. need to get ball pos.
        # Just use bx/bz/zn for ranking; gap not critical here
        zn_s = f"{zn:.3f}" if zn is not None else "miss"
        xt_s = f"{xt:+.2f}" if xt is not None else "n/a"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        print(f"{label:<60} ({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})  "
              f"({face_n[0]:+.2f},{face_n[1]:+.2f},{face_n[2]:+.2f}) "
              f"   - {bx:>+6.2f} {bz:>+6.2f} {zn_s:>6} {xt_s:>7} {clears}")
        rows.append((label, v, face_n, bx, bz, zn, xt))

    print(f"\n=== Sorted by x_land (most -X = farthest) with zn > 0.94 ===")
    valid = [r for r in rows if r[5] is not None and r[5] > 0.94 and r[6] is not None]
    valid.sort(key=lambda r: r[6])
    for label, v, fn, bx, bz, zn, xt in valid[:8]:
        print(f"  x_land={xt:+.2f}  bx={bx:+.2f}  bz={bz:+.2f}  zn={zn:.3f}  vx={v[0]:+.2f}  fz={fn[2]:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
