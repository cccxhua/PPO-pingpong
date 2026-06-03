"""V67: V65 base 上 sweep yb_5 snap (其他不动) — 找 bx-bz Pareto.

V65: yb_5 mid=-0.315, windup=-0.315, PIN=-0.345, snap=-0.165
  bx=-1.37, bz=+3.18, zn=1.298, gap=2.9 — face 太竖, bx 不够
v65b 在 V64 base 上测 yb_5 snap -0.65 给 bx=-2.45 但 bz 较小
要 bx 大同时 bz 大, snap 应在 -0.30 ~ -0.50 范围 (PIN 时上仰, snap 后朝 -X 转)

也测 PIN 单独调 (面在击球瞬间朝向更朝前+上).
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


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


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
        return hit, face_n, bx, bz, zn, x_table

    print(f"\n{'variant':<60} {'face_n':<22} {'gap':>5} {'bx':>6} {'bz':>6} {'zn':>6} {'x_land':>7}")

    probes = [
        ("V65 baseline", V65),
        # Sweep yb_5 snap @0.550 only (mid/windup/PIN unchanged at -0.315/-0.315/-0.345)
        ("yb_5 snap -0.30 (was -0.165)", edit_kf(V65, 0.550, 4, -0.30)),
        ("yb_5 snap -0.40", edit_kf(V65, 0.550, 4, -0.40)),
        ("yb_5 snap -0.50", edit_kf(V65, 0.550, 4, -0.50)),
        ("yb_5 snap -0.60", edit_kf(V65, 0.550, 4, -0.60)),
        ("yb_5 snap -0.70", edit_kf(V65, 0.550, 4, -0.70)),
        # Adjust PIN — control face at hit moment (PIN is closest to actual hit time)
        ("yb_5 PIN -0.45 (was -0.345)", edit_kf(V65, 0.475, 4, -0.45)),
        ("yb_5 PIN -0.55", edit_kf(V65, 0.475, 4, -0.55)),
        ("yb_5 PIN -0.65", edit_kf(V65, 0.475, 4, -0.65)),
        # PIN+snap combo: PIN slightly more -X, snap also -X
        ("yb_5 PIN -0.45 + snap -0.40",
            edits(V65, [(0.475, 4, -0.45), (0.550, 4, -0.40)])),
        ("yb_5 PIN -0.50 + snap -0.50",
            edits(V65, [(0.475, 4, -0.50), (0.550, 4, -0.50)])),
        # Reduce hit-window offset: from +0.20 down
        ("yb_5 hw +0.10 (less aggressive face up)",
            [(t, [v[0], v[1], v[2], v[3], v[4]+0.10 if 0.29<t<0.56 else v[4], v[5], v[6]])
             for t, v in [(t, list(vals)) for t, vals in
                          [(0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
                           (0.300, [+1.127, +0.198, -1.904, +1.027, -0.515, -1.045, +1.000]),
                           (0.400, [+1.187, +0.103, -1.979, +0.657, -0.515, -1.245, +1.000]),
                           (0.475, [+1.287, +0.103, -1.979, +0.607, -0.545, -1.020, +1.000]),
                           (0.550, [+1.437, +0.103, -1.979, +0.557, -0.365, -0.495, +1.000]),
                           (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
                           (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
                           (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000])]]]),
        # PIN/snap mix to get face_n with more -X
        ("V65 + yb_5 PIN -0.45 + snap -0.30 (compromise)",
            edits(V65, [(0.475, 4, -0.45), (0.550, 4, -0.30)])),
        ("V65 + yb_5 PIN -0.50 + snap -0.40",
            edits(V65, [(0.475, 4, -0.50), (0.550, 4, -0.40)])),
        ("V65 + yb_5 PIN -0.55 + snap -0.50",
            edits(V65, [(0.475, 4, -0.55), (0.550, 4, -0.50)])),
    ]

    rows = []
    for label, keys in probes:
        (t_, p, v, q, bp, bv), face_n, bx, bz, zn, xt = run(keys)
        gap = np.linalg.norm(p - bp) * 100
        zn_s = f"{zn:.3f}" if zn is not None else "miss"
        xt_s = f"{xt:+.2f}" if xt is not None else "n/a"
        clears = "✓" if (zn is not None and zn > 0.94) else " "
        print(f"{label:<60} ({face_n[0]:+.2f},{face_n[1]:+.2f},{face_n[2]:+.2f})  "
              f"{gap:>4.1f} {bx:>+6.2f} {bz:>+6.2f} {zn_s:>6} {xt_s:>7} {clears}")
        rows.append((label, face_n, bx, bz, zn, gap, xt))

    print(f"\n=== Top 8 by combined score (bx weight + bz weight) with zn > 0.94, gap < 5 ===")
    valid = [r for r in rows if r[4] is not None and r[4] > 0.94 and r[5] < 5.0]
    # score: more negative bx is better, more positive bz is better, penalize lower zn
    valid.sort(key=lambda r: r[2] - 0.3 * r[3])  # bx + small bz preference
    for label, fn, bx, bz, zn, gap, xt in valid[:8]:
        zn_s = f"{zn:.3f}"
        xt_s = f"{xt:+.2f}" if xt else "n/a"
        print(f"  bx={bx:+.2f}  bz={bz:+.2f}  zn={zn_s}  x_land={xt_s}  gap={gap:.1f}  fz={fn[2]:+.2f}  {label}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
