"""V59 yb_2/yb_3 sweep: find lift on shoulder joints that puts dynamic paddle
Y on ball Y (≈0). Face plane must intersect ball trajectory for clean hit.
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


V58_BASE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def apply_lift(keys, j_idx, lift, t_lo=0.39, t_hi=0.56):
    out = []
    for t, vals in keys:
        v = list(vals)
        if t_lo < t < t_hi:
            v[j_idx] += lift
        out.append((t, v))
    return out


def quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)


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
    env_origin = scene.env_origins[0]

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
        robot.set_joint_position_target(full, env_ids=ids)
        for _ in range(150):
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
        for step in range(int(0.60 / sim_dt)):
            t = step * sim_dt
            target = spline(min(t, 1.0))
            full_target = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                full_target[0, jid] = float(target[k])
            robot.set_joint_position_target(full_target, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)
            p = robot.data.body_pos_w[0, paddle_idx]
            q_ = robot.data.body_quat_w[0, paddle_idx]
            b = ball.data.root_pos_w[0]
            R = quat_to_R(q_)
            b_local = R.T @ (b - p)
            log.append((t, p.cpu().numpy(), b.cpu().numpy(), b_local.cpu().numpy()))
        return log

    print(f"\n{'variant':<32}  ball_local samples (mm) at t=0.45,0.46,...,0.50  (* = in face)")

    def apply_lift_wide(keys, j_idx, lift, t_lo=0.29, t_hi=0.56):
        out = []
        for t, vals in keys:
            v = list(vals)
            if t_lo < t < t_hi:
                v[j_idx] += lift
            out.append((t, v))
        return out

    base_y4y5 = apply_lift(apply_lift(V58_BASE, 4, -0.30), 3, -0.20)

    def make(yb1=0, yb2=0, yb3=0, yb4=-0.20, yb5=-0.30, yb6=0, yb7=0, t_lo=0.39, early=()):
        keys = V58_BASE
        # narrow window (t_lo=0.39) for "snap" joints
        if yb3: keys = apply_lift(keys, 2, yb3)
        if yb4: keys = apply_lift(keys, 3, yb4)
        if yb5: keys = apply_lift(keys, 4, yb5)
        if yb6: keys = apply_lift(keys, 5, yb6)
        if yb7: keys = apply_lift(keys, 6, yb7)
        # wide window (t_lo=0.29) for "early" joints (PD-saturated)
        for j_idx, lift in early:
            keys = apply_lift_wide(keys, j_idx, lift)
        return keys

    variants = [
        # candidates very close to face hit
        ("yb_5 -0.40, yb_3 +0.12, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.12, yb5=-0.40, yb4=-0.05, early=[(1, +0.05)])),
        ("yb_5 -0.42, yb_3 +0.12, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.12, yb5=-0.42, yb4=-0.05, early=[(1, +0.05)])),
        ("yb_5 -0.43, yb_3 +0.12, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.12, yb5=-0.43, yb4=-0.05, early=[(1, +0.05)])),
        ("yb_5 -0.42, yb_3 +0.13, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.13, yb5=-0.42, yb4=-0.05, early=[(1, +0.05)])),
        ("yb_5 -0.42, yb_3 +0.12, yb_4 -0.07, yb_2 +0.05",  make(yb3=+0.12, yb5=-0.42, yb4=-0.07, early=[(1, +0.05)])),
        ("yb_5 -0.42, yb_3 +0.12, yb_4 -0.05, yb_2 +0.06",  make(yb3=+0.12, yb5=-0.42, yb4=-0.05, early=[(1, +0.06)])),
        ("yb_5 -0.42, yb_3 +0.12, yb_4 -0.05, yb_2 +0.07",  make(yb3=+0.12, yb5=-0.42, yb4=-0.05, early=[(1, +0.07)])),
        ("yb_5 -0.42, yb_3 +0.13, yb_4 -0.07, yb_2 +0.06",  make(yb3=+0.13, yb5=-0.42, yb4=-0.07, early=[(1, +0.06)])),
        ("yb_5 -0.43, yb_3 +0.13, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.13, yb5=-0.43, yb4=-0.05, early=[(1, +0.05)])),
        ("yb_5 -0.45, yb_3 +0.12, yb_4 -0.05, yb_2 +0.05",  make(yb3=+0.12, yb5=-0.45, yb4=-0.05, early=[(1, +0.05)])),
    ]

    for label, keys in variants:
        log = run(keys)
        # search the full trajectory for any moment ball is INSIDE face region
        face_moments = []
        for (t, p, b, b_local) in log:
            if not (0.30 < t < 0.60):
                continue
            bx, by, bz = b_local * 1000
            in_x = abs(bx) < 75
            in_y = abs(by) < 15
            in_z = -47 <= bz <= 58
            if in_x and in_y and in_z:
                face_moments.append((t, bx, by, bz))

        # also report ball local at fixed time samples
        samples = [0.450, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485, 0.490, 0.500]
        rows = []
        for ts in samples:
            i = min(range(len(log)), key=lambda i: abs(log[i][0] - ts))
            _, _, _, bl = log[i]
            bx, by, bz = bl * 1000
            in_face = (abs(bx) < 75) and (abs(by) < 15) and (-47 <= bz <= 58)
            mark = "*" if in_face else " "
            rows.append(f"{ts:.3f}{mark}({bx:+4.0f},{by:+4.0f},{bz:+4.0f})")

        if face_moments:
            n = len(face_moments)
            t0 = face_moments[0][0]
            print(f"{label:<32}  ✓✓✓ FACE HIT [{n} pts from t={t0:.3f}]  " + "  ".join(rows))
        else:
            print(f"{label:<32}  " + "  ".join(rows))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
