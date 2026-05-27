"""Probe LEFT forehand motion: diagnose paddle position/velocity at ball arrival.

LEFT motion (motion_id=1) is triggered when predicted_y > +0.05 at robot_x.
Ball: pos=(-0.35, +0.03, 1.3), vel=(3.5, +0.10, 0.5)
predicted_y @ robot_x=1.5: 0.03 + 0.10*(1.85/3.5) = +0.083 → triggers LEFT.

Key questions:
  1. Where does the paddle end up during the "hit" window?
  2. Where does the ball actually arrive (with linear_damping=0.5)?
  3. What's the gap between paddle and ball?
  4. What's the paddle velocity and face normal at closest approach?

Also tests MIDDLE keyframes with yb_2 shift to LEFT side for comparison.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_left_motion.py --task X1-TableTennis
"""

import argparse, sys
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
from scipy.spatial.transform import Rotation
import isaaclab_tasks, unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# Ball params for LEFT (positive vy = toward robot's +Y / left side)
BALL_POS = np.array([-0.35, +0.03, 1.3])
BALL_VEL = np.array([3.5, +0.10, 0.5])

DURATION = 1.0
HIT_PHASE = 0.475
BALL_ARRIVE_TIME_EST = 0.55
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

# Current LEFT keyframes from create_forehand.py
LEFT_CURRENT = [
    (0.00, [1.56, 0.00, -1.80, 1.10, -0.82, 0.00, 0.1]),
    (0.35, [1.56, -0.25, -1.80, 1.10, -0.82, 0.00, 0.1]),
    (0.50, [1.56, -0.45, -1.80, 1.10, -0.82, 0.00, 0.1]),
    (0.60, [1.56, -0.45, -1.80, 1.10, -0.82, 0.00, 0.1]),
    (1.00, [1.56, 0.00, -1.80, 1.10, -0.82, 0.00, 0.1]),
]

# MIDDLE v74 for comparison (with yb_2 shift +0.15 to reach +Y ball)
MIDDLE_SHIFTED = [
    (0.000, [+1.000, +0.450, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.348, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.253, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.253, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.253, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.250, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.450, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.450, -2.000, +1.400, +0.000, -1.000, +1.000]),
]

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def analyze_trajectory(x0, z0, vx, vz):
    if vx >= 0:
        return None, None, False, False
    t_net = x0 / (-vx)
    z_at_net = z0 + vz * t_net - 0.5 * G * t_net * t_net
    a, b, c = 0.5 * G, -vz, TABLE_Z - z0
    disc = b*b - 4*a*c
    if disc < 0:
        return z_at_net, None, False, False
    t_bounce = (-b + np.sqrt(disc)) / (2*a)
    x_bounce = x0 + vx * t_bounce
    clears = z_at_net > NET_Z and t_net < t_bounce
    valid = x_bounce < NET_X
    return z_at_net, x_bounce, clears, valid


def check_limits(keys):
    times = np.array([k[0] for k in keys])
    angs = np.array([k[1] for k in keys], dtype=np.float64)
    cs = CubicSpline(times, angs, bc_type="clamped")
    t_dense = np.linspace(0, times[-1], 1001)
    y = cs(t_dense)
    viol = []
    for i in range(7):
        lo, hi = LIMITS[i]
        if y[:, i].min() < lo - 0.01 or y[:, i].max() > hi + 0.01:
            viol.append(f"yb{i+1}=[{y[:, i].min():+.3f},{y[:, i].max():+.3f}] limit=[{lo:.3f},{hi:.3f}]")
    return viol


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
    yb_joint_ids = [robot.find_joints(f"joint_yb_{i}")[0][0] for i in range(1, 8)]
    env_origin = scene.env_origins[0].cpu().numpy()
    sim_dt = float(env.unwrapped.sim.get_physics_dt())

    print(f"\n{'='*100}")
    print(f"  LEFT MOTION DIAGNOSTIC")
    print(f"{'='*100}")
    print(f"  Ball: pos={BALL_POS}, vel={BALL_VEL}")
    predicted_y = BALL_POS[1] + BALL_VEL[1] * (1.85 / BALL_VEL[0])
    print(f"  predicted_y @ robot_x=1.5: {predicted_y:+.3f} (threshold=+0.05 → motion_id=1 'left')")
    print(f"  HIT_PHASE={HIT_PHASE}, BALL_ARRIVE_TIME_EST={BALL_ARRIVE_TIME_EST}")

    def run_motion(name, keyframes):
        lim = check_limits(keyframes)
        if lim:
            print(f"\n  [{name}] JOINT LIMIT VIOLATION: {lim}")

        times = np.array([k[0] for k in keyframes])
        angs = np.array([k[1] for k in keyframes], dtype=np.float64)
        spline = CubicSpline(times, angs, bc_type="clamped")

        initial_phase = (HIT_PHASE - BALL_ARRIVE_TIME_EST / DURATION) % 1.0
        q0 = spline(initial_phase)
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q0[k])
        v0 = torch.zeros_like(full)
        ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        for _ in range(200):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

        ball_state = ball.data.default_root_state.clone()
        ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device) + scene.env_origins[0]
        ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
        ball_state[0, 10:13] = torch.zeros(3, device=device)
        ball.write_root_state_to_sim(ball_state, env_ids=ids)
        scene.write_data_to_sim()

        n_steps = int(1.2 / sim_dt)
        min_gap, min_t = 1e9, -1
        hit = False
        pv_hit = np.zeros(3)
        pp_hit = np.zeros(3)
        bp_hit = np.zeros(3)
        bv_post = np.zeros(3)
        face_hit = np.zeros(3)

        paddle_traj = []

        for step in range(n_steps):
            t = step * sim_dt
            phase = (initial_phase + t / DURATION) % 1.0
            target = spline(phase)
            ft = robot.data.default_joint_pos[0:1].clone()
            for k, jid in enumerate(yb_joint_ids):
                ft[0, jid] = float(target[k])
            robot.set_joint_position_target(ft, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

            p = robot.data.body_pos_w[0, paddle_idx].cpu().numpy() - env_origin
            pv = robot.data.body_lin_vel_w[0, paddle_idx].cpu().numpy()
            bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
            bv = ball.data.root_lin_vel_w[0].cpu().numpy()
            gap = float(np.linalg.norm(p - bp))

            if 0.30 < phase < 0.65:
                paddle_traj.append(p.copy())

            if gap < min_gap:
                min_gap, min_t = gap, t
                pp_hit = p.copy()
                bp_hit = bp.copy()
                pv_hit = pv.copy()
                pq = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()
                rot = Rotation.from_quat([pq[1], pq[2], pq[3], pq[0]])
                face_hit = rot.apply([1, 0, 0])

            if not hit and bv[0] < -0.5 and t > 0.3:
                hit = True
                bv_post = bv.copy()

        if not hit:
            bv_post = bv.copy()

        paddle_traj = np.array(paddle_traj) if paddle_traj else np.zeros((1, 3))
        zn, xb, clr, val = analyze_trajectory(pp_hit[0], pp_hit[2], bv_post[0], bv_post[2])

        return {
            "name": name, "min_gap": min_gap, "min_t": min_t, "hit": hit,
            "paddle_pos": pp_hit, "ball_pos": bp_hit,
            "paddle_vel": pv_hit, "ball_vel_post": bv_post,
            "face_normal": face_hit,
            "paddle_x_range": (paddle_traj[:, 0].min(), paddle_traj[:, 0].max()),
            "paddle_y_range": (paddle_traj[:, 1].min(), paddle_traj[:, 1].max()),
            "paddle_z_range": (paddle_traj[:, 2].min(), paddle_traj[:, 2].max()),
            "z_at_net": zn, "x_bounce": xb, "clears": clr and val,
        }

    variants = [
        ("LEFT_current", LEFT_CURRENT),
        ("MIDDLE_yb2+0.15", MIDDLE_SHIFTED),
    ]

    for name, keys in variants:
        r = run_motion(name, keys)
        pp, bp = r["paddle_pos"], r["ball_pos"]
        pv, bv = r["paddle_vel"], r["ball_vel_post"]
        fn = r["face_normal"]
        print(f"\n{'─'*80}")
        print(f"  [{r['name']}]")
        print(f"  Hit: {'YES' if r['hit'] else 'NO'}, min_gap={r['min_gap']:.3f}m @ t={r['min_t']:.3f}s")
        print(f"  Paddle pos: ({pp[0]:+.3f}, {pp[1]:+.3f}, {pp[2]:+.3f})")
        print(f"  Ball pos:   ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")
        print(f"  Gap vector: (Δx={pp[0]-bp[0]:+.3f}, Δy={pp[1]-bp[1]:+.3f}, Δz={pp[2]-bp[2]:+.3f})")
        print(f"  Paddle vel: (vx={pv[0]:+.2f}, vy={pv[1]:+.2f}, vz={pv[2]:+.2f}), |v|={np.linalg.norm(pv):.2f}")
        print(f"  Face normal: ({fn[0]:+.3f}, {fn[1]:+.3f}, {fn[2]:+.3f})")
        if r['hit']:
            print(f"  Ball post-hit: (vx={bv[0]:+.2f}, vy={bv[1]:+.2f}, vz={bv[2]:+.2f}), |v|={np.linalg.norm(bv):.2f}")
            zns = f"{r['z_at_net']:+.2f}" if r['z_at_net'] is not None else " - "
            xbs = f"{r['x_bounce']:+.2f}" if r['x_bounce'] is not None else " - "
            print(f"  Trajectory: z@net={zns}, x_bounce={xbs}, clears={'Y' if r['clears'] else 'N'}")
        xr, yr, zr = r["paddle_x_range"], r["paddle_y_range"], r["paddle_z_range"]
        print(f"  Paddle sweep (phase 0.30-0.65):")
        print(f"    X: [{xr[0]:+.3f}, {xr[1]:+.3f}] (span={xr[1]-xr[0]:.3f}m)")
        print(f"    Y: [{yr[0]:+.3f}, {yr[1]:+.3f}] (span={yr[1]-yr[0]:.3f}m)")
        print(f"    Z: [{zr[0]:+.3f}, {zr[1]:+.3f}] (span={zr[1]-zr[0]:.3f}m)")

    # Ball trajectory without paddle interference
    print(f"\n{'─'*80}")
    print(f"  Ball free-flight analysis (with linear_damping=0.5):")
    ball_state = ball.data.default_root_state.clone()
    ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device) + scene.env_origins[0]
    ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
    ball_state[0, 10:13] = torch.zeros(3, device=device)
    ids = torch.tensor([0], device=device)

    full = robot.data.default_joint_pos[0:1].clone()
    full[0, yb_joint_ids[0]] = 0.0
    full[0, yb_joint_ids[1]] = 0.0
    full[0, yb_joint_ids[2]] = -2.0
    full[0, yb_joint_ids[3]] = 1.5
    robot.write_joint_state_to_sim(full, torch.zeros_like(full), env_ids=ids)
    for _ in range(100):
        robot.set_joint_position_target(full, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)

    ball.write_root_state_to_sim(ball_state, env_ids=ids)
    scene.write_data_to_sim()

    print(f"  {'t':>5} {'bx':>7} {'by':>7} {'bz':>7} {'bvx':>7} {'bvy':>7} {'bvz':>7}")
    for step in range(int(0.8 / sim_dt)):
        t = step * sim_dt
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)
        bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
        bv = ball.data.root_lin_vel_w[0].cpu().numpy()
        if step % 10 == 0 or abs(t - 0.55) < sim_dt:
            print(f"  {t:>5.3f} {bp[0]:>+7.3f} {bp[1]:>+7.3f} {bp[2]:>+7.3f} "
                  f"{bv[0]:>+7.2f} {bv[1]:>+7.2f} {bv[2]:>+7.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
