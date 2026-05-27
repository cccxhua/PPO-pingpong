"""LEFT v2: Design LEFT keyframes based on MIDDLE v74 with yb_2 shift to reach +Y ball.

Analysis from probe:
- Ball at t=0.55: X≈1.11, Y≈+0.07, Z≈1.08
- MIDDLE v74 paddle at hit: X≈1.28, Y≈-0.02 (from compare_middle_right data)
- Need: paddle Y shift of +0.09 to reach ball Y=+0.07

MIDDLE yb_2 at PIN = +0.103. Shifting to +0.20~0.25 should push paddle +Y.
But yb_2 upper limit is +0.314, so max shift ≈ +0.21.

Also: ball X only reaches 1.11 (not 1.28 like MIDDLE ball at center).
Wait — MIDDLE ball also has damping=0.5. Let me check what X the MIDDLE ball reaches.
If MIDDLE ball also reaches ~1.1, then MIDDLE paddle at X≈1.28 is already ahead of ball X,
and the paddle sweeps THROUGH the ball. Same should work for LEFT.

Strategy: Start from MIDDLE v74, shift yb_2 by +0.10 (conservative, stays within limit).
This should move paddle from Y≈-0.02 to Y≈+0.05~+0.08 (closer to ball Y=+0.07).

Also test: yb_2 +0.15, +0.20 shifts to find optimal.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_left_v2.py --task X1-TableTennis
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

BALL_POS = np.array([-0.35, +0.03, 1.3])
BALL_VEL = np.array([3.5, +0.10, 0.5])

DURATION = 1.0
HIT_PHASE = 0.475
BALL_ARRIVE_TIME_EST = 0.55
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

# MIDDLE v74 baseline
MIDDLE_BASE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
]

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])


def make_shifted(base, yb2_shift):
    """Shift yb_2 (index 1) by constant across all keyframes."""
    result = []
    for t, angles in base:
        new_angles = angles.copy()
        new_angles[1] += yb2_shift
        result.append((t, new_angles))
    return result


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
        mn, mx = y[:, i].min(), y[:, i].max()
        if mn < lo - 0.01 or mx > hi + 0.01:
            viol.append(f"yb{i+1}=[{mn:+.3f},{mx:+.3f}] lim=[{lo:.3f},{hi:.3f}]")
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

    def run_motion(name, keyframes):
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
        pv_hit, pp_hit, bp_hit = np.zeros(3), np.zeros(3), np.zeros(3)
        bv_post = np.zeros(3)
        face_hit = np.zeros(3)

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

            if gap < min_gap:
                min_gap, min_t = gap, t
                pp_hit, bp_hit, pv_hit = p.copy(), bp.copy(), pv.copy()
                pq = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()
                rot = Rotation.from_quat([pq[1], pq[2], pq[3], pq[0]])
                face_hit = rot.apply([1, 0, 0])

            if not hit and bv[0] < -0.5 and t > 0.3:
                hit = True
                bv_post = bv.copy()

        if not hit:
            bv_post = bv.copy()
        bp_post = bp_hit
        zn, xb, clr, val = analyze_trajectory(bp_post[0], bp_post[2], bv_post[0], bv_post[2])

        return {
            "name": name, "min_gap": min_gap, "min_t": min_t, "hit": hit,
            "paddle_pos": pp_hit, "ball_pos": bp_hit,
            "paddle_vel": pv_hit, "ball_vel_post": bv_post,
            "face_normal": face_hit,
            "z_at_net": zn, "x_bounce": xb, "clears": clr and val,
        }

    print(f"\n{'='*100}")
    print(f"  LEFT v2: MIDDLE v74 + yb_2 shift sweep")
    print(f"{'='*100}")
    print(f"  Ball: pos={BALL_POS}, vel={BALL_VEL}")
    print(f"  Ball @ t=0.55: X≈1.11, Y≈+0.07, Z≈1.08")
    print(f"\n{'name':<20} {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'pp_x':>6} {'pp_y':>6} {'pp_z':>6} | "
          f"{'pv_x':>6} {'pv_z':>6} | "
          f"{'fn_x':>6} {'fn_z':>6} | "
          f"{'bvx':>6} {'bvz':>6} {'CLR':>3} | lim")
    print("-" * 120)

    shifts = [0.00, +0.02, +0.05, +0.08, +0.10, +0.12, +0.15]
    for shift in shifts:
        keys = make_shifted(MIDDLE_BASE, shift)
        name = f"yb2+{shift:.2f}"
        lim = check_limits(keys)
        lim_str = ",".join(lim) if lim else "ok"
        r = run_motion(name, keys)
        pp, pv, fn = r["paddle_pos"], r["paddle_vel"], r["face_normal"]
        bv = r["ball_vel_post"]
        hit_s = "Y" if r["hit"] else "N"
        clr_s = "Y" if r["clears"] else "N"
        print(f"{name:<20} {r['min_gap']:>5.3f} {r['min_t']:>5.3f} {hit_s:>3} | "
              f"{pp[0]:>+6.3f} {pp[1]:>+6.3f} {pp[2]:>+6.3f} | "
              f"{pv[0]:>+6.2f} {pv[2]:>+6.2f} | "
              f"{fn[0]:>+6.3f} {fn[2]:>+6.3f} | "
              f"{bv[0]:>+6.2f} {bv[2]:>+6.2f} {clr_s:>3} | {lim_str}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
