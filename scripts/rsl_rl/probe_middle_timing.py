"""Diagnose MIDDLE timing issue: test with both ball_arrive_time_est values.

Compares MIDDLE v74 with:
  A) ball_arrive_time_est=0.55  (current env_cfg, after linear_damping=0.5 was added)
  B) ball_arrive_time_est=0.5205 (original value when v74 was developed)

Uses MIDDLE ball (y=0, vy=0) — same as play_pure_ref --ball_preset middle.

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/probe_middle_timing.py --task X1-TableTennis
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

# MIDDLE ball preset (same as play_pure_ref --ball_preset middle)
BALL_POS = np.array([-0.35, 0.0, 1.3])
BALL_VEL = np.array([3.5, 0.0, 0.5])
DURATION = 1.0
HIT_PHASE = 0.475
NET_X, NET_Z = 0.0, 0.9125
TABLE_Z = 0.76
G = 9.81

LIMITS = np.array([
    [-1.053, 3.169], [-3.081, 0.314], [-2.777, 2.762],
    [-1.911, 1.948], [-2.789, 2.761], [-1.288, 1.508], [-3.14, 3.14],
])

# MIDDLE v74 keyframes (from create_forehand.py)
MIDDLE = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (0.300, [+1.127, +0.198, -1.904, +0.877, -0.315, -1.045, +1.000]),
    (0.400, [+1.087, +0.103, -1.979, +0.507, -0.315, -1.150, +1.000]),
    (0.475, [+1.387, +0.103, -1.850, +0.457, -0.900, -0.400, +1.000]),
    (0.550, [+1.437, +0.103, -1.979, +0.407, -0.165, -0.495, +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000, +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000, +1.000]),
]


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

    times_kf = np.array([k[0] for k in MIDDLE])
    angs_kf = np.array([k[1] for k in MIDDLE], dtype=np.float64)
    spline = CubicSpline(times_kf, angs_kf, bc_type="clamped")

    def run(ball_arrive_time_est, hit_phase=HIT_PHASE):
        initial_phase = (hit_phase - ball_arrive_time_est / DURATION) % 1.0
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
        bv_post, bp_post = np.zeros(3), np.zeros(3)
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
                bp_post = bp.copy()

        if not hit:
            bv_post = bv.copy()
            bp_post = bp.copy()
        bp_post -= env_origin
        zn, xb, clr, val = analyze_trajectory(bp_post[0], bp_post[2], bv_post[0], bv_post[2])
        return dict(bat=ball_arrive_time_est, hp=hit_phase, ip=initial_phase,
                    gap=min_gap, t=min_t, hit=hit,
                    pp=pp_hit, bp=bp_hit, pv=pv_hit, fn=face_hit,
                    bv=bv_post, zn=zn, xb=xb, clr=clr and val)

    print(f"\n{'='*120}")
    print(f"  MIDDLE v74 timing diagnostic")
    print(f"  Ball: pos={BALL_POS}, vel={BALL_VEL} (middle preset, y=0)")
    print(f"  hit_phase={HIT_PHASE}, duration={DURATION}")
    print(f"{'='*120}")
    print(f"\n{'bat':>6} {'ip':>6} {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'pp_x':>6} {'pp_y':>6} {'pp_z':>6} | "
          f"{'bp_x':>6} {'bp_y':>6} {'bp_z':>6} | "
          f"{'pvx':>6} {'pvz':>6} | "
          f"{'bvx':>6} {'bvz':>6} | {'zn':>6} {'xb':>6} {'CLR':>3}")
    print("-" * 120)

    # Test with different ball_arrive_time_est values
    for bat in [0.5205, 0.53, 0.54, 0.55, 0.56, 0.57]:
        r = run(bat)
        pp, pv, fn, bv = r['pp'], r['pv'], r['fn'], r['bv']
        hit_s = "Y" if r['hit'] else "N"
        zns = f"{r['zn']:+.2f}" if r['zn'] is not None else "  -  "
        xbs = f"{r['xb']:+.2f}" if r['xb'] is not None else "  -  "
        clr_s = "Y" if r['clr'] else "N"
        print(f"{r['bat']:>6.4f} {r['ip']:>6.4f} {r['gap']:>5.3f} {r['t']:>5.3f} {hit_s:>3} | "
              f"{pp[0]:>+6.3f} {pp[1]:>+6.3f} {pp[2]:>+6.3f} | "
              f"{r['bp'][0]:>+6.3f} {r['bp'][1]:>+6.3f} {r['bp'][2]:>+6.3f} | "
              f"{pv[0]:>+6.2f} {pv[2]:>+6.2f} | "
              f"{bv[0]:>+6.2f} {bv[2]:>+6.2f} | {zns:>6} {xbs:>6} {clr_s:>3}")

    # Also test phase sweep at both timing values
    print(f"\n\n--- Phase sweep (hit_phase ± 0.05) ---")
    print(f"{'bat':>6} {'hp':>5} {'gap':>5} {'t':>5} {'HIT':>3} | "
          f"{'bvx':>6} {'bvz':>6} | {'zn':>6} {'CLR':>3}")
    print("-" * 70)
    for bat in [0.5205, 0.55]:
        for hp in [0.425, 0.450, 0.475, 0.500, 0.525]:
            r = run(bat, hit_phase=hp)
            hit_s = "Y" if r['hit'] else "N"
            zns = f"{r['zn']:+.2f}" if r['zn'] is not None else "  -  "
            clr_s = "Y" if r['clr'] else "N"
            print(f"{bat:>6.4f} {hp:>5.3f} {r['gap']:>5.3f} {r['t']:>5.3f} {hit_s:>3} | "
                  f"{r['bv'][0]:>+6.2f} {r['bv'][2]:>+6.2f} | {zns:>6} {clr_s:>3}")
        print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
