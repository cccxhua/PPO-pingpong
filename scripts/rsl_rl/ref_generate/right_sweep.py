"""Sweep yb6 and yb2 at HIT to find combo that gives good face + contacts ball.

Tries different yb6 values at the HIT keyframe (t=0.420) with yb2 compensation,
measures gap and face normal, finds best combo for ball return.
"""

import sys
sys.argv = ['diag', '--headless', '--task', 'X1-TableTennis', '--num_envs', '1']

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--num_envs', type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
import numpy as np
from scipy.interpolate import CubicSpline

import isaaclab_tasks
import unitree_rl_lab.tasks
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

JOINT_NAMES = np.array(["joint_yb_1","joint_yb_2","joint_yb_3","joint_yb_4","joint_yb_5","joint_yb_6","joint_yb_7"])

def make_npz(yb2_hit, yb6_hit, path):
    """Generate motion npz with specific yb2/yb6 at HIT frame."""
    keyframes = [
        (0.000, [+0.440, +0.900, -0.400, -1.350, +0.000, +0.720, -0.140]),
        (0.300, [+0.440, +1.400, -0.400, -1.850, +0.000, +0.720, -0.140]),
        (0.380, [+0.440, +1.100, -0.400, -1.700, +0.000, +0.720, -0.140]),
        (0.420, [+0.480, yb2_hit, -0.400, -1.100, +0.000, yb6_hit, -0.140]),  # HIT
        (0.450, [+0.500, yb2_hit-0.20, -0.400, -0.600, +0.000, yb6_hit-0.10, -0.140]),
        (0.500, [+0.520, yb2_hit-0.35, -0.400, -0.400, +0.000, yb6_hit, -0.140]),
        (0.700, [+0.440, +0.900, -0.400, -1.350, +0.000, +0.720, -0.140]),
        (1.000, [+0.440, +0.900, -0.400, -1.350, +0.000, +0.720, -0.140]),
    ]
    times = np.array([kf[0] for kf in keyframes])
    angles = np.array([kf[1] for kf in keyframes], dtype=np.float32)
    cs = CubicSpline(times, angles, bc_type="clamped")
    t_interp = np.linspace(0, 1.0, 31)
    dof = cs(t_interp).astype(np.float32)
    np.savez(path, fps=np.float64(30), upper_body_dof=dof,
             base_y=np.zeros(31, dtype=np.float32), joint_names=JOINT_NAMES)


# Sweep parameters
# yb6: 0.72 (original, face DOWN) → try 0.4, 0.2, 0.0, -0.2, -0.4
# yb2: compensate for yb6-induced shift. Original 0.70.
# Empirically yb6 -0.72 rad change → ~13cm Y shift. Per 0.1 rad yb6: ~1.8cm Y shift.
# yb2 0.1 rad change → ~5cm Y shift. So compensation ratio: delta_yb2 ≈ -delta_yb6 * 0.36

SWEEP = [
    # (yb6_hit, yb2_hit, label)
    (0.72, 0.70, "baseline"),
    (0.40, 0.58, "yb6=0.4"),
    (0.20, 0.51, "yb6=0.2"),
    (0.00, 0.44, "yb6=0.0"),
    (-0.20, 0.37, "yb6=-0.2"),
    (-0.40, 0.30, "yb6=-0.4"),
]

print("\n=== YB6/YB2 SWEEP FOR RIGHT FOREHAND ===\n")
print(f"{'Label':<12} {'yb6':>5} {'yb2':>5} {'gap':>7} {'rv_x':>7} {'rv_y':>7} {'|rv|':>7} {'fn_x':>6} {'fn_z':>6} {'bvx_aft':>8}")
print("-" * 90)

for yb6_hit, yb2_hit, label in SWEEP:
    npz_path = f'/tmp/right_sweep_{label}.npz'
    make_npz(yb2_hit, yb6_hit, npz_path)

    env_cfg = parse_env_cfg('X1-TableTennis', device='cuda:0', num_envs=1,
                            use_fabric=True, entry_point_key='play_env_cfg_entry_point')
    env_cfg.commands.motion.motion_files = [npz_path, npz_path, npz_path]
    env_cfg.commands.motion.hit_phase_noise = 0.0
    env_cfg.commands.motion.ball_arrive_time_est = 0.47
    env_cfg.commands.motion.ball_arrive_time_noise = 0.0

    bp = dict(x_range=(-0.35, -0.35), y_range=(+0.45, +0.45),
              z_range=(1.35, 1.35), vx_range=(3.5, 3.5),
              vy_range=(+0.2, +0.2), vz_range=(0.3, 0.3))
    env_cfg.events.reset_ball.params.update({"ball_cfg": env_cfg.events.reset_ball.params["ball_cfg"], **bp})
    env_cfg.events.relaunch_ball.params.update({"ball_cfg": env_cfg.events.relaunch_ball.params["ball_cfg"], **bp})
    env_cfg.terminations.ball_on_own_table = None
    env_cfg.terminations.ball_missed_paddle = None

    env = gym.make('X1-TableTennis', cfg=env_cfg)
    obs, _ = env.reset()

    scene = env.unwrapped.scene
    ball = scene["ball"]
    robot = scene["robot"]
    racket_body_idx = robot.find_bodies("Link_yb_paddle")[0][0]
    device = env.unwrapped.device
    zero_action = torch.zeros((1, env.action_space.shape[-1]), device=device)

    min_gap = 999.0
    best_data = {}
    for step in range(60):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(zero_action)
        ball_pos = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
        racket_pos = (robot.data.body_pos_w[0, racket_body_idx] - scene.env_origins[0]).cpu().numpy()
        racket_vel = robot.data.body_lin_vel_w[0, racket_body_idx].cpu().numpy()
        racket_quat = robot.data.body_quat_w[0, racket_body_idx].cpu().numpy()
        ball_vel = ball.data.root_lin_vel_w[0].cpu().numpy()
        gap = float(np.linalg.norm(ball_pos - racket_pos))
        if gap < min_gap:
            min_gap = gap
            best_data = {'rv': racket_vel.copy(), 'rq': racket_quat.copy(), 'bv': ball_vel.copy()}

    env.close()

    # Compute face normal
    q = best_data['rq']
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    face_z = np.array([2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx*qx + qy*qy)])

    rv = best_data['rv']
    bv = best_data['bv']
    n = face_z / np.linalg.norm(face_z)
    e = 0.92
    v_ball_n = np.dot(bv, n)
    v_rack_n = np.dot(rv, n)
    v_ball_after_n = (1+e)*v_rack_n - e*v_ball_n
    v_ball_t = bv - v_ball_n * n
    v_ball_after = v_ball_after_n * n + v_ball_t

    print(f"{label:<12} {yb6_hit:>5.2f} {yb2_hit:>5.2f} {min_gap:>7.3f} {rv[0]:>7.3f} {rv[1]:>7.3f} "
          f"{np.linalg.norm(rv):>7.3f} {face_z[0]:>6.3f} {face_z[2]:>6.3f} {v_ball_after[0]:>8.3f}")

print("\n=== Target: gap < 0.05, fn_x < -0.3, bvx_aft < -1.0 ===")

simulation_app.close()
