"""Fast face sweep: test 3 yb5 values in single sim session.

Tests pronation (yb5) effect on face normal starting from v31 base config.
Runs 3 sequential episodes in same env, changing motion file between resets.

Usage:
    cd /root/unitree_rl_lab
    /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/right_face_fast.py
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


def make_npz(keyframes, path):
    times = np.array([kf[0] for kf in keyframes])
    angles = np.array([kf[1] for kf in keyframes], dtype=np.float32)
    cs = CubicSpline(times, angles, bc_type="clamped")
    t_interp = np.linspace(0, 1.0, 31)
    dof = cs(t_interp).astype(np.float32)
    np.savez(path, fps=np.float64(30), upper_body_dof=dof,
             base_y=np.zeros(31, dtype=np.float32), joint_names=JOINT_NAMES)


# Test configs: (yb3, yb5, yb6, yb2, label)
# Based on insight: yb5 (pronation) rotates face without moving position much.
# MIDDLE uses yb5=-0.9 at PIN with yb6=-0.4 — face points toward -X.
# Our v31 has yb5=0.0, yb6=0.72 — face points DOWN.
# Try adding pronation while keeping yb6=0.72 (proven good position).
CONFIGS = [
    # yb3,    yb5,    yb6,    yb2,   label
    (-0.40,  -0.80,  +0.72,  +0.70, "A: yb5=-0.8 only"),
    (-0.40,  -1.20,  +0.72,  +0.70, "B: yb5=-1.2 only"),
    (-1.00,  -0.80,  +0.72,  +0.70, "C: yb3=-1.0 + yb5=-0.8"),
]

# Pre-generate all NPZs
npz_paths = []
for i, (yb3, yb5, yb6, yb2, label) in enumerate(CONFIGS):
    kf = [
        (0.000, [+0.440, +0.750, -0.400, -1.350, +0.000, +0.720, -0.140]),
        (0.250, [+0.440, +1.200, -0.400, -1.850, +0.000, +0.720, -0.140]),
        (0.350, [+0.440, +0.900, -0.400, -1.700, +0.000, +0.720, -0.140]),
        (0.420, [+0.480, yb2,    yb3,    -1.100, yb5,    yb6,    -0.140]),  # HIT
        (0.460, [+0.500, yb2-0.20, yb3*0.8, -0.700, yb5*0.5, yb6-0.10, -0.140]),
        (0.530, [+0.520, yb2-0.40, -0.400, -0.500, +0.000, +0.720, -0.140]),
        (0.750, [+0.440, +0.750, -0.400, -1.350, +0.000, +0.720, -0.140]),
        (1.000, [+0.440, +0.750, -0.400, -1.350, +0.000, +0.720, -0.140]),
    ]
    p = f'/tmp/right_face_cfg{i}.npz'
    make_npz(kf, p)
    npz_paths.append(p)

# Use first config for env creation; we'll swap motion data between episodes
env_cfg = parse_env_cfg('X1-TableTennis', device='cuda:0', num_envs=1,
                        use_fabric=True, entry_point_key='play_env_cfg_entry_point')
env_cfg.commands.motion.motion_files = [npz_paths[0], npz_paths[0], npz_paths[0]]
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

print("\n=== FAST YB5 FACE SWEEP FOR RIGHT FOREHAND ===\n")
print(f"{'Label':<24} {'gap':>6} {'fn_x':>6} {'fn_y':>6} {'fn_z':>6} {'rv_x':>7} {'|rv|':>6} {'bvx_aft':>8}")
print("-" * 80)

for i, (yb3, yb5, yb6, yb2, label) in enumerate(CONFIGS):
    # Hot-swap motion data in the commands manager
    cmd_mgr = env.unwrapped.command_manager
    for term in cmd_mgr._terms:
        if hasattr(term, '_motions'):
            # Reload motion from new npz
            data = np.load(npz_paths[i])
            for m_idx in range(len(term._motions)):
                term._motions[m_idx]['upper_body_dof'] = torch.tensor(
                    data['upper_body_dof'], device=device, dtype=torch.float32)

    # Reset env
    obs, _ = env.reset()

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
            best_data = {'rv': racket_vel.copy(), 'rq': racket_quat.copy(),
                         'bv': ball_vel.copy(), 'bp': ball_pos.copy(), 'rp': racket_pos.copy()}

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

    print(f"{label:<24} {min_gap:>6.3f} {face_z[0]:>6.3f} {face_z[1]:>6.3f} {face_z[2]:>6.3f} "
          f"{rv[0]:>7.3f} {np.linalg.norm(rv):>6.3f} {v_ball_after[0]:>8.3f}")
    print(f"  racket=({best_data['rp'][0]:.3f},{best_data['rp'][1]:.3f},{best_data['rp'][2]:.3f}) "
          f"ball=({best_data['bp'][0]:.3f},{best_data['bp'][1]:.3f},{best_data['bp'][2]:.3f})")

print("\n=== TARGET: fn_x < -0.3, gap < 0.05 ===")

env.close()
simulation_app.close()
