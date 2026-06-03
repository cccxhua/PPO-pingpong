"""Diagnose racket velocity and face normal at ball contact for RIGHT motion.

Runs v31 (known to contact ball at gap~3cm) and prints:
- Paddle linear velocity at closest approach
- Paddle angular velocity
- Face normal direction (body Z axis in world frame)
- Ball velocity before/after contact

Usage:
    cd /root/unitree_rl_lab
    /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/right_diag.py
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

import isaaclab_tasks
import unitree_rl_lab.tasks
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

env_cfg = parse_env_cfg('X1-TableTennis', device='cuda:0', num_envs=1,
                        use_fabric=True, entry_point_key='play_env_cfg_entry_point')

# Use current v35 motion
npz_path = os.path.abspath('source/unitree_rl_lab/unitree_rl_lab/tasks/table_tennis/robots/x1/forehand/forehand_right_v35.npz')
env_cfg.commands.motion.motion_files = [npz_path, npz_path, npz_path]
env_cfg.commands.motion.hit_phase_noise = 0.0
env_cfg.commands.motion.ball_arrive_time_est = 0.55
env_cfg.commands.motion.ball_arrive_time_noise = 0.0

# Ball preset "right" - same as play_pure_ref right preset
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

action_dim = env.action_space.shape[-1]
device = env.unwrapped.device
zero_action = torch.zeros((1, action_dim), device=device)

prev_ball_pos = None
prev_racket_pos = None
min_gap = 999.0
min_gap_step = 0
contact_data = {}
steps = 0

print("\n=== RIGHT MOTION DIAGNOSTICS ===\n")

for step in range(200):  # ~4s of simulation
    with torch.inference_mode():
        obs, _, _, _, _ = env.step(zero_action)

    ball_pos = (ball.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
    ball_vel = ball.data.root_lin_vel_w[0].cpu().numpy()

    racket_pos = (robot.data.body_pos_w[0, racket_body_idx] - scene.env_origins[0]).cpu().numpy()
    racket_vel = robot.data.body_lin_vel_w[0, racket_body_idx].cpu().numpy()
    racket_quat = robot.data.body_quat_w[0, racket_body_idx].cpu().numpy()  # w,x,y,z

    gap = float(np.linalg.norm(ball_pos - racket_pos))

    if gap < min_gap:
        min_gap = gap
        min_gap_step = step
        contact_data = {
            'ball_pos': ball_pos.copy(),
            'ball_vel': ball_vel.copy(),
            'racket_pos': racket_pos.copy(),
            'racket_vel': racket_vel.copy(),
            'racket_quat': racket_quat.copy(),
        }

    # Print every 5 steps near expected contact (steps 20-35)
    if 18 <= step <= 35 and step % 2 == 0:
        print(f"  step {step:3d}: gap={gap:.3f}m  racket=({racket_pos[0]:.3f},{racket_pos[1]:.3f},{racket_pos[2]:.3f})  "
              f"ball=({ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f})  "
              f"rv=({racket_vel[0]:.2f},{racket_vel[1]:.2f},{racket_vel[2]:.2f})  "
              f"bv=({ball_vel[0]:.2f},{ball_vel[1]:.2f},{ball_vel[2]:.2f})")

print(f"\n=== CLOSEST APPROACH: step {min_gap_step}, gap = {min_gap:.4f}m ===")
print(f"  Ball pos:    ({contact_data['ball_pos'][0]:.4f}, {contact_data['ball_pos'][1]:.4f}, {contact_data['ball_pos'][2]:.4f})")
print(f"  Racket pos:  ({contact_data['racket_pos'][0]:.4f}, {contact_data['racket_pos'][1]:.4f}, {contact_data['racket_pos'][2]:.4f})")
print(f"  Ball vel:    ({contact_data['ball_vel'][0]:.3f}, {contact_data['ball_vel'][1]:.3f}, {contact_data['ball_vel'][2]:.3f})")
print(f"  Racket vel:  ({contact_data['racket_vel'][0]:.3f}, {contact_data['racket_vel'][1]:.3f}, {contact_data['racket_vel'][2]:.3f})")
print(f"  Racket |v|:  {np.linalg.norm(contact_data['racket_vel']):.3f} m/s")
print(f"  Racket vx:   {contact_data['racket_vel'][0]:.3f} m/s (need < -1.68 for return)")

# Compute face normal from quaternion (paddle body Z-axis in world frame)
q = contact_data['racket_quat']  # w,x,y,z
# Rotate [0,0,1] by quaternion
qw, qx, qy, qz = q[0], q[1], q[2], q[3]
# Body Z in world: using quaternion rotation formula
face_z = np.array([
    2*(qx*qz + qw*qy),
    2*(qy*qz - qw*qx),
    1 - 2*(qx*qx + qy*qy)
])
face_x = np.array([
    1 - 2*(qy*qy + qz*qz),
    2*(qx*qy + qw*qz),
    2*(qx*qz - qw*qy)
])
face_y = np.array([
    2*(qx*qy - qw*qz),
    1 - 2*(qx*qx + qz*qz),
    2*(qy*qz + qw*qx)
])
print(f"\n  Face normal (body +Z): ({face_z[0]:.3f}, {face_z[1]:.3f}, {face_z[2]:.3f})")
print(f"  Face X-axis:           ({face_x[0]:.3f}, {face_x[1]:.3f}, {face_x[2]:.3f})")
print(f"  Face Y-axis:           ({face_y[0]:.3f}, {face_y[1]:.3f}, {face_y[2]:.3f})")

# Expected ball reflection
rv = contact_data['racket_vel']
bv = contact_data['ball_vel']
e = 0.92  # restitution
# For face normal collision: v_ball_after = (1+e)*v_racket_n - e*v_ball_n + tangential component
# Project onto face normal
n = face_z / np.linalg.norm(face_z)
v_ball_n = np.dot(bv, n)
v_rack_n = np.dot(rv, n)
v_ball_after_n = (1+e)*v_rack_n - e*v_ball_n
# Tangential unchanged
v_ball_t = bv - v_ball_n * n
v_ball_after = v_ball_after_n * n + v_ball_t
print(f"\n  Expected ball vel after (if face normal collision):")
print(f"    v_ball = ({v_ball_after[0]:.3f}, {v_ball_after[1]:.3f}, {v_ball_after[2]:.3f})")
print(f"    |v_ball| = {np.linalg.norm(v_ball_after):.3f} m/s")
print(f"    vx_after = {v_ball_after[0]:.3f} (need < 0 for return)")

env.close()
simulation_app.close()
