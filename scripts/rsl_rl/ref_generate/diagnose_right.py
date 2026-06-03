"""诊断 RIGHT motion: 测量球到达位置 + 当前 PIN pose 下 paddle 位置/朝向.

报告:
  1. 球弹桌后到达 robot 附近的精确 (x, y, z) 和时刻
  2. PIN pose 下 paddle 的 3D 位置 + face normal
  3. 两者的 gap 向量 (需要怎么调整 paddle 才能碰到球)
  4. face normal 是否朝向对方球台

Usage:
  /isaac-sim/python.sh scripts/rsl_rl/diagnose_right.py --task X1-TableTennis
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
from scipy.spatial.transform import Rotation

import isaaclab_tasks  # noqa
import unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# Current RIGHT v14 keyframes (PIN plateau: hold PIN for 65ms)
RIGHT = [
    (0.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # hold
    (0.350, [+1.400, -0.250, -2.050, +0.300, -0.300, -1.000, +0.800]),  # windup
    (0.475, [+1.500, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),  # PIN start
    (0.540, [+1.500, -0.250, -1.800, +0.200, -0.700, -0.500, +0.800]),  # PIN hold (plateau)
    (0.650, [+1.450, -0.250, -1.950, +0.300, -0.300, -0.800, +0.800]),  # follow
    (1.000, [+1.400, -0.250, -2.000, +0.300, -0.300, -0.800, +0.800]),  # return hold
]

BALL_POS = np.array([-0.35, -0.03, 1.3])
BALL_VEL = np.array([3.5, -0.10, 0.5])

BALL_ARRIVE_TIME_EST = 0.55
HIT_PHASE = 0.475
DURATION = 1.0


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
    env_origin = scene.env_origins[0].cpu().numpy()

    print(f"\n{'='*60}")
    print(f"  RIGHT MOTION DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"\nEnv origin: {env_origin}")
    print(f"Robot root pos: {robot.data.root_pos_w[0].cpu().numpy()}")

    # === Part 1: Ball trajectory ===
    print(f"\n--- BALL TRAJECTORY ---")
    print(f"Launch: pos={BALL_POS}, vel={BALL_VEL}")

    # Simulate ball to find its position at various times
    ball_state = ball.data.default_root_state.clone()
    ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device) + scene.env_origins[0]
    ball_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
    ball_state[0, 10:13] = torch.zeros(3, device=device)
    ids = torch.tensor([0], device=device)
    ball.write_root_state_to_sim(ball_state, env_ids=ids)
    scene.write_data_to_sim()

    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    ball_positions = []
    ball_velocities = []
    ball_times = []

    # Let ball fly for 1.0s, record trajectory
    for step in range(int(1.0 / sim_dt)):
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)
        t = (step + 1) * sim_dt
        bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
        bv = ball.data.root_lin_vel_w[0].cpu().numpy()
        ball_positions.append(bp.copy())
        ball_velocities.append(bv.copy())
        ball_times.append(t)

    ball_positions = np.array(ball_positions)
    ball_velocities = np.array(ball_velocities)
    ball_times = np.array(ball_times)

    # Find ball position at key times
    print(f"\nBall position over time (local frame):")
    print(f"{'t':>6} {'x':>7} {'y':>7} {'z':>7} {'vx':>7} {'vy':>7} {'vz':>7}")
    for t_target in [0.3, 0.4, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        idx = np.argmin(np.abs(ball_times - t_target))
        bp = ball_positions[idx]
        bv = ball_velocities[idx]
        print(f"{ball_times[idx]:>6.3f} {bp[0]:>+7.3f} {bp[1]:>+7.3f} {bp[2]:>+7.3f} "
              f"{bv[0]:>+7.2f} {bv[1]:>+7.2f} {bv[2]:>+7.2f}")

    # Find when ball is closest to x=1.3 (typical paddle reach)
    target_x_values = [1.2, 1.25, 1.3, 1.35, 1.4]
    print(f"\nBall position when crossing key x values:")
    for tx in target_x_values:
        crossings = np.where((ball_positions[:-1, 0] < tx) & (ball_positions[1:, 0] >= tx))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            bp = ball_positions[idx]
            bv = ball_velocities[idx]
            print(f"  x={tx:.2f}: t={ball_times[idx]:.3f}s, pos=({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f}), "
                  f"vel=({bv[0]:+.2f}, {bv[1]:+.2f}, {bv[2]:+.2f})")

    # === Part 2: Paddle position at PIN pose (static FK) ===
    print(f"\n--- PADDLE POSITION AT EACH KEYFRAME (static settle) ---")

    times_kf = np.array([k[0] for k in RIGHT])
    angs_kf = np.array([k[1] for k in RIGHT], dtype=np.float64)
    spline = CubicSpline(times_kf, angs_kf, bc_type="clamped")

    for kf_name, phase_val in [("HOLD", 0.0), ("WINDUP", 0.350), ("PIN", 0.475), ("FOLLOW", 0.600)]:
        q = spline(phase_val)
        full = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full[0, jid] = float(q[k])
        v0 = torch.zeros_like(full)
        robot.write_joint_state_to_sim(full, v0, env_ids=ids)
        # Let PD settle
        for _ in range(300):
            robot.set_joint_position_target(full, env_ids=ids)
            scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            scene.update(sim_dt)

        paddle_pos = robot.data.body_pos_w[0, paddle_idx].cpu().numpy() - env_origin
        paddle_quat = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()  # (w, x, y, z)
        # Convert to rotation matrix to get face normal
        # Isaac uses (w, x, y, z) quaternion
        rot = Rotation.from_quat([paddle_quat[1], paddle_quat[2], paddle_quat[3], paddle_quat[0]])
        # Face normal = local Z axis of paddle (or X depending on mesh)
        face_z = rot.apply([0, 0, 1])
        face_x = rot.apply([1, 0, 0])
        face_y = rot.apply([0, 1, 0])

        print(f"\n  [{kf_name}] phase={phase_val:.3f}")
        print(f"    paddle pos: ({paddle_pos[0]:+.4f}, {paddle_pos[1]:+.4f}, {paddle_pos[2]:+.4f})")
        print(f"    face_normal (local +Z): ({face_z[0]:+.3f}, {face_z[1]:+.3f}, {face_z[2]:+.3f})")
        print(f"    face_normal (local +X): ({face_x[0]:+.3f}, {face_x[1]:+.3f}, {face_x[2]:+.3f})")
        print(f"    face_normal (local +Y): ({face_y[0]:+.3f}, {face_y[1]:+.3f}, {face_y[2]:+.3f})")
        print(f"    joints: [{', '.join(f'{v:+.3f}' for v in q)}]")

    # === Part 3: Dynamic simulation — where is paddle during ball arrival ===
    print(f"\n--- DYNAMIC: PADDLE TRAJECTORY DURING BALL FLIGHT ---")

    # Reset robot to initial phase
    initial_phase = (HIT_PHASE - BALL_ARRIVE_TIME_EST / DURATION) % 1.0
    print(f"  initial_phase = {initial_phase:.4f}")
    print(f"  ball arrives at t={BALL_ARRIVE_TIME_EST:.4f}s, phase should be {HIT_PHASE:.3f}")

    q0 = spline(initial_phase)
    full = robot.data.default_joint_pos[0:1].clone()
    for k, jid in enumerate(yb_joint_ids):
        full[0, jid] = float(q0[k])
    v0 = torch.zeros_like(full)
    robot.write_joint_state_to_sim(full, v0, env_ids=ids)
    for _ in range(200):
        robot.set_joint_position_target(full, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)

    # Launch ball
    ball_state[0, 0:3] = torch.tensor(BALL_POS, dtype=torch.float32, device=device) + scene.env_origins[0]
    ball_state[0, 7:10] = torch.tensor(BALL_VEL, dtype=torch.float32, device=device)
    ball_state[0, 10:13] = torch.zeros(3, device=device)
    ball.write_root_state_to_sim(ball_state, env_ids=ids)
    scene.write_data_to_sim()

    print(f"\n{'t':>6} {'phase':>6} | {'pad_x':>7} {'pad_y':>7} {'pad_z':>7} | "
          f"{'ball_x':>7} {'ball_y':>7} {'ball_z':>7} | {'gap':>5} {'gap_xyz':>20}")
    print("-" * 100)

    min_gap = 1e9
    min_gap_t = -1
    for step in range(int(0.8 / sim_dt)):
        t = step * sim_dt
        phase = (initial_phase + t / DURATION) % 1.0
        target = spline(phase)
        full_target = robot.data.default_joint_pos[0:1].clone()
        for k, jid in enumerate(yb_joint_ids):
            full_target[0, jid] = float(target[k])
        robot.set_joint_position_target(full_target, env_ids=ids)
        scene.write_data_to_sim()
        env.unwrapped.sim.step(render=False)
        scene.update(sim_dt)

        pp = robot.data.body_pos_w[0, paddle_idx].cpu().numpy() - env_origin
        bp = ball.data.root_pos_w[0].cpu().numpy() - env_origin
        gap_vec = pp - bp
        gap = float(np.linalg.norm(gap_vec))

        if gap < min_gap:
            min_gap = gap
            min_gap_t = t
            min_gap_paddle = pp.copy()
            min_gap_ball = bp.copy()
            min_gap_vec = gap_vec.copy()

        # Print at key times
        if t > 0.35 and step % 10 == 0:
            print(f"{t:>6.3f} {phase:>6.3f} | {pp[0]:>+7.3f} {pp[1]:>+7.3f} {pp[2]:>+7.3f} | "
                  f"{bp[0]:>+7.3f} {bp[1]:>+7.3f} {bp[2]:>+7.3f} | {gap:>5.3f} "
                  f"({gap_vec[0]:+.3f},{gap_vec[1]:+.3f},{gap_vec[2]:+.3f})")

    print(f"\n--- MINIMUM GAP ---")
    print(f"  gap = {min_gap:.4f}m at t = {min_gap_t:.4f}s")
    print(f"  paddle @ min: ({min_gap_paddle[0]:+.4f}, {min_gap_paddle[1]:+.4f}, {min_gap_paddle[2]:+.4f})")
    print(f"  ball   @ min: ({min_gap_ball[0]:+.4f}, {min_gap_ball[1]:+.4f}, {min_gap_ball[2]:+.4f})")
    print(f"  gap vector (paddle - ball): dx={min_gap_vec[0]:+.4f}, dy={min_gap_vec[1]:+.4f}, dz={min_gap_vec[2]:+.4f}")
    print(f"\n  TO HIT: paddle needs to move by ({-min_gap_vec[0]:+.4f}, {-min_gap_vec[1]:+.4f}, {-min_gap_vec[2]:+.4f})")

    # Get face normal at closest approach
    paddle_quat = robot.data.body_quat_w[0, paddle_idx].cpu().numpy()
    rot = Rotation.from_quat([paddle_quat[1], paddle_quat[2], paddle_quat[3], paddle_quat[0]])
    face_z = rot.apply([0, 0, 1])
    face_x = rot.apply([1, 0, 0])
    print(f"\n  Face normal (local +Z) @ min gap: ({face_z[0]:+.3f}, {face_z[1]:+.3f}, {face_z[2]:+.3f})")
    print(f"  Face normal (local +X) @ min gap: ({face_x[0]:+.3f}, {face_x[1]:+.3f}, {face_x[2]:+.3f})")
    print(f"  IDEAL face: should point toward (-X, ~0Y, slight +Z) = toward opponent table")

    # Check if face is pointing toward opponent
    # Opponent is in -X direction
    toward_opp = face_z[0] < -0.3 or face_x[0] < -0.3
    print(f"\n  Face toward opponent (-X): {'YES' if toward_opp else 'NO — PROBLEM!'}")
    if not toward_opp:
        print(f"  Face is pointing toward: ", end="")
        if abs(face_z[1]) > 0.5:
            print(f"{'+ Y (left)' if face_z[1] > 0 else '- Y (outward/right)'}", end=" ")
        if face_z[2] > 0.5:
            print(f"+ Z (up/ceiling)", end=" ")
        if face_z[0] > 0.3:
            print(f"+ X (backward)", end=" ")
        print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
