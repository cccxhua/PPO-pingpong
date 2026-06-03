"""V59 paddle-frame diagnostic: at hit moment, transform ball position
into paddle local frame to see where on paddle the ball actually hits.

Paddle local frame (from STL):
  +x_local: face width (±75mm)
  +y_local: face normal (±5mm thickness)
  +z_local: handle direction (face: -47..+58, handle: +78..+170)

If ball's local z > +58: hits handle.
If ball's local z in [-47, +58]: hits face (good).
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


V58 = [
    (0.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (0.300, [+1.400, +0.185, -2.025, +1.050, +0.000, -1.000,  +1.000]),
    (0.400, [+1.460, +0.090, -2.100, +0.680, +0.000, -1.200,  +1.000]),
    (0.475, [+1.560, +0.090, -2.100, +0.630, -0.030, -0.975,  +1.000]),
    (0.550, [+1.710, +0.090, -2.100, +0.580, +0.000, -0.450,  +1.000]),
    (0.700, [+1.450, +0.100, -2.000, +0.850, +0.000, -1.000,  +1.000]),
    (0.900, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
    (1.000, [+1.000, +0.300, -2.000, +1.400, +0.000, -1.000,  +1.000]),
]


def quat_inv_rotate(q, v):
    """Rotate vector v by inverse of quaternion q (w,x,y,z) — i.e., world->local."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # q_inv = (w, -x, -y, -z) for unit quat
    # v_local = q_inv * v * q
    # Use formula: v_local = v + 2 * q_vec × (q_vec × v - w*v)  (with q_inv -> -q_vec)
    qv = torch.tensor([-x, -y, -z], device=v.device, dtype=v.dtype)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + (-w) * t + torch.cross(qv, t, dim=-1)
    # Simpler: use full rotation matrix.


def quat_to_R(q):
    """quat (w,x,y,z) -> 3x3 rotation matrix (world<-local)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], device=q.device, dtype=q.dtype)
    return R


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

    times = np.array([k[0] for k in V58])
    angs = np.array([k[1] for k in V58], dtype=np.float64)
    spline = CubicSpline(times, angs, bc_type="clamped")

    # init
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

    ball_root_state = ball.data.default_root_state.clone()
    ball_root_state[0, 0:3] = torch.tensor([-0.35, 0.0, 1.3], device=device)
    ball_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_root_state[0, 7:10] = torch.tensor([3.5, 0.0, 0.5], device=device)
    ball_root_state[0, 10:13] = torch.tensor([0.0, 0.0, 0.0], device=device)
    ball.write_root_state_to_sim(ball_root_state, env_ids=ids)
    scene.write_data_to_sim()

    sim_dt = float(env.unwrapped.sim.get_physics_dt())
    log = []
    for step in range(int(0.65 / sim_dt)):
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
        q = robot.data.body_quat_w[0, paddle_idx]  # (w,x,y,z)
        b = ball.data.root_pos_w[0]

        # Ball in paddle local frame: b_local = R^T (b - p)
        R = quat_to_R(q)
        delta = (b - p)
        b_local = R.T @ delta  # world->local
        log.append((t, p.cpu().numpy(), q.cpu().numpy(), b.cpu().numpy(), b_local.cpu().numpy()))

    print(f"\n=== Ball position in PADDLE LOCAL frame (mm) ===")
    print(f"  Face region:   x_local ±75, y_local ±5,  z_local -47 to +58")
    print(f"  Handle region: x_local ±9,  y_local ±10, z_local +78 to +170")
    print(f"")
    print(f"  {'t':>5}  {'paddle world (xyz)':<24}  {'ball local (xyz mm)':<24}  {'region'}")
    for tt in np.arange(0.40, 0.60, 0.01):
        idx = min(range(len(log)), key=lambda i: abs(log[i][0] - tt))
        t_log, p, q, b, b_local = log[idx]
        bx, by, bz = b_local * 1000  # mm
        # Decide region
        in_face = (abs(bx) < 80) and (abs(by) < 30) and (-50 <= bz <= 65)
        in_handle = (abs(bx) < 20) and (abs(by) < 20) and (75 <= bz <= 175)
        gap_to_face_center = np.sqrt(bx**2 + by**2 + (bz - 5.5)**2)  # face center at z=+5.5
        if in_face:
            region = f"FACE  (off-center {gap_to_face_center:.0f}mm)"
        elif in_handle:
            region = f"HANDLE  ★"
        else:
            in_y = abs(by) < 50
            region = f"miss (Δy={by:.0f}mm)" if not in_y else f"outside  (z={bz:.0f}mm)"
        print(f"  {tt:>5.3f}  ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  ({bx:+5.0f},{by:+5.0f},{bz:+5.0f})  {region}")

    # Find when ball is closest to face center
    print(f"\n=== Closest approach to FACE CENTER (paddle local origin) ===")
    best_t, best_d = None, 1e9
    for entry in log:
        t, p, q, b, b_local = entry
        if 0.30 < t < 0.65:
            d = np.linalg.norm(b_local)
            if d < best_d:
                best_d = d
                best_t = t
                best_entry = entry
    if best_t:
        t, p, q, b, b_local = best_entry
        bx, by, bz = b_local * 1000
        print(f"  t={t:.3f}, gap to origin = {best_d*100:.2f}cm")
        print(f"  ball local = ({bx:+.0f}, {by:+.0f}, {bz:+.0f}) mm")
        print(f"  paddle world = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        print(f"  paddle quat (wxyz) = ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
