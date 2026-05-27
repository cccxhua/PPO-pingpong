from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis.mdp.commands import UpperBodyMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

RACKET_OFFSET_Z = 0.0  # X1: Link_yb_paddle 本身就是球拍, 无偏移. (历史: G1 腕→球拍 16cm)


def upper_body_joint_pos_rel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ids = command.upper_body_joint_ids
    return (env.scene["robot"].data.joint_pos[:, ids] - env.scene["robot"].data.default_joint_pos[:, ids])


def upper_body_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    return env.scene["robot"].data.joint_vel[:, command.upper_body_joint_ids]


def base_y_pos(env: ManagerBasedEnv) -> torch.Tensor:
    return (env.scene["robot"].data.root_pos_w[:, 1:2] - env.scene.env_origins[:, 1:2])


def base_y_vel(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_w[:, 1:2]


def ball_pos_relative(env: ManagerBasedEnv, ball_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    robot_root = env.scene["robot"].data.root_pos_w
    return (ball.data.root_pos_w - robot_root).clamp(-5.0, 5.0)


def ball_vel_relative(env: ManagerBasedEnv, ball_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    robot_vel = env.scene["robot"].data.root_lin_vel_w
    return (ball.data.root_lin_vel_w - robot_vel).clamp(-10.0, 10.0)


def racket_pos(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    wrist_pos = robot.data.body_pos_w[:, body_idx]
    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_offset = torch.zeros_like(wrist_pos)
    local_offset[:, 2] = RACKET_OFFSET_Z
    world_pos = wrist_pos + quat_rotate(wrist_quat, local_offset)
    return world_pos - env.scene.env_origins


def racket_ori(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    return robot.data.body_quat_w[:, body_idx]


def racket_normal(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    """Racket face normal direction in world frame (3D)."""
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_normal = torch.zeros(wrist_quat.shape[0], 3, device=wrist_quat.device)
    local_normal[:, 1] = 1.0
    return quat_rotate(wrist_quat, local_normal)


def racket_vel(env: ManagerBasedEnv, racket_body_name: str) -> torch.Tensor:
    """Racket center linear velocity in world frame (3D), including angular velocity contribution."""
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)
    wrist_vel = robot.data.body_lin_vel_w[:, body_idx]
    wrist_ang_vel = robot.data.body_ang_vel_w[:, body_idx]
    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_offset = torch.zeros_like(wrist_vel)
    local_offset[:, 2] = RACKET_OFFSET_Z
    world_offset = quat_rotate(wrist_quat, local_offset)
    return wrist_vel + torch.cross(wrist_ang_vel, world_offset, dim=-1)


def racket_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact force magnitude on the racket (1D), normalized by 10N."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    force_mag = torch.norm(net_forces, dim=-1)
    return (force_mag / 10.0).clamp(max=1.0)


def ball_spin_relative(env: ManagerBasedEnv, ball_name: str) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    return ball.data.root_ang_vel_w.clamp(-50.0, 50.0)


def motion_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    return command.phase.unsqueeze(-1)


def compute_ideal_ball_velocity(
    ball_pos_local: torch.Tensor,
    target_x: float = -0.7,
    target_z: float = 0.76,
    net_x: float = 0.0,
    net_z_top: float = 0.9125,
    clearance: float = 0.02,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Minimum-speed post-hit ball velocity to land on target while clearing net."""
    N = ball_pos_local.shape[0]
    device = ball_pos_local.device

    x_h = ball_pos_local[:, 0]
    y_h = ball_pos_local[:, 1]
    z_h = ball_pos_local[:, 2]

    dx = target_x - x_h
    dy = -y_h
    dz = target_z - z_h

    T = torch.linspace(0.15, 1.0, 18, device=device)
    vx = dx.unsqueeze(-1) / T.unsqueeze(0)
    vy = dy.unsqueeze(-1) / T.unsqueeze(0)
    vz = dz.unsqueeze(-1) / T.unsqueeze(0) + 0.5 * gravity * T.unsqueeze(0)

    t_net = (net_x - x_h).unsqueeze(-1) / vx.clamp(max=-0.1)
    z_net = z_h.unsqueeze(-1) + vz * t_net - 0.5 * gravity * t_net ** 2

    valid = (t_net > 0) & (t_net < T.unsqueeze(0)) & (z_net > net_z_top + clearance)
    speed_sq = vx ** 2 + vy ** 2 + vz ** 2
    speed_sq = torch.where(valid, speed_sq, torch.full_like(speed_sq, 1e6))

    best_idx = speed_sq.argmin(dim=-1)
    arange = torch.arange(N, device=device)
    result = torch.stack(
        [vx[arange, best_idx], vy[arange, best_idx], vz[arange, best_idx]], dim=-1
    )

    no_valid = speed_sq.min(dim=-1).values >= 1e6
    if torch.any(no_valid):
        result[no_valid] = torch.tensor([-3.5, 0.0, 1.5], device=device)

    return result


def ideal_hit_velocity(
    env: ManagerBasedEnv,
    ball_name: str,
    target_x: float = -0.7,
    target_z: float = 0.76,
) -> torch.Tensor:
    """Ideal post-hit ball velocity to land on opponent table (3D)."""
    ball: RigidObject = env.scene[ball_name]
    ball_pos_local = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    return compute_ideal_ball_velocity(ball_pos_local, target_x, target_z)


def ball_time_to_arrive(env: ManagerBasedEnv, ball_name: str, robot_x: float = 1.5) -> torch.Tensor:
    """Estimated time for ball to reach robot x position (1D, clamped to [0, 3])."""
    ball: RigidObject = env.scene[ball_name]
    ball_x = ball.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_vx = ball.data.root_lin_vel_w[:, 0]
    dx = robot_x - ball_x
    time = torch.where(
        ball_vx > 0.1,
        dx / ball_vx,
        torch.full_like(dx, 3.0),
    )
    return time.clamp(0.0, 3.0).unsqueeze(-1)


def ball_predicted_hit_point(
    env: ManagerBasedEnv,
    ball_name: str,
    robot_x: float = 1.5,
    table_z: float = 0.76,
    restitution: float = 0.9,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Predict where the ball will be when it reaches robot_x, accounting for table bounce.

    Returns (predicted_y, predicted_z, time_to_hit) relative to env origin.
    The prediction considers:
      - If ball is above table and moving down with vx > 0: compute bounce point, then trajectory after bounce
      - If ball is already rising after bounce (vz > 0, x on robot side): direct parabolic prediction
    """
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    bx, by, bz = ball_pos[:, 0], ball_pos[:, 1], ball_pos[:, 2]
    vx, vy, vz = ball_vel[:, 0], ball_vel[:, 1], ball_vel[:, 2]

    N = bx.shape[0]
    device = bx.device

    pred_y = torch.zeros(N, device=device)
    pred_z = torch.zeros(N, device=device)
    pred_t = torch.full((N,), 3.0, device=device)

    moving_toward = vx > 0.1
    dx = robot_x - bx

    # Case 1: ball above table, moving down, hasn't bounced yet (on opponent side or in flight)
    # Estimate time to hit table: bz + vz*t - 0.5*g*t^2 = table_z
    # 0.5*g*t^2 - vz*t + (table_z - bz) = 0
    a_coef = 0.5 * gravity
    b_coef = -vz
    c_coef = table_z - bz

    discriminant = b_coef ** 2 - 4 * a_coef * c_coef
    disc_valid = discriminant >= 0
    disc_safe = discriminant.clamp(min=0)

    # Time to bounce (take the positive root)
    t_bounce = (b_coef + torch.sqrt(disc_safe)) / (2 * a_coef)
    t_bounce = t_bounce.clamp(min=0.0)

    # Position at bounce
    x_bounce = bx + vx * t_bounce
    y_bounce = by + vy * t_bounce

    # After bounce: vz flips and reduces by restitution
    vz_after = torch.abs(vz + (-gravity) * t_bounce) * restitution
    # vx and vy approximately unchanged after bounce
    vx_after = vx
    vy_after = vy

    # Time from bounce to reach robot_x
    dx_after = robot_x - x_bounce
    t_after = torch.where(
        vx_after > 0.1,
        dx_after / vx_after,
        torch.full_like(dx_after, 3.0),
    ).clamp(min=0.0)

    # Predicted position at robot_x
    pred_z_bounce = table_z + vz_after * t_after - 0.5 * gravity * t_after ** 2
    pred_y_bounce = y_bounce + vy_after * t_after
    pred_t_bounce = t_bounce + t_after

    # Case 2: ball already rising after bounce (bz > table_z, vz > 0, on robot's half)
    # Direct parabolic prediction
    t_direct = torch.where(
        vx > 0.1,
        dx / vx,
        torch.full_like(dx, 3.0),
    ).clamp(min=0.0)
    pred_z_direct = bz + vz * t_direct - 0.5 * gravity * t_direct ** 2
    pred_y_direct = by + vy * t_direct

    # Select which case applies
    already_bounced = (bx > 0.0) & (vz > 0.0) & (bz > table_z - 0.05)
    needs_bounce = moving_toward & (~already_bounced) & disc_valid

    # Apply predictions
    pred_y = torch.where(already_bounced & moving_toward, pred_y_direct, pred_y)
    pred_z = torch.where(already_bounced & moving_toward, pred_z_direct, pred_z)
    pred_t = torch.where(already_bounced & moving_toward, t_direct, pred_t)

    pred_y = torch.where(needs_bounce, pred_y_bounce, pred_y)
    pred_z = torch.where(needs_bounce, pred_z_bounce, pred_z)
    pred_t = torch.where(needs_bounce, pred_t_bounce, pred_t)

    pred_t = pred_t.clamp(0.0, 3.0)
    pred_z = pred_z.clamp(table_z, 2.0)

    return torch.stack([pred_y, pred_z, pred_t], dim=-1)


def ball_bounce_state(
    env: ManagerBasedEnv,
    ball_name: str,
    table_z: float = 0.76,
    bounce_z_thresh: float = 0.05,
) -> torch.Tensor:
    """Ball bounce state indicator (3D):
      [has_bounced_on_own_side, is_rising_after_bounce, urgency]

    - has_bounced: 1.0 if ball is on robot's half (x > 0) and vz > 0 and z is near/above table
    - is_rising: 1.0 if ball is going up after bounce
    - urgency: 1.0 - normalized time to reach robot (0=far, 1=imminent)
    """
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    bx, bz = ball_pos[:, 0], ball_pos[:, 2]
    vx, vz = ball_vel[:, 0], ball_vel[:, 2]

    on_own_side = bx > 0.0
    near_table = (bz - table_z).abs() < bounce_z_thresh
    is_rising = vz > 0.0
    moving_toward = vx > 0.1

    has_bounced = (on_own_side & is_rising & (bz < table_z + 0.3)).float()
    rising = (is_rising & on_own_side).float()

    # Urgency: based on time to reach robot
    dx = 1.5 - bx
    time_to_arrive = torch.where(vx > 0.1, dx / vx, torch.full_like(dx, 3.0)).clamp(0.0, 3.0)
    urgency = (1.0 - time_to_arrive / 3.0).clamp(0.0, 1.0)

    return torch.stack([has_bounced, rising, urgency], dim=-1)
