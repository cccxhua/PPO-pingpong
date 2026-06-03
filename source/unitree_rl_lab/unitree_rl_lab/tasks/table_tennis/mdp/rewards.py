from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate

from unitree_rl_lab.tasks.table_tennis.mdp.commands import UpperBodyMotionCommand
from unitree_rl_lab.tasks.table_tennis.mdp.observations import compute_ideal_ball_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RACKET_OFFSET_Z = 0.0  # X1: Link_yb_paddle 本身就是球拍, 无偏移. (历史: G1 腕→球拍 16cm)


def _racket_world_pos(robot, racket_body_name: str) -> torch.Tensor:
    body_idx = robot.body_names.index(racket_body_name)
    wrist_pos = robot.data.body_pos_w[:, body_idx]
    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_offset = torch.zeros_like(wrist_pos)
    local_offset[:, 2] = RACKET_OFFSET_Z
    return wrist_pos + quat_rotate(wrist_quat, local_offset)


def _racket_pos_from_sensor(env, sensor_cfg) -> torch.Tensor:
    # NOTE: sensor.body_names 与 robot.body_names 顺序不同 (sensor 按字母重排).
    # 用 sensor 的索引去查 robot.body_names 会拿到错误 link (历史 bug).
    # body_names 是 @property, 每次访问都做 PhysX view + 字符串解析, 缓存到 sensor_cfg 上避免每步重算.
    racket_body_name = getattr(sensor_cfg, "_cached_racket_body_name", None)
    if racket_body_name is None:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        racket_body_name = contact_sensor.body_names[sensor_cfg.body_ids[0]]
        sensor_cfg._cached_racket_body_name = racket_body_name
    return _racket_world_pos(env.scene["robot"], racket_body_name)


def upper_body_pose_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float
) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    cur_dof = command.robot_upper_body_joint_pos()
    error = torch.sum((cur_dof - command.ref_dof) ** 2, dim=-1)
    return torch.exp(-sigma * error)


def single_joint_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, joint_indices: List[int], sigma: float
) -> torch.Tensor:
    """Track specific joints in the upper body reference motion."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    cur_dof = command.robot_upper_body_joint_pos()
    idx = torch.tensor(joint_indices, device=cur_dof.device)
    error = torch.sum((cur_dof[:, idx] - command.ref_dof[:, idx]) ** 2, dim=-1)
    return torch.exp(-sigma * error)


def upper_body_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float
) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    cur_vel = command.robot_upper_body_joint_vel()
    error = torch.sum((cur_vel - command.ref_dof_vel) ** 2, dim=-1)
    return torch.exp(-sigma * error)


def base_y_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float
) -> torch.Tensor:
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    cur_base_y = env.scene["robot"].data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    error = (cur_base_y - command.ref_base_y) ** 2
    return torch.exp(-sigma * error)


def racket_ball_proximity(
    env: ManagerBasedRLEnv, ball_name: str, racket_body_name: str, sigma: float
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    error = torch.sum((racket_pos - ball_pos) ** 2, dim=-1)
    return torch.exp(-sigma * error)


def ball_hit_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    proximity_threshold: float = 0.15,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces[:, 0], dim=-1).squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    racket_pos = _racket_pos_from_sensor(env, sensor_cfg)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    return (force_magnitude > 0.1).float() * (dist < proximity_threshold).float()


def ball_return_reward(
    env: ManagerBasedRLEnv, ball_name: str, command_name: str = "motion", net_x: float = 0.0,
    net_z: float = 0.9125, robot_side: int = 1,
) -> torch.Tensor:
    """Only reward ball crossing net if it was hit AND is above net height (valid return)."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vx = ball.data.root_lin_vel_w[:, 0]
    crossing = ((ball_pos[:, 0] - net_x) * robot_side < 0) & (ball_vx * robot_side < -0.5)
    above_net = ball_pos[:, 2] > net_z
    return (crossing & above_net & command.ball_was_hit).float()


def ball_hit_toward_opponent(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    proximity_threshold: float = 0.25,
    optimal_vx: float = -3.0,
    sigma: float = 1.5,
    robot_side: int = 1,
) -> torch.Tensor:
    """Bell-curve direction reward: peaks when ball vx matches optimal return velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces[:, 0], dim=-1).squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    racket_pos = _racket_pos_from_sensor(env, sensor_cfg)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    hit = (force_magnitude > 0.1) & (dist < proximity_threshold)
    ball_vx = ball.data.root_lin_vel_w[:, 0]

    direction_reward = torch.exp(-((ball_vx - optimal_vx) ** 2) / (2 * sigma ** 2))
    direction_reward = torch.where(ball_vx * robot_side < 0, direction_reward, torch.zeros_like(direction_reward))
    return hit.float() * direction_reward


def ball_speed_after_hit(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    proximity_threshold: float = 0.25,
    optimal_speed: float = 3.5,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Bell-curve reward: peaks at optimal_speed, decays for too slow or too fast."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces[:, 0], dim=-1).squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    racket_pos = _racket_pos_from_sensor(env, sensor_cfg)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    hit = (force_magnitude > 0.1) & (dist < proximity_threshold)
    ball_speed = torch.norm(ball.data.root_lin_vel_w, dim=-1)
    speed_reward = torch.exp(-((ball_speed - optimal_speed) ** 2) / (2 * sigma ** 2))
    return hit.float() * speed_reward


def ball_land_on_opponent_table(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str = "motion",
    table_z: float = 0.76,
    table_x_min: float = -1.37,
    table_x_max: float = 0.0,
    table_y_half: float = 0.7625,
) -> torch.Tensor:
    """Reward ball landing on opponent table after racket hit (valid score)."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    near_table = (ball_pos[:, 2] > table_z) & (ball_pos[:, 2] < table_z + 0.08)
    on_opponent = (ball_pos[:, 0] > table_x_min) & (ball_pos[:, 0] < table_x_max)
    in_bounds_y = ball_pos[:, 1].abs() < table_y_half
    going_down = ball_vel[:, 2] < 0

    landing = near_table & on_opponent & in_bounds_y & going_down
    return (landing & command.ball_was_hit).float()


def ball_land_on_own_table(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str = "motion",
    table_z: float = 0.76,
    table_x_min: float = 0.0,
    table_x_max: float = 1.37,
    table_y_half: float = 0.7625,
) -> torch.Tensor:
    """Penalize ball landing on own table side after racket contact (illegal return)."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    near_table = (ball_pos[:, 2] > table_z) & (ball_pos[:, 2] < table_z + 0.08)
    on_own_side = (ball_pos[:, 0] > table_x_min) & (ball_pos[:, 0] < table_x_max)
    in_bounds_y = ball_pos[:, 1].abs() < table_y_half
    going_down = ball_vel[:, 2] < 0

    landing = near_table & on_own_side & in_bounds_y & going_down
    return (landing & command.ball_was_hit).float()


def ball_land_placement_reward(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str = "motion",
    table_z: float = 0.76,
    table_x_min: float = -1.37,
    table_x_max: float = 0.0,
    table_y_half: float = 0.7625,
    sigma_x: float = 0.7,
    sigma_y: float = 0.7,
    out_of_bounds_penalty: float = -1.0,
) -> torch.Tensor:
    """Shaped reward for ball landing placement: center=1, edge→0, out-of-bounds=penalty."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    near_table = (ball_pos[:, 2] > table_z) & (ball_pos[:, 2] < table_z + 0.08)
    going_down = ball_vel[:, 2] < 0
    landing = near_table & going_down & command.ball_was_hit

    center_x = (table_x_min + table_x_max) / 2.0
    center_y = 0.0
    half_x = (table_x_max - table_x_min) / 2.0

    dx = (ball_pos[:, 0] - center_x) / half_x
    dy = (ball_pos[:, 1] - center_y) / table_y_half

    in_x = (ball_pos[:, 0] > table_x_min) & (ball_pos[:, 0] < table_x_max)
    in_y = ball_pos[:, 1].abs() < table_y_half
    in_bounds = in_x & in_y

    placement_reward = torch.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
    reward = torch.where(in_bounds, placement_reward, torch.full_like(placement_reward, out_of_bounds_penalty))

    return landing.float() * reward


def _racket_world_vel(robot, racket_body_name: str) -> torch.Tensor:
    body_idx = robot.body_names.index(racket_body_name)
    wrist_vel = robot.data.body_lin_vel_w[:, body_idx]
    wrist_ang_vel = robot.data.body_ang_vel_w[:, body_idx]
    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_offset = torch.zeros_like(wrist_vel)
    local_offset[:, 2] = RACKET_OFFSET_Z
    world_offset = quat_rotate(wrist_quat, local_offset)
    return wrist_vel + torch.cross(wrist_ang_vel, world_offset, dim=-1)


def racket_approach_ball_vel(
    env: ManagerBasedRLEnv, ball_name: str, racket_body_name: str,
    optimal_vel: float = 1.0, sigma: float = 0.8,
) -> torch.Tensor:
    """Bell-curve reward: peaks at optimal approach velocity, decays for too slow or too fast."""
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]

    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]

    direction = ball_pos - racket_pos
    dist = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_norm = direction / dist

    racket_vel = _racket_world_vel(robot, racket_body_name)
    approach_vel = torch.sum(racket_vel * direction_norm, dim=-1)

    reward = torch.exp(-((approach_vel - optimal_vel) ** 2) / (2 * sigma ** 2))
    reward = torch.where(approach_vel > 0, reward, torch.zeros_like(reward))
    return reward


def racket_face_toward_target(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_x: float = -0.7,
    target_z: float = 0.9,
    proximity_gate: float = 0.5,
) -> torch.Tensor:
    """Dense reward: when ball is near, reward racket normal pointing toward target on opponent table."""
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    body_idx = robot.body_names.index(racket_body_name)

    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    target = torch.zeros_like(racket_pos)
    target[:, 0] = env.scene.env_origins[:, 0] + target_x
    target[:, 1] = env.scene.env_origins[:, 1]
    target[:, 2] = env.scene.env_origins[:, 2] + target_z

    desired_dir = target - racket_pos
    desired_dir = desired_dir / torch.norm(desired_dir, dim=-1, keepdim=True).clamp(min=1e-6)

    wrist_quat = robot.data.body_quat_w[:, body_idx]
    local_normal = torch.zeros(wrist_quat.shape[0], 3, device=wrist_quat.device)
    local_normal[:, 1] = 1.0
    racket_normal = quat_rotate(wrist_quat, local_normal)

    alignment = torch.sum(racket_normal * desired_dir, dim=-1).clamp(min=0.0)

    gate = (dist < proximity_gate).float()
    return gate * alignment


def ball_toward_target_after_hit(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    target_x: float = -0.7,
    target_z: float = 0.9,
    proximity_threshold: float = 0.25,
) -> torch.Tensor:
    """After contact, reward ball velocity pointing toward opponent table target."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces[:, 0], dim=-1).squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    racket_pos = _racket_pos_from_sensor(env, sensor_cfg)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    hit = (force_magnitude > 0.1) & (dist < proximity_threshold)

    target = torch.zeros(ball_pos.shape[0], 3, device=ball_pos.device)
    target[:, 0] = env.scene.env_origins[:, 0] + target_x
    target[:, 1] = env.scene.env_origins[:, 1]
    target[:, 2] = env.scene.env_origins[:, 2] + target_z

    desired_dir = target - ball_pos
    desired_dir = desired_dir / torch.norm(desired_dir, dim=-1, keepdim=True).clamp(min=1e-6)

    ball_vel = ball.data.root_lin_vel_w
    ball_speed = torch.norm(ball_vel, dim=-1, keepdim=True).clamp(min=1e-6)
    ball_dir = ball_vel / ball_speed

    alignment = torch.sum(ball_dir * desired_dir, dim=-1).clamp(min=0.0)
    return hit.float() * alignment


def ball_velocity_match_ideal(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    target_x: float = -0.7,
    target_z: float = 0.76,
    proximity_threshold: float = 0.25,
    speed_sigma: float = 2.0,
    robot_side: int = 1,
) -> torch.Tensor:
    """After contact, reward ball velocity matching the ideal parabolic trajectory."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.norm(net_forces[:, 0], dim=-1).squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    racket_pos = _racket_pos_from_sensor(env, sensor_cfg)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    hit = (force_magnitude > 0.1) & (dist < proximity_threshold)

    ball_pos_local = ball_pos - env.scene.env_origins
    ideal_vel = compute_ideal_ball_velocity(ball_pos_local, target_x, target_z, robot_side=robot_side)
    actual_vel = ball.data.root_lin_vel_w

    cos_sim = torch.nn.functional.cosine_similarity(actual_vel, ideal_vel, dim=-1)
    actual_speed = torch.norm(actual_vel, dim=-1)
    ideal_speed = torch.norm(ideal_vel, dim=-1)
    speed_match = torch.exp(-((actual_speed - ideal_speed) ** 2) / (speed_sigma ** 2))

    return hit.float() * cos_sim.clamp(min=0.0) * speed_match


def racket_swing_toward_ideal(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    target_x: float = -0.7,
    target_z: float = 0.76,
    proximity_gate: float = 0.5,
    optimal_speed: float = 1.5,
    sigma: float = 1.0,
    robot_side: int = 1,
) -> torch.Tensor:
    """Dense reward: when ball is near, reward racket swinging in the ideal hit direction at optimal speed."""
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]

    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    ball_pos_local = ball_pos - env.scene.env_origins
    ideal_vel = compute_ideal_ball_velocity(ball_pos_local, target_x, target_z, robot_side=robot_side)
    ideal_dir = ideal_vel / torch.norm(ideal_vel, dim=-1, keepdim=True).clamp(min=1e-6)

    racket_vel = _racket_world_vel(robot, racket_body_name)
    racket_speed = torch.norm(racket_vel, dim=-1)
    racket_dir = racket_vel / racket_speed.unsqueeze(-1).clamp(min=1e-6)

    alignment = torch.sum(racket_dir * ideal_dir, dim=-1).clamp(min=0.0)
    speed_reward = torch.exp(-((racket_speed - optimal_speed) ** 2) / (2 * sigma ** 2))

    gate = (dist < proximity_gate).float()
    return gate * alignment * speed_reward


def phase_speed_regularization(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize deviation from normal playback speed (1.0). Use with negative weight."""
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    return (command.phase_speed - 1.0) ** 2


def swing_timing_reward(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    command_name: str,
    hit_phase_start: float = 0.30,
    hit_phase_end: float = 0.55,
    proximity_gate: float = 0.6,
) -> torch.Tensor:
    """Dense reward for correct swing timing: reward when in hitting phase AND ball is close.

    This bridges the gap between "move toward ball" and "contact". It rewards the agent
    for being in the right phase of the swing when the ball is within striking range.
    """
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]

    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    in_hit_phase = (command.phase >= hit_phase_start) & (command.phase <= hit_phase_end)
    ball_close = dist < proximity_gate

    proximity_reward = torch.exp(-dist / 0.2)
    return in_hit_phase.float() * ball_close.float() * proximity_reward


def phase_ball_alignment_reward(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str,
    robot_x: float = 1.5,
    hit_phase: float = 0.43,
    sigma: float = 0.15,
    robot_side: int = 1,
) -> torch.Tensor:
    """Reward for phase being close to hit_phase when ball is approaching.

    Encourages the policy to use phase_speed to align the swing timing with ball arrival.
    The reward is: exp(-((phase - hit_phase)^2) / sigma^2) * urgency
    where urgency increases as ball gets closer.
    """
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]

    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    bx, vx = ball_pos[:, 0], ball_vel[:, 0]

    # Time for ball to reach robot
    t_arrive = torch.where(vx * robot_side > 0.1, (robot_x - bx) / vx, torch.full_like(bx, 3.0)).clamp(0.0, 3.0)

    # Urgency: high when ball is about to arrive (t < 0.5s)
    urgency = (1.0 - t_arrive / 1.0).clamp(0.0, 1.0)

    # Phase alignment: reward when phase is near hit_phase as ball approaches
    phase_diff = (command.phase - hit_phase).abs()
    phase_diff = torch.min(phase_diff, 1.0 - phase_diff)  # handle wrap-around
    phase_reward = torch.exp(-(phase_diff ** 2) / (sigma ** 2))

    return phase_reward * urgency


def racket_at_predicted_hit(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    robot_x: float = 1.5,
    sigma: float = 20.0,
    urgency_window: float = 0.5,
    z_offset: float = 0.10,
    robot_side: int = 1,
) -> torch.Tensor:
    """Reward racket being at the predicted ball arrival point, offset below by z_offset.

    Predicts where the ball will cross x=robot_x using ballistic motion, then rewards
    racket position matching (pred_x, pred_y, pred_z - z_offset). The downward offset
    aligns this reward with `racket_below_ball_when_close` so both pull the racket below
    the ball (forehand under-swing setup). Urgency ramps from 0 (ball >urgency_window s
    away) to 1 (ball about to arrive).
    """
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]

    ball_pos_local = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w
    bx, by, bz = ball_pos_local[:, 0], ball_pos_local[:, 1], ball_pos_local[:, 2]
    vx, vy, vz = ball_vel[:, 0], ball_vel[:, 1], ball_vel[:, 2]

    safe_vx = torch.where(vx * robot_side > 0.1, vx, torch.ones_like(vx) * robot_side)
    t_arrive = (robot_x - bx) / safe_vx
    valid = (vx * robot_side > 0.1) & (t_arrive > 0.0) & (t_arrive < 1.0)

    g = 9.81
    pred_y = by + vy * t_arrive
    pred_z = bz + vz * t_arrive - 0.5 * g * t_arrive * t_arrive - z_offset
    pred_x = torch.full_like(bx, robot_x)
    pred_pos_local = torch.stack([pred_x, pred_y, pred_z], dim=-1)
    pred_pos_world = pred_pos_local + env.scene.env_origins

    racket_pos = _racket_world_pos(robot, racket_body_name)
    dist_sq = torch.sum((racket_pos - pred_pos_world) ** 2, dim=-1)

    pos_reward = torch.exp(-sigma * dist_sq)
    urgency = (1.0 - t_arrive / urgency_window).clamp(0.0, 1.0)

    return valid.float() * urgency * pos_reward


def racket_below_ball_when_close(
    env: ManagerBasedRLEnv,
    ball_name: str,
    racket_body_name: str,
    gate: float = 0.4,
    optimal_below: float = 0.10,
    penalty_scale: float = 0.10,
) -> torch.Tensor:
    """Reward racket at `optimal_below` meters below ball, penalize on both sides.

    Triangular reward, range [-1, +1], peaks at z_below = optimal_below:
      - +1.0 at exactly optimal_below (e.g. 10cm below ball)
      -  0.0 at distance == penalty_scale from peak (e.g. at ball height, or 20cm below)
      - -1.0 at distance >= 2*penalty_scale from peak (clamped)

    Active only when 3D distance to ball is within `gate` meters.
    z_below = ball_z - racket_z (positive = racket below ball).
    """
    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]

    racket_pos = _racket_world_pos(robot, racket_body_name)
    ball_pos = ball.data.root_pos_w[:, :3]

    dist = torch.norm(racket_pos - ball_pos, dim=-1)
    z_below = ball_pos[:, 2] - racket_pos[:, 2]

    distance_from_optimal = (z_below - optimal_below).abs()
    score = 1.0 - distance_from_optimal / penalty_scale
    score = score.clamp(min=-1.0, max=1.0)

    gate_mask = (dist < gate).float()
    return gate_mask * score
