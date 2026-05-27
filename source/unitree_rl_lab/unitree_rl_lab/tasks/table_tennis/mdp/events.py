from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from unitree_rl_lab.tasks.table_tennis.mdp.commands import UpperBodyMotionCommand


def launch_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-1.0, -0.2),
    y_range: tuple[float, float] = (-0.5, 0.5),
    z_range: tuple[float, float] = (1.0, 1.5),
    vx_range: tuple[float, float] = (2.0, 4.0),
    vy_range: tuple[float, float] = (-0.5, 0.5),
    vz_range: tuple[float, float] = (0.0, 2.0),
):
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    ball: RigidObject = env.scene[ball_cfg.name]
    num = len(env_ids)

    pos = torch.zeros(num, 3, device=env.device)
    pos[:, 0] = sample_uniform(*x_range, (num,), device=env.device)
    pos[:, 1] = sample_uniform(*y_range, (num,), device=env.device)
    pos[:, 2] = sample_uniform(*z_range, (num,), device=env.device)
    pos += env.scene.env_origins[env_ids]

    quat = torch.zeros(num, 4, device=env.device)
    quat[:, 0] = 1.0

    vel = torch.zeros(num, 3, device=env.device)
    vel[:, 0] = sample_uniform(*vx_range, (num,), device=env.device)
    vel[:, 1] = sample_uniform(*vy_range, (num,), device=env.device)
    vel[:, 2] = sample_uniform(*vz_range, (num,), device=env.device)

    ang_vel = torch.zeros(num, 3, device=env.device)

    root_state = torch.cat([pos, quat, vel, ang_vel], dim=-1)
    ball.write_root_state_to_sim(root_state, env_ids=env_ids)


def relaunch_ball_if_out(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    z_min: float = 0.5,
    x_limit: float = 3.0,
    table_z: float = 0.745,
    slow_thresh: float = 0.5,
    x_range: tuple[float, float] = (0.3, 1.0),
    y_range: tuple[float, float] = (-0.5, 0.5),
    z_range: tuple[float, float] = (0.9, 1.2),
    vx_range: tuple[float, float] = (1.5, 3.0),
    vy_range: tuple[float, float] = (-0.3, 0.3),
    vz_range: tuple[float, float] = (-4.0, -2.0),
):
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_local = ball.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    ball_vel = ball.data.root_lin_vel_w[env_ids]

    z_bad = ball_pos_local[:, 2] < z_min
    x_bad = torch.abs(ball_pos_local[:, 0]) > x_limit
    slow_low = (ball_pos_local[:, 2] < table_z) & (torch.norm(ball_vel, dim=-1) < slow_thresh)

    out_mask = z_bad | x_bad | slow_low
    if not torch.any(out_mask):
        return

    out_ids = env_ids[out_mask]
    command: UpperBodyMotionCommand = env.command_manager.get_term("motion")
    command.ball_was_hit[out_ids] = False
    command.swing_done[out_ids] = False
    launch_ball(env, out_ids, ball_cfg, x_range, y_range, z_range, vx_range, vy_range, vz_range)

    # 重新对齐 motion phase: 让新球的过网时刻对应 phase=0, 球到达 robot 时对应 hit_phase
    duration = command.motion.motions[0]["duration"]
    aligned_phase = (command.cfg.hit_phase - command.cfg.ball_arrive_time_est / duration) % 1.0
    command.phase[out_ids] = aligned_phase
    command.phase_speed[out_ids] = 1.0


def reset_robot_on_rail(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    fixed_x: float = 1.5,
    fixed_z: float = 0.76,
):
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = env.scene.env_origins[env_ids, 0] + fixed_x
    root_state[:, 1] = env.scene.env_origins[env_ids, 1] + command.ref_base_y[env_ids]
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + fixed_z
    root_state[:, 3] = 0.0
    root_state[:, 4:6] = 0.0
    root_state[:, 6] = 1.0
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, command.upper_body_joint_ids] = command.ref_dof[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def track_ball_hit(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    ball_name: str,
    command_name: str = "motion",
    proximity_threshold: float = 0.25,
):
    """Interval event: detect racket-ball contact and set ball_was_hit flag."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # 检查整个 history 内的最大力, 否则瞬时接触 (1-2 substep) 会因仅看最新帧而漏判
    force_magnitude = torch.norm(net_forces, dim=-1).max(dim=1).values.squeeze(-1)

    ball: RigidObject = env.scene[ball_name]
    robot = env.scene["robot"]
    # sensor body_ids 与 robot body_ids 索引顺序不同 (sensor 按字母重排), 不能混用.
    # 用 sensor 的 body_names[body_ids[0]] 拿到名字, 再到 robot.body_names 里查实际索引.
    racket_name = getattr(sensor_cfg, "_cached_racket_body_name", None)
    if racket_name is None:
        racket_name = contact_sensor.body_names[sensor_cfg.body_ids[0]]
        sensor_cfg._cached_racket_body_name = racket_name
    robot_body_idx = robot.body_names.index(racket_name)
    racket_pos = robot.data.body_pos_w[:, robot_body_idx]
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.norm(racket_pos - ball_pos, dim=-1)

    hit = (force_magnitude > 0.1) & (dist < proximity_threshold)
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    command.ball_was_hit = command.ball_was_hit | hit
