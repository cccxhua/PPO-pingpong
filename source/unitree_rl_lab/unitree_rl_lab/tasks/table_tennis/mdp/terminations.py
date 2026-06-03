from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from unitree_rl_lab.tasks.table_tennis.mdp.commands import UpperBodyMotionCommand


def bad_torso_orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, limit_angle: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = quat_apply_inverse(asset.data.root_quat_w, asset.data.GRAVITY_VEC_W)
    return torch.acos((-projected_gravity[:, 2]).clamp(-1, 1)) > limit_angle


def joint_state_nan(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    pos_nan = torch.any(torch.isnan(robot.data.joint_pos), dim=-1)
    vel_nan = torch.any(torch.isnan(robot.data.joint_vel), dim=-1)
    return pos_nan | vel_nan


def base_y_out_of_bounds(
    env: ManagerBasedRLEnv, y_limit: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    base_y = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.abs(base_y) > y_limit


def ball_out_of_play(
    env: ManagerBasedRLEnv, ball_name: str, z_min: float = 0.0, x_limit: float = 3.0
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_name]
    ball_pos_local = ball.data.root_pos_w - env.scene.env_origins
    z_bad = ball_pos_local[:, 2] < z_min
    x_bad = torch.abs(ball_pos_local[:, 0]) > x_limit
    return z_bad | x_bad


def ball_landed_on_own_table(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str = "motion",
    table_z: float = 0.76,
    table_x_min: float = 0.0,
    table_x_max: float = 1.37,
    table_y_half: float = 0.7625,
    z_margin: float = 0.08,
) -> torch.Tensor:
    """End episode when ball (after racket contact) actually lands on own table side.

    Triggers only when ball_was_hit=True AND ball is in a narrow band at table top
    (table_z .. table_z+z_margin) AND moving downward AND on own side. The downward +
    narrow-band conditions (aligned with the ball_land_on_own_table reward) avoid the
    earlier false positives where a returned ball flying over own-side airspace dipped
    below a single z threshold mid-flight and was wrongly counted as a landing.
    """
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    ball_vel = ball.data.root_lin_vel_w

    near_table = (ball_pos[:, 2] > table_z) & (ball_pos[:, 2] < table_z + z_margin)
    on_own_side = (ball_pos[:, 0] > table_x_min) & (ball_pos[:, 0] < table_x_max)
    in_bounds_y = ball_pos[:, 1].abs() < table_y_half
    going_down = ball_vel[:, 2] < 0

    return near_table & on_own_side & in_bounds_y & going_down & command.ball_was_hit


def ball_missed_paddle(
    env: ManagerBasedRLEnv,
    ball_name: str,
    command_name: str = "motion",
    robot_x: float = 1.5,
    margin: float = 0.15,
    robot_side: int = 1,
) -> torch.Tensor:
    """End episode when ball flies past the robot without being hit.

    Catches the "swing-and-miss" failure mode where paddle never contacts the ball,
    so ball_landed_on_own_table (which requires ball_was_hit) doesn't trigger.
    Without this, ref play stalls until the ball drops below z<0.5 (relaunch_ball_if_out).
    """
    command: UpperBodyMotionCommand = env.command_manager.get_term(command_name)
    ball: RigidObject = env.scene[ball_name]
    ball_pos = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    past_robot = (ball_pos[:, 0] - robot_x) * robot_side > margin
    return past_robot & ~command.ball_was_hit
