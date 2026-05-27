from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.table_tennis.mdp as mdp
from unitree_rl_lab.tasks.table_tennis.robots.g1_29dof.forehand.env_cfg import (
    UPPER_BODY_JOINT_NAMES,
    TableTennisSceneCfg,
    EventCfg,
    TerminationsCfg,
)

FOREHAND_DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "forehand"
)

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NON_RIGHT_ARM_JOINT_NAMES = [
    n for n in UPPER_BODY_JOINT_NAMES if n not in RIGHT_ARM_JOINT_NAMES
]

##
# Action: right arm = RL residual (7 DOF), waist+left arm = auto-track reference (10 DOF)
##


@configclass
class SacActionsCfg:
    right_arm = mdp.ReferenceResidualJointActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        command_name="motion",
        residual_scale=[0.5, 0.5, 0.5, 0.5, 0.3, 0.15, 0.15],
    )
    other_joints = mdp.ReferenceTrackingJointActionCfg(
        asset_name="robot",
        joint_names=NON_RIGHT_ARM_JOINT_NAMES,
        command_name="motion",
    )
    base_y_slider = mdp.BaseYSliderActionCfg(
        asset_name="robot",
        scale=0.2,
        fixed_x=1.5,
        fixed_z=0.76,
        y_min=-1.0,
        y_max=1.0,
    )
    phase_speed = mdp.PhaseSpeedActionCfg(
        asset_name="robot",
        command_name="motion",
        speed_min=0.2,
        speed_max=3.0,
    )

EASY_BALL = {
    "x_range": (-0.5, -0.3),
    "y_range": (-0.15, 0.15),
    "z_range": (1.25, 1.45),
    "vx_range": (2.0, 2.8),
    "vy_range": (-0.1, 0.1),
    "vz_range": (0.5, 1.2),
}

HARD_BALL = {
    "x_range": (-0.8, -0.4),
    "y_range": (-0.5, 0.5),
    "z_range": (1.2, 1.45),
    "vx_range": (2.8, 4.2),
    "vy_range": (-0.3, 0.3),
    "vz_range": (0.5, 1.3),
}


@configclass
class SacTerminationsCfg(TerminationsCfg):
    ball_out = None


@configclass
class SacEventCfg(EventCfg):
    reset_ball = EventTerm(
        func=mdp.launch_ball,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            **EASY_BALL,
        },
    )
    relaunch_ball = EventTerm(
        func=mdp.relaunch_ball_if_out,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            **EASY_BALL,
        },
    )
    track_hit = EventTerm(
        func=mdp.track_ball_hit,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "command_name": "motion",
        },
    )


@configclass
class CurriculumCfg:
    ball_difficulty = CurrTerm(
        func=mdp.ball_difficulty_curriculum,
        params={
            "ramp_steps": 200_000,
            "easy_x": EASY_BALL["x_range"],
            "easy_y": EASY_BALL["y_range"],
            "easy_z": EASY_BALL["z_range"],
            "easy_vx": EASY_BALL["vx_range"],
            "easy_vy": EASY_BALL["vy_range"],
            "easy_vz": EASY_BALL["vz_range"],
            "hard_x": HARD_BALL["x_range"],
            "hard_y": HARD_BALL["y_range"],
            "hard_z": HARD_BALL["z_range"],
            "hard_vx": HARD_BALL["vx_range"],
            "hard_vy": HARD_BALL["vy_range"],
            "hard_vz": HARD_BALL["vz_range"],
        },
    )


@configclass
class SacRewardsCfg:
    # -- imitation rewards (dominant: robot must learn swing motion first)
    pose_tracking = RewTerm(
        func=mdp.upper_body_pose_tracking_exp,
        weight=0.50,
        params={"command_name": "motion", "sigma": 0.2},
    )
    vel_tracking = RewTerm(
        func=mdp.upper_body_vel_tracking_exp,
        weight=0.20,
        params={"command_name": "motion", "sigma": 0.1},
    )
    base_y_tracking = RewTerm(
        func=mdp.base_y_tracking_exp,
        weight=0.10,
        params={"command_name": "motion", "sigma": 0.5},
    )

    # -- task rewards
    racket_ball_proximity = RewTerm(
        func=mdp.racket_ball_proximity,
        weight=0.20,
        params={
            "ball_name": "ball",
            "racket_body_name": "right_wrist_yaw_link",
            "sigma": 0.3,
        },
    )
    racket_approach = RewTerm(
        func=mdp.racket_approach_ball_vel,
        weight=0.08,
        params={
            "ball_name": "ball",
            "racket_body_name": "right_wrist_yaw_link",
            "max_vel": 2.0,
        },
    )
    ball_hit = RewTerm(
        func=mdp.ball_hit_reward,
        weight=0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "proximity_threshold": 0.25,
        },
    )
    ball_hit_direction = RewTerm(
        func=mdp.ball_hit_toward_opponent,
        weight=0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "proximity_threshold": 0.25,
            "max_vel": 3.0,
        },
    )
    ball_return = RewTerm(
        func=mdp.ball_return_reward,
        weight=0.10,
        params={"ball_name": "ball", "command_name": "motion", "net_x": 0.0},
    )
    ball_land_opponent = RewTerm(
        func=mdp.ball_land_on_opponent_table,
        weight=0.05,
        params={"ball_name": "ball"},
    )
    ball_land_own_table = RewTerm(
        func=mdp.ball_land_on_own_table,
        weight=-0.5,
        params={"ball_name": "ball", "command_name": "motion"},
    )

    # -- regularization (reduced for SAC: high-entropy exploration causes large penalties)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
    )
    phase_speed_reg = RewTerm(
        func=mdp.phase_speed_regularization,
        weight=-0.01,
        params={"command_name": "motion"},
    )


@configclass
class CommandsCfg:
    motion = mdp.UpperBodyMotionCommandCfg(
        asset_name="robot",
        motion_files=[
            os.path.join(FOREHAND_DATA_DIR, "forehand_upper.npz"),
            os.path.join(FOREHAND_DATA_DIR, "backhand_upper.npz"),
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        base_y_noise_range=(-0.02, 0.02),
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        upper_body_joint_pos_rel = ObsTerm(
            func=mdp.upper_body_joint_pos_rel,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        upper_body_joint_vel = ObsTerm(
            func=mdp.upper_body_joint_vel,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        base_y_pos = ObsTerm(func=mdp.base_y_pos)
        base_y_vel = ObsTerm(func=mdp.base_y_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        ball_pos_relative = ObsTerm(func=mdp.ball_pos_relative, params={"ball_name": "ball"})
        ball_vel_relative = ObsTerm(func=mdp.ball_vel_relative, params={"ball_name": "ball"})
        ball_spin = ObsTerm(
            func=mdp.ball_spin_relative,
            params={"ball_name": "ball"},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        ball_time_to_arrive = ObsTerm(
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": 1.5}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        upper_body_joint_pos_rel = ObsTerm(
            func=mdp.upper_body_joint_pos_rel, params={"command_name": "motion"}
        )
        upper_body_joint_vel = ObsTerm(
            func=mdp.upper_body_joint_vel, params={"command_name": "motion"}
        )
        base_y_pos = ObsTerm(func=mdp.base_y_pos)
        base_y_vel = ObsTerm(func=mdp.base_y_vel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        ball_pos_relative = ObsTerm(func=mdp.ball_pos_relative, params={"ball_name": "ball"})
        ball_vel_relative = ObsTerm(func=mdp.ball_vel_relative, params={"ball_name": "ball"})
        ball_spin = ObsTerm(func=mdp.ball_spin_relative, params={"ball_name": "ball"})
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_ori = ObsTerm(
            func=mdp.racket_ori, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        ball_time_to_arrive = ObsTerm(
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": 1.5}
        )
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RobotSacEnvCfg(ManagerBasedRLEnvCfg):
    scene: TableTennisSceneCfg = TableTennisSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: SacActionsCfg = SacActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SacRewardsCfg = SacRewardsCfg()
    terminations: SacTerminationsCfg = SacTerminationsCfg()
    events: SacEventCfg = SacEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class RobotSacPlayEnvCfg(RobotSacEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
