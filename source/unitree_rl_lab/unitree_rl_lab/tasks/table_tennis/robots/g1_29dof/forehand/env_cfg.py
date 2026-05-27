from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg

import unitree_rl_lab.tasks.table_tennis.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG

##
# Robot configuration: G1-29DOF with locked legs
##

ROBOT_CFG = UNITREE_G1_29DOF_MIMIC_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.5, 0.0, 0.76),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_hip_pitch_joint": -0.312,
            "right_hip_pitch_joint": -0.312,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.669,
            "right_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "right_ankle_pitch_joint": -0.363,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.6,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
)
ROBOT_CFG.actuators["legs"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_joint",
    ],
    effort_limit_sim=500.0,
    velocity_limit_sim=0.1,
    stiffness=10000.0,
    damping=1000.0,
    armature=0.01,
)
ROBOT_CFG.actuators["feet"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit_sim=500.0,
    velocity_limit_sim=0.1,
    stiffness=10000.0,
    damping=1000.0,
    armature=0.01,
)

##
# Upper body action scale (effort / stiffness * 0.25)
##

UPPER_BODY_ACTION_SCALE = {}
for _actuator_name in ["waist", "waist_yaw", "arms"]:
    _a = UNITREE_G1_29DOF_MIMIC_CFG.actuators[_actuator_name]
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for n in _names:
        if n in _e and n in _s and _s[n]:
            UPPER_BODY_ACTION_SCALE[n] = 0.25 * _e[n] / _s[n]

UPPER_BODY_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

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

##
# Scene definition
##


@configclass
class TableTennisSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG

    table_surface = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableSurface",
        spawn=sim_utils.CuboidCfg(
            size=(2.74, 1.525, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.6)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.745)),
    )

    net = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableNet",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.83, 0.1525),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76 + 0.1525 / 2)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.9,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.0, 1.35)),
    )

    racket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/Racket",
        spawn=sim_utils.CylinderCfg(
            radius=0.075,
            height=0.005,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.16), rot=(0.7071, 0.7071, 0.0, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=0.1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##

DATA_DIR = os.path.dirname(__file__)


@configclass
class CommandsCfg:
    motion = mdp.UpperBodyMotionCommandCfg(
        asset_name="robot",
        motion_files=[
            os.path.join(DATA_DIR, "forehand_upper.npz"),
            os.path.join(DATA_DIR, "backhand_upper.npz"),
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        base_y_noise_range=(-0.02, 0.02),
    )


@configclass
class ActionsCfg:
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
        ideal_hit_vel = ObsTerm(
            func=mdp.ideal_hit_velocity,
            params={"ball_name": "ball", "target_x": -0.7, "target_z": 0.76},
        )
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_normal = ObsTerm(
            func=mdp.racket_normal, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_vel = ObsTerm(
            func=mdp.racket_vel, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_hit_force = ObsTerm(
            func=mdp.racket_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"])},
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
        ideal_hit_vel = ObsTerm(
            func=mdp.ideal_hit_velocity,
            params={"ball_name": "ball", "target_x": -0.7, "target_z": 0.76},
        )
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_ori = ObsTerm(
            func=mdp.racket_ori, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_normal = ObsTerm(
            func=mdp.racket_normal, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_vel = ObsTerm(
            func=mdp.racket_vel, params={"racket_body_name": "right_wrist_yaw_link"}
        )
        racket_hit_force = ObsTerm(
            func=mdp.racket_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"])},
        )
        motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        ball_time_to_arrive = ObsTerm(
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": 1.5}
        )
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    reset_ball = EventTerm(
        func=mdp.launch_ball,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "x_range": (0.3, 1.0),
            "y_range": (-0.5, 0.5),
            "z_range": (0.9, 1.2),
            "vx_range": (1.5, 3.0),
            "vy_range": (-0.3, 0.3),
            "vz_range": (-4.0, -2.0),
        },
    )


@configclass
class PpoEventCfg(EventCfg):
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
class RewardsCfg:

    # -- imitation rewards (reduced: already learned, avoid dominating gradient)
    pose_tracking = RewTerm(
        func=mdp.upper_body_pose_tracking_exp,
        weight=0.05,
        params={"command_name": "motion", "sigma": 0.2},
    )
    vel_tracking = RewTerm(
        func=mdp.upper_body_vel_tracking_exp,
        weight=0.03,
        params={"command_name": "motion", "sigma": 0.1},
    )
    base_y_tracking = RewTerm(
        func=mdp.base_y_tracking_exp,
        weight=0.02,
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
        weight=0.10,
        params={
            "ball_name": "ball",
            "racket_body_name": "right_wrist_yaw_link",
            "max_vel": 2.0,
        },
    )
    ball_hit = RewTerm(
        func=mdp.ball_hit_reward,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "proximity_threshold": 0.25,
        },
    )
    ball_hit_direction = RewTerm(
        func=mdp.ball_hit_toward_opponent,
        weight=0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "proximity_threshold": 0.25,
            "max_vel": 3.0,
        },
    )
    racket_face_target = RewTerm(
        func=mdp.racket_face_toward_target,
        weight=0.10,
        params={
            "ball_name": "ball",
            "racket_body_name": "right_wrist_yaw_link",
            "target_x": -0.7,
            "target_z": 0.9,
            "proximity_gate": 1.5,
        },
    )
    racket_swing_ideal = RewTerm(
        func=mdp.racket_swing_toward_ideal,
        weight=0.25,
        params={
            "ball_name": "ball",
            "racket_body_name": "right_wrist_yaw_link",
            "target_x": -0.7,
            "target_z": 0.76,
            "proximity_gate": 1.5,
            "max_vel": 3.0,
        },
    )
    ball_toward_target = RewTerm(
        func=mdp.ball_velocity_match_ideal,
        weight=0.30,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "target_x": -0.7,
            "target_z": 0.76,
        },
    )
    ball_hit_speed = RewTerm(
        func=mdp.ball_speed_after_hit,
        weight=0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_wrist_yaw_link"]),
            "ball_name": "ball",
            "proximity_threshold": 0.25,
            "min_speed": 1.0,
            "max_speed": 5.0,
        },
    )
    ball_return = RewTerm(
        func=mdp.ball_return_reward,
        weight=1.50,
        params={"ball_name": "ball", "command_name": "motion", "net_x": 0.0},
    )
    ball_land_opponent = RewTerm(
        func=mdp.ball_land_on_opponent_table,
        weight=1.50,
        params={"ball_name": "ball"},
    )
    ball_land_own_table = RewTerm(
        func=mdp.ball_land_on_own_table,
        weight=-1.0,
        params={"ball_name": "ball", "command_name": "motion"},
    )

    # -- regularization
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_torso_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.5},
    )
    base_y_bounds = DoneTerm(
        func=mdp.base_y_out_of_bounds,
        params={"y_limit": 1.2},
    )
    ball_out = DoneTerm(
        func=mdp.ball_out_of_play,
        params={"ball_name": "ball", "z_min": 0.0, "x_limit": 3.0},
    )


@configclass
class PpoTerminationsCfg(TerminationsCfg):
    ball_out = None


@configclass
class CurriculumCfg:
    ball_difficulty = CurrTerm(
        func=mdp.ball_difficulty_curriculum,
        params={
            "ramp_steps": 1_000_000,
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


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: TableTennisSceneCfg = TableTennisSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: PpoTerminationsCfg = PpoTerminationsCfg()
    events: PpoEventCfg = PpoEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
