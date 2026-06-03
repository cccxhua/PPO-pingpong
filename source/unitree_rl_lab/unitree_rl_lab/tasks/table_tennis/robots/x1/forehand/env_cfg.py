"""X1 table tennis environment configuration.

使用参考动作 (residual + phase speed) 模式。
X1 结构: 固定底盘 + 升降柱(锁定) + 7DOF右臂(yb) + 球拍(Link_yb_paddle)
仅右臂 7DOF 可控，通过 residual 叠加在参考动作上。

回退基线: 2026-05-12_04-10-52 (历史唯一 ball_hit 非零的从头训运行)
保留改进: ball USD + 全局 CCD + racket_approach 负区间惩罚后退。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

import unitree_rl_lab.tasks.table_tennis.mdp as mdp
from unitree_rl_lab.assets.robots.x1 import X1_TABLE_TENNIS_CFG

ROBOT_CFG = X1_TABLE_TENNIS_CFG

RIGHT_ARM_JOINT_NAMES = [
    "joint_yb_1",
    "joint_yb_2",
    "joint_yb_3",
    "joint_yb_4",
    "joint_yb_5",
    "joint_yb_6",
    "joint_yb_7",
]

RACKET_BODY_NAME = "Link_yb_paddle"

FOREHAND_DATA_DIR = os.path.dirname(__file__)

EASY_BALL = {
    "x_range": (-0.33, -0.37),
    "y_range": (0.0, 0.0),
    "z_range": (1.28, 1.32),
    "vx_range": (3.3, 3.7),
    "vy_range": (-0.2, 0.2),
    "vz_range": (0.3, 0.7),
}

TRAIN_BALL = {
    "x_range": (-0.28, -0.42),
    "y_range": (-0.10, 0.10),
    "z_range": (1.23, 1.37),
    "vx_range": (2.9, 4.1),
    "vy_range": (-0.35, 0.35),
    "vz_range": (0.1, 0.9),
}

##
# Scene
##


@configclass
class X1TableTennisSceneCfg(InteractiveSceneCfg):

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
                friction_combine_mode="max",
                restitution_combine_mode="max",
                static_friction=0.35,
                dynamic_friction=0.25,
                restitution=0.905,
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
        spawn=sim_utils.UsdFileCfg(
            usd_path="/root/x1/ping_pong_ball.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                linear_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 1.2)),
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
# MDP
##


@configclass
class ActionsCfg:
    """Residual control on right arm + phase speed."""
    right_arm = mdp.ReferenceResidualJointActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        command_name="motion",
        residual_scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        action_delay_steps_min=0,
        action_delay_steps_max=2,
    )
    phase_speed = mdp.PhaseSpeedActionCfg(
        asset_name="robot",
        command_name="motion",
        speed_min=0.85,  # 收紧范围, 避免 policy 慢到破坏 ball relaunch 时序 (历史: [0.5, 1.5])
        speed_max=1.15,
    )


@configclass
class CommandsCfg:
    motion = mdp.UpperBodyMotionCommandCfg(
        asset_name="robot",
        motion_files=[
            os.path.join(FOREHAND_DATA_DIR, "forehand_middle.npz"),
            os.path.join(FOREHAND_DATA_DIR, "forehand_left.npz"),
            os.path.join(FOREHAND_DATA_DIR, "forehand_right_v52.npz"),
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        base_y_noise_range=(0.0, 0.0),
        fixed_base=True,
        hit_phase=0.475,  # v58: 真实 sim 球到达 paddle 时刻 (probe 实测最近接 t=0.480 gap 4.75cm).
        hit_phase_noise=0.05,
        ball_arrive_time_est=0.55,  # 延迟 swing 使拍面在接触瞬间朝上 (damping=0, 实测过网)
        ball_arrive_time_noise=0.05,  # ±50ms 随机化, 避免依赖精确时间估计
        match_ball_direction=True,
        ball_y_threshold=0.05,
        axis_flip_indices=None,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        ball_pos_relative = ObsTerm(
            func=mdp.ball_pos_relative, params={"ball_name": "ball"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        ball_vel_relative = ObsTerm(
            func=mdp.ball_vel_relative, params={"ball_name": "ball"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        ball_spin = ObsTerm(
            func=mdp.ball_spin_zero,
            params={"ball_name": "ball"},
        )
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME}
        )
        motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        ball_time_to_arrive = ObsTerm(
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": 1.5}
        )
        ball_predicted_hit = ObsTerm(
            func=mdp.ball_predicted_hit_point,
            params={"ball_name": "ball", "robot_x": 1.5},
        )
        ball_bounce = ObsTerm(
            func=mdp.ball_bounce_state, params={"ball_name": "ball"}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
        )
        ball_pos_relative = ObsTerm(func=mdp.ball_pos_relative, params={"ball_name": "ball"})
        ball_vel_relative = ObsTerm(func=mdp.ball_vel_relative, params={"ball_name": "ball"})
        ball_spin = ObsTerm(func=mdp.ball_spin_zero, params={"ball_name": "ball"})
        racket_pos = ObsTerm(
            func=mdp.racket_pos, params={"racket_body_name": RACKET_BODY_NAME}
        )
        racket_ori = ObsTerm(
            func=mdp.racket_ori, params={"racket_body_name": RACKET_BODY_NAME}
        )
        motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
        ball_time_to_arrive = ObsTerm(
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": 1.5}
        )
        ball_predicted_hit = ObsTerm(
            func=mdp.ball_predicted_hit_point,
            params={"ball_name": "ball", "robot_x": 1.5},
        )
        ball_bounce = ObsTerm(
            func=mdp.ball_bounce_state, params={"ball_name": "ball"}
        )
        last_action = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # ============================================================
    # Stage 2: 解锁 residual, 加重击球奖励, pose 降为锚点
    #   - residual_scale 放回 0.4 (见 ActionsCfg)
    #   - pose_tracking 1.5 → 0.5 (仍保留, 防漂移)
    #   - 球相关奖励恢复并适度加重
    # 上一版 (Stage 1) 配置:
    #   pose_tracking=1.5, vel_tracking=0.3, residual_scale=0.15...
    # ============================================================

    # -- 模仿奖励: pose 锚点 (V66 ref motion 已能 1/2 过网, 加大跟踪权重让 policy 紧贴 ref)
    pose_tracking = RewTerm(
        func=mdp.upper_body_pose_tracking_exp,
        weight=0.80,  # V66 retrain: 0.50 → 1.50. ref 已是好动作, 加大模仿权重让 policy 学会 V66 整套挥拍.
        params={"command_name": "motion", "sigma": 0.3},
    )
    vel_tracking = RewTerm(
        func=mdp.upper_body_vel_tracking_exp,
        weight=0.50,  # V66 retrain: 0.30 → 0.90
        params={"command_name": "motion", "sigma": 0.2},
    )
    # joint2_tracking: 禁用
    # joint2_tracking = RewTerm(
    #     func=mdp.single_joint_tracking_exp,
    #     weight=0.00,
    #     params={"command_name": "motion", "joint_indices": [1], "sigma": 2.0},
    # )

    # -- 引导奖励: 球拍靠近球 (Stage 2 恢复)
    racket_ball_proximity = RewTerm(
        func=mdp.racket_ball_proximity,
        weight=0.60,  # Stage 2: 0.20 → 0.60
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "sigma": 0.3,
        },
    )
    racket_approach = RewTerm(
        func=mdp.racket_approach_ball_vel,
        weight=0.15,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "optimal_vel": 1.0,
            "sigma": 0.8,
        },
    )

    # -- 挥拍时机
    swing_timing = RewTerm(
        func=mdp.swing_timing_reward,
        weight=1.50,  # Fix G: 0.80 → 1.50, 推 contact 进窗口避免过晚相位
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "command_name": "motion",
            "hit_phase_start": 0.40,  # v58: 提前 0.05 (PIN 0.55→0.475)
            "hit_phase_end": 0.60,    # v58: 提前 0.10 (覆盖 PIN→snap)
            "proximity_gate": 0.6,
        },
    )

    # phase_ball_alignment: 禁用, swing_timing + racket_ball_proximity 已覆盖
    # phase_ball_alignment = RewTerm(
    #     func=mdp.phase_ball_alignment_reward,
    #     weight=0.0,
    #     params={...},
    # )

    # -- 挥拍引导: 拍面朝向 + 挥拍方向 (Fix J: 主升级 — pre-contact 几何条件)
    # 旧 weight 太低 (0.10 / 0.25), 训 600 iter 后 episode_reward 完全没动 (0.047 / 0.020).
    # 这两个是"球能反弹回去"的几何前提: 拍面对着对手 + 拍朝目标方向挥. 提到与 swing_timing
    # 同档, 让 policy 在接近球时主动调拍. 不动 pose_tracking 避免破坏 ref 锚定.
    # Fix K: gate 0.5 → 0.25, reward 集中在真正接触距离附近. 0.5m 时 alignment 高 ≠ 0.15m
    # 接触时 alignment 高 — 收紧后强制 policy 把 alignment 调到接触瞬间最优.
    racket_face_target = RewTerm(
        func=mdp.racket_face_toward_target,
        weight=1.50,  # Fix J: 0.10 → 1.50 (15×)
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "target_x": -0.7,
            "target_z": 0.9,
            "proximity_gate": 0.25,  # Fix K: 0.5 → 0.25
        },
    )
    racket_swing_ideal = RewTerm(
        func=mdp.racket_swing_toward_ideal,
        weight=1.50,
        params={
            "ball_name": "ball",
            "racket_body_name": RACKET_BODY_NAME,
            "target_x": -0.7,
            "target_z": 0.76,
            "proximity_gate": 0.25,
            "optimal_speed": 1.5,
            "sigma": 1.0,
        },
    )

    # -- 击球奖励 (Stage 2 加重, 推 policy 突破"接近但不击"局部最优)
    # Fix F (重平衡 hit 质量): ball_hit 退为事件标记, speed/direction 升为主要信号.
    # 历史: ball_hit(1.5) > speed(0.2)+dir(1.0) → policy 优先保 contact 率, 软碰也满分,
    # 没动力学硬挥. 改后: 软碰 ≈ 0.5, 硬挥 ≈ 0.5+1.5+2.0=4.0 → 8× 差距推 policy 学 peak 时机.
    ball_hit = RewTerm(
        func=mdp.ball_hit_reward,
        weight=0.50,  # 1.50 → 0.50 (Fix F)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "proximity_threshold": 0.15,
        },
    )
    ball_hit_speed = RewTerm(
        func=mdp.ball_speed_after_hit,
        weight=3.00,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "proximity_threshold": 0.15,
            "optimal_speed": 3.5,
            "sigma": 1.5,
        },
    )
    ball_hit_direction = RewTerm(
        func=mdp.ball_hit_toward_opponent,
        weight=0.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "proximity_threshold": 0.15,
            "optimal_vx": -3.0,
            "sigma": 1.5,
        },
    )
    # ball_toward_target: 禁用, 功能由 ball_land_placement 替代
    # ball_toward_target = RewTerm(
    #     func=mdp.ball_velocity_match_ideal,
    #     weight=0.30,
    #     params={...},
    # )

    # -- 回球奖励
    ball_return = RewTerm(
        func=mdp.ball_return_reward,
        weight=3.00,
        params={"ball_name": "ball", "command_name": "motion", "net_x": 0.0},
    )
    ball_land_opponent = RewTerm(
        func=mdp.ball_land_on_opponent_table,
        weight=2.00,
        params={"ball_name": "ball", "command_name": "motion"},
    )
    ball_land_placement = RewTerm(
        func=mdp.ball_land_placement_reward,
        weight=2.00,
        params={"ball_name": "ball", "command_name": "motion", "sigma_x": 0.7, "sigma_y": 0.7, "out_of_bounds_penalty": -0.5},
    )
    ball_land_own_table = RewTerm(
        func=mdp.ball_land_on_own_table,
        weight=-0.50,
        params={"ball_name": "ball", "command_name": "motion"},
    )

    # -- 正则化
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,  # 起步值, 抑制过大力矩利于 sim2real; 若挥拍速度被压制可再调小
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    phase_speed_reg = RewTerm(
        func=mdp.phase_speed_regularization,
        weight=-0.01,
        params={"command_name": "motion"},
    )
    self_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Link_yb_.*"]),
            "threshold": 1.0,
        },
    )


@configclass
class EventCfg:
    reset_ball = EventTerm(
        func=mdp.launch_ball,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            **TRAIN_BALL,
        },
    )
    relaunch_ball = EventTerm(
        func=mdp.relaunch_ball_if_out,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            **TRAIN_BALL,
        },
    )
    track_hit = EventTerm(
        func=mdp.track_ball_hit,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "command_name": "motion",
        },
    )
    randomize_gains = EventTerm(
        func=mdp.randomize_pd_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_range": (0.70, 1.30),
            "damping_range": (0.70, 1.30),
        },
    )
    randomize_effort = EventTerm(
        func=mdp.randomize_effort_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "effort_range": (0.90, 1.10),
        },
    )
    # TODO: PhysX tensor API set_masses/set_material_properties 兼容性问题, 暂时禁用
    # randomize_ball = EventTerm(
    #     func=mdp.randomize_ball_mass,
    #     mode="reset",
    #     params={
    #         "ball_cfg": SceneEntityCfg("ball"),
    #         "mass_range": (0.9, 1.1),
    #     },
    # )
    # randomize_table = EventTerm(
    #     func=mdp.randomize_table_physics,
    #     mode="reset",
    #     params={
    #         "table_cfg": SceneEntityCfg("table_surface"),
    #         "friction_range": (0.8, 1.2),
    #         "restitution_range": (0.85, 1.0),
    #     },
    # )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.joint_state_nan)
    ball_on_own_table = DoneTerm(
        func=mdp.ball_landed_on_own_table,
        params={"ball_name": "ball", "command_name": "motion"},
    )
    ball_missed_paddle = DoneTerm(
        func=mdp.ball_missed_paddle,
        params={"ball_name": "ball", "command_name": "motion", "robot_x": 1.5},
    )


##
# Environment
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: X1TableTennisSceneCfg = X1TableTennisSceneCfg(num_envs=2048, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    obs_delay_min: int = 0
    obs_delay_max: int = 3

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = True


class RobotPlayEnvCfg(RobotEnvCfg):

    zero_ball_spin: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.viewer.eye = (2.5, 2.2, 1.4)
        self.viewer.lookat = (1.0, 0.0, 1.0)
        self.commands.motion.hit_phase_noise = 0.0
        self.obs_delay_min = 0
        self.obs_delay_max = 0

        if self.zero_ball_spin:
            self.observations.policy.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
            self.observations.critic.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
