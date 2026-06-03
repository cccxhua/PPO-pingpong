"""A1 table tennis environment configuration.

基于 X1 任务，替换为 A1 手臂。
结构: 固定底盘 + 升降柱(锁定) + 7DOF右臂(yb) + 球拍(Link_yb_paddle)
仅右臂 7DOF 可控，通过 residual 叠加在参考动作上。
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
from unitree_rl_lab.assets.robots.a1 import A1_TABLE_TENNIS_CFG

ROBOT_CFG = A1_TABLE_TENNIS_CFG

TABLE_USD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
    "data", "robots", "a1", "table_tennis_table.usd"
)

# 乒乓球资源: 与球桌同目录, 用相对路径避免硬编码绝对路径
BALL_USD_PATH = os.path.join(os.path.dirname(TABLE_USD_PATH), "ping_pong_ball.usd")

# 分阶段域随机化开关 (累积式), 由环境变量 DR_STAGE 控制:
#   0 = 确定性基线 (无随机化)   1 = + 发球范围   2 = + 动作延迟   3 = + PD/力矩/观测噪声/观测延迟
DR_STAGE = int(os.environ.get("DR_STAGE", "0"))

# Robot side: +1 = robot at +X (original), -1 = robot at -X (flipped)
ROBOT_SIDE = -1
ROBOT_X = 1.5 * ROBOT_SIDE
ROBOT_BASE_X = 1.7 * ROBOT_SIDE
OWN_TABLE_X = (min(0.0, 1.37 * ROBOT_SIDE), max(0.0, 1.37 * ROBOT_SIDE))
OPP_TABLE_X = (min(0.0, -1.37 * ROBOT_SIDE), max(0.0, -1.37 * ROBOT_SIDE))
TARGET_X = -0.7 * ROBOT_SIDE
OPTIMAL_VX = -3.0 * ROBOT_SIDE

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
    "x_range": (min(-0.37 * ROBOT_SIDE, -0.33 * ROBOT_SIDE), max(-0.37 * ROBOT_SIDE, -0.33 * ROBOT_SIDE)),
    "y_range": (0.0, 0.0),
    "z_range": (1.28, 1.32),
    "vx_range": (min(2.3 * ROBOT_SIDE, 2.6 * ROBOT_SIDE), max(2.3 * ROBOT_SIDE, 2.6 * ROBOT_SIDE)),
    "vy_range": (-0.2, 0.2),
    "vz_range": (0.1, 0.3),
}

TRAIN_BALL = {
    "x_range": (min(-0.42 * ROBOT_SIDE, -0.28 * ROBOT_SIDE), max(-0.42 * ROBOT_SIDE, -0.28 * ROBOT_SIDE)),
    "y_range": (-0.10, 0.10),
    "z_range": (1.23, 1.37),
    "vx_range": (min(2.0 * ROBOT_SIDE, 2.9 * ROBOT_SIDE), max(2.0 * ROBOT_SIDE, 2.9 * ROBOT_SIDE)),
    "vy_range": (-0.35, 0.35),
    "vz_range": (0.05, 0.35),
}

# Fixed "middle" serve preset (play_pure_ref.py middle), single-point ranges = no randomization.
# ROBOT_SIDE=-1 flips X vs X1: ball spawns at +X (x=0.35) and travels -X (vx=-3.0) toward A1.
MIDDLE_BALL = {
    "x_range": (-0.35 * ROBOT_SIDE, -0.35 * ROBOT_SIDE),
    "y_range": (0.0, 0.0),
    "z_range": (1.10, 1.10),
    "vx_range": (3.0 * ROBOT_SIDE, 3.0 * ROBOT_SIDE),
    "vy_range": (0.0, 0.0),
    "vz_range": (0.2, 0.2),
}

# 以 MIDDLE_BALL 为中心向外随机 (第 1 步发球随机化):
# 中幅扩大档: 纵深 x±0.15、vx 2.6~3.4, 高度 z 0.90~1.30、vz 渐进, 横向 y±0.25 / vy±0.35.
# 配合修正后的 ball_landed_on_own_table 终止判定 (不再误杀回球过网低空帧)。
# ROBOT_SIDE=-1: x 在 +X 出生, vx 朝 -X 飞向 A1。
MIDDLE_DR_BALL = {
    "x_range":  (min(-0.50 * ROBOT_SIDE, -0.20 * ROBOT_SIDE), max(-0.50 * ROBOT_SIDE, -0.20 * ROBOT_SIDE)),
    "y_range":  (-0.25, 0.25),
    "z_range":  (0.90, 1.30),
    "vx_range": (min(2.6 * ROBOT_SIDE, 3.4 * ROBOT_SIDE), max(2.6 * ROBOT_SIDE, 3.4 * ROBOT_SIDE)),
    "vy_range": (-0.35, 0.35),
    "vz_range": (0.10, 0.30),
}

# whip_high3 serve: moderate-high/slow ball that arrives near apex in the paddle's
# far-reach zone. Mirrors play_pure_ref.py "high" preset, paired with the v5 whip
# motion + hit_phase 0.54 + ball_arrive_time_est 0.70 (clean high contact z~1.02).
HIGH_BALL = {
    "x_range": (-0.35 * ROBOT_SIDE, -0.35 * ROBOT_SIDE),
    "y_range": (0.0, 0.0),
    "z_range": (1.16, 1.16),
    "vx_range": (2.45 * ROBOT_SIDE, 2.45 * ROBOT_SIDE),
    "vy_range": (0.0, 0.0),
    "vz_range": (0.55, 0.55),
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

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BALL_USD_PATH,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0 * ROBOT_SIDE, 0.0, 1.2)),
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
        action_delay_steps_max=0,  # 关闭所有随机化: 无动作延迟
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
            os.path.join(FOREHAND_DATA_DIR, "forehand_middle_a1_whip.npz"),
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        base_y_noise_range=(0.0, 0.0),
        fixed_base=True,
        hit_phase=0.54,  # 对齐 pure ref play (--hit_phase 0.54)
        hit_phase_noise=0.0,  # 关闭所有随机化: 纯模仿确定性环境
        ball_arrive_time_est=0.51,  # 对齐 pure ref play (--arrive_time 0.51, middle ball)
        ball_arrive_time_noise=0.0,  # 关闭所有随机化
        match_ball_direction=True,
        ball_y_threshold=0.05,
        robot_x=ROBOT_X,
        robot_side=ROBOT_SIDE,
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
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE}
        )
        ball_predicted_hit = ObsTerm(
            func=mdp.ball_predicted_hit_point,
            params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE},
        )
        ball_bounce = ObsTerm(
            func=mdp.ball_bounce_state, params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # 关闭所有随机化: 无观测噪声
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
            func=mdp.ball_time_to_arrive, params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE}
        )
        ball_predicted_hit = ObsTerm(
            func=mdp.ball_predicted_hit_point,
            params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE},
        )
        ball_bounce = ObsTerm(
            func=mdp.ball_bounce_state, params={"ball_name": "ball", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE}
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
        weight=0.8,  # 模仿为主, 对齐参考 run 2026-05-19_10-01-14
        params={"command_name": "motion", "sigma": 0.3},
    )
    vel_tracking = RewTerm(
        func=mdp.upper_body_vel_tracking_exp,
        weight=0.80,  # 对齐参考 run 2026-05-19_10-01-14
        params={"command_name": "motion", "sigma": 0.2},
    )
    # joint2_tracking 已移除: yb2(joint_yb_2) 对 A1 不重要, 无需单独跟踪

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
            "target_x": TARGET_X,
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
            "target_x": TARGET_X,
            "target_z": 0.76,
            "proximity_gate": 0.25,
            "optimal_speed": 1.5,
            "sigma": 1.0,
            "robot_side": ROBOT_SIDE,
        },
    )

    # -- 击球奖励 (Stage 2 加重, 推 policy 突破"接近但不击"局部最优)
    # Fix F (重平衡 hit 质量): ball_hit 退为事件标记, speed/direction 升为主要信号.
    # 历史: ball_hit(1.5) > speed(0.2)+dir(1.0) → policy 优先保 contact 率, 软碰也满分,
    # 没动力学硬挥. 改后: 软碰 ≈ 0.5, 硬挥 ≈ 0.5+1.5+2.0=4.0 → 8× 差距推 policy 学 peak 时机.
    ball_hit = RewTerm(
        func=mdp.ball_hit_reward,
        weight=1.50,  # 1.50 → 0.50 (Fix F)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[RACKET_BODY_NAME]),
            "ball_name": "ball",
            "proximity_threshold": 0.15,
        },
    )
    ball_hit_speed = RewTerm(
        func=mdp.ball_speed_after_hit,
        weight=1.50,
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
            "optimal_vx": OPTIMAL_VX,
            "sigma": 1.5,
            "robot_side": ROBOT_SIDE,
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
        weight=2.00,  # 对齐参考 run 2026-05-19_10-01-14
        params={"ball_name": "ball", "command_name": "motion", "net_x": 0.0, "robot_side": ROBOT_SIDE},
    )
    ball_land_opponent = RewTerm(
        func=mdp.ball_land_on_opponent_table,
        weight=2.00,  # 对齐参考 run 2026-05-19_10-01-14
        params={"ball_name": "ball", "command_name": "motion",
                "table_x_min": OPP_TABLE_X[0], "table_x_max": OPP_TABLE_X[1]},
    )
    ball_land_placement = RewTerm(
        func=mdp.ball_land_placement_reward,
        weight=2.00,
        params={"ball_name": "ball", "command_name": "motion", "sigma_x": 0.7, "sigma_y": 0.7, "out_of_bounds_penalty": -0.5,
                "table_x_min": OPP_TABLE_X[0], "table_x_max": OPP_TABLE_X[1]},
    )
    ball_land_own_table = RewTerm(
        func=mdp.ball_land_on_own_table,
        weight=-0.50,
        params={"ball_name": "ball", "command_name": "motion",
                "table_x_min": OWN_TABLE_X[0], "table_x_max": OWN_TABLE_X[1]},
    )

    # -- 正则化
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
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
            **MIDDLE_BALL,
        },
    )
    relaunch_ball = EventTerm(
        func=mdp.relaunch_ball_if_out,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            **MIDDLE_BALL,
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
    # 关闭所有随机化: PD 增益 / 力矩域随机化禁用 (见 RobotEnvCfg.__post_init__)
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
        params={"ball_name": "ball", "command_name": "motion",
                "table_x_min": OWN_TABLE_X[0], "table_x_max": OWN_TABLE_X[1]},
    )
    ball_missed_paddle = DoneTerm(
        func=mdp.ball_missed_paddle,
        params={"ball_name": "ball", "command_name": "motion", "robot_x": ROBOT_X, "robot_side": ROBOT_SIDE},
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
    obs_delay_max: int = 1

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.enable_ccd = True
        self.scene.robot.init_state.pos = (ROBOT_BASE_X, 0.0, 0.0)
        if ROBOT_SIDE < 0:
            self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # 分阶段域随机化 (DR_STAGE 累积式, 见文件顶部):
        #   0 = 确定性基线   1 = + 发球范围   2 = + 动作延迟   3 = + PD/力矩/观测噪声/观测延迟
        # 第 1 步: 发球范围 (middle 为中心, 横向放大)
        if DR_STAGE >= 1:
            self.events.reset_ball.params.update(MIDDLE_DR_BALL)
            self.events.relaunch_ball.params.update(MIDDLE_DR_BALL)
        # 第 2 步: 动作延迟 (min=0,max=1 → 0~20ms, 均值 10ms ≈ 真实延迟)
        self.actions.right_arm.action_delay_steps_max = 1 if DR_STAGE >= 2 else 0
        # 第 3 步: PD 增益 / 力矩 / 观测噪声 / 观测延迟 (obs_delay min=0,max=1 → 均值 10ms)
        if DR_STAGE >= 3:
            self.observations.policy.enable_corruption = True
            self.obs_delay_max = 1
        else:
            self.events.randomize_gains = None
            self.events.randomize_effort = None
            self.observations.policy.enable_corruption = False
            self.obs_delay_max = 0


class RobotPlayEnvCfg(RobotEnvCfg):

    zero_ball_spin: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.viewer.eye = (-2.5, -2.0, 1.4)
        self.viewer.lookat = (-1.0, 0.0, 1.0)
        self.commands.motion.hit_phase_noise = 0.0
        self.actions.right_arm.action_delay_steps_max = 0  # 回放确定性: 无动作延迟
        self.obs_delay_min = 0
        self.obs_delay_max = 0

        if self.zero_ball_spin:
            self.observations.policy.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
            self.observations.critic.ball_spin = ObsTerm(
                func=mdp.ball_spin_zero, params={"ball_name": "ball"},
            )
