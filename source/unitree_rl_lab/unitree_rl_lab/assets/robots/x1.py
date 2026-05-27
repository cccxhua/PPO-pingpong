"""X1 robot configuration for table tennis."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

X1_USD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "data", "robots", "x1", "ee_with_lift.usd"
)

# 参考实机电机规格 (yb_1~3 大关节, yb_4~7 小关节/腕)
X1_ARM_STIFFNESS = {
    "joint_yb_1": 250.0, "joint_yb_2": 250.0, "joint_yb_3": 250.0,
    "joint_yb_4": 150.0, "joint_yb_5": 150.0, "joint_yb_6": 150.0, "joint_yb_7": 150.0,
}
X1_ARM_DAMPING = {
    "joint_yb_1": 1.0, "joint_yb_2": 1.0, "joint_yb_3": 1.0,
    "joint_yb_4": 0.5, "joint_yb_5": 0.5, "joint_yb_6": 0.5, "joint_yb_7": 0.5,
}
X1_ARM_EFFORT = {
    "joint_yb_1": 28.0, "joint_yb_2": 28.0, "joint_yb_3": 28.0,
    "joint_yb_4": 8.0, "joint_yb_5": 8.0, "joint_yb_6": 8.0, "joint_yb_7": 8.0,
}
X1_ARM_VELOCITY = {
    "joint_yb_1": 8.0, "joint_yb_2": 8.0, "joint_yb_3": 8.0,
    "joint_yb_4": 20.0, "joint_yb_5": 20.0, "joint_yb_6": 20.0, "joint_yb_7": 20.0,
}
X1_LIFT_EFFORT = 1000.0    # N (升降柱力限制)
X1_LIFT_STIFFNESS = 5000.0
X1_LIFT_DAMPING = 500.0

X1_TABLE_TENNIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=X1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.7, 0.14, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # TODO: 设置 X1 打球初始姿态
            "joint_lift": -0.28,
            "joint_yb_1": 1.56,
            "joint_yb_2": -0.12,
            "joint_yb_3": -1.70,
            "joint_yb_4": 1.50,
            "joint_yb_5": 2.03,
            "joint_yb_6": 0.00,
            "joint_yb_7": -0.39,
            "joint_zb_1": 0.0,
            "joint_zb_2": 0.0,
            "joint_zb_3": 0.0,
            "joint_zb_4": -1.3,
            "joint_zb_5": 0.0,
            "joint_zb_6": 0.0,
            "joint_zb_7": 0.0,
            "joint_head_lr": 0.0,
            "joint_head_ud": 0.0,
            "joint_left_wheel": 0.0,
            "joint_right_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_yb_[1-7]"],
            effort_limit_sim=X1_ARM_EFFORT,
            velocity_limit_sim=X1_ARM_VELOCITY,
            stiffness=X1_ARM_STIFFNESS,
            damping=X1_ARM_DAMPING,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_zb_[1-7]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit_sim=X1_LIFT_EFFORT,
            velocity_limit_sim=0.0,
            stiffness=X1_LIFT_STIFFNESS,
            damping=X1_LIFT_DAMPING,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*_wheel"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)
