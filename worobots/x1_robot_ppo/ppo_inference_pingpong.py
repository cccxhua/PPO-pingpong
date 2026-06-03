#!/usr/bin/env python3
"""
ppo_inference_pingpong.py

PPO 乒乓球策略推理节点，50Hz 控制循环。
订阅关节状态和球感知话题，运行策略推理，发布关节位置指令。

订阅话题:
  /joint_states                (sensor_msgs/JointState)        — 右臂关节位置/速度
  /kalman/pingpong_pos         (geometry_msgs/PoseStamped)     — 球 3D 位置
  /kalman/pingpong_vel         (geometry_msgs/TwistStamped)    — 球 3D 速度 + 角速度
  /kalman/racket_pos           (geometry_msgs/PoseStamped)     — 球拍 3D 位置 (动捕)

发布话题:
  x1/sent_actions              (sensor_msgs/JointState)        — 推理输出的关节位置指令

注: ball_bounce 状态由 ball_pos + ball_vel 内部计算，无需外部话题。
"""

from __future__ import annotations

import os
import time
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# PPO deploy modules (from x1_table_tennis_deploy.py)
# ---------------------------------------------------------------------------

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from x1_table_tennis_deploy import (
    MotionLoader,
    PolicyRunner,
    PhaseStateMachine,
    ObservationAssembler,
    X1TableTennisController,
    process_action,
    DEFAULT_MOTION_FILES,
    RIGHT_ARM_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    ROBOT_X,
    STEP_DT,
    HIT_PHASE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 50.0  # 50Hz control loop (matches training)

# /joint_states 中右臂关节名称 → 内部索引 (0~6)
RIGHT_ARM_SOURCE_NAMES = {
    "joint1-r": 0,
    "joint2-r": 1,
    "joint3-r": 2,
    "joint4-r": 3,
    "joint5-r": 4,
    "joint6-r": 5,
    "joint7-r": 6,
}

# 推理输出中的关节名 (发布到 x1/sent_actions)
OUTPUT_JOINT_NAMES = [
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
]

# 默认策略路径
DEFAULT_POLICY_PATH = os.path.join(
    SCRIPT_DIR, "..", "..", "unitree_rl_lab", "logs", "rsl_rl",
    "x1_tabletennis", "2026-05-21_02-19-50", "exported", "policy.pt"
)
DEFAULT_POLICY_PATH = os.path.abspath(DEFAULT_POLICY_PATH)


# ---------------------------------------------------------------------------
# Rally State Machine
# ---------------------------------------------------------------------------

class RallyState:
    IDLE = 0        # 等待位姿，等球来
    TRACKING = 1    # 检测到球，phase 运行中
    SWINGING = 2    # 正在挥拍 (phase > HIT_PHASE)
    RECOVERING = 3  # 挥拍结束，回归等待位姿


class PPOPingPongNode(Node):
    """PPO 乒乓球部署推理节点。"""

    RECOVER_DURATION = 0.5  # 回归等待位姿的时间(秒)

    def __init__(self):
        super().__init__("ppo_inference_pingpong")

        # ---- Parameters ----
        self.declare_parameter("policy_path", DEFAULT_POLICY_PATH)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("ball_incoming_vx_threshold", 1.0)
        self.declare_parameter("ball_incoming_x_threshold", 1.2)

        policy_path = self.get_parameter("policy_path").value
        device = self.get_parameter("device").value
        self._ball_vx_thresh = self.get_parameter("ball_incoming_vx_threshold").value
        self._ball_x_thresh = self.get_parameter("ball_incoming_x_threshold").value

        # ---- PPO Controller ----
        self.get_logger().info(f"Loading PPO policy from: {policy_path}")
        self.controller = X1TableTennisController(
            policy_path=policy_path, device=device
        )

        # ---- State ----
        self._lock = Lock()
        self._joint_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._joint_vel = np.zeros(7, dtype=np.float32)
        self._ball_pos = np.zeros(3, dtype=np.float32)
        self._ball_vel = np.zeros(3, dtype=np.float32)
        self._ball_ang_vel = np.zeros(3, dtype=np.float32)
        self._racket_pos = np.array([1.35, 0.0, 1.1], dtype=np.float32)
        self._has_joint_data = False
        self._has_ball_data = False
        self._has_racket_data = False
        self._ball_last_time = 0.0

        # Rally state
        self._state = RallyState.IDLE
        self._recover_t = 0.0
        self._recover_start_pos = None
        self._default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._last_target = self._default_pos.copy()
        self._hit_count = 0
        self._swing_count = 0

        # ---- ROS2 Subscriptions ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, qos
        )
        self.create_subscription(
            PoseStamped, "/kalman/pingpong_pos", self._ball_pos_cb, qos
        )
        self.create_subscription(
            TwistStamped, "/kalman/pingpong_vel", self._ball_vel_cb, qos
        )
        self.create_subscription(
            PoseStamped, "/kalman/racket_pos", self._racket_pos_cb, qos
        )

        # ---- ROS2 Publisher ----
        self._action_pub = self.create_publisher(
            JointState, "x1/sent_actions", qos
        )

        # ---- 50Hz Control Timer ----
        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._control_loop)
        self._last_log_time = 0.0

        self.get_logger().info(
            f"PPO PingPong node started @ {CONTROL_HZ} Hz. "
            f"State: IDLE. Waiting for ball..."
        )

    # ------------------------------------------------------------------
    # ROS2 Callbacks
    # ------------------------------------------------------------------

    def _joint_states_cb(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    idx = RIGHT_ARM_SOURCE_NAMES[name]
                    self._joint_pos[idx] = float(pos)
            # 速度从 velocity 字段获取
            for name, vel in zip(msg.name, msg.velocity):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    idx = RIGHT_ARM_SOURCE_NAMES[name]
                    self._joint_vel[idx] = float(vel)
            self._has_joint_data = True

    def _ball_pos_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._ball_pos[0] = msg.pose.position.x
            self._ball_pos[1] = msg.pose.position.y
            self._ball_pos[2] = msg.pose.position.z
            self._has_ball_data = True
            self._ball_last_time = time.time()

    def _ball_vel_cb(self, msg: TwistStamped) -> None:
        with self._lock:
            self._ball_vel[0] = msg.twist.linear.x
            self._ball_vel[1] = msg.twist.linear.y
            self._ball_vel[2] = msg.twist.linear.z
            self._ball_ang_vel[0] = msg.twist.angular.x
            self._ball_ang_vel[1] = msg.twist.angular.y
            self._ball_ang_vel[2] = msg.twist.angular.z

    def _racket_pos_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._racket_pos[0] = msg.pose.position.x
            self._racket_pos[1] = msg.pose.position.y
            self._racket_pos[2] = msg.pose.position.z
            self._has_racket_data = True

    # ------------------------------------------------------------------
    # Control Loop (50Hz)
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        if not self._has_joint_data:
            return

        with self._lock:
            joint_pos = self._joint_pos.copy()
            joint_vel = self._joint_vel.copy()
            ball_pos = self._ball_pos.copy()
            ball_vel = self._ball_vel.copy()
            ball_ang_vel = self._ball_ang_vel.copy()
            has_ball = self._has_ball_data
            ball_age = time.time() - self._ball_last_time

        # 球数据超过 1 秒视为丢失
        if ball_age > 1.0:
            has_ball = False

        # 球拍位置 (来自动捕)
        racket_pos = self._racket_pos.copy()

        # ---- 坐标变换: 部署X轴与训练相反，取反X分量 ----
        ball_pos[0] = -ball_pos[0]
        ball_vel[0] = -ball_vel[0]
        ball_ang_vel[1] = -ball_ang_vel[1]
        ball_ang_vel[2] = -ball_ang_vel[2]
        racket_pos[0] = -racket_pos[0]

        # ---- Rally State Machine ----
        target = self._run_state_machine(
            joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos, has_ball
        )

        # ---- 发布关节指令 ----
        self._publish_action(target)

        # ---- 日志 ----
        now = time.time()
        if now - self._last_log_time >= 2.0:
            self._last_log_time = now
            state_names = ["IDLE", "TRACKING", "SWINGING", "RECOVERING"]
            phase = self.controller.phase_machine.phase
            self.get_logger().info(
                f"State={state_names[self._state]} | phase={phase:.3f} | "
                f"swings={self._swing_count} | ball_pos={ball_pos} | "
                f"has_ball={has_ball}"
            )

    def _run_state_machine(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        ball_ang_vel: np.ndarray,
        racket_pos: np.ndarray,
        has_ball: bool,
    ) -> np.ndarray:
        """Rally 状态机，返回关节目标位置 (7,)."""

        if self._state == RallyState.IDLE:
            self._last_target = self._default_pos.copy()
            if has_ball and self._ball_incoming(ball_pos, ball_vel):
                self._start_tracking(ball_pos, ball_vel)

        elif self._state == RallyState.TRACKING:
            target, info = self.controller.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_target = target

            if info["phase"] > HIT_PHASE:
                self._state = RallyState.SWINGING

        elif self._state == RallyState.SWINGING:
            target, info = self.controller.step(
                joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos
            )
            self._last_target = target

            if self.controller.is_swing_done:
                self._swing_count += 1
                self._start_recovery(joint_pos)

        elif self._state == RallyState.RECOVERING:
            self._recover_t += STEP_DT
            alpha = min(1.0, self._recover_t / self.RECOVER_DURATION)
            alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
            self._last_target = (
                (1 - alpha_smooth) * self._recover_start_pos
                + alpha_smooth * self._default_pos
            )
            if alpha >= 1.0:
                self._state = RallyState.IDLE

        return self._last_target

    def _ball_incoming(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
        """判断球是否正朝机器人飞来。"""
        return (
            ball_vel[0] > self._ball_vx_thresh
            and ball_pos[0] < self._ball_x_thresh
            and ball_pos[2] > 0.5
        )

    def _start_tracking(self, ball_pos: np.ndarray, ball_vel: np.ndarray):
        """从 IDLE 进入 TRACKING，初始化 phase。"""
        self.controller.reset(ball_pos, ball_vel)
        self._state = RallyState.TRACKING
        self.get_logger().info(
            f"Ball incoming! vx={ball_vel[0]:.2f} pos={ball_pos} → TRACKING"
        )

    def _start_recovery(self, current_pos: np.ndarray):
        """挥拍完成，开始回归。"""
        self._state = RallyState.RECOVERING
        self._recover_t = 0.0
        self._recover_start_pos = current_pos.copy()

    def _publish_action(self, targets: np.ndarray):
        """发布 7 个关节目标位置到 x1/sent_actions。"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = OUTPUT_JOINT_NAMES
        msg.position = [float(t) for t in targets]
        msg.velocity = []
        msg.effort = []
        self._action_pub.publish(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = PPOPingPongNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
