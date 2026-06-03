#!/usr/bin/env python3
"""
test_inference_dry_run.py

干跑测试: 订阅真实传感器数据，跑策略推理，只打印关节目标，不发指令给电机。
用于真机部署前验证策略输出是否合理。

用法:
    ros2 run <package> test_inference_dry_run
    或
    python test_inference_dry_run.py [--policy <path>]
"""

from __future__ import annotations

import os
import sys
import time
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from x1_table_tennis_deploy import (
    X1TableTennisController,
    DEFAULT_JOINT_POS,
    RIGHT_ARM_JOINT_NAMES,
    JOINT_LIMITS,
    ROBOT_X,
    STEP_DT,
    HIT_PHASE,
)

from ppo_inference_pingpong import (
    RIGHT_ARM_SOURCE_NAMES,
    DEFAULT_POLICY_PATH,
    CONTROL_HZ,
    RallyState,
)

LIMIT_LO = np.array([JOINT_LIMITS[n][0] for n in RIGHT_ARM_JOINT_NAMES])
LIMIT_HI = np.array([JOINT_LIMITS[n][1] for n in RIGHT_ARM_JOINT_NAMES])


class DryRunNode(Node):
    """干跑测试节点: 只推理 + 打印，不发指令。"""

    RECOVER_DURATION = 0.5

    def __init__(self):
        super().__init__("test_inference_dry_run")

        self.declare_parameter("policy_path", DEFAULT_POLICY_PATH)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("ball_incoming_vx_threshold", 1.0)
        self.declare_parameter("ball_incoming_x_threshold", 1.2)

        policy_path = self.get_parameter("policy_path").value
        device = self.get_parameter("device").value
        self._ball_vx_thresh = self.get_parameter("ball_incoming_vx_threshold").value
        self._ball_x_thresh = self.get_parameter("ball_incoming_x_threshold").value

        self.get_logger().info(f"[DRY RUN] Loading policy: {policy_path}")
        self.controller = X1TableTennisController(
            policy_path=policy_path, device=device
        )

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
        self._ball_last_time = 0.0

        self._state = RallyState.IDLE
        self._recover_t = 0.0
        self._recover_start_pos = None
        self._default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._last_target = self._default_pos.copy()
        self._swing_count = 0
        self._step_count = 0
        self._max_delta = 0.0
        self._limit_violations = 0

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

        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        self.get_logger().info(
            f"[DRY RUN] Node started @ {CONTROL_HZ} Hz. "
            "NO commands will be sent to motors."
        )

    def _joint_states_cb(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    idx = RIGHT_ARM_SOURCE_NAMES[name]
                    self._joint_pos[idx] = float(pos)
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

        if ball_age > 1.0:
            has_ball = False

        racket_pos = self._racket_pos.copy()

        # 坐标变换: 部署X轴与训练相反
        ball_pos[0] = -ball_pos[0]
        ball_vel[0] = -ball_vel[0]
        ball_ang_vel[1] = -ball_ang_vel[1]
        ball_ang_vel[2] = -ball_ang_vel[2]
        racket_pos[0] = -racket_pos[0]

        target = self._run_state_machine(
            joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos, has_ball
        )

        # 检查
        self._step_count += 1
        delta = np.max(np.abs(target - joint_pos))
        self._max_delta = max(self._max_delta, delta)
        within = np.all(target >= LIMIT_LO - 0.01) and np.all(target <= LIMIT_HI + 0.01)
        if not within:
            self._limit_violations += 1

        # 每 0.5 秒打印一次详细信息
        if self._step_count % 25 == 0:
            state_names = ["IDLE", "TRACKING", "SWINGING", "RECOVERING"]
            phase = self.controller.phase_machine.phase
            self.get_logger().info(
                f"[DRY RUN] State={state_names[self._state]} | "
                f"phase={phase:.3f} | swings={self._swing_count} | "
                f"delta={delta:.4f} | limits={'OK' if within else 'FAIL'}\n"
                f"  ball_pos={ball_pos} | ball_vel={ball_vel} | has_ball={has_ball}\n"
                f"  target={np.array2string(target, precision=4)}\n"
                f"  joint ={np.array2string(joint_pos, precision=4)}"
            )

        # 关键事件: 超限警告
        if not within:
            violations = []
            for i, n in enumerate(RIGHT_ARM_JOINT_NAMES):
                if target[i] < LIMIT_LO[i] - 0.01:
                    violations.append(f"{n}: {target[i]:.3f} < {LIMIT_LO[i]:.3f}")
                if target[i] > LIMIT_HI[i] + 0.01:
                    violations.append(f"{n}: {target[i]:.3f} > {LIMIT_HI[i]:.3f}")
            self.get_logger().warn(f"[DRY RUN] LIMIT VIOLATION: {violations}")

        if delta > 0.3:
            self.get_logger().warn(
                f"[DRY RUN] Large step delta={delta:.4f} rad, may cause motor jerk"
            )

    def _run_state_machine(
        self, joint_pos, joint_vel, ball_pos, ball_vel, ball_ang_vel, racket_pos, has_ball
    ) -> np.ndarray:
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

    def _ball_incoming(self, ball_pos, ball_vel) -> bool:
        return (
            ball_vel[0] > self._ball_vx_thresh
            and ball_pos[0] < self._ball_x_thresh
            and ball_pos[2] > 0.5
        )

    def _start_tracking(self, ball_pos, ball_vel):
        self.controller.reset(ball_pos, ball_vel)
        self._state = RallyState.TRACKING
        self.get_logger().info(
            f"[DRY RUN] Ball incoming! vx={ball_vel[0]:.2f} pos={ball_pos} → TRACKING"
        )

    def _start_recovery(self, current_pos):
        self._state = RallyState.RECOVERING
        self._recover_t = 0.0
        self._recover_start_pos = current_pos.copy()

    def destroy_node(self):
        self.get_logger().info(
            f"[DRY RUN] Summary: steps={self._step_count} | "
            f"swings={self._swing_count} | max_delta={self._max_delta:.4f} | "
            f"limit_violations={self._limit_violations}"
        )
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DryRunNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
