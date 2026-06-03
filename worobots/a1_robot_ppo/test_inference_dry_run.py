#!/usr/bin/env python3
"""
test_inference_dry_run.py

A1 干跑测试: 订阅真实传感器数据，跑策略推理，只打印关节目标，不发指令给电机。
用于真机部署前验证策略输出是否合理。

用法:
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

from a1_table_tennis_deploy import (
    A1TableTennisController,
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
    """A1 干跑测试节点: 只推理 + 打印，不发指令。"""

    RECOVER_DURATION = 0.5

    def __init__(self):
        super().__init__("test_inference_dry_run_a1")

        self.declare_parameter("policy_path", DEFAULT_POLICY_PATH)
        self.declare_parameter("device", "cpu")

        policy_path = self.get_parameter("policy_path").value
        device = self.get_parameter("device").value

        self.get_logger().info(f"[DRY RUN] Loading A1 policy: {policy_path}")
        self.controller = A1TableTennisController(
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
        self._racket_pos = np.array([-1.35, 0.0, 1.1], dtype=np.float32)
        self._has_joint = False
        self._has_ball = False
        self._ball_last_time = 0.0

        self._state = RallyState.IDLE
        self._recover_t = 0.0
        self._recover_start_pos = None
        self._default_pos = np.array(
            [DEFAULT_JOINT_POS[n] for n in RIGHT_ARM_JOINT_NAMES], dtype=np.float32
        )
        self._last_target = self._default_pos.copy()
        self._swing_count = 0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(JointState, "/right_joint_states", self._joint_cb, qos)
        self.create_subscription(PoseStamped, "/kalman/pingpong_pos", self._ball_pos_cb, qos)
        self.create_subscription(TwistStamped, "/kalman/pingpong_vel", self._ball_vel_cb, qos)
        self.create_subscription(PoseStamped, "/kalman/racket_pos", self._racket_cb, qos)

        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._loop)
        self._last_log = 0.0

        self.get_logger().info("[DRY RUN] A1 node ready. No commands will be sent.")

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    self._joint_pos[RIGHT_ARM_SOURCE_NAMES[name]] = float(pos)
            if msg.velocity:
                for name, vel in zip(msg.name, msg.velocity):
                    if name in RIGHT_ARM_SOURCE_NAMES:
                        self._joint_vel[RIGHT_ARM_SOURCE_NAMES[name]] = float(vel)
            self._has_joint = True

    def _ball_pos_cb(self, msg: PoseStamped):
        with self._lock:
            self._ball_pos[:] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self._has_ball = True
            self._ball_last_time = time.time()

    def _ball_vel_cb(self, msg: TwistStamped):
        with self._lock:
            self._ball_vel[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
            self._ball_ang_vel[:] = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]

    def _racket_cb(self, msg: PoseStamped):
        with self._lock:
            self._racket_pos[:] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def _loop(self):
        if not self._has_joint:
            return

        with self._lock:
            jp = self._joint_pos.copy()
            jv = self._joint_vel.copy()
            bp = self._ball_pos.copy()
            bv = self._ball_vel.copy()
            ba = self._ball_ang_vel.copy()
            has_ball = self._has_ball and (time.time() - self._ball_last_time < 1.0)

        rp = self._racket_pos.copy()

        # State machine (same as inference node)
        if self._state == RallyState.IDLE:
            self._last_target = self._default_pos.copy()
            if has_ball and bv[0] < -1.0 and bp[0] > ROBOT_X + 0.3 and bp[2] > 0.5:
                self.controller.reset(bp, bv)
                self._state = RallyState.TRACKING
                self.get_logger().info(f"[DRY] Ball incoming → TRACKING")

        elif self._state == RallyState.TRACKING:
            target, info = self.controller.step(jp, jv, bp, bv, ba, rp)
            self._last_target = target
            if info["phase"] > HIT_PHASE:
                self._state = RallyState.SWINGING

        elif self._state == RallyState.SWINGING:
            target, info = self.controller.step(jp, jv, bp, bv, ba, rp)
            self._last_target = target
            if self.controller.is_swing_done:
                self._swing_count += 1
                self._recover_start_pos = jp.copy()
                self._recover_t = 0.0
                self._state = RallyState.RECOVERING

        elif self._state == RallyState.RECOVERING:
            self._recover_t += STEP_DT
            alpha = min(1.0, self._recover_t / self.RECOVER_DURATION)
            a = 0.5 * (1.0 - np.cos(np.pi * alpha))
            self._last_target = (1 - a) * self._recover_start_pos + a * self._default_pos
            if alpha >= 1.0:
                self._state = RallyState.IDLE

        # Safety check
        target = self._last_target
        oob = np.any(target < LIMIT_LO - 0.05) or np.any(target > LIMIT_HI + 0.05)

        now = time.time()
        if now - self._last_log >= 0.5:
            self._last_log = now
            state_names = ["IDLE", "TRACKING", "SWINGING", "RECOVERING"]
            tgt_str = " ".join(f"{t:+.3f}" for t in target)
            self.get_logger().info(
                f"[DRY] {state_names[self._state]} | swings={self._swing_count} | "
                f"{'OOB!' if oob else 'OK'} | target=[{tgt_str}]"
            )


def main():
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
