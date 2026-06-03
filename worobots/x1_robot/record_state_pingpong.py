#!/usr/bin/env python3
"""
record_state_pingpong.py

乒乓球专用状态记录节点，只记录右臂 7 关节状态，无夹爪。
以 30 Hz 定时发布。

发布话题: x1/recorded_joint_states_pingpong  (sensor_msgs/JointState, 7 joints)
    - position : 全零（无主臂指令）
    - velocity : 右臂从臂关节实际位置 (来自 /joint_states 的 joint1-r ~ joint7-r)
    - effort   : 全零

订阅话题:
  /joint_states    (sensor_msgs/JointState)  — 从臂关节实际位置
"""

import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


RIGHT_ARM_JOINT_NAMES = [
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
]

# /joint_states 中右臂关节名称 → 输出索引（0~6）
RIGHT_ARM_SOURCE_NAMES = {
    "joint1-r": 0,
    "joint2-r": 1,
    "joint3-r": 2,
    "joint4-r": 3,
    "joint5-r": 4,
    "joint6-r": 5,
    "joint7-r": 6,
}

PUBLISH_HZ = 30.0


class RightArmStateRecorder(Node):
    def __init__(self):
        super().__init__("right_arm_state_recorder")

        self.last_log_time = 0.0
        self.last_state_time = -1.0

        # 频率监测
        self.target_hz = PUBLISH_HZ
        self.freq_window_size = 10
        self.joint_state_timestamps: deque[float] = deque(maxlen=self.freq_window_size)
        self.freq_check_interval = 2.0
        self.last_freq_check_time = time.time()

        # 右臂从臂关节实际位置 (来自 /joint_states)
        self.right_arm_state: list[float] = [0.0] * 7
        self.has_state = False

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # 发布者
        self.joint_state_pub = self.create_publisher(
            JointState, "x1/recorded_joint_states_pingpong", qos_profile
        )

        # 订阅者
        self.create_subscription(
            JointState, "/joint_states", self.state_callback, qos_profile
        )

        # 30 Hz 定时器
        self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"RightArmStateRecorder started @ {PUBLISH_HZ} Hz. "
            "Publishing to x1/recorded_joint_states_pingpong"
        )

    # ------------------------------------------------------------------ #
    #  频率监测
    # ------------------------------------------------------------------ #
    def calculate_frequency(self, timestamps: list[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0

    def check_and_log_frequencies(self) -> None:
        current_time = time.time()
        if current_time - self.last_freq_check_time < self.freq_check_interval:
            return
        self.last_freq_check_time = current_time

        freq = self.calculate_frequency(list(self.joint_state_timestamps))
        if 0 < freq < self.target_hz:
            self.get_logger().warn(
                f"⚠️  Joint state frequency too low: {freq:.1f} Hz (target: {self.target_hz} Hz)"
            )

    # ------------------------------------------------------------------ #
    #  回调
    # ------------------------------------------------------------------ #
    def state_callback(self, msg: JointState) -> None:
        """从 /joint_states 取右臂从臂实际位置。"""
        current_time = time.time()
        self.joint_state_timestamps.append(current_time)

        if self.last_state_time >= 0 and current_time - self.last_state_time > 1.0:
            self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_state_time = current_time

        for name, position in zip(msg.name, msg.position):
            if name in RIGHT_ARM_SOURCE_NAMES:
                self.right_arm_state[RIGHT_ARM_SOURCE_NAMES[name]] = float(position)

        self.has_state = True
        self.check_and_log_frequencies()

    def timer_callback(self) -> None:
        """组装并发布 JointState 消息（30 Hz）。"""
        if not self.has_state:
            self.get_logger().warn("No joint state data yet, skipping.")
            return

        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = RIGHT_ARM_JOINT_NAMES

        # position = 全零（仅记录状态时无主臂指令）
        msg.position = [0.0] * 7
        # velocity = 从臂关节实际位置 (state)
        msg.velocity = list(map(float, self.right_arm_state))
        msg.effort = [0.0] * 7

        self.joint_state_pub.publish(msg)

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"Timestamp: {timestamp_ms} ms")
            self.get_logger().info(
                f"Right arm state (velocity): {[f'{v:.4f}' for v in msg.velocity]}"
            )

    def destroy_node(self):
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RightArmStateRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
