#!/usr/bin/env python3
"""
record_action_pingpong.py

只记录右臂（主臂指令位置 + 从臂实际位置），忽略左臂和夹爪。

发布话题: x1/recorded_joint_states_pingpong  (sensor_msgs/JointState, 7 joints)
    - position : 右臂主臂关节指令位置 (来自 /record_data 的 [14:21])
    - velocity : 右臂从臂关节实际位置 (来自 /joint_states 的 joint1-r ~ joint7-r)
    - effort   : 全零

订阅话题:
  /record_data     (std_msgs/Float64MultiArray)  — 主臂关节指令
  /joint_states    (sensor_msgs/JointState)       — 从臂关节实际位置
"""

import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
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


class RightArmRecorder(Node):
    def __init__(self):
        super().__init__("right_arm_recorder")

        self.last_log_time = 0.0
        self.last_action_time = -1.0
        self.last_state_time = -1.0

        # 右臂主臂关节指令位置 (来自 /record_data [14:21])
        self.right_arm_action: list[float] | None = None

        # 右臂从臂关节实际位置 (来自 /joint_states)
        self.right_arm_state: list[float] = [0.0] * 7

        # 发布者
        self.joint_state_pub = self.create_publisher(
            JointState, "x1/recorded_joint_states_pingpong", 10
        )

        # 订阅者
        self.create_subscription(
            Float64MultiArray,
            "/record_data",
            self.action_callback,
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.state_callback,
            10,
        )

        # 100 Hz 定时器
        self.create_timer(1.0 / 100.0, self.timer_callback)

        self.get_logger().info("RightArmRecorder started. Publishing to x1/recorded_joint_states_pingpong")

    def action_callback(self, msg: Float64MultiArray) -> None:
        """从 /record_data 取右臂主臂指令位置 (索引 14~20)。"""
        current_time = time.time()
        if self.last_action_time >= 0 and current_time - self.last_action_time > 1.0:
            self.get_logger().warn("No joint action received for more than 1 second.")
        self.last_action_time = current_time

        data = msg.data
        if len(data) < 21:
            self.get_logger().warn(
                f"/record_data has only {len(data)} elements, expected at least 21. Skipping."
            )
            return

        self.right_arm_action = list(data[14:21])

    def state_callback(self, msg: JointState) -> None:
        """从 /joint_states 取右臂从臂实际位置。"""
        current_time = time.time()
        if self.last_state_time >= 0 and current_time - self.last_state_time > 1.0:
            self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_state_time = current_time

        for name, position in zip(msg.name, msg.position):
            if name in RIGHT_ARM_SOURCE_NAMES:
                self.right_arm_state[RIGHT_ARM_SOURCE_NAMES[name]] = float(position)

    def timer_callback(self) -> None:
        """组装并发布 JointState 消息（100 Hz）。"""
        if self.right_arm_action is None or self.right_arm_state is None:
            self.get_logger().warn("Incomplete data, skipping this cycle.")
            return

        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = RIGHT_ARM_JOINT_NAMES

        # position = 主臂关节指令位置 (action)
        msg.position = list(map(float, self.right_arm_action))
        # velocity = 从臂关节实际位置 (state)
        msg.velocity = list(map(float, self.right_arm_state))
        msg.effort = [0.0] * 7

        self.joint_state_pub.publish(msg)

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"Timestamp: {timestamp_ms} ms")
            self.get_logger().info(f"Right arm action (position): {msg.position}")
            self.get_logger().info(f"Right arm state  (velocity): {msg.velocity}")

    def destroy_node(self):
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record right arm action and state for ping-pong robot.")
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = RightArmRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
