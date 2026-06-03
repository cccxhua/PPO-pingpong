#!/usr/bin/env python3
"""
play_action_pingpong.py

乒乓球专用动作播放节点，只控制右臂 7 关节，无夹爪。
以 30 Hz 定时发布最新的推理动作到机器人底层。

订阅话题:
  x1/sent_actions   (sensor_msgs/JointState)  — 推理输出的动作

发布话题:
  /record_data       (std_msgs/Float64MultiArray, 28 elements)  — 底层关节指令
      索引 14~20 为右臂 7 关节，其余为 0
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# 推理输出中的关节名 → /record_data 中的索引
RIGHT_ARM_JOINT_TO_INDEX = {
    "right_joint_1": 14,
    "right_joint_2": 15,
    "right_joint_3": 16,
    "right_joint_4": 17,
    "right_joint_5": 18,
    "right_joint_6": 19,
    "right_joint_7": 20,
}

PUBLISH_HZ = 30.0


class ActionPlayerPingPong(Node):
    def __init__(self):
        super().__init__("action_player_pingpong")

        self.last_log_time = 0.0
        # 缓存最新的关节指令（28 维，只填右臂 7 关节）
        self.joint_command: list[float] = [0.0] * 28
        self.has_action = False

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # 订阅推理输出
        self.create_subscription(
            JointState, "x1/sent_actions", self.action_callback, qos_profile
        )

        # 发布底层关节指令
        self.joint_commands_pub = self.create_publisher(
            Float64MultiArray, "/record_data", qos_profile
        )

        # 30 Hz 定时发布
        self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"ActionPlayerPingPong started @ {PUBLISH_HZ} Hz. "
            "Subscribing x1/sent_actions, publishing /record_data (right arm only)"
        )

    def action_callback(self, msg: JointState) -> None:
        """缓存最新的右臂 7 关节动作。"""
        for name, position in zip(msg.name, msg.position):
            if name in RIGHT_ARM_JOINT_TO_INDEX:
                self.joint_command[RIGHT_ARM_JOINT_TO_INDEX[name]] = float(position)
        self.has_action = True

    def timer_callback(self) -> None:
        """以 30 Hz 发布缓存的关节指令。"""
        if not self.has_action:
            return

        joint_msg = Float64MultiArray()
        joint_msg.data = list(self.joint_command)
        self.joint_commands_pub.publish(joint_msg)

        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            right_arm = self.joint_command[14:21]
            self.get_logger().info(f"right arm command: {[f'{v:.4f}' for v in right_arm]}")

    def destroy_node(self):
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ActionPlayerPingPong()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
