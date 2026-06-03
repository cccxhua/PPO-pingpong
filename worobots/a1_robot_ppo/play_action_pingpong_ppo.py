#!/usr/bin/env python3
"""
play_action_pingpong_ppo.py

A1 PPO 乒乓球专用动作播放节点，50Hz 控制频率（匹配训练）。
只控制右臂 7 关节，无夹爪。

订阅话题:
  a1/sent_actions   (sensor_msgs/JointState)  — PPO 推理输出的关节目标位置

发布话题:
  /record_data      (std_msgs/Float64MultiArray, 28 elements)  — 底层关节指令
      索引 14~20 为右臂 7 关节，其余为 0
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


RIGHT_ARM_JOINT_TO_INDEX = {
    "right_joint_1": 14,
    "right_joint_2": 15,
    "right_joint_3": 16,
    "right_joint_4": 17,
    "right_joint_5": 18,
    "right_joint_6": 19,
    "right_joint_7": 20,
}

PUBLISH_HZ = 100.0  # 100Hz 控制频率


class ActionPlayerPingPongPPO(Node):
    def __init__(self):
        super().__init__("action_player_pingpong_ppo_a1")

        self.last_log_time = 0.0
        self.joint_command: list[float] = [0.0] * 28
        self.has_action = False

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(
            JointState, "a1/sent_actions", self.action_callback, qos_profile
        )

        self.joint_commands_pub = self.create_publisher(
            Float64MultiArray, "/record_data", qos_profile
        )

        self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"A1 ActionPlayerPingPongPPO started @ {PUBLISH_HZ} Hz. "
            "Subscribing a1/sent_actions, publishing /record_data (right arm only)"
        )

    def action_callback(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in RIGHT_ARM_JOINT_TO_INDEX:
                self.joint_command[RIGHT_ARM_JOINT_TO_INDEX[name]] = float(position)
        self.has_action = True

    def timer_callback(self) -> None:
        if not self.has_action:
            return

        joint_msg = Float64MultiArray()
        joint_msg.data = list(self.joint_command)
        self.joint_commands_pub.publish(joint_msg)

        current_time = time.time()
        if current_time - self.last_log_time >= 2.0:
            self.last_log_time = current_time
            right_arm = self.joint_command[14:21]
            self.get_logger().info(f"Right arm targets: {[f'{v:.3f}' for v in right_arm]}")


def main() -> None:
    rclpy.init()
    node = ActionPlayerPingPongPPO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
