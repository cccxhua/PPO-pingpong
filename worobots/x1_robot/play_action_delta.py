#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
import time
import argparse
import sys
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ActionPlayer(Node):
    def __init__(self):
        super().__init__('action_player')

        self.last_log_time = 0
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(JointState, 'x1/sent_actions', self.action_callback, qos_profile)

        self.joint_commands_pub = self.create_publisher(Float64MultiArray, '/record_data', qos_profile)
        self.joint_delta_commands_pub = self.create_publisher(Float64MultiArray, '/record_data_delta', qos_profile)

        self.gripper_command_pub = self.create_publisher(Int32, '/joystick_info', qos_profile)

    def action_callback(self, msg: JointState):
        current_time = time.time() 
        joint_command = [0.0] * 28
        left_gripper_pos = 0.0
        right_gripper_pos = 0.0

        joint_name_to_index = {
            "left_joint_1": 0, "left_joint_2": 1, "left_joint_3": 2, "left_joint_4": 3, "left_joint_5": 4, "left_joint_6": 5, "left_joint_7": 6,
            "right_joint_1": 14, "right_joint_2": 15, "right_joint_3": 16, "right_joint_4": 17, "right_joint_5": 18, "right_joint_6": 19, "right_joint_7": 20
        }

        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                joint_command[joint_name_to_index[name]] = position
            elif name == "left_gripper":
                left_gripper_pos = position
            elif name == "right_gripper":
                right_gripper_pos = position
        
        left_gripper_pos = 0.99 if left_gripper_pos >= 1.0 else left_gripper_pos
        right_gripper_pos = 0.99 if right_gripper_pos >= 1.0 else right_gripper_pos

        left_gripper_pos = 0.0 if left_gripper_pos <= 0.0 else left_gripper_pos
        right_gripper_pos = 0.0 if right_gripper_pos <= 0.0 else right_gripper_pos

        # 发布关节控制命令
        joint_msg = Float64MultiArray()
        joint_msg.data = joint_command
        self.joint_commands_pub.publish(joint_msg)

        # 先发左手
        gripper_info = Int32()
        gripper_info.data = int(left_gripper_pos * 100.0 + 100.0)
        self.gripper_command_pub.publish(gripper_info)

        # 再发右手
        gripper_info = Int32()
        gripper_info.data = int(right_gripper_pos * 100.0 + 200.0)
        self.gripper_command_pub.publish(gripper_info)

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"publish joint commands:  {joint_command}")
            self.get_logger().info(f"publish left gripper command:  {left_gripper_pos}")
            self.get_logger().info(f"publish right gripper command:  {right_gripper_pos}")

    def destroy_node(self):
        # self.file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = ActionPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
