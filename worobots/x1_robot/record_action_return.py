#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8
from std_msgs.msg import Int32
import time
import argparse
import sys

class FullCommandRecorder(Node):
    def __init__(self, output_file):
        super().__init__('full_command_recorder')

        self.last_log_time = 0
        self.last_action_time = -1
        self.last_state_time = -1
        self.last_left_joint_time = -1
        self.last_right_joint_time = -1
        self.last_left_gripper_time = -1
        self.last_right_gripper_time = -1

        # self.file = open(output_file, 'w')
        # self.get_logger().info(f"Recording to {output_file}")

        self.joint_action = None
        self.left_gripper_action = 0
        self.right_gripper_action = 0

        self.joint_state = None
        self.left_gripper_state = 0
        self.right_gripper_state = 0

        self.joint_state_pub = self.create_publisher(JointState, 'x1/recorded_joint_states', 10)

        # self.create_subscription(
        #     Float64MultiArray,
        #     '/record_data',
        #     self.joint_action_callback,
        #     10
        # )
        self.create_subscription(
            JointState,
            '/left_joint_states',
            self.joint_left_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/right_joint_states',
            self.joint_right_callback,
            10
        )

        self.create_subscription(
            Int32,
            '/joystick_info',
            self.gripper_action_callback,
            10
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.create_subscription(
            UInt8,
            '/left_gripper_state',
            self.left_gripper_state_callback,
            10
        )
        self.create_subscription(
            UInt8,
            '/right_gripper_state',
            self.right_gripper_state_callback,
            10
        )

        self.create_timer(1.0 / 100.0, self.timer_callback)

    # def joint_action_callback(self, msg):
    #     current_time = time.time()
    #     if self.last_action_time >= 0:
    #         if current_time - self.last_action_time > 1.0:
    #             self.get_logger().warn("No joint action received for more than 1 second.")
    #     self.last_action_time = current_time
    #     self.joint_action = msg.data[0:7] + msg.data[14:21]
    def joint_left_callback(self, msg):
        current_time = time.time()
        if self.last_left_joint_time >= 0:
            if current_time - self.last_left_joint_time > 1.0:
                self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_left_joint_time = current_time

        if self.joint_action is None:
            self.joint_action = [0.0] * 14

        # 定义关节名称与索引的映射
        joint_name_to_index = {
            "joint1-l": 0, "joint2-l": 1, "joint3-l": 2, "joint4-l": 3, "joint5-l": 4, "joint6-l": 5, "joint7-l": 6,
            "joint1-r": 7, "joint2-r": 8, "joint3-r": 9, "joint4-r": 10, "joint5-r": 11, "joint6-r": 12, "joint7-r": 13
        }

        # 遍历接收到的关节名称和位置
        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                index = joint_name_to_index[name]
                self.joint_action[index] = position

    def joint_right_callback(self, msg):
        current_time = time.time()
        if self.last_right_joint_time >= 0:
            if current_time - self.last_right_joint_time > 1.0:
                self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_right_joint_time = current_time

        if self.joint_action is None:
            self.joint_action = [0.0] * 14

        # 定义关节名称与索引的映射
        joint_name_to_index = {
            "joint1-l": 0, "joint2-l": 1, "joint3-l": 2, "joint4-l": 3, "joint5-l": 4, "joint6-l": 5, "joint7-l": 6,
            "joint1-r": 7, "joint2-r": 8, "joint3-r": 9, "joint4-r": 10, "joint5-r": 11, "joint6-r": 12, "joint7-r": 13
        }

        # 遍历接收到的关节名称和位置
        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                index = joint_name_to_index[name]
                self.joint_action[index] = position

    def gripper_action_callback(self, msg):
        command = msg.data
        if command >= 100 and command < 200:
            self.left_gripper_action = (command-100) / 100.0
        elif command >= 200 and command < 300:
            self.right_gripper_action = (command-200) / 100.0

    def joint_state_callback(self, msg):
        current_time = time.time()
        if self.last_state_time >= 0:
            if current_time - self.last_state_time > 1.0:
                self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_state_time = current_time

        if self.joint_state is None:
            self.joint_state = [0.0] * 14

        # 定义关节名称与索引的映射
        joint_name_to_index = {
            "joint1-l": 0, "joint2-l": 1, "joint3-l": 2, "joint4-l": 3, "joint5-l": 4, "joint6-l": 5, "joint7-l": 6,
            "joint1-r": 7, "joint2-r": 8, "joint3-r": 9, "joint4-r": 10, "joint5-r": 11, "joint6-r": 12, "joint7-r": 13
        }

        # 遍历接收到的关节名称和位置
        for name, position in zip(msg.name, msg.position):
            if name in joint_name_to_index:
                index = joint_name_to_index[name]
                self.joint_state[index] = position

    def left_gripper_state_callback(self, msg):
        state = msg.data
        self.left_gripper_state = state / 255.0
        if self.left_gripper_state > 1.0:
            self.left_gripper_state = 1.0

    def right_gripper_state_callback(self, msg):
        state = msg.data
        self.right_gripper_state = state / 255.0
        if self.right_gripper_state > 1.0:
            self.right_gripper_state = 1.0

    def timer_callback(self):
        if self.joint_action is None or self.joint_state is None or len(self.joint_action) < 14 or self.left_gripper_action < 0 or self.right_gripper_action < 0:
            self.get_logger().warn("Incomplete data, skipping this cycle.")
            return

        timestamp_ms = int(time.time() * 1000)
        current_time = time.time()  # 当前时间（单位：秒）

        left_arm_action = self.joint_action[0:7]
        right_arm_action = self.joint_action[7:14]

        # 创建 JointState 消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [
            "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_joint_7", "left_gripper",
            "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7", "right_gripper"
        ]

        # 确保所有元素都是 float 类型
        joint_state_msg.position = list(map(float, left_arm_action)) + [float(self.left_gripper_action)] + list(map(float, right_arm_action)) + [float(self.right_gripper_action)]
        joint_state_msg.velocity = list(map(float, self.joint_state[0:7])) + [float(self.left_gripper_state)] + list(map(float, self.joint_state[7:14])) + [float(self.right_gripper_state)]
        joint_state_msg.effort = [0.0] * 16  # 假设没有力矩信息，这里填充为0

        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            self.get_logger().info(f"Timestamp: {timestamp_ms} ms")
            self.get_logger().info(f"JointState name: {joint_state_msg.name}")
            self.get_logger().info(f"JointState position (action): {joint_state_msg.position}")
            self.get_logger().info(f"JointState velocity (state): {joint_state_msg.velocity}")

        # 发布 JointState 消息
        self.joint_state_pub.publish(joint_state_msg)
        

        # todo: # 记录到文件
        # left_arm_str = ' '.join([f'{x:.6f}' for x in left_arm_action])
        # right_arm_str = ' '.join([f'{x:.6f}' for x in right_arm_action])

        # line = f'{timestamp_ms} {left_arm_str} {self.left_gripper_action:.6f} {right_arm_str} {self.right_gripper_action:.6f}'
        # # print(line, file=sys.stderr)
        # self.get_logger().info(f'Recorded: {line}')
        # self.file.write(line + '\n')
        # self.file.flush()

    def destroy_node(self):
        # self.file.close()
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='full_joint_gripper_commands.txt',
                        help='Output file name')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)  # 把 ROS 参数和 argparse 参数分开
    node = FullCommandRecorder(args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
