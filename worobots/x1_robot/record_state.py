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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from collections import deque

class FullCommandRecorder(Node):
    def __init__(self, output_file):
        super().__init__('full_command_recorder')

        self.last_log_time = 0
        self.last_action_time = -1
        self.last_state_time = -1
        self.last_left_gripper_time = -1
        self.last_right_gripper_time = -1

        # 频率监测相关变量
        self.target_hz = 50.0  # 目标频率
        self.freq_window_size = 10  # 用于计算频率的窗口大小
        # 为每个数据流维护时间戳队列
        self.joint_state_timestamps = deque(maxlen=self.freq_window_size)
        self.left_gripper_timestamps = deque(maxlen=self.freq_window_size)
        self.right_gripper_timestamps = deque(maxlen=self.freq_window_size)
        # 频率检查间隔（秒）
        self.freq_check_interval = 2.0
        self.last_freq_check_time = time.time()

        # self.file = open(output_file, 'w')
        # self.get_logger().info(f"Recording to {output_file}")

        self.joint_action = None
        self.left_gripper_action = 0
        self.right_gripper_action = 0

        self.joint_state = None
        self.left_gripper_state = 0
        self.right_gripper_state = 0

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.joint_state_pub = self.create_publisher(JointState, 'x1/recorded_joint_states', qos_profile)

        # self.create_subscription(
        #     Float64MultiArray,
        #     '/record_data',
        #     self.joint_action_callback,
        #     qos_profile
        # )
        # self.create_subscription(
        #     Int32,
        #     '/joystick_info',
        #     self.gripper_action_callback,
        #     qos_profile
        # )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )

        self.create_subscription(
            UInt8,
            '/left_gripper_state',
            self.left_gripper_state_callback,
            qos_profile
        )
        self.create_subscription(
            UInt8,
            '/right_gripper_state',
            self.right_gripper_state_callback,
            qos_profile
        )

        self.create_timer(1.0 / 100.0, self.timer_callback)

    # def joint_action_callback(self, msg):
    #     current_time = time.time()
    #     if self.last_action_time >= 0:
    #         if current_time - self.last_action_time > 1.0:
    #             self.get_logger().warn("No joint action received for more than 1 second.")
    #     self.last_action_time = current_time
    #     self.joint_action = msg.data[0:7] + msg.data[14:21]

    # def gripper_action_callback(self, msg):
    #     command = msg.data
    #     if command > 100 and command < 200:
    #         self.left_gripper_action = (command-100) / 100.0
    #     elif command >= 200 and command < 300:
    #         self.right_gripper_action = (command-200) / 100.0

    # 新增频率计算和日志打印功能
    def calculate_frequency(self, timestamps):
        """计算频率"""
        if len(timestamps) < 2:
            return 0.0
        
        # 计算时间间隔的平均值
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        if not intervals:
            return 0.0
            
        avg_interval = sum(intervals) / len(intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    # 新增频率检查和日志打印功能
    def check_and_log_frequencies(self):
        """检查各个数据流的频率，如果低于50Hz则打印数据"""
        current_time = time.time()
        
        # 每隔一定时间检查一次频率
        if current_time - self.last_freq_check_time < self.freq_check_interval:
            return
            
        self.last_freq_check_time = current_time
        
        # 检查关节状态频率
        joint_freq = self.calculate_frequency(list(self.joint_state_timestamps))
        if joint_freq > 0 and joint_freq < self.target_hz:
            self.get_logger().warn(f"⚠️  Joint state frequency too low: {joint_freq:.1f}Hz (target: {self.target_hz}Hz)")
            self.get_logger().info(f"📊 Current joint state data: {self.joint_state}")
        
        # 检查左夹爪频率
        left_gripper_freq = self.calculate_frequency(list(self.left_gripper_timestamps))
        if left_gripper_freq > 0 and left_gripper_freq < self.target_hz:
            self.get_logger().warn(f"⚠️  Left gripper frequency too low: {left_gripper_freq:.1f}Hz (target: {self.target_hz}Hz)")
            self.get_logger().info(f"📊 Current left gripper state: {self.left_gripper_state}")
        
        # 检查右夹爪频率
        right_gripper_freq = self.calculate_frequency(list(self.right_gripper_timestamps))
        if right_gripper_freq > 0 and right_gripper_freq < self.target_hz:
            self.get_logger().warn(f"⚠️  Right gripper frequency too low: {right_gripper_freq:.1f}Hz (target: {self.target_hz}Hz)")
            self.get_logger().info(f"📊 Current right gripper state: {self.right_gripper_state}")


    def joint_state_callback(self, msg):
        current_time = time.time()
        # 记录时间戳用于频率计算
        self.joint_state_timestamps.append(current_time)

        if self.last_state_time >= 0:
            if current_time - self.last_state_time > 1.0:
                self.get_logger().warn("No joint state received for more than 1 second.")
        self.last_state_time = current_time

        if self.joint_state is None:
            self.joint_state = [0.0] * 14
        
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
                self.joint_state[index] = position

        # 检查并打印频率
        self.check_and_log_frequencies()

    def left_gripper_state_callback(self, msg):
        current_time = time.time()
        # 记录时间戳用于频率计算
        self.left_gripper_timestamps.append(current_time)

        state = msg.data
        self.left_gripper_state = state / 255.0
        if self.left_gripper_state > 1.0:
            self.left_gripper_state = 1.0

        # 检查并打印频率
        self.check_and_log_frequencies()

    def right_gripper_state_callback(self, msg):
        current_time = time.time()
        # 记录时间戳用于频率计算
        self.right_gripper_timestamps.append(current_time)

        state = msg.data
        self.right_gripper_state = state / 255.0
        if self.right_gripper_state > 1.0:
            self.right_gripper_state = 1.0

        # 检查并打印频率
        self.check_and_log_frequencies()

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
