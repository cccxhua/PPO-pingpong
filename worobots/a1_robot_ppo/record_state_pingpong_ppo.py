#!/usr/bin/env python3
"""
record_state_pingpong_ppo.py

A1 PPO 乒乓球状态记录节点，50Hz。
记录右臂 7 关节的位置和速度（PPO 策略需要两者）。

发布话题: a1/recorded_joint_states_pingpong  (sensor_msgs/JointState, 7 joints)
    - position : 右臂关节位置 (rad)
    - velocity : 右臂关节速度 (rad/s)
    - effort   : 全零

订阅话题:
  /joint_states    (sensor_msgs/JointState)  — 从臂关节实际位置和速度
"""

import time

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

# /joint_states 中右臂关节名称 → 输出索引 (0~6)
RIGHT_ARM_SOURCE_NAMES = {
    "joint1-a1_r": 0,
    "joint2-a1_r": 1,
    "joint3-a1_r": 2,
    "joint4-a1_r": 3,
    "joint5-a1_r": 4,
    "joint6-a1_r": 5,
    "joint7-a1_r": 6,
}

PUBLISH_HZ = 50.0  # PPO 控制频率


class RightArmStateRecorderPPO(Node):
    def __init__(self):
        super().__init__("right_arm_state_recorder_ppo_a1")

        self.last_log_time = 0.0
        self.last_state_time = -1.0

        self.right_arm_pos: list[float] = [0.0] * 7
        self.right_arm_vel: list[float] = [0.0] * 7
        self.has_state = False

        self._prev_pos: list[float] | None = None
        self._prev_time: float = 0.0

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.joint_state_pub = self.create_publisher(
            JointState, "a1/recorded_joint_states_pingpong", qos_profile
        )

        self.create_subscription(
            JointState, "/right_joint_states", self.state_callback, qos_profile
        )

        self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"A1 RightArmStateRecorderPPO started @ {PUBLISH_HZ} Hz. "
            "Publishing pos+vel to a1/recorded_joint_states_pingpong"
        )

    def state_callback(self, msg: JointState) -> None:
        current_time = time.time()

        for name, pos in zip(msg.name, msg.position):
            if name in RIGHT_ARM_SOURCE_NAMES:
                idx = RIGHT_ARM_SOURCE_NAMES[name]
                self.right_arm_pos[idx] = float(pos)

        # 尝试从 velocity 字段获取速度
        if msg.velocity:
            for name, vel in zip(msg.name, msg.velocity):
                if name in RIGHT_ARM_SOURCE_NAMES:
                    idx = RIGHT_ARM_SOURCE_NAMES[name]
                    self.right_arm_vel[idx] = float(vel)
        else:
            # 数值微分
            if self._prev_pos is not None and current_time > self._prev_time:
                dt = current_time - self._prev_time
                for i in range(7):
                    self.right_arm_vel[i] = (self.right_arm_pos[i] - self._prev_pos[i]) / dt

        self._prev_pos = list(self.right_arm_pos)
        self._prev_time = current_time
        self.has_state = True
        self.last_state_time = current_time

    def timer_callback(self) -> None:
        if not self.has_state:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = RIGHT_ARM_JOINT_NAMES
        msg.position = list(self.right_arm_pos)
        msg.velocity = list(self.right_arm_vel)
        msg.effort = [0.0] * 7
        self.joint_state_pub.publish(msg)

        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:
            self.last_log_time = current_time
            self.get_logger().info(
                f"pos: {[f'{v:.3f}' for v in self.right_arm_pos]} | "
                f"vel: {[f'{v:.2f}' for v in self.right_arm_vel]}"
            )


def main() -> None:
    rclpy.init()
    node = RightArmStateRecorderPPO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
