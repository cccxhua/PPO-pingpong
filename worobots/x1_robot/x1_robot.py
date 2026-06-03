#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
# from lerobot.motors import Motor, MotorCalibration, MotorNormMode
# from lerobot.motors.feetech import (
#     FeetechMotorsBus,
#     OperatingMode,
# )
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
# from ..utils import ensure_safe_goal_position
from .config_x1_robot import X1RobotConfig

logger = logging.getLogger(__name__)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import numpy as np

import cv2
from sensor_msgs.msg import CompressedImage
from functools import partial
from threading import Lock

from ..ros_camera_subscriber import ROSCameraSubscriber, ROSCameraConfig, create_head_camera_subscriber


class AdaptiveKalmanFilter:
    def __init__(self, dim, process_variance, measurement_variance, threshold=5.0, scale_factor=10.0):
        """
        自适应卡尔曼过滤器。
        Args:
            dim (int): 状态变量的维度
            process_variance (float): 初始过程噪声的协方差
            measurement_variance (float): 初始测量噪声的协方差
            threshold (float): 判断变化幅度是否较大的阈值
            scale_factor (float): 自适应变化的缩放因子
        """
        self.dim = dim
        self.x = np.zeros(dim)  # 状态向量初始化
        self.P = np.ones(dim)   # 协方差矩阵初始化
        self.Q = process_variance * np.eye(dim)  # 系统过程噪声
        self.R = measurement_variance * np.eye(1)  # 测量噪声
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.prev_measurement = np.zeros(1)
        self.delta_t = 0.03333

        # 状态转移矩阵（角度和速度的预测公式）
        self.A = np.array([[1, self.delta_t],   # position = position + velocity * delta_t
                           [0, 1]])             # velocity = velocity (保持速度)

    def initialize_state_from_measurement(self, initial_measurement):
        """
        根据初始观测初始化状态。
        Args:
            initial_measurement (np.ndarray): 初始测量值
        """
        self.x[0] = initial_measurement[0]
        self.x[1] = 0.0  # 假设初始速度为零
        self.P = np.array([0.1, 0.1])  # 初始化较小协方差
        self.prev_measurement[0] = initial_measurement[0]

    def predict(self):
        """
        使用运动模型预测当前状态。
        """
        # 预测下一状态（基于状态转移矩阵）
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q  # 更新协方差矩阵

    def update(self, measurement):
        """
        更新状态估计。
        Args:
            measurement (np.ndarray): 测量值（如角度）
        Returns:
            np.ndarray: 更新后的状态 [位置, 速度]
        """
        # 动态调整测量噪声 R
        # 动态调整测量噪声 R
        delta = np.abs(measurement - self.prev_measurement)
        # delta_rate = delta / self.delta_t  # 引入速率变化判断
        if np.any(delta > self.threshold):  # 如果变化速率较大，增大测量噪声
            self.R = self.scale_factor * np.eye(1)
        else:  # 如果变化速率较小，恢复测量噪声
            self.R = 7.5 * np.eye(1)
 
        # 卡尔曼增益计算
        H = np.array([[1, 0]])  # 观测矩阵，仅观测位置
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
 
        # 更新状态
        y = measurement - H @ self.x  # 残差
        self.x += K @ y
        self.P = (np.eye(self.dim) - K @ H) @ self.P
 
        # 保存当前测量值
        self.prev_measurement = measurement
 
        return self.x

class X1Robot(Robot, Node):
    config_class = X1RobotConfig
    name = "x1_robot"

    def __init__(self, config: X1RobotConfig):
        Robot.__init__(self, config)
        Node.__init__(self, "x1_robot_node")
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self.camera_images = {}
        self.camera_locks = {}
        
        self.camera_subscribers = {}
        for cam_key in self.cameras.keys():
            topic = f"/{cam_key}/camera/rgb"

            self.camera_images[cam_key] = None
            self.camera_locks[cam_key] = Lock()

            # CompressedImage 订阅
            callback = partial(self._camera_callback, cam_key=cam_key)
            self.camera_subscribers[cam_key] = self.create_subscription(
                CompressedImage,   
                topic,
                callback,
                10
            )

            logger.info(f"Subscribed to camera topic: {topic}")

        # 头部相机订阅器 (RealSense D435)
        # 默认启用，可通过config.enable_head_camera控制
        self.head_camera_subscriber = None
        if getattr(config, 'enable_head_camera', True):
            head_cam_config = {
                "head_rgb": ROSCameraConfig(
                    topic=getattr(config, 'head_rgb_topic', "/head/camera/rgb"),
                    width=getattr(config, 'head_camera_width', 640),
                    height=getattr(config, 'head_camera_height', 480),
                    msg_type="compressed",
                    channels=3,
                    is_depth=False
                ),
                "head_depth": ROSCameraConfig(
                    topic=getattr(config, 'head_depth_topic', "/head/camera/depth/image_raw"),
                    width=getattr(config, 'head_camera_depth_width', 320),
                    height=getattr(config, 'head_camera_depth_height', 240),
                    msg_type="raw",
                    channels=3,  # 3通道灰度图
                    is_depth=True,
                    max_depth_mm=getattr(config, 'head_depth_max_mm', 3000)  # 默认3米范围
                )
            }
            self.head_camera_subscriber = ROSCameraSubscriber(head_cam_config)
            self.head_camera_subscriber.setup_subscriptions(self)
            logger.info("Head camera subscriber initialized (head_rgb, head_depth)")

        self._is_connected = False

        self.joint_states = None
        self.joint_actions = None

        if config.pingpong_mode:
            # 乒乓球模式：只记录右臂 7 个关节，忽略左臂和夹爪
            self.motors = [
                "right_joint_1", "right_joint_2", "right_joint_3",
                "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7",
            ]
        else:
            self.motors = [
                "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_joint_7",
                "left_gripper",
                "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_joint_7",
                "right_gripper"
            ]

        self.action_publisher = None

        if not config.teleop:
            self.action_publisher = self.create_publisher(JointState, "x1/sent_actions", 10)

        # kalman滤波初始化标志
        self._kalman_initialized = False

        # 为每个关节创建自适应卡尔曼滤波器实例
        self.kalman_filters = {
            motor: AdaptiveKalmanFilter(dim=2, process_variance=0.5, measurement_variance=5, 
                                         threshold=0.075, scale_factor=15.0)
            for motor in self.motors
        }

        joint_states_topic = (
            "x1/recorded_joint_states_pingpong"
            if config.pingpong_mode
            else "x1/recorded_joint_states"
        )
        self.create_subscription(JointState, joint_states_topic, self._joint_states_callback, 10)
        logger.info(f"Subscribing to joint states topic: {joint_states_topic}")

    
    def _joint_states_callback(self, msg: JointState):
        if not self._is_connected:
            return
        # logger.info(f"Received joint states: {msg.name} with positions {msg.position}")
        # logger.info(f"Received joint velocities: {msg.name} with velocities {msg.velocity}")
        
        self.joint_actions = {f"{name}.pos": position for name, position in zip(msg.name, msg.position)}
        self.joint_states = {f"{name}.pos": position for name, position in zip(msg.name, msg.velocity)}

    def _camera_callback(self, msg: CompressedImage, cam_key: str):
        '''
        RealSense 直连 read()，→ 获取 RAW RGB → 若 ColorMode=BGR → 转成 BGR，直连编码流向是：RGB → BGR
        CompressedImage 解码后是 BGR，不用人工做 BGR→RGB→BGR 这种绕路
        为保持一致，最终模型的图像也必须是 BGR（保持现有流向）
        '''
        # CompressedImage → numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            logger.warning(f"Failed to decode image for camera {cam_key}")
            return

        # resize 到 config 确定大小
        cam_cfg = self.config.cameras[cam_key]
        if cv_image.shape[1] != cam_cfg.width or cv_image.shape[0] != cam_cfg.height:
            cv_image = cv2.resize(cv_image, (cam_cfg.width, cam_cfg.height))

        # 写入图像（需要加锁）
        with self.camera_locks[cam_key]:
            self.camera_images[cam_key] = cv_image

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # 基础相机特征
        features = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        # 添加头部相机特征
        if self.head_camera_subscriber is not None:
            features.update(self.head_camera_subscriber.camera_features)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        # return self._is_connected and all(cam.is_connected for cam in self.cameras.values())
        return self._is_connected # 只检查机器人连接状态，不检查相机物理连接状态

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 不连接物理相机,使用ROS订阅相机
        # for cam in self.cameras.values():
        #     cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected.")
        logger.info(self._is_connected)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 不断开物理相机,使用ROS订阅相机
        # for cam in self.cameras.values():
        #     cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        

        if self.joint_states is None:
            return {}

        obs_dict = self.joint_states.copy()

        # for cam_key, cam in self.cameras.items():
        #     obs_dict[cam_key] = cam.async_read()

        # 使用ROS订阅的相机图像
        for cam_key in self.cameras.keys():
            with self.camera_locks[cam_key]:
                img = self.camera_images[cam_key]

                if img is None:
                    # 占位黑图（符合 BGR）
                    cam_cfg = self.config.cameras[cam_key]
                    obs_dict[cam_key] = np.zeros(
                        (cam_cfg.height, cam_cfg.width, 3),
                        dtype=np.uint8
                    )
                else:
                    # 拷贝（避免被回调线程修改）
                    obs_dict[cam_key] = img.copy()

        # 添加头部相机图像
        if self.head_camera_subscriber is not None:
            for cam_key in self.head_camera_subscriber.config.cameras.keys():
                obs_dict[cam_key] = self.head_camera_subscriber.get_image_or_placeholder(cam_key)

        return obs_dict

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.joint_actions is None:
            return {}

        action_dict = self.joint_actions.copy()

        return action_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.enable_kalman_filter:
            # 如果还未初始化，则用第一个动作来初始化卡尔曼滤波器的状态
            if not self._kalman_initialized:
                for key, value in action.items():
                    if key.endswith(".pos"):
                        motor = key.removesuffix(".pos")
                        if motor in self.kalman_filters:
                            self.kalman_filters[motor].initialize_state_from_measurement(np.array([value]))
                self._kalman_initialized = True
            
            # 对每个动作进行自适应卡尔曼滤波
            filtered_action = {}
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor = key.removesuffix(".pos")
                    if motor in self.kalman_filters:
                        kf = self.kalman_filters[motor]
                        kf.predict()
                        filtered_value = float(kf.update(np.array([value]))[0])
                        filtered_action[key] = filtered_value
                    else:
                        filtered_action[key] = value
        else:
            filtered_action = action

        goal_pos = {key.removesuffix(".pos"): val for key, val in filtered_action.items() if key.endswith(".pos")}

        # 发布 JointState（除 gripper 外的关节动作）
        joint_state_msg = JointState()
        joint_state_msg.name = list(goal_pos.keys())
        # joint_state_msg.position = list(goal_pos.values())
        joint_state_msg.position = [float(val) for val in goal_pos.values()]
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()

        if self.action_publisher is not None:
            self.action_publisher.publish(joint_state_msg)


        # 返回合并动作
        result = {f"{motor}.pos": val for motor, val in goal_pos.items()}

        return result
    
