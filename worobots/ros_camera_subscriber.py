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

"""
ROS2相机订阅器模块，用于订阅ROS话题中的相机数据。
支持sensor_msgs/Image和sensor_msgs/CompressedImage类型。
"""

import logging
from threading import Lock
from typing import Any, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ROSCameraConfig:
    """ROS相机配置"""
    topic: str
    width: int = 640
    height: int = 480
    # 消息类型: "compressed" 或 "raw"
    msg_type: str = "compressed"
    # 图像通道数，RGB/BGR为3，深度图为1
    channels: int = 3
    # 是否为深度图
    is_depth: bool = False
    # 深度图的缩放因子（将raw值转换为米）
    depth_scale: float = 0.001
    # 深度图的最大范围（毫米），用于固定范围归一化，避免波动
    max_depth_mm: int = 5000


@dataclass
class ROSCameraSubscriberConfig:
    """ROS相机订阅器配置"""
    # 相机配置字典，key为相机名称
    cameras: dict[str, ROSCameraConfig] = field(default_factory=dict)


class ROSCameraSubscriber:
    """
    ROS2相机订阅器类，用于从ROS话题订阅相机图像数据。
    
    支持:
    - sensor_msgs/msg/CompressedImage (压缩图像)
    - sensor_msgs/msg/Image (原始图像)
    
    使用方式:
    1. 在ROS2节点中创建实例
    2. 调用setup_subscriptions(node)来创建订阅
    3. 调用get_images()获取当前图像
    """
    
    def __init__(self, config: ROSCameraSubscriberConfig | dict[str, ROSCameraConfig]):
        """
        初始化ROS相机订阅器。
        
        Args:
            config: ROSCameraSubscriberConfig对象或相机配置字典
        """
        if isinstance(config, dict):
            self.config = ROSCameraSubscriberConfig(cameras=config)
        else:
            self.config = config
            
        self.camera_images: dict[str, np.ndarray | None] = {}
        self.camera_locks: dict[str, Lock] = {}
        self.subscribers: dict[str, Any] = {}
        
        # 初始化图像存储和锁
        for cam_key in self.config.cameras.keys():
            self.camera_images[cam_key] = None
            self.camera_locks[cam_key] = Lock()
    
    def setup_subscriptions(self, node) -> None:
        """
        在ROS2节点上创建订阅。
        
        Args:
            node: ROS2节点实例 (rclpy.node.Node)
        """
        from functools import partial
        from sensor_msgs.msg import CompressedImage, Image
        
        for cam_key, cam_cfg in self.config.cameras.items():
            if cam_cfg.msg_type == "compressed":
                msg_class = CompressedImage
                callback = partial(self._compressed_image_callback, cam_key=cam_key)
            else:
                msg_class = Image
                callback = partial(self._raw_image_callback, cam_key=cam_key)
            
            self.subscribers[cam_key] = node.create_subscription(
                msg_class,
                cam_cfg.topic,
                callback,
                10
            )
            logger.info(f"ROSCameraSubscriber: Subscribed to {cam_cfg.topic} ({cam_cfg.msg_type}) as '{cam_key}'")
    
    def _compressed_image_callback(self, msg, cam_key: str) -> None:
        """
        处理CompressedImage消息。
        
        Args:
            msg: sensor_msgs/msg/CompressedImage消息
            cam_key: 相机键名
        """
        cam_cfg = self.config.cameras[cam_key]
        
        # 解码压缩图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            logger.warning(f"Failed to decode compressed image for camera {cam_key}")
            return
        
        # 调整大小
        if cv_image.shape[1] != cam_cfg.width or cv_image.shape[0] != cam_cfg.height:
            cv_image = cv2.resize(cv_image, (cam_cfg.width, cam_cfg.height))
        
        with self.camera_locks[cam_key]:
            self.camera_images[cam_key] = cv_image
    
    def _raw_image_callback(self, msg, cam_key: str) -> None:
        """
        处理Image消息（原始图像）。
        
        Args:
            msg: sensor_msgs/msg/Image消息
            cam_key: 相机键名
        """
        cam_cfg = self.config.cameras[cam_key]
        
        # 根据编码格式解码图像
        encoding = msg.encoding.lower()
        
        if cam_cfg.is_depth:
            # 深度图处理
            if encoding in ['16uc1', 'mono16']:
                # 16位深度图 (单位通常是毫米)
                cv_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif encoding in ['32fc1']:
                # 32位浮点深度图 (单位通常是米)
                cv_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                # 转换为毫米并转uint16
                cv_image = (cv_image * 1000).astype(np.uint16)
            else:
                logger.warning(f"Unsupported depth encoding: {encoding} for camera {cam_key}")
                return
            
            # 使用固定范围归一化，避免波动 (0-5000mm 映射到 0-255)
            # depth_scale 可通过配置调整，默认5米范围
            max_depth_mm = getattr(cam_cfg, 'max_depth_mm', 5000)
            cv_image = cv_image.astype(np.float32)
            cv_image = np.clip(cv_image, 0, max_depth_mm)
            cv_image = (cv_image / max_depth_mm * 255).astype(np.uint8)
            
            # 转换为3通道灰度图 (不使用伪彩色)
            if cam_cfg.channels == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        else:
            # RGB/BGR图像处理
            if encoding in ['rgb8']:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif encoding in ['bgr8']:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif encoding in ['mono8']:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                if cam_cfg.channels == 3:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif encoding in ['bgra8', 'rgba8']:
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                if encoding == 'rgba8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
                else:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            else:
                logger.warning(f"Unsupported image encoding: {encoding} for camera {cam_key}")
                return
        
        if cv_image is None:
            logger.warning(f"Failed to decode raw image for camera {cam_key}")
            return
        
        # 调整大小
        if cv_image.shape[1] != cam_cfg.width or cv_image.shape[0] != cam_cfg.height:
            cv_image = cv2.resize(cv_image, (cam_cfg.width, cam_cfg.height))
        
        with self.camera_locks[cam_key]:
            self.camera_images[cam_key] = cv_image
    
    def get_image(self, cam_key: str) -> np.ndarray | None:
        """
        获取指定相机的当前图像。
        
        Args:
            cam_key: 相机键名
            
        Returns:
            图像numpy数组，如果无图像则返回None
        """
        if cam_key not in self.camera_locks:
            logger.warning(f"Camera '{cam_key}' not found in subscriber")
            return None
            
        with self.camera_locks[cam_key]:
            if self.camera_images[cam_key] is not None:
                return self.camera_images[cam_key].copy()
            return None
    
    def get_images(self) -> dict[str, np.ndarray | None]:
        """
        获取所有相机的当前图像。
        
        Returns:
            字典，key为相机名称，value为图像numpy数组（或None）
        """
        result = {}
        for cam_key in self.config.cameras.keys():
            result[cam_key] = self.get_image(cam_key)
        return result
    
    def get_image_or_placeholder(self, cam_key: str) -> np.ndarray:
        """
        获取指定相机的图像，如果无图像则返回黑色占位图。
        
        Args:
            cam_key: 相机键名
            
        Returns:
            图像numpy数组
        """
        img = self.get_image(cam_key)
        if img is not None:
            return img
        
        # 返回占位黑图
        cam_cfg = self.config.cameras.get(cam_key)
        if cam_cfg:
            return np.zeros(
                (cam_cfg.height, cam_cfg.width, cam_cfg.channels),
                dtype=np.uint8
            )
        else:
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @property
    def camera_features(self) -> dict[str, tuple]:
        """
        获取相机特征字典，用于LeRobot数据集。
        
        Returns:
            字典，key为相机名称，value为(height, width, channels)元组
        """
        return {
            cam_key: (cam_cfg.height, cam_cfg.width, cam_cfg.channels)
            for cam_key, cam_cfg in self.config.cameras.items()
        }
    
    def destroy_subscriptions(self, node) -> None:
        """
        销毁所有订阅。
        
        Args:
            node: ROS2节点实例
        """
        for cam_key, sub in self.subscribers.items():
            node.destroy_subscription(sub)
            logger.info(f"ROSCameraSubscriber: Destroyed subscription for '{cam_key}'")
        self.subscribers.clear()


def create_head_camera_subscriber(
    rgb_topic: str = "/head/camera/rgb",
    depth_topic: str = "/head/camera/depth/image_raw",
    width: int = 640,
    height: int = 480
) -> ROSCameraSubscriber:
    """
    创建头部相机订阅器的便捷函数。
    
    Args:
        rgb_topic: RGB图像话题
        depth_topic: 深度图像话题
        width: 图像宽度
        height: 图像高度
        
    Returns:
        配置好的ROSCameraSubscriber实例
    """
    config = {
        "head_rgb": ROSCameraConfig(
            topic=rgb_topic,
            width=width,
            height=height,
            msg_type="compressed",  # /head/camera/rgb 是 CompressedImage
            channels=3,
            is_depth=False
        ),
        "head_depth": ROSCameraConfig(
            topic=depth_topic,
            width=width,
            height=height,
            msg_type="raw",  # /head/camera/depth/image_raw 是 Image
            channels=3,  # 转换为伪彩色
            is_depth=True
        )
    }
    return ROSCameraSubscriber(config)
