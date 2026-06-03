#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("x1_robot")
@dataclass
class X1RobotConfig(RobotConfig):
    # # Port to connect to the arm
    # port: str

    # disable_torque_on_disconnect: bool = True

    # # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # # names to the max_relative_target value for that motor.
    # max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    #teleop mode
    teleop: bool =  False

    # 头部相机配置 (RealSense D435)
    # 是否启用头部相机订阅
    enable_head_camera: bool = True
    # 头部RGB相机话题
    head_rgb_topic: str = "/head/camera/rgb"
    # 头部深度相机话题
    head_depth_topic: str = "/head/camera/depth/image_raw"
    # 头部相机图像宽度
    head_camera_width: int = 640
    # 头部相机图像高度
    head_camera_height: int = 480
    # 头部相机深度图像宽度
    head_camera_depth_width: int = 320
    # 头部相机深度图像高度
    head_camera_depth_height: int = 240

    # 头部深度相机最大深度范围(毫米)，用于归一化
    head_depth_max_mm: int = 3000

    # 乒乓球模式：只记录右臂（7关节），忽略左臂和夹爪
    # 订阅 x1/recorded_joint_states_pingpong 而非 x1/recorded_joint_states
    pingpong_mode: bool = False

    # 是否启用卡尔曼滤波（对快速动作如乒乓球建议关闭）
    enable_kalman_filter: bool = True
