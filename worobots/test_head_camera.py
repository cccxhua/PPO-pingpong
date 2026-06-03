#!/usr/bin/env python3
"""
测试头部相机订阅器是否正常工作。
运行方式: python -m woanlerobot.worobots.test_head_camera
"""

import logging
import time
import threading

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from woanlerobot.worobots.ros_camera_subscriber import (
    ROSCameraSubscriber, 
    ROSCameraConfig,
    create_head_camera_subscriber
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNode(Node):
    def __init__(self):
        super().__init__('test_head_camera_node')
        
        # 使用便捷函数创建头部相机订阅器
        self.head_cam_subscriber = create_head_camera_subscriber(
            rgb_topic="/head/camera/rgb",
            depth_topic="/head/camera/depth/image_raw",
            width=640,
            height=480
        )
        
        # 设置订阅
        self.head_cam_subscriber.setup_subscriptions(self)
        
        logger.info("Head camera subscriber initialized")
        logger.info(f"Camera features: {self.head_cam_subscriber.camera_features}")


def main():
    rclpy.init()
    
    node = TestNode()
    
    # 创建executor并在后台线程中运行
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    logger.info("Waiting for images... (Press Ctrl+C to exit)")
    
    try:
        while rclpy.ok():
            # 获取图像
            images = node.head_cam_subscriber.get_images()
            
            for cam_key, img in images.items():
                if img is not None:
                    logger.info(f"[{cam_key}] Image shape: {img.shape}, dtype: {img.dtype}")
                    
                    # 可选：显示图像
                    cv2.imshow(cam_key, img)
                else:
                    logger.info(f"[{cam_key}] No image received yet")
            
            key = cv2.waitKey(100)
            if key == ord('q'):
                break
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
