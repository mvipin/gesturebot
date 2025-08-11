#!/usr/bin/env python3
"""
Image Viewer Node for GestureBot Vision System
Displays annotated object detection images using OpenCV.
"""

import cv2
import numpy as np
from typing import Optional
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor


class ImageViewerNode(Node):
    """
    Node for displaying annotated images using OpenCV cv2.imshow().
    Optimized for Raspberry Pi 5 with minimal resource usage.
    """

    def __init__(self):
        super().__init__('image_viewer_node')
        
        # Declare parameters
        self.declare_parameter(
            'window_name', 
            'GestureBot Object Detection',
            ParameterDescriptor(description='OpenCV window name for display')
        )
        
        self.declare_parameter(
            'display_fps', 
            10.0,
            ParameterDescriptor(description='Maximum display refresh rate (FPS) to limit CPU usage')
        )
        
        self.declare_parameter(
            'window_width', 
            640,
            ParameterDescriptor(description='Display window width (0 = original size)')
        )
        
        self.declare_parameter(
            'window_height', 
            480,
            ParameterDescriptor(description='Display window height (0 = original size)')
        )
        
        self.declare_parameter(
            'show_fps_overlay', 
            True,
            ParameterDescriptor(description='Show FPS overlay on displayed image')
        )
        
        # Get parameters
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.display_fps = self.get_parameter('display_fps').get_parameter_value().double_value
        self.window_width = self.get_parameter('window_width').get_parameter_value().integer_value
        self.window_height = self.get_parameter('window_height').get_parameter_value().integer_value
        self.show_fps_overlay = self.get_parameter('show_fps_overlay').get_parameter_value().bool_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Display control
        self.last_display_time = 0.0
        self.display_interval = 1.0 / self.display_fps if self.display_fps > 0 else 0.0
        
        # FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Try to set OpenCV backend to avoid Qt issues
        try:
            # Try to use GTK backend if available
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            self.get_logger().error(f"Failed to create OpenCV window: {e}")
            self.get_logger().info("Trying alternative display method...")
            # Set environment variable to use different backend
            import os
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # Subscribe to annotated images topic
        self.image_subscription = self.create_subscription(
            Image,
            '/vision/objects/annotated',
            self.image_callback,
            10  # QoS depth
        )
        
        self.get_logger().info(f"Image viewer initialized:")
        self.get_logger().info(f"  - Window: {self.window_name}")
        self.get_logger().info(f"  - Max display FPS: {self.display_fps}")
        self.get_logger().info(f"  - Window size: {self.window_width}x{self.window_height}")
        self.get_logger().info(f"  - FPS overlay: {self.show_fps_overlay}")
        self.get_logger().info(f"  - Subscribed to: /vision/objects/annotated")

    def image_callback(self, msg: Image) -> None:
        """
        Callback for receiving annotated images.
        Implements frame rate limiting to reduce CPU usage.
        """
        try:
            current_time = time.time()
            
            # Frame rate limiting
            if self.display_interval > 0:
                if current_time - self.last_display_time < self.display_interval:
                    return  # Skip this frame to maintain target FPS
            
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Resize if specified
            if self.window_width > 0 and self.window_height > 0:
                cv_image = cv2.resize(cv_image, (self.window_width, self.window_height))
            
            # Add FPS overlay if enabled
            if self.show_fps_overlay:
                cv_image = self.add_fps_overlay(cv_image)
            
            # Display image
            cv2.imshow(self.window_name, cv_image)
            
            # Process OpenCV events (required for display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                self.get_logger().info("Quit key pressed, shutting down...")
                rclpy.shutdown()
            elif key == ord('s'):  # 's' to save screenshot
                self.save_screenshot(cv_image)
            
            # Update display timing
            self.last_display_time = current_time
            
            # Update FPS calculation
            self.update_fps_calculation()
            
        except Exception as e:
            self.get_logger().error(f"Error displaying image: {e}")

    def add_fps_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add FPS overlay to the image."""
        # Create overlay text
        fps_text = f"Display FPS: {self.current_fps:.1f}"
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(image, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, fps_text, (15, 10 + text_height + 5), font, font_scale, color, thickness)
        
        return image

    def update_fps_calculation(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time

    def save_screenshot(self, image: np.ndarray) -> None:
        """Save current image as screenshot."""
        timestamp = int(time.time())
        filename = f"gesturebot_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Screenshot saved: {filename}")

    def destroy_node(self) -> None:
        """Clean up OpenCV windows when node is destroyed."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main function for image viewer node."""
    rclpy.init(args=args)
    
    try:
        node = ImageViewerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
