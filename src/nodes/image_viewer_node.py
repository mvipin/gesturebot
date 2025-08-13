#!/usr/bin/env python3
"""
Unified Image Viewer Node for GestureBot Vision System
Displays annotated images from multiple vision topics using OpenCV.
Supports object detection, gesture recognition, and other vision outputs.
"""

import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class UnifiedImageViewerNode(Node):
    """
    Unified node for displaying annotated images using OpenCV cv2.imshow().
    Supports multiple image topics with configurable display windows.
    Optimized for Raspberry Pi 5 with minimal resource usage.
    """

    def __init__(self):
        super().__init__('unified_image_viewer')

        # Declare parameters
        self.declare_parameter(
            'window_name',
            'GestureBot Vision System',
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

        self.declare_parameter(
            'image_topics',
            '["/vision/objects/annotated"]',
            ParameterDescriptor(description='List of ROS image topics to subscribe to (JSON array string format)')
        )

        self.declare_parameter(
            'topic_window_names',
            '{}',
            ParameterDescriptor(description='Optional mapping of topic names to custom window names (JSON string format)')
        )

        # Get parameters
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.display_fps = self.get_parameter('display_fps').get_parameter_value().double_value
        self.window_width = self.get_parameter('window_width').get_parameter_value().integer_value
        self.window_height = self.get_parameter('window_height').get_parameter_value().integer_value
        self.show_fps_overlay = self.get_parameter('show_fps_overlay').get_parameter_value().bool_value

        # Parse image topics from JSON string
        image_topics_str = self.get_parameter('image_topics').get_parameter_value().string_value
        try:
            import json
            self.image_topics = json.loads(image_topics_str)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON for image_topics: {e}")
            self.image_topics = ['/vision/objects/annotated']  # fallback

        # Parse topic window names if provided (JSON format)
        self.topic_window_names = self.get_parameter('topic_window_names').get_parameter_value().string_value
        self.custom_window_names = {}
        if self.topic_window_names:
            try:
                self.custom_window_names = json.loads(self.topic_window_names)
            except json.JSONDecodeError as e:
                self.get_logger().warn(f"Invalid JSON for topic_window_names: {e}")

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Display control (per topic)
        self.last_display_times = {}
        self.display_interval = 1.0 / self.display_fps if self.display_fps > 0 else 0.0

        # FPS calculation (per topic)
        self.frame_counts = {}
        self.fps_start_times = {}
        self.current_fps_values = {}

        # Window management
        self.active_windows = {}
        self.topic_subscriptions = {}

        # Initialize windows and subscriptions for each topic
        self._setup_topic_subscriptions()

        self.get_logger().info(f"Unified image viewer initialized:")
        self.get_logger().info(f"  - Base window name: {self.window_name}")
        self.get_logger().info(f"  - Max display FPS: {self.display_fps}")
        self.get_logger().info(f"  - Window size: {self.window_width}x{self.window_height}")
        self.get_logger().info(f"  - FPS overlay: {self.show_fps_overlay}")
        self.get_logger().info(f"  - Subscribed topics: {self.image_topics}")

    def _setup_topic_subscriptions(self):
        """Set up subscriptions and windows for all configured topics."""
        # Use BEST_EFFORT reliability to match image publishers
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        for topic in self.image_topics:
            # Determine window name for this topic
            if topic in self.custom_window_names:
                window_name = self.custom_window_names[topic]
            else:
                # Generate window name from topic
                topic_suffix = topic.split('/')[-1] if '/' in topic else topic
                window_name = f"{self.window_name} - {topic_suffix}"

            # Initialize tracking variables for this topic
            self.last_display_times[topic] = 0.0
            self.frame_counts[topic] = 0
            self.fps_start_times[topic] = time.time()
            self.current_fps_values[topic] = 0.0

            # Create OpenCV window
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                self.active_windows[topic] = window_name
            except Exception as e:
                self.get_logger().error(f"Failed to create OpenCV window for {topic}: {e}")
                self.get_logger().info("Trying alternative display method...")
                import os
                os.environ['QT_QPA_PLATFORM'] = 'offscreen'
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                self.active_windows[topic] = window_name

            # Create subscription
            callback = lambda msg, t=topic: self.image_callback(msg, t)
            subscription = self.create_subscription(
                Image,
                topic,
                callback,
                image_qos
            )
            self.topic_subscriptions[topic] = subscription

            self.get_logger().info(f"  - {topic} -> {window_name}")

    def image_callback(self, msg: Image, topic: str) -> None:
        """
        Callback for receiving annotated images from a specific topic.
        Implements frame rate limiting to reduce CPU usage.
        """
        try:
            current_time = time.time()

            # Frame rate limiting per topic
            if self.display_interval > 0:
                if current_time - self.last_display_times[topic] < self.display_interval:
                    return  # Skip this frame to maintain target FPS

            # Convert ROS Image to OpenCV format
            # Handle different input formats (RGB888 from camera, BGR8 from annotated images)
            if msg.encoding == 'rgb8':
                # Raw camera feed is RGB, convert to BGR for OpenCV display
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                # Annotated images are already BGR8
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize if specified
            if self.window_width > 0 and self.window_height > 0:
                cv_image = cv2.resize(cv_image, (self.window_width, self.window_height))

            # Add FPS overlay if enabled
            if self.show_fps_overlay:
                cv_image = self.add_fps_overlay(cv_image, topic)

            # Display image in the appropriate window
            window_name = self.active_windows[topic]
            cv2.imshow(window_name, cv_image)

            # Process OpenCV events (required for display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                self.get_logger().info("Quit key pressed, shutting down...")
                rclpy.shutdown()
            elif key == ord('s'):  # 's' to save screenshot
                self.save_screenshot(cv_image, topic)

            # Update display timing for this topic
            self.last_display_times[topic] = current_time

            # Update FPS calculation for this topic
            self.update_fps_calculation(topic)

        except Exception as e:
            self.get_logger().error(f"Error displaying image from {topic}: {e}")

    def add_fps_overlay(self, image: np.ndarray, topic: str) -> np.ndarray:
        """Add FPS overlay to the image for a specific topic in the lower-right corner."""
        # Create overlay text with topic info
        fps_text = f"Display FPS: {self.current_fps_values[topic]:.1f}"
        topic_text = f"Topic: {topic.split('/')[-1]}"

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green
        thickness = 1

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Get text sizes for background rectangles
        (fps_width, fps_height), fps_baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        (topic_width, topic_height), topic_baseline = cv2.getTextSize(topic_text, font, font_scale, thickness)

        # Calculate dimensions for background rectangle
        max_width = max(fps_width, topic_width)
        total_height = fps_height + topic_height + fps_baseline + topic_baseline + 15
        padding = 10  # Padding from image edges

        # Calculate lower-right corner position
        rect_x2 = img_width - padding
        rect_y2 = img_height - padding
        rect_x1 = rect_x2 - max_width - 10  # 10px internal padding
        rect_y1 = rect_y2 - total_height

        # Ensure rectangle stays within image bounds
        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)

        # Draw background rectangle
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

        # Calculate text positions (with internal padding)
        text_x = rect_x1 + 5  # 5px internal padding from rectangle edge
        fps_text_y = rect_y1 + fps_height + 5  # 5px from top of rectangle
        topic_text_y = fps_text_y + topic_height + 5  # 5px spacing between lines

        # Draw text
        cv2.putText(image, fps_text, (text_x, fps_text_y), font, font_scale, color, thickness)
        cv2.putText(image, topic_text, (text_x, topic_text_y), font, font_scale, color, thickness)

        return image

    def update_fps_calculation(self, topic: str) -> None:
        """Update FPS calculation for a specific topic."""
        self.frame_counts[topic] += 1
        current_time = time.time()

        # Calculate FPS every second
        if current_time - self.fps_start_times[topic] >= 1.0:
            self.current_fps_values[topic] = self.frame_counts[topic] / (current_time - self.fps_start_times[topic])
            self.frame_counts[topic] = 0
            self.fps_start_times[topic] = current_time

    def save_screenshot(self, image: np.ndarray, topic: str) -> None:
        """Save current image as screenshot with topic identifier."""
        timestamp = int(time.time())
        topic_name = topic.replace('/', '_').replace('~', '')
        filename = f"gesturebot_screenshot_{topic_name}_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Screenshot saved: {filename}")

    def destroy_node(self) -> None:
        """Clean up OpenCV windows when node is destroyed."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main function for unified image viewer node."""
    rclpy.init(args=args)

    try:
        node = UnifiedImageViewerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
