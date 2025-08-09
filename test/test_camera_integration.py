#!/usr/bin/env python3
"""
Test Camera Integration and Image Publishing
Validates real Pi Camera hardware, image capture, and ROS 2 image publishing functionality.
Tests actual camera_ros node integration with Raspberry Pi Camera Module.
"""

import os
import sys
import time
import unittest
import threading
import subprocess
import signal
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge


class CameraTestSubscriber(Node):
    """Test subscriber to verify real camera image publishing."""

    def __init__(self):
        super().__init__('camera_test_subscriber')

        self.cv_bridge = CvBridge()
        self.received_images = []
        self.received_compressed = []
        self.received_camera_info = []
        self.last_raw_time = None
        self.last_compressed_time = None
        self.last_info_time = None

        # Subscribers for real camera topics
        self.raw_sub = self.create_subscription(
            Image, '/camera/image_raw', self.raw_image_callback, 10
        )
        self.compressed_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self.compressed_image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.camera_info_callback, 10
        )

        self.get_logger().info('Camera test subscriber initialized')
    
    def raw_image_callback(self, msg: Image):
        """Handle raw image messages from real camera."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.received_images.append({
                'timestamp': time.time(),
                'image': cv_image,
                'header': msg.header,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding
            })
            self.last_raw_time = time.time()

        except Exception as e:
            self.get_logger().error(f'Failed to process raw image: {e}')

    def compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed image messages from real camera."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is not None:
                self.received_compressed.append({
                    'timestamp': time.time(),
                    'image': cv_image,
                    'header': msg.header,
                    'format': msg.format,
                    'data_size': len(msg.data)
                })
                self.last_compressed_time = time.time()

        except Exception as e:
            self.get_logger().error(f'Failed to process compressed image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages from real camera."""
        self.received_camera_info.append({
            'timestamp': time.time(),
            'width': msg.width,
            'height': msg.height,
            'distortion_model': msg.distortion_model,
            'header': msg.header
        })
        self.last_info_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about received images from real camera."""
        latest_raw = self.received_images[-1] if self.received_images else None
        latest_compressed = self.received_compressed[-1] if self.received_compressed else None
        latest_info = self.received_camera_info[-1] if self.received_camera_info else None

        return {
            'raw_images_received': len(self.received_images),
            'compressed_images_received': len(self.received_compressed),
            'camera_info_received': len(self.received_camera_info),
            'last_raw_time': self.last_raw_time,
            'last_compressed_time': self.last_compressed_time,
            'last_info_time': self.last_info_time,
            'latest_raw_resolution': (latest_raw['width'], latest_raw['height']) if latest_raw else None,
            'latest_compressed_size': latest_compressed['data_size'] if latest_compressed else None,
            'camera_resolution': (latest_info['width'], latest_info['height']) if latest_info else None
        }


class TestCameraIntegration(unittest.TestCase):
    """Test real camera integration and image publishing."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.package_dir = Path(__file__).parent.parent

        # Initialize ROS 2
        rclpy.init()

        # Camera process placeholder
        cls.camera_process = None

        print("Real camera integration test setup complete")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Ensure camera process is terminated
        if cls.camera_process and cls.camera_process.poll() is None:
            cls.camera_process.terminate()
            cls.camera_process.wait(timeout=5)

        rclpy.shutdown()

    def start_camera_node(self, timeout: int = 10) -> subprocess.Popen:
        """Start the camera_ros node and wait for it to be ready."""
        print("🚀 Starting camera_ros node...")

        # Start camera_ros node
        process = subprocess.Popen(
            ['ros2', 'run', 'camera_ros', 'camera_node'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for clean termination
        )

        # Wait for node to start and begin publishing
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Camera node failed to start: {stderr.decode()}")

            # Check if topics are available
            result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True, text=True
            )
            if '/camera/image_raw' in result.stdout:
                print("✅ Camera node started and topics available")
                time.sleep(2)  # Give it time to stabilize
                return process

            time.sleep(0.5)

        # Timeout reached
        process.terminate()
        raise TimeoutError(f"Camera node failed to start within {timeout} seconds")

    def stop_camera_node(self, process: subprocess.Popen):
        """Stop the camera_ros node cleanly."""
        if process and process.poll() is None:
            print("🛑 Stopping camera_ros node...")
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)
                print("✅ Camera node stopped cleanly")
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
                print("⚠️ Camera node force-killed")
    
    def test_rpicam_hardware_detection(self):
        """Test if Pi Camera hardware is properly detected."""
        print("🔍 Testing Pi Camera hardware detection...")

        # Test rpicam-still availability
        try:
            result = subprocess.run(['rpicam-still', '--list-cameras'],
                                  capture_output=True, text=True, timeout=10)

            self.assertEqual(result.returncode, 0, "rpicam-still command failed")
            self.assertIn("Available cameras", result.stdout, "No cameras detected")
            print("✅ Pi Camera hardware detected")
            print(f"Camera info: {result.stdout.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.fail(f"rpicam-still not available: {e}")

    def test_rpicam_hardware_availability(self):
        """Test that rpicam-still is available (without capturing to avoid resource conflicts)."""
        print("🔧 Testing rpicam-still availability...")

        try:
            # Test rpicam-still help command (doesn't access camera hardware)
            result = subprocess.run(['rpicam-still', '--help'],
                                  capture_output=True, text=True, timeout=10)

            self.assertEqual(result.returncode, 0, "rpicam-still command not available")
            self.assertIn("--help", result.stdout.lower(), "rpicam-still help output invalid")

            print("✅ rpicam-still tool available and functional")
            print("ℹ️  Note: Direct capture test skipped to avoid resource conflicts with camera_ros")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.fail(f"rpicam-still not available: {e}")

    def test_camera_ros_node_publishing(self):
        """Test real camera_ros node image publishing."""
        print("📡 Testing camera_ros node with real hardware...")

        # Start camera_ros node
        camera_process = None
        try:
            camera_process = self.start_camera_node(timeout=15)

            # Create subscriber
            subscriber = CameraTestSubscriber()
            executor = SingleThreadedExecutor()
            executor.add_node(subscriber)

            # Run for test duration to collect images
            test_duration = 5.0  # seconds
            start_time = time.time()

            print(f"📊 Collecting camera data for {test_duration} seconds...")
            while (time.time() - start_time) < test_duration:
                executor.spin_once(timeout_sec=0.1)

            # Check results
            stats = subscriber.get_stats()
            print(f"📈 Camera statistics: {stats}")

            # Assertions for real camera
            self.assertGreater(stats['raw_images_received'], 0,
                              "No raw images received from camera")
            self.assertGreater(stats['compressed_images_received'], 0,
                              "No compressed images received from camera")
            self.assertGreater(stats['camera_info_received'], 0,
                              "No camera info received")

            # Verify image properties from real camera
            if stats['latest_raw_resolution']:
                width, height = stats['latest_raw_resolution']
                self.assertGreater(width, 0, "Invalid image width")
                self.assertGreater(height, 0, "Invalid image height")
                print(f"✅ Real camera resolution: {width}x{height}")

            # Verify compressed image size is reasonable
            if stats['latest_compressed_size']:
                size_kb = stats['latest_compressed_size'] / 1024
                self.assertGreater(size_kb, 10, "Compressed image too small")
                self.assertLess(size_kb, 1000, "Compressed image too large")
                print(f"✅ Compressed image size: {size_kb:.1f} KB")

            # Clean up
            executor.shutdown()

        finally:
            # Ensure camera process is stopped
            if camera_process:
                self.stop_camera_node(camera_process)
    
    def test_camera_topic_discovery(self):
        """Test that real camera topics can be discovered."""
        print("🔍 Testing camera topic discovery...")

        camera_process = None
        try:
            # Start camera_ros node
            camera_process = self.start_camera_node(timeout=15)

            # Wait a moment for topics to be established
            time.sleep(2)

            # Check topic discovery using ros2 topic list
            result = subprocess.run(['ros2', 'topic', 'list'],
                                  capture_output=True, text=True, timeout=10)

            self.assertEqual(result.returncode, 0, "Failed to list topics")

            topics = result.stdout.strip().split('\n')

            # Verify expected camera topics exist
            self.assertIn('/camera/image_raw', topics, "Raw image topic not found")
            self.assertIn('/camera/image_raw/compressed', topics, "Compressed image topic not found")
            self.assertIn('/camera/camera_info', topics, "Camera info topic not found")

            print(f"✅ Camera topics discovered: {[t for t in topics if 'camera' in t]}")

        finally:
            if camera_process:
                self.stop_camera_node(camera_process)

    def test_real_camera_image_properties(self):
        """Test real camera image format and properties."""
        print("📊 Testing real camera image properties...")

        camera_process = None
        try:
            # Start camera_ros node
            camera_process = self.start_camera_node(timeout=15)

            # Create subscriber
            subscriber = CameraTestSubscriber()
            executor = SingleThreadedExecutor()
            executor.add_node(subscriber)

            # Collect a few images
            print("📸 Collecting sample images...")
            for _ in range(30):  # About 3 seconds
                executor.spin_once(timeout_sec=0.1)
                if len(subscriber.received_images) >= 3:
                    break

            # Validate received images
            self.assertGreater(len(subscriber.received_images), 0, "No images received from camera")

            # Check image properties
            first_image_data = subscriber.received_images[0]
            image = first_image_data['image']

            # Real camera image validation
            self.assertEqual(len(image.shape), 3, "Image should be 3-channel (BGR)")
            self.assertEqual(image.dtype, np.uint8, f"Wrong image data type: {image.dtype}")
            self.assertGreater(image.shape[0], 100, "Image height too small")
            self.assertGreater(image.shape[1], 100, "Image width too small")

            # Check that image has actual content (not all zeros)
            self.assertGreater(np.sum(image), 1000, "Image appears to be blank")

            print(f"✅ Real camera image properties validated:")
            print(f"   Resolution: {image.shape[1]}x{image.shape[0]}")
            print(f"   Encoding: {first_image_data['encoding']}")
            print(f"   Data type: {image.dtype}")

            # Clean up
            executor.shutdown()

        finally:
            if camera_process:
                self.stop_camera_node(camera_process)

    def test_cv_bridge_with_real_camera(self):
        """Test OpenCV-ROS bridge functionality with real camera data."""
        print("🌉 Testing CV Bridge with real camera...")

        camera_process = None
        try:
            # Start camera_ros node
            camera_process = self.start_camera_node(timeout=15)

            # Create subscriber
            subscriber = CameraTestSubscriber()
            executor = SingleThreadedExecutor()
            executor.add_node(subscriber)

            # Collect one image
            print("📸 Capturing real camera image for CV Bridge test...")
            for _ in range(20):
                executor.spin_once(timeout_sec=0.1)
                if len(subscriber.received_images) >= 1:
                    break

            self.assertGreater(len(subscriber.received_images), 0, "No real camera images received")

            # Test CV Bridge conversion with real camera data
            cv_bridge = CvBridge()
            real_image = subscriber.received_images[0]['image']

            # Convert real camera image to ROS message and back
            ros_image = cv_bridge.cv2_to_imgmsg(real_image, 'bgr8')
            converted_image = cv_bridge.imgmsg_to_cv2(ros_image, 'bgr8')

            # Verify conversion preserves image properties
            self.assertEqual(real_image.shape, converted_image.shape,
                           "Image shape changed during conversion")
            self.assertEqual(real_image.dtype, converted_image.dtype,
                           "Image data type changed during conversion")

            # Check conversion accuracy (should be identical for lossless conversion)
            diff = cv2.absdiff(real_image, converted_image)
            max_diff = np.max(diff)
            self.assertEqual(max_diff, 0, f"CV Bridge conversion not lossless: max_diff={max_diff}")

            print("✅ CV Bridge conversion test passed with real camera data")

            # Clean up
            executor.shutdown()

        finally:
            if camera_process:
                self.stop_camera_node(camera_process)


def main():
    """Run camera integration tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
