#!/usr/bin/env python3
"""
Test Camera Integration and Image Publishing
Validates camera input, image capture, and ROS 2 image publishing functionality.
Updated for gesturebot package (renamed from gesturebot_vision).
"""

import os
import sys
import time
import unittest
import threading
import subprocess
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class MockCameraPublisher(Node):
    """Mock camera publisher for testing without physical hardware."""
    
    def __init__(self, test_images: List[np.ndarray]):
        super().__init__('mock_camera_publisher')
        
        self.test_images = test_images
        self.current_image_idx = 0
        self.cv_bridge = CvBridge()
        
        # Publishers
        self.raw_image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.compressed_image_pub = self.create_publisher(
            CompressedImage, '/camera/image_raw/compressed', 10
        )
        
        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz
        
        self.frame_count = 0
        self.get_logger().info(f'Mock camera initialized with {len(test_images)} test images')
    
    def publish_image(self):
        """Publish current test image."""
        if not self.test_images:
            return
        
        # Get current image
        current_img = self.test_images[self.current_image_idx]
        self.current_image_idx = (self.current_image_idx + 1) % len(self.test_images)
        
        # Add frame counter
        img_with_counter = current_img.copy()
        cv2.putText(img_with_counter, f'Frame: {self.frame_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        try:
            # Publish raw image
            raw_msg = self.cv_bridge.cv2_to_imgmsg(img_with_counter, 'bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = 'camera_frame'
            self.raw_image_pub.publish(raw_msg)
            
            # Publish compressed image
            compressed_msg = CompressedImage()
            compressed_msg.header = raw_msg.header
            compressed_msg.format = 'jpeg'
            _, compressed_data = cv2.imencode('.jpg', img_with_counter)
            compressed_msg.data = compressed_data.tobytes()
            self.compressed_image_pub.publish(compressed_msg)
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')


class ImageSubscriber(Node):
    """Test subscriber to verify image publishing."""
    
    def __init__(self):
        super().__init__('image_subscriber')
        
        self.cv_bridge = CvBridge()
        self.received_images = []
        self.received_compressed = []
        self.last_raw_time = None
        self.last_compressed_time = None
        
        # Subscribers
        self.raw_sub = self.create_subscription(
            Image, '/camera/image_raw', self.raw_image_callback, 10
        )
        self.compressed_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', 
            self.compressed_image_callback, 10
        )
        
        self.get_logger().info('Image subscriber initialized')
    
    def raw_image_callback(self, msg: Image):
        """Handle raw image messages."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.received_images.append({
                'timestamp': time.time(),
                'image': cv_image,
                'header': msg.header
            })
            self.last_raw_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process raw image: {e}')
    
    def compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed image messages."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.received_compressed.append({
                'timestamp': time.time(),
                'image': cv_image,
                'header': msg.header
            })
            self.last_compressed_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process compressed image: {e}')
    
    def get_stats(self) -> dict:
        """Get reception statistics."""
        return {
            'raw_images_received': len(self.received_images),
            'compressed_images_received': len(self.received_compressed),
            'last_raw_time': self.last_raw_time,
            'last_compressed_time': self.last_compressed_time
        }


class TestCameraIntegration(unittest.TestCase):
    """Test camera integration and image publishing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.package_dir = Path(__file__).parent.parent
        
        # Initialize ROS 2
        rclpy.init()
        
        # Create test images
        cls.test_images = cls.create_test_images()
        
        print("Camera integration test setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        rclpy.shutdown()
    
    @classmethod
    def create_test_images(cls) -> List[np.ndarray]:
        """Create test images for camera simulation."""
        images = []
        
        # Create different test patterns
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if i == 0:  # Checkerboard
                square_size = 40
                for row in range(0, 480, square_size):
                    for col in range(0, 640, square_size):
                        if (row // square_size + col // square_size) % 2 == 0:
                            img[row:row+square_size, col:col+square_size] = [255, 255, 255]
            
            elif i == 1:  # Gradient
                for col in range(640):
                    intensity = int(255 * col / 640)
                    img[:, col] = [intensity, intensity, intensity]
            
            elif i == 2:  # Shapes
                cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
                cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
                cv2.ellipse(img, (500, 100), (60, 40), 0, 0, 360, (0, 0, 255), -1)
            
            elif i == 3:  # Text
                cv2.putText(img, 'CAMERA TEST', (100, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(img, f'Pattern {i}', (100, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            else:  # Random noise
                img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            
            images.append(img)
        
        return images
    
    def test_rpicam_availability(self):
        """Test if rpicam-still is available on the system."""
        try:
            result = subprocess.run(['rpicam-still', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("rpicam-still is available")
            else:
                print("rpicam-still not available, will use mock camera")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("rpicam-still not found, will use mock camera for testing")
    
    def test_mock_camera_publishing(self):
        """Test mock camera image publishing."""
        # Create nodes
        publisher = MockCameraPublisher(self.test_images)
        subscriber = ImageSubscriber()
        
        # Create executor
        executor = SingleThreadedExecutor()
        executor.add_node(publisher)
        executor.add_node(subscriber)
        
        # Run for test duration
        test_duration = 3.0  # seconds
        start_time = time.time()
        
        while (time.time() - start_time) < test_duration:
            executor.spin_once(timeout_sec=0.1)
        
        # Check results
        stats = subscriber.get_stats()
        
        # Assertions
        self.assertGreater(stats['raw_images_received'], 0, 
                          "No raw images received")
        self.assertGreater(stats['compressed_images_received'], 0, 
                          "No compressed images received")
        
        # Check message rate (should be around 10 Hz)
        expected_messages = int(test_duration * 10 * 0.8)  # 80% of expected
        self.assertGreater(stats['raw_images_received'], expected_messages,
                          f"Too few raw images: {stats['raw_images_received']}")
        self.assertGreater(stats['compressed_images_received'], expected_messages,
                          f"Too few compressed images: {stats['compressed_images_received']}")
        
        print(f"Test results: {stats}")
        
        # Clean up
        executor.shutdown()
    
    def test_image_format_validation(self):
        """Test that published images have correct format and properties."""
        publisher = MockCameraPublisher(self.test_images)
        subscriber = ImageSubscriber()
        
        executor = SingleThreadedExecutor()
        executor.add_node(publisher)
        executor.add_node(subscriber)
        
        # Run briefly to get some images
        for _ in range(20):  # About 2 seconds at 10 Hz
            executor.spin_once(timeout_sec=0.1)
        
        # Validate received images
        self.assertGreater(len(subscriber.received_images), 0, "No images received")
        
        # Check first received image
        first_image_data = subscriber.received_images[0]
        image = first_image_data['image']
        header = first_image_data['header']
        
        # Image format checks
        self.assertEqual(image.shape, (480, 640, 3), 
                        f"Wrong image dimensions: {image.shape}")
        self.assertEqual(image.dtype, np.uint8, 
                        f"Wrong image data type: {image.dtype}")
        
        # Header checks
        self.assertEqual(header.frame_id, 'camera_frame', 
                        f"Wrong frame_id: {header.frame_id}")
        self.assertIsNotNone(header.stamp, "Missing timestamp")
        
        # Compressed image checks
        self.assertGreater(len(subscriber.received_compressed), 0, 
                          "No compressed images received")
        
        first_compressed = subscriber.received_compressed[0]
        compressed_image = first_compressed['image']
        
        self.assertEqual(compressed_image.shape, (480, 640, 3), 
                        f"Wrong compressed image dimensions: {compressed_image.shape}")
        
        print(f"Image validation passed - Shape: {image.shape}, Type: {image.dtype}")
        
        # Clean up
        executor.shutdown()
    
    def test_camera_topic_discovery(self):
        """Test that camera topics can be discovered."""
        publisher = MockCameraPublisher(self.test_images)
        
        executor = SingleThreadedExecutor()
        executor.add_node(publisher)
        
        # Let publisher start
        for _ in range(5):
            executor.spin_once(timeout_sec=0.1)
        
        # Check topic discovery
        topic_names = publisher.get_topic_names_and_types()
        topic_dict = dict(topic_names)
        
        # Verify expected topics exist
        self.assertIn('/camera/image_raw', topic_dict, 
                     "Raw image topic not found")
        self.assertIn('/camera/image_raw/compressed', topic_dict, 
                     "Compressed image topic not found")
        
        # Verify topic types
        self.assertEqual(topic_dict['/camera/image_raw'], ['sensor_msgs/msg/Image'],
                        "Wrong raw image topic type")
        self.assertEqual(topic_dict['/camera/image_raw/compressed'], 
                        ['sensor_msgs/msg/CompressedImage'],
                        "Wrong compressed image topic type")
        
        print(f"Camera topics discovered: {list(topic_dict.keys())}")
        
        # Clean up
        executor.shutdown()
    
    def test_cv_bridge_functionality(self):
        """Test OpenCV-ROS bridge functionality."""
        cv_bridge = CvBridge()
        
        # Test image conversion
        test_image = self.test_images[0]
        
        try:
            # Convert to ROS message
            ros_image = cv_bridge.cv2_to_imgmsg(test_image, 'bgr8')
            
            # Convert back to OpenCV
            converted_image = cv_bridge.imgmsg_to_cv2(ros_image, 'bgr8')
            
            # Verify conversion
            self.assertEqual(test_image.shape, converted_image.shape,
                           "Image shape changed during conversion")
            
            # Check if images are identical (allowing for small differences)
            diff = cv2.absdiff(test_image, converted_image)
            max_diff = np.max(diff)
            self.assertLess(max_diff, 5, f"Images differ too much: {max_diff}")
            
            print("CV Bridge conversion test passed")
            
        except Exception as e:
            self.fail(f"CV Bridge conversion failed: {e}")


def main():
    """Run camera integration tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
