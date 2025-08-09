#!/usr/bin/env python3
"""
Test Vision Node Functionality
Validates individual vision processing nodes and their ROS 2 integration.
"""

import os
import sys
import time
import unittest
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig
from vision_core.message_converter import MessageConverter
from gesturebot.msg import DetectedObjects, HandGesture, VisionPerformance


class TestVisionNode(MediaPipeBaseNode):
    """Test implementation of MediaPipeBaseNode for testing."""
    
    def __init__(self):
        config = ProcessingConfig(
            enabled=True,
            max_fps=10.0,
            frame_skip=1,
            confidence_threshold=0.5
        )
        super().__init__('test_vision_node', 'test_feature', config)
        
        self.processed_frames = 0
        self.processing_times = []
        
        # Test publisher
        self.test_results_pub = self.create_publisher(
            DetectedObjects, '/test/results', 10
        )
    
    def initialize_mediapipe(self) -> bool:
        """Initialize test MediaPipe components."""
        # Simulate MediaPipe initialization
        time.sleep(0.1)  # Simulate loading time
        return True
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Any:
        """Process frame for testing."""
        start_time = time.time()
        
        # Simulate processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        self.processed_frames += 1
        
        return {
            'contours': contours,
            'processing_time': processing_time,
            'frame_number': self.processed_frames
        }
    
    def publish_results(self, results: Any, timestamp: float) -> None:
        """Publish test results."""
        # Create test message
        msg = DetectedObjects()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.detector_name = 'test_detector'
        msg.processing_time = results['processing_time']
        msg.total_detections = len(results['contours'])
        
        self.test_results_pub.publish(msg)


class VisionResultsCollector(Node):
    """Collector node for vision processing results."""
    
    def __init__(self):
        super().__init__('vision_results_collector')
        
        self.detected_objects = []
        self.hand_gestures = []
        self.performance_data = []
        
        # Subscribers
        self.objects_sub = self.create_subscription(
            DetectedObjects, '/vision/objects', self.objects_callback, 10
        )
        self.test_objects_sub = self.create_subscription(
            DetectedObjects, '/test/results', self.test_objects_callback, 10
        )
        self.gestures_sub = self.create_subscription(
            HandGesture, '/vision/gestures', self.gestures_callback, 10
        )
        self.performance_sub = self.create_subscription(
            VisionPerformance, '/vision/test_feature/performance', 
            self.performance_callback, 10
        )
        
        self.get_logger().info('Vision results collector initialized')
    
    def objects_callback(self, msg: DetectedObjects):
        """Handle object detection results."""
        self.detected_objects.append({
            'timestamp': time.time(),
            'total_detections': msg.total_detections,
            'processing_time': msg.processing_time,
            'detector_name': msg.detector_name
        })
    
    def test_objects_callback(self, msg: DetectedObjects):
        """Handle test object detection results."""
        self.detected_objects.append({
            'timestamp': time.time(),
            'total_detections': msg.total_detections,
            'processing_time': msg.processing_time,
            'detector_name': msg.detector_name,
            'test_result': True
        })
    
    def gestures_callback(self, msg: HandGesture):
        """Handle gesture recognition results."""
        self.hand_gestures.append({
            'timestamp': time.time(),
            'gesture_name': msg.gesture_name,
            'confidence': msg.confidence,
            'handedness': msg.handedness
        })
    
    def performance_callback(self, msg: VisionPerformance):
        """Handle performance metrics."""
        self.performance_data.append({
            'timestamp': time.time(),
            'feature_name': msg.feature_name,
            'frames_processed': msg.frames_processed,
            'avg_processing_time': msg.avg_processing_time,
            'current_fps': msg.current_fps,
            'processing_healthy': msg.processing_healthy
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'objects_received': len(self.detected_objects),
            'gestures_received': len(self.hand_gestures),
            'performance_updates': len(self.performance_data),
            'latest_objects': self.detected_objects[-1] if self.detected_objects else None,
            'latest_gesture': self.hand_gestures[-1] if self.hand_gestures else None,
            'latest_performance': self.performance_data[-1] if self.performance_data else None
        }


class MockImagePublisher(Node):
    """Mock image publisher for vision node testing."""
    
    def __init__(self, test_images: list):
        super().__init__('mock_image_publisher')
        
        self.test_images = test_images
        self.current_idx = 0
        self.cv_bridge = CvBridge()
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz
        
        self.get_logger().info('Mock image publisher initialized')
    
    def publish_image(self):
        """Publish test image."""
        if not self.test_images:
            return
        
        current_img = self.test_images[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.test_images)
        
        try:
            msg = self.cv_bridge.cv2_to_imgmsg(current_img, 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.image_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')


class TestVisionNodes(unittest.TestCase):
    """Test vision node functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.package_dir = Path(__file__).parent.parent
        
        # Initialize ROS 2
        rclpy.init()
        
        # Create test images
        cls.test_images = cls.create_test_images()
        
        print("Vision nodes test setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        rclpy.shutdown()
    
    @classmethod
    def create_test_images(cls) -> list:
        """Create test images with various features."""
        images = []
        
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add different features for detection
            if i == 0:  # Geometric shapes
                cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
                cv2.circle(img, (400, 150), 50, (0, 255, 0), -1)
                cv2.ellipse(img, (300, 350), (80, 60), 0, 0, 360, (0, 0, 255), -1)
            
            elif i == 1:  # Lines and contours
                cv2.line(img, (50, 50), (590, 50), (255, 255, 255), 5)
                cv2.line(img, (50, 430), (590, 430), (255, 255, 255), 5)
                cv2.line(img, (50, 50), (50, 430), (255, 255, 255), 5)
                cv2.line(img, (590, 50), (590, 430), (255, 255, 255), 5)
                
                # Add some internal shapes
                cv2.rectangle(img, (200, 200), (440, 280), (128, 128, 128), -1)
            
            else:  # Complex pattern
                # Create a more complex pattern with multiple contours
                for j in range(5):
                    x = 50 + j * 120
                    y = 100 + (j % 2) * 200
                    cv2.rectangle(img, (x, y), (x + 80, y + 80), 
                                (255 - j * 40, j * 50, 128), -1)
            
            images.append(img)
        
        return images
    
    def test_base_node_initialization(self):
        """Test MediaPipeBaseNode initialization."""
        try:
            test_node = TestVisionNode()
            
            # Check node properties
            self.assertEqual(test_node.get_name(), 'test_vision_node')
            self.assertEqual(test_node.feature_name, 'test_feature')
            self.assertTrue(test_node.config.enabled)
            
            # Check that node has required components
            self.assertIsNotNone(test_node.cv_bridge)
            self.assertIsNotNone(test_node.stats)
            
            print("Base node initialization test passed")
            
        except Exception as e:
            self.fail(f"Base node initialization failed: {e}")
    
    def test_vision_processing_pipeline(self):
        """Test complete vision processing pipeline."""
        # Create nodes
        image_publisher = MockImagePublisher(self.test_images)
        vision_node = TestVisionNode()
        results_collector = VisionResultsCollector()
        
        # Create executor
        executor = SingleThreadedExecutor()
        executor.add_node(image_publisher)
        executor.add_node(vision_node)
        executor.add_node(results_collector)
        
        # Run processing
        test_duration = 3.0  # seconds
        start_time = time.time()
        
        while (time.time() - start_time) < test_duration:
            executor.spin_once(timeout_sec=0.1)
        
        # Check results
        stats = results_collector.get_stats()
        
        # Assertions
        self.assertGreater(stats['objects_received'], 0, 
                          "No object detection results received")
        self.assertGreater(stats['performance_updates'], 0, 
                          "No performance updates received")
        
        # Check processing performance
        self.assertGreater(vision_node.processed_frames, 0, 
                          "No frames processed")
        
        if vision_node.processing_times:
            avg_processing_time = sum(vision_node.processing_times) / len(vision_node.processing_times)
            self.assertLess(avg_processing_time, 100, 
                           f"Processing too slow: {avg_processing_time:.1f}ms")
        
        print(f"Vision processing test results: {stats}")
        print(f"Processed {vision_node.processed_frames} frames")
        
        # Clean up
        executor.shutdown()
    
    def test_message_converter_functionality(self):
        """Test message converter utility functions."""
        try:
            # Test point conversion
            test_point = MessageConverter.pixel_coords_to_point(320, 240, 640, 480)
            self.assertAlmostEqual(test_point.x, 0.5, places=2)
            self.assertAlmostEqual(test_point.y, 0.5, places=2)
            
            # Test reverse conversion
            pixel_coords = MessageConverter.point_to_pixel_coords(test_point, 640, 480)
            self.assertEqual(pixel_coords, (320, 240))
            
            # Test OpenCV contour conversion
            test_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
            detected_obj = MessageConverter.opencv_contour_to_detected_object(
                test_contour, 'test_object', 0.8
            )
            
            self.assertEqual(detected_obj.class_name, 'test_object')
            self.assertEqual(detected_obj.confidence, 0.8)
            self.assertGreater(detected_obj.bbox_width, 0)
            self.assertGreater(detected_obj.bbox_height, 0)
            
            print("Message converter functionality test passed")
            
        except Exception as e:
            self.fail(f"Message converter test failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        vision_node = TestVisionNode()
        results_collector = VisionResultsCollector()
        
        executor = SingleThreadedExecutor()
        executor.add_node(vision_node)
        executor.add_node(results_collector)
        
        # Simulate some processing by calling the performance timer
        for _ in range(3):
            vision_node.publish_performance_stats()
            executor.spin_once(timeout_sec=0.1)
        
        # Check performance data
        stats = results_collector.get_stats()
        
        self.assertGreater(stats['performance_updates'], 0, 
                          "No performance updates received")
        
        if stats['latest_performance']:
            perf_data = stats['latest_performance']
            self.assertEqual(perf_data['feature_name'], 'test_feature')
            self.assertIsInstance(perf_data['frames_processed'], int)
            self.assertIsInstance(perf_data['avg_processing_time'], float)
            self.assertIsInstance(perf_data['current_fps'], float)
        
        print("Performance monitoring test passed")
        
        # Clean up
        executor.shutdown()
    
    def test_node_lifecycle_management(self):
        """Test node lifecycle and cleanup."""
        try:
            vision_node = TestVisionNode()
            
            # Test enable/disable functionality
            self.assertTrue(vision_node.config.enabled)
            
            vision_node.disable_feature()
            self.assertFalse(vision_node.config.enabled)
            
            vision_node.enable_feature()
            self.assertTrue(vision_node.config.enabled)
            
            # Test configuration update
            new_config = ProcessingConfig(
                enabled=True,
                max_fps=15.0,
                confidence_threshold=0.7
            )
            vision_node.update_config(new_config)
            
            self.assertEqual(vision_node.config.max_fps, 15.0)
            self.assertEqual(vision_node.config.confidence_threshold, 0.7)
            
            # Test cleanup
            vision_node.cleanup()
            
            print("Node lifecycle management test passed")
            
        except Exception as e:
            self.fail(f"Node lifecycle test failed: {e}")


def main():
    """Run vision node tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
