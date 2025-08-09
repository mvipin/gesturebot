#!/usr/bin/env python3
"""
Object Detection Node for GestureBot Vision System
Enhanced MediaPipe object detection with navigation integration.
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig, MediaPipeCallbackMixin
from vision_core.message_converter import MessageConverter
from gesturebot.msg import DetectedObjects


class ObjectDetectionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time object detection using MediaPipe EfficientDet.
    Integrates with navigation system for obstacle avoidance and object tracking.
    """
    
    def __init__(self):
        # Configuration for object detection
        config = ProcessingConfig(
            enabled=True,
            max_fps=15.0,
            frame_skip=1,
            confidence_threshold=0.5,
            max_results=5,
            priority=0  # Critical priority for navigation safety
        )
        
        super().__init__('object_detection_node', 'object_detection', config)
        MediaPipeCallbackMixin.__init__(self)
        
        # MediaPipe components
        self.detector = None
        
        # Model path
        self.model_path = self.get_model_path()
        
        # Publishers
        self.detections_publisher = self.create_publisher(
            DetectedObjects,
            '/vision/objects',
            self.result_qos
        )
        
        # Parameters
        self.declare_parameter('model_path', str(self.model_path))
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('max_results', 5)
        
        self.get_logger().info('Object Detection Node initialized')
    
    def get_model_path(self) -> Path:
        """Get the path to the EfficientDet model."""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent.parent / 'models' / 'efficientdet.tflite',
            Path.home() / 'GestureBot' / 'mediapipe-test' / 'efficientdet.tflite',
            Path('/opt/ros/jazzy/share/gesturebot/models/efficientdet.tflite'),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default path (may not exist yet)
        return Path(__file__).parent.parent.parent / 'models' / 'efficientdet.tflite'
    
    def initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe object detector."""
        try:
            model_path = Path(self.get_parameter('model_path').value)
            
            if not model_path.exists():
                self.get_logger().error(f'Model file not found: {model_path}')
                return False
            
            # Initialize object detector
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=mp_vis.RunningMode.LIVE_STREAM,
                max_results=self.get_parameter('max_results').value,
                score_threshold=self.get_parameter('confidence_threshold').value,
                result_callback=self.create_callback('detection')
            )
            
            self.detector = mp_vis.ObjectDetector.create_from_options(options)
            
            self.get_logger().info('MediaPipe object detector initialized successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MediaPipe: {e}')
            return False
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """Process frame for object detection."""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run detection asynchronously
            timestamp_ms = int(timestamp * 1000)
            if self.detector:
                self.detector.detect_async(mp_image, timestamp_ms)
            
            # Get results from callback
            callback_results = self.get_callback_results()
            
            if callback_results and callback_results['result'].detections:
                return {
                    'detections': callback_results['result'].detections,
                    'timestamp': timestamp,
                    'processing_time': (time.time() - timestamp) * 1000
                }
            
            return None
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            return None
    
    def publish_results(self, results: Dict, timestamp: float) -> None:
        """Publish object detection results."""
        try:
            # Convert to ROS message
            msg = MessageConverter.mediapipe_detections_to_ros(
                results['detections'],
                'efficientdet',
                results['processing_time']
            )
            
            # Set header
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            
            # Publish
            self.detections_publisher.publish(msg)
            
            # Log significant detections
            if msg.total_detections > 0:
                detection_summary = []
                for obj in msg.objects:
                    if obj.confidence > 0.7:  # High confidence detections
                        detection_summary.append(f"{obj.class_name}({obj.confidence:.2f})")
                
                if detection_summary:
                    self.get_logger().info(f'High confidence detections: {", ".join(detection_summary)}')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing results: {e}')
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        try:
            if self.detector:
                self.detector.close()
            super().cleanup()
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')


def main(args=None):
    """Main function for object detection node."""
    import rclpy
    
    rclpy.init(args=args)
    
    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in object detection node: {e}')
    finally:
        if 'node' in locals():
            node.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
