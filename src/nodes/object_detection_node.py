#!/usr/bin/env python3
"""
Object Detection Node for GestureBot Vision System
Enhanced MediaPipe object detection with navigation integration.
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path
# Removed deque, threading, datetime imports - now using BufferedLogger from base_node

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig, MediaPipeCallbackMixin, PerformanceStats, PipelineTimer
from vision_core.message_converter import MessageConverter
from gesturebot.msg import DetectedObjects


# BufferedLogger is now imported from vision_core.base_node


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

        # MediaPipe components
        self.detector = None

        # Model path (get default before calling parent init)
        self.model_path = self.get_model_path()

        # Initialize parent with default values (will be updated after parameter declaration)
        super().__init__(
            'object_detection_node',
            'object_detection',
            config,
            enable_buffered_logging=True,  # Default, will be updated
            unlimited_buffer_mode=False,   # Default, will be updated
            enable_performance_tracking=False  # Default, will be updated
        )
        MediaPipeCallbackMixin.__init__(self)

        # Declare parameters for this node
        self.declare_parameter('unlimited_buffer_mode', False)
        self.declare_parameter('buffer_logging_enabled', True)
        self.declare_parameter('enable_performance_tracking', False)

        # Update performance tracking setting from parameter
        performance_tracking_enabled = self.get_parameter('enable_performance_tracking').get_parameter_value().bool_value

        # Update parent class performance tracking setting
        self.enable_performance_tracking = performance_tracking_enabled

        # Re-initialize performance tracking components if enabled
        if self.enable_performance_tracking:
            self.stats = PerformanceStats()
            self.stats.period_start_time = time.perf_counter()
            self.pipeline_timer = PipelineTimer()
            self.timing_history = []
            self.max_timing_history = 20
            self.get_logger().info("Performance tracking enabled")
        else:
            self.get_logger().info("Performance tracking disabled")

        # Update buffered logging setting from parameter
        buffer_logging_enabled = self.get_parameter('buffer_logging_enabled').get_parameter_value().bool_value

        # Update buffered logger enabled state
        if hasattr(self, 'buffered_logger'):
            self.buffered_logger.enabled = buffer_logging_enabled
            if buffer_logging_enabled:
                self.get_logger().info("Buffered logging enabled")
            else:
                self.get_logger().info("Buffered logging disabled - only critical errors will be logged directly")

        # Log the buffer configuration (using inherited buffered logger)
        buffer_stats = self.get_buffer_stats()
        self.get_logger().info(f"BufferedLogger initialized: {buffer_stats}")

        # Parameters (declare after parent init)
        self.declare_parameter('model_path', str(self.model_path))
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('max_results', 5)
        self.declare_parameter('publish_annotated_images', False)

        # Publishers
        self.detections_publisher = self.create_publisher(
            DetectedObjects,
            '/vision/objects',
            self.result_qos
        )

        # Conditional annotated image publisher
        self.annotated_image_publisher = None
        # Note: cv_bridge is already initialized in the base class

        if self.get_parameter('publish_annotated_images').value:
            self.annotated_image_publisher = self.create_publisher(
                Image,
                '/vision/objects/annotated',
                self.result_qos
            )
            self.get_logger().info('Annotated image publishing enabled on /vision/objects/annotated')
        else:
            self.get_logger().info('Annotated image publishing disabled')

        self.get_logger().info('Object Detection Node initialized')
    
    def get_model_path(self) -> Path:
        """Get the path to the EfficientDet model."""
        # Try multiple possible locations
        possible_paths = [
            # Installed location (preferred)
            Path('/home/pi/GestureBot/gesturebot_ws/install/gesturebot/share/gesturebot/models/efficientdet.tflite'),
            # Source location (development)
            Path(__file__).parent.parent.parent / 'models' / 'efficientdet.tflite',
            # Alternative locations
            Path.home() / 'GestureBot' / 'mediapipe-test' / 'efficientdet.tflite',
            Path('/opt/ros/jazzy/share/gesturebot/models/efficientdet.tflite'),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Default path (may not exist yet)
        return Path('/home/pi/GestureBot/gesturebot_ws/install/gesturebot/share/gesturebot/models/efficientdet.tflite')
    
    def initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe object detector."""
        try:
            # Try to get parameter, fall back to default if not declared yet
            try:
                model_path = Path(self.get_parameter('model_path').value)
            except:
                # Parameter not declared yet, use default
                model_path = self.model_path
            
            if not model_path.exists():
                self.get_logger().error(f'Model file not found: {model_path}')
                return False
            
            # Initialize object detector
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))

            # Try to get parameters, fall back to defaults if not declared yet
            try:
                max_results = self.get_parameter('max_results').value
            except:
                max_results = 5  # Default value

            try:
                confidence_threshold = self.get_parameter('confidence_threshold').value
            except:
                confidence_threshold = 0.5  # Default value

            options = mp_vis.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=mp_vis.RunningMode.LIVE_STREAM,
                max_results=max_results,
                score_threshold=confidence_threshold,
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
            # Optimization: Check if frame is already in RGB format (camera_format=RGB888)
            # If camera outputs RGB888, we can skip the expensive BGR→RGB conversion
            if frame.shape[2] == 3:  # Ensure it's a 3-channel image
                # Assume RGB input from camera (when camera_format=RGB888)
                # This eliminates the expensive cv2.cvtColor() preprocessing step
                rgb_frame = frame
                self.log_buffered_event(
                    'PREPROCESSING_OPTIMIZED',
                    'Using direct RGB input - skipping BGR→RGB conversion',
                    frame_shape=str(frame.shape)
                )
            else:
                # Fallback: Convert BGR to RGB for MediaPipe (legacy support)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.log_buffered_event(
                    'PREPROCESSING_FALLBACK',
                    'Applied BGR→RGB conversion (fallback mode)',
                    frame_shape=str(frame.shape)
                )

            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run detection asynchronously
            timestamp_ms = int(timestamp * 1000)
            if self.detector:
                self.detector.detect_async(mp_image, timestamp_ms)
            else:
                self.get_logger().error(f'No detector available!')
                return None

            # Get results from callback
            callback_results = self.get_callback_results()

            if callback_results and 'result' in callback_results and callback_results['result']:
                detections = callback_results['result'].detections

                if detections:
                    result_dict = {
                        'detections': detections,
                        'timestamp': timestamp,
                        'processing_time': (time.time() - timestamp) * 1000
                    }

                    # Include output_image if available (for annotated image publishing)
                    if 'output_image' in callback_results:
                        result_dict['output_image'] = callback_results['output_image']
                        self.log_buffered_event(
                            'OUTPUT_IMAGE_RECEIVED',
                            'MediaPipe output image included in results',
                            image_valid=callback_results["output_image"] is not None,
                            image_type=type(callback_results["output_image"]).__name__
                        )
                    else:
                        self.log_buffered_event(
                            'OUTPUT_IMAGE_MISSING',
                            'No output_image in callback_results'
                        )

                    return result_dict
                else:
                    self.get_logger().debug(f'[ObjectDetection] No detections found')
            else:
                self.get_logger().debug(f'[ObjectDetection] No callback results received')

            return None

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            return None
    
    def publish_results(self, results: Dict, timestamp: float) -> None:
        """Publish object detection results and optionally annotated images."""
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

            # Publish detection results
            self.detections_publisher.publish(msg)

            # Conditional annotated image publishing (Optimization: Check subscriber count first)
            if (self.annotated_image_publisher is not None and
                'output_image' in results and
                results['output_image'] is not None):

                # Optimization: Skip expensive postprocessing if no subscribers
                subscriber_count = self.annotated_image_publisher.get_subscription_count()

                self.log_buffered_event(
                    'PUBLISH_CONDITIONS_CHECK',
                    'Checking annotated image publishing conditions',
                    publisher_exists=self.annotated_image_publisher is not None,
                    output_image_in_results="output_image" in results,
                    output_image_not_none=results.get("output_image") is not None,
                    subscriber_count=subscriber_count
                )

                if subscriber_count == 0:
                    self.log_buffered_event(
                        'ANNOTATED_PROCESSING_SKIPPED',
                        'Skipping annotated image processing - no subscribers',
                        subscriber_count=subscriber_count
                    )
                    return

                # Continue with annotated image processing (subscribers present)
                self.log_buffered_event(
                    'ANNOTATED_PROCESSING_CONTINUE',
                    'Continuing with annotated image processing - subscribers present',
                    subscriber_count=subscriber_count
                )

                try:
                    self.log_buffered_event(
                        'IMAGE_PROCESSING_START',
                        'Starting annotated image processing'
                    )

                    output_image = results['output_image']
                    self.log_buffered_event(
                        'OUTPUT_IMAGE_TYPE',
                        'Retrieved output image from results',
                        image_type=type(output_image).__name__
                    )

                    # Convert MediaPipe image to OpenCV format
                    if hasattr(output_image, 'numpy_view'):
                        # MediaPipe Image object
                        cv_image = output_image.numpy_view()
                        self.log_buffered_event(
                            'MEDIAPIPE_TO_NUMPY',
                            'Converted MediaPipe image to numpy view',
                            shape=str(cv_image.shape) if cv_image is not None else "None",
                            success=cv_image is not None
                        )
                    else:
                        # Already numpy array
                        cv_image = np.array(output_image)
                        self.log_buffered_event(
                            'ARRAY_CONVERSION',
                            'Converted to numpy array',
                            shape=str(cv_image.shape) if cv_image is not None else "None",
                            success=cv_image is not None
                        )

                    # Ensure we have a valid image
                    if cv_image is None or cv_image.size == 0:
                        self.log_buffered_event(
                            'IMAGE_VALIDATION_FAILED',
                            'Empty or invalid annotated image, skipping publish',
                            image_none=cv_image is None,
                            image_size=cv_image.size if cv_image is not None else 0
                        )
                        return

                    # Optimization: Smart color conversion based on camera format
                    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                        # When camera_format=RGB888, MediaPipe outputs RGB, need to convert to BGR for ROS
                        # When camera_format=YUYV (BGR), MediaPipe outputs RGB, need to convert to BGR for ROS
                        # In both cases, we need RGB→BGR conversion for ROS publishing
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                        self.log_buffered_event(
                            'COLOR_CONVERSION_RGB2BGR',
                            'Applied RGB to BGR conversion for ROS publishing',
                            final_shape=str(cv_image.shape)
                        )
                    else:
                        self.log_buffered_event(
                            'COLOR_CONVERSION_SKIPPED',
                            'Skipping color conversion - unexpected image format',
                            final_shape=str(cv_image.shape)
                        )

                    # Convert to ROS Image message
                    annotated_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                    annotated_msg.header.stamp = msg.header.stamp  # Same timestamp
                    annotated_msg.header.frame_id = 'camera_frame'
                    self.log_buffered_event(
                        'ROS_MESSAGE_CREATED',
                        'Created ROS Image message for annotated image',
                        width=annotated_msg.width,
                        height=annotated_msg.height,
                        encoding=annotated_msg.encoding
                    )

                    # Publish annotated image
                    self.annotated_image_publisher.publish(annotated_msg)
                    self.log_buffered_event(
                        'PUBLISH_SUCCESS',
                        'Successfully published annotated image',
                        message_size=len(annotated_msg.data)
                    )

                except Exception as img_error:
                    self.log_buffered_event(
                        'PUBLISH_ERROR',
                        f'Error publishing annotated image: {img_error}',
                        error_type=type(img_error).__name__
                    )
                    # Still log critical errors to main logger
                    self.get_logger().error(f'Critical error in annotated image pipeline: {img_error}')
                    import traceback
                    self.log_buffered_event(
                        'PUBLISH_ERROR_TRACEBACK',
                        'Full error traceback',
                        traceback=traceback.format_exc()
                    )
            # Log high-confidence detections occasionally (every 10th detection)
            if hasattr(self, '_detection_count'):
                self._detection_count += 1
            else:
                self._detection_count = 1

            if self._detection_count % 10 == 0:
                high_conf_detections = []
                for detection in results['detections']:
                    if detection.categories:
                        best_category = max(detection.categories, key=lambda c: c.score if c.score else 0)
                        if best_category.score and best_category.score >= 0.7:
                            class_name = best_category.category_name or 'unknown'
                            high_conf_detections.append(f"{class_name}({best_category.score:.2f})")

                if high_conf_detections:
                    self.get_logger().info(f'High confidence detections: {", ".join(high_conf_detections)}')



        except Exception as e:
            self.get_logger().error(f'Error publishing results: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def flush_annotated_diagnostics(self) -> None:
        """Manually flush the annotated image diagnostics buffer."""
        try:
            stats = self.get_buffer_stats()
            self.get_logger().info(f"Manual flush requested - Buffer stats: {stats}")
            self.buffered_logger.flush()
        except Exception as e:
            self.get_logger().error(f'Error in manual flush: {e}')

    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        try:
            if self.detector:
                self.detector.close()
            # Parent cleanup will handle buffer flushing
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
