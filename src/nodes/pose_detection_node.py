#!/usr/bin/env python3
"""
Pose Detection Node for GestureBot Vision System
MediaPipe pose detection with callback-based architecture and 33-point pose landmark tracking.
"""

import time
import os
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from rcl_interfaces.msg import ParameterDescriptor

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig, MediaPipeCallbackMixin
from vision_core.controller import PoseDetectionController
from vision_core.message_converter import MessageConverter
from gesturebot.msg import PoseLandmarks
from geometry_msgs.msg import Point


class PoseDetectionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time pose detection using MediaPipe PoseLandmarker.
    Uses callback-based architecture for optimal performance and 33-point pose tracking.
    Includes pose-to-action mapping for navigation control.
    """

    # Simplified 4-pose navigation control system
    POSE_ACTION_MAP = {
        'arms_raised': 'forward',      # Both arms raised → move forward
        'pointing_left': 'left',       # Left arm pointing → turn left
        'pointing_right': 'right',     # Right arm pointing → turn right
        't_pose': 'stop',              # T-pose → stop
        'no_pose': 'stop'              # No clear pose → stop (default)
    }

    def __init__(self):
        # Configuration for pose detection
        config = ProcessingConfig(
            enabled=True,
            max_fps=15.0,
            frame_skip=1,
            confidence_threshold=0.5,
            max_results=2,  # Support up to 2 poses
            priority=2  # Medium priority
        )

        # MediaPipe controller (composition)
        self.controller = None

        # Model path will be set after parameter declaration
        self.model_path = None

        # Current pose landmarks for visualization
        self._current_pose_landmarks = None

        # Pose classification testing - simplified state tracking
        self.current_pose_action = None
        self.last_pose_action = None
        self.pose_change_count = 0

        # Monotonic timestamp tracking for MediaPipe
        self._last_timestamp = 0
        self._timestamp_increment = 33  # ~30 FPS (33ms between frames)

        # Initialize MediaPipeCallbackMixin first (required for base class initialization)
        MediaPipeCallbackMixin.__init__(self)

        # Initialize parent with default values (will be updated after parameter declaration)
        super().__init__(
            'pose_detection_node',
            'pose_detection',
            config,
            enable_buffered_logging=True,  # Default, will be updated
            unlimited_buffer_mode=False,   # Default, will be updated
            enable_performance_tracking=False,  # Default, will be updated
            controller=None  # Will set after model_path is resolved
        )

        # Set base node reference for callback publishing
        self._set_base_node_reference(self)

        # Declare parameters for this node (only those not provided by launch file)
        self.declare_parameter('unlimited_buffer_mode', False)
        self.declare_parameter('buffer_logging_enabled', True)
        self.declare_parameter('enable_performance_tracking', False)

        # Pose classification testing parameters (declared in launch file)
        # Note: These parameters are declared in pose_detection.launch.py
        self.declare_parameter('enable_detailed_coordinates', False)     # Enable detailed coordinate logging
        self.declare_parameter('debug_rate_limit', 2.0)                  # Max debug messages per second

        # Note: num_poses, min_pose_detection_confidence, min_pose_presence_confidence,
        # min_tracking_confidence, output_segmentation_masks, publish_annotated_images,
        # debug_mode are declared by launch file
        
        # Declare model path parameter with descriptor (if not already declared)
        try:
            if not self.has_parameter('model_path'):
                model_path_descriptor = ParameterDescriptor(
                    description='Path to the pose landmarker model file (.task). Can be absolute path or relative to package share directory.',
                    additional_constraints='Must be a valid path to a .task model file'
                )
                self.declare_parameter('model_path', 'models/pose_landmarker.task', model_path_descriptor)
        except Exception as e:
            self.get_logger().debug(f"Model path parameter handling: {e}")

        # Declare visualization parameters
        try:
            if not self.has_parameter('publish_annotated_images'):
                self.declare_parameter('publish_annotated_images', True)
            if not self.has_parameter('show_landmark_indices'):
                self.declare_parameter('show_landmark_indices', False)
        except Exception as e:
            self.get_logger().debug(f"Visualization parameter handling: {e}")

        # Update logging configuration from parameters
        try:
            buffer_logging_enabled = self.get_parameter('buffer_logging_enabled').value
            unlimited_buffer_mode = self.get_parameter('unlimited_buffer_mode').value
            enable_performance_tracking = self.get_parameter('enable_performance_tracking').value

            # Update buffered logger configuration
            self.buffered_logger.enabled = buffer_logging_enabled
            self.buffered_logger.unlimited_mode = unlimited_buffer_mode
            self.enable_performance_tracking = enable_performance_tracking
        except Exception as e:
            self.get_logger().warn(f'Error configuring logging, using defaults: {e}')

        # Update pose classification debug settings
        try:
            self.enable_pose_debug = self.get_parameter('enable_pose_classification_debug').value
            self.enable_detailed_coordinates = self.get_parameter('enable_detailed_coordinates').value
            self.debug_rate_limit = self.get_parameter('debug_rate_limit').value
        except Exception as e:
            self.enable_pose_debug = True  # Default to enabled for testing
            self.enable_detailed_coordinates = False  # Default to disabled
            self.debug_rate_limit = 2.0  # Default rate limit
            self.get_logger().warn(f'Error reading pose debug parameters, using defaults: {e}')

        # Debug rate limiting
        self.last_debug_time = 0.0
        self.last_debug_pose = None

        # Log the buffer configuration
        buffer_stats = self.get_buffer_stats()
        self.get_logger().info(f"BufferedLogger initialized: {buffer_stats}")

        # Publishers
        self.poses_publisher = self.create_publisher(
            PoseLandmarks,
            '/vision/poses',
            self.result_qos
        )

        # Conditional annotated image publisher
        self.annotated_image_publisher = None

        # Check if annotated image publishing is enabled
        try:
            publish_annotated = self.get_parameter('publish_annotated_images').value
        except Exception as e:
            self.get_logger().warn(f'Could not read publish_annotated_images parameter: {e}')
            publish_annotated = True  # Default to enabled (consistent with other vision nodes)

        if publish_annotated:
            from sensor_msgs.msg import Image
            self.annotated_image_publisher = self.create_publisher(
                Image,
                '/vision/pose/annotated',
                self.result_qos
            )
            self.get_logger().info("Annotated image publishing enabled -> /vision/pose/annotated")
        else:
            self.get_logger().info("Annotated image publishing disabled")

        # Initialize controller after all parameters are available
        self._initialize_controller()

        self.get_logger().info("Pose detection node initialized successfully")

    def _initialize_controller(self):
        """Initialize the MediaPipe controller with resolved model path."""
        try:
            # Resolve model path
            self.model_path = self._resolve_model_path()
            if not self.model_path:
                self.get_logger().error("Failed to resolve model path")
                return

            # Get pose detection parameters
            num_poses = self.get_parameter('num_poses').value if self.has_parameter('num_poses') else 2
            min_pose_detection_confidence = self.get_parameter('min_pose_detection_confidence').value if self.has_parameter('min_pose_detection_confidence') else 0.5
            min_pose_presence_confidence = self.get_parameter('min_pose_presence_confidence').value if self.has_parameter('min_pose_presence_confidence') else 0.5
            min_tracking_confidence = self.get_parameter('min_tracking_confidence').value if self.has_parameter('min_tracking_confidence') else 0.5
            output_segmentation_masks = self.get_parameter('output_segmentation_masks').value if self.has_parameter('output_segmentation_masks') else False

            # Create controller
            self.controller = PoseDetectionController(
                model_path=self.model_path,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=output_segmentation_masks,
                result_callback=self._process_callback_results
            )

            self.get_logger().info(f"PoseDetectionController initialized with model: {self.model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize controller: {e}")
            self.controller = None

    def _resolve_model_path(self) -> Optional[str]:
        """Resolve the model path from parameter, checking package share directory."""
        try:
            model_path_param = self.get_parameter('model_path').value
        except Exception as e:
            self.get_logger().error(f"Failed to get model_path parameter: {e}")
            return None

        # If absolute path, check if it exists
        if os.path.isabs(model_path_param):
            if os.path.exists(model_path_param):
                return model_path_param
            else:
                self.get_logger().error(f"Absolute model path does not exist: {model_path_param}")
                return None

        # Try relative to package share directory
        try:
            package_share_dir = get_package_share_directory('gesturebot')
            full_path = os.path.join(package_share_dir, model_path_param)
            if os.path.exists(full_path):
                return full_path
        except PackageNotFoundError:
            pass

        # Try relative to source directory (for development)
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent.parent.parent
            source_path = script_dir / model_path_param
            if source_path.exists():
                return str(source_path.absolute())
        except Exception:
            pass

        # Try relative to current working directory
        if os.path.exists(model_path_param):
            return os.path.abspath(model_path_param)

        self.get_logger().error(f"Could not resolve model path: {model_path_param}")
        return None

    def initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe components (delegated to controller)."""
        return self.controller is not None and self.controller.is_ready()

    def _generate_monotonic_timestamp(self, input_timestamp: int) -> int:
        """Generate monotonic timestamp for MediaPipe."""
        # Convert input to int if needed
        current_timestamp = int(input_timestamp) if isinstance(input_timestamp, (int, float)) else int(input_timestamp)

        # Ensure timestamp is always increasing
        if current_timestamp <= self._last_timestamp:
            # If timestamp is not increasing, increment by fixed amount
            self._last_timestamp += self._timestamp_increment
        else:
            # Use the actual timestamp if it's properly increasing
            self._last_timestamp = current_timestamp

        return self._last_timestamp

    def process_frame(self, frame: np.ndarray, timestamp: int) -> Optional[Dict]:
        """Process frame using MediaPipe controller."""
        if not self.controller or not self.controller.is_ready():
            return None

        try:
            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Generate monotonic timestamp for MediaPipe
            monotonic_timestamp = self._generate_monotonic_timestamp(timestamp)

            # Submit for async processing
            self.controller.detect_async(mp_image, monotonic_timestamp)

            # Return None as results come via callback
            return None

        except Exception as e:
            self.log_buffered_event('FRAME_PROCESSING_ERROR', f'Error processing frame: {str(e)}')
            return None

    def publish_results(self, results: Dict, timestamp: int) -> None:
        """Publish pose detection results (required by base class)."""
        # This method is required by MediaPipeBaseNode but we handle publishing
        # in the callback method _process_callback_results instead
        pass

    def _process_callback_results(self, result, output_image: mp.Image, timestamp_ms: int):
        """Process MediaPipe pose detection results from callback - TESTING MODE."""
        try:
            # Store current pose landmarks for visualization
            self._current_pose_landmarks = result.pose_landmarks if result.pose_landmarks else None

            # Process pose classification if poses detected
            if result.pose_landmarks:
                # Classify pose action from landmarks
                pose_action = self._classify_pose_action(result.pose_landmarks)

                # Log debug information (rate-limited and concise)
                self._log_pose_debug(pose_action, result.pose_landmarks)

                # Track pose changes for statistics
                if pose_action != self.current_pose_action:
                    self.last_pose_action = self.current_pose_action
                    self.current_pose_action = pose_action
                    self.pose_change_count += 1

                # Create and publish pose landmarks message with classification
                pose_msg = self._create_pose_landmarks_message(result, timestamp_ms, pose_action)
                self.poses_publisher.publish(pose_msg)

            else:
                # No poses detected
                pose_action = 'no_pose'

                # Log debug information (rate-limited)
                self._log_pose_debug(pose_action, [])

                # Track pose changes
                if self.current_pose_action != 'no_pose':
                    self.last_pose_action = self.current_pose_action
                    self.current_pose_action = 'no_pose'
                    self.pose_change_count += 1

                # Publish empty message with no_pose action
                pose_msg = self._create_pose_landmarks_message(result, timestamp_ms, pose_action)
                self.poses_publisher.publish(pose_msg)

            # Create and publish annotated image if enabled
            if self.annotated_image_publisher and result.pose_landmarks:
                annotated_frame = self._create_annotated_image(output_image.numpy_view(), result)
                annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                annotated_msg.header.stamp = self.get_clock().now().to_msg()
                annotated_msg.header.frame_id = 'camera_frame'
                self.annotated_image_publisher.publish(annotated_msg)

            # Log successful processing (reduced verbosity)
            self.log_buffered_event('POSE_DETECTION_SUCCESS',
                                  f'Detected {len(result.pose_landmarks) if result.pose_landmarks else 0} poses')

        except Exception as e:
            self.log_buffered_event('CALLBACK_PROCESSING_ERROR', f'Error in callback: {str(e)}')

    def _classify_pose_action(self, pose_landmarks_list) -> str:
        """
        Classify pose action from MediaPipe pose landmarks with concise debug logging.
        Returns action string for navigation control.
        """
        try:
            if not pose_landmarks_list or len(pose_landmarks_list) == 0:
                return 'no_pose'

            # Use the first detected pose for action classification
            pose_landmarks = pose_landmarks_list[0]

            # Handle both possible MediaPipe pose landmark structures
            landmarks_list = None
            try:
                if hasattr(pose_landmarks, '__iter__') and not hasattr(pose_landmarks, 'landmark'):
                    landmarks_list = pose_landmarks
                else:
                    landmarks_list = pose_landmarks.landmark
            except Exception:
                return 'no_pose'

            if not landmarks_list or len(landmarks_list) < 33:
                return 'no_pose'

            # Extract key landmarks (MediaPipe pose landmark indices)
            left_shoulder = landmarks_list[11]
            right_shoulder = landmarks_list[12]
            left_wrist = landmarks_list[15]
            right_wrist = landmarks_list[16]

            # Calculate adaptive thresholds based on shoulder width
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            extension_threshold = max(0.08, shoulder_width * 0.6)
            horizontal_tolerance = max(0.08, shoulder_width * 0.4)
            pointing_extension_threshold = max(0.06, shoulder_width * 0.4)
            pointing_horizontal_tolerance = max(0.12, shoulder_width * 0.8)

            # POSE CLASSIFICATION LOGIC

            # 1. Check for ARMS RAISED (both wrists above shoulders)
            left_wrist_raised = left_wrist.y < left_shoulder.y - 0.05
            right_wrist_raised = right_wrist.y < right_shoulder.y - 0.05

            if left_wrist_raised and right_wrist_raised:
                detected_pose = 'arms_raised'
                self._log_pose_debug(detected_pose, {
                    'shoulder_width': shoulder_width,
                    'left_wrist_diff': left_shoulder.y - left_wrist.y,
                    'right_wrist_diff': right_shoulder.y - right_wrist.y
                })
                return detected_pose

            # 2. Check for T-POSE (both arms horizontal and extended)
            left_arm_horizontal = abs(left_wrist.y - left_shoulder.y) < horizontal_tolerance
            right_arm_horizontal = abs(right_wrist.y - right_shoulder.y) < horizontal_tolerance
            left_arm_extended_out = left_wrist.x < left_shoulder.x - extension_threshold
            right_arm_extended_out = right_wrist.x > right_shoulder.x + extension_threshold

            if (left_arm_horizontal and right_arm_horizontal and
                left_arm_extended_out and right_arm_extended_out):
                detected_pose = 't_pose'
                self._log_pose_debug(detected_pose, {
                    'shoulder_width': shoulder_width,
                    'extension_threshold': extension_threshold,
                    'horizontal_tolerance': horizontal_tolerance,
                    'left_horizontal': left_arm_horizontal,
                    'right_horizontal': right_arm_horizontal,
                    'left_extended': left_arm_extended_out,
                    'right_extended': right_arm_extended_out
                })
                return detected_pose

            # 3. Check for POINTING gestures (single arm extended horizontally)
            # Note: Removed pointing_forward as it was unreliable - keeping only left/right
            left_arm_pointing = (left_wrist.x < left_shoulder.x - pointing_extension_threshold and
                               abs(left_wrist.y - left_shoulder.y) < pointing_horizontal_tolerance)
            right_arm_pointing = (right_wrist.x > right_shoulder.x + pointing_extension_threshold and
                                abs(right_wrist.y - right_shoulder.y) < pointing_horizontal_tolerance)

            # Only detect single-arm pointing (left OR right, not both)
            if left_arm_pointing and not right_arm_pointing:
                detected_pose = 'pointing_left'
                self._log_pose_debug(detected_pose, {
                    'shoulder_width': shoulder_width,
                    'pointing_threshold': pointing_extension_threshold,
                    'left_extended': True,
                    'right_extended': False
                })
                return detected_pose
            elif right_arm_pointing and not left_arm_pointing:
                detected_pose = 'pointing_right'
                self._log_pose_debug(detected_pose, {
                    'shoulder_width': shoulder_width,
                    'pointing_threshold': pointing_extension_threshold,
                    'left_extended': False,
                    'right_extended': True
                })
                return detected_pose
            # Note: If both arms are pointing, treat as no_pose (ambiguous gesture)

            # 4. Default to no clear pose detected
            detected_pose = 'no_pose'
            self._log_pose_debug(detected_pose, {
                'shoulder_width': shoulder_width,
                'reason': 'No pose criteria met (4-pose system: arms_raised, pointing_left, pointing_right, t_pose)'
            })
            return detected_pose

        except Exception as e:
            self.log_buffered_event('POSE_CLASSIFICATION_ERROR',
                                  f'Error classifying pose: {str(e)}')
            return 'no_pose'

    def _log_pose_debug(self, pose_action: str, debug_info: dict) -> None:
        """Log pose classification debug information with rate limiting and concise format."""
        if not self.enable_pose_debug:
            return

        # Check if we should log based on rate limiting and pose changes
        if not self._should_log_debug(pose_action):
            return

        # Create concise debug message
        if pose_action == 'no_pose':
            debug_msg = f"🎯 POSE: {pose_action.upper()} | {debug_info.get('reason', 'Unknown')}"
        else:
            # Format key metrics in a single line
            metrics = []
            if 'shoulder_width' in debug_info:
                metrics.append(f"width={debug_info['shoulder_width']:.3f}")
            if 'extension_threshold' in debug_info:
                metrics.append(f"ext_thresh={debug_info['extension_threshold']:.3f}")
            if 'pointing_threshold' in debug_info:
                metrics.append(f"point_thresh={debug_info['pointing_threshold']:.3f}")

            debug_msg = f"🎯 POSE: {pose_action.upper()} | {' | '.join(metrics)}"

            # Add detailed coordinates only if enabled
            if self.enable_detailed_coordinates:
                coord_info = []
                for key, value in debug_info.items():
                    if key not in ['shoulder_width', 'extension_threshold', 'pointing_threshold']:
                        if isinstance(value, bool):
                            coord_info.append(f"{key}={value}")
                        elif isinstance(value, float):
                            coord_info.append(f"{key}={value:.3f}")
                if coord_info:
                    debug_msg += f" | {' | '.join(coord_info)}"

        self.get_logger().info(debug_msg)

    def _should_log_debug(self, pose_action: str) -> bool:
        """Check if we should log debug information based on rate limiting and pose changes."""
        import time
        current_time = time.time()

        # Only log if pose changed or enough time has passed
        pose_changed = pose_action != self.last_debug_pose
        time_elapsed = current_time - self.last_debug_time >= (1.0 / self.debug_rate_limit)

        if pose_changed or time_elapsed:
            self.last_debug_time = current_time
            self.last_debug_pose = pose_action
            return True

        return False

    def _create_pose_landmarks_message(self, result, timestamp_ms: int, pose_action: str = 'no_pose') -> PoseLandmarks:
        """Create ROS message from MediaPipe pose detection results with pose classification."""
        msg = PoseLandmarks()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.timestamp_ms = timestamp_ms

        # Add pose classification and navigation command
        msg.pose_action = pose_action
        msg.nav_command = self.POSE_ACTION_MAP.get(pose_action, 'stop')

        if result.pose_landmarks:
            msg.num_poses = len(result.pose_landmarks)

            # Flatten all pose landmarks into a single array
            for pose_landmarks in result.pose_landmarks:
                # Handle both possible MediaPipe pose landmark structures
                try:
                    # Try accessing as list directly (newer format)
                    if hasattr(pose_landmarks, '__iter__') and not hasattr(pose_landmarks, 'landmark'):
                        # pose_landmarks is already a list of landmarks
                        for landmark in pose_landmarks:
                            point = Point()
                            point.x = landmark.x
                            point.y = landmark.y
                            point.z = landmark.z
                            msg.landmarks.append(point)
                    else:
                        # Try accessing via .landmark attribute (older format)
                        for landmark in pose_landmarks.landmark:
                            point = Point()
                            point.x = landmark.x
                            point.y = landmark.y
                            point.z = landmark.z
                            msg.landmarks.append(point)
                except Exception as e:
                    self.log_buffered_event('LANDMARK_STRUCTURE_ERROR', f'Unexpected pose landmark structure: {type(pose_landmarks)}, error: {str(e)}')
                    continue
        else:
            msg.num_poses = 0

        return msg

    def _create_annotated_image(self, frame: np.ndarray, result) -> np.ndarray:
        """Create annotated image with pose landmarks and information."""
        annotated_frame = frame.copy()

        if result.pose_landmarks:
            # Draw pose landmarks for each detected pose
            for pose_index, pose_landmarks in enumerate(result.pose_landmarks):
                self._draw_pose_landmarks(annotated_frame, pose_landmarks, pose_index)

            # Add pose count text
            pose_count_text = f"Poses: {len(result.pose_landmarks)}"
            self._draw_text_with_background(
                annotated_frame,
                pose_count_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Standardized font scale
                (0, 255, 0),  # Green text
                1  # Standardized thickness
            )

        return annotated_frame

    def _draw_pose_landmarks(self, image: np.ndarray, pose_landmarks, pose_index: int):
        """Draw pose landmarks and connections on the image."""
        try:
            height, width = image.shape[:2]

            # Define pose connections (MediaPipe pose connections)
            pose_connections = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                # Torso
                (9, 10), (11, 12), (11, 13), (13, 15),
                (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20),
                (16, 22), (18, 20), (11, 23), (12, 24),
                (23, 24), (23, 25), (24, 26), (25, 27),
                (26, 28), (27, 29), (28, 30), (29, 31),
                (30, 32), (27, 31), (28, 32)
            ]

            # Color for this pose (different colors for multiple poses)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            color = colors[pose_index % len(colors)]

            # Handle both possible MediaPipe pose landmark structures
            landmarks_list = None
            try:
                # Try accessing as list directly (newer format)
                if hasattr(pose_landmarks, '__iter__') and not hasattr(pose_landmarks, 'landmark'):
                    landmarks_list = pose_landmarks
                else:
                    # Try accessing via .landmark attribute (older format)
                    landmarks_list = pose_landmarks.landmark
            except Exception as e:
                self.get_logger().error(f"Error accessing pose landmarks structure: {e}")
                return

            if landmarks_list is None:
                return

            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                    start_landmark = landmarks_list[start_idx]
                    end_landmark = landmarks_list[end_idx]

                    start_x = int(start_landmark.x * width)
                    start_y = int(start_landmark.y * height)
                    end_x = int(end_landmark.x * width)
                    end_y = int(end_landmark.y * height)

                    cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)

            # Draw landmarks
            for i, landmark in enumerate(landmarks_list):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Draw landmark point
                cv2.circle(image, (x, y), 4, color, -1)
                cv2.circle(image, (x, y), 6, (255, 255, 255), 2)  # White border

                # Optional: Add landmark indices if enabled
                try:
                    show_indices = self.get_parameter('show_landmark_indices').value
                    if show_indices:
                        cv2.putText(image, str(i), (x+8, y-8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except:
                    pass  # Parameter not available

        except Exception as e:
            self.get_logger().error(f"Error drawing pose landmarks: {e}")

    def _draw_text_with_background(self, image, text, position, font, font_scale, color, thickness):
        """Draw text with background rectangle for better visibility."""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        x, y = position
        cv2.rectangle(image, (x, y - text_height - 10), (x + text_width + 10, y + baseline), (0, 0, 0), -1)

        # Draw text
        cv2.putText(image, text, (x + 5, y - 5), font, font_scale, color, thickness)

    def cleanup_mediapipe(self):
        """Clean up MediaPipe resources."""
        if self.controller:
            self.controller.close()
            self.controller = None

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.cleanup_mediapipe()
        super().destroy_node()


def main(args=None):
    """Main function for pose detection node."""
    import rclpy

    rclpy.init(args=args)

    try:
        node = PoseDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
