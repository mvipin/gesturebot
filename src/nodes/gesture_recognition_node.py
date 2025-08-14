#!/usr/bin/env python3
"""
Gesture Recognition Node for GestureBot Vision System
MediaPipe gesture recognition with callback-based architecture and navigation command mapping.
"""

import time
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from rcl_interfaces.msg import ParameterDescriptor

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig, MediaPipeCallbackMixin
from vision_core.controller import GestureRecognitionController
from vision_core.message_converter import MessageConverter
from gesturebot.msg import HandGesture
from geometry_msgs.msg import Point


class GestureRecognitionNode(MediaPipeBaseNode, MediaPipeCallbackMixin):
    """
    ROS 2 node for real-time gesture recognition using MediaPipe GestureRecognizer.
    Uses callback-based architecture for optimal performance and navigation command mapping.
    """

    # Gesture to navigation command mapping
    GESTURE_COMMANDS = {
        'thumbs_up': 'start_navigation',
        'thumbs_down': 'stop_navigation',
        'open_palm': 'pause_navigation',
        'pointing_up': 'move_forward',
        'pointing_left': 'turn_left',
        'pointing_right': 'turn_right',
        'peace': 'follow_person',
        'fist': 'emergency_stop',
        'wave': 'return_home'
    }

    def __init__(self):
        # Configuration for gesture recognition
        config = ProcessingConfig(
            enabled=True,
            max_fps=15.0,
            frame_skip=1,
            confidence_threshold=0.7,
            max_results=2,  # Support up to 2 hands
            priority=1  # High priority for navigation
        )

        # MediaPipe controller (composition)
        self.controller = None

        # Model path will be set after parameter declaration
        self.model_path = None

        # Enhanced gesture state tracking for MAXIMUM responsiveness
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_stability_threshold = 0.1  # seconds (minimum viable for fastest confirmation)
        self.gesture_consistency_count = 1  # single detection for immediate response
        self.gesture_transition_delay = 0.05  # minimum viable delay for fastest switching
        self.last_transition_time = 0.0  # track last gesture change time

        # Consistency tracking (minimal for maximum speed)
        self.gesture_detection_history = []  # recent detections for consistency
        self.max_history_size = 2  # minimal history for fastest processing

        # Initialize MediaPipeCallbackMixin first (required for base class initialization)
        MediaPipeCallbackMixin.__init__(self)

        # Initialize parent with default values (will be updated after parameter declaration)
        super().__init__(
            'gesture_recognition_node',
            'gesture_recognition',
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

        # Note: confidence_threshold, max_hands, gesture_stability_threshold,
        # publish_annotated_images, debug_mode are declared by launch file
        self.declare_parameter('gesture_consistency_count', 1)
        self.declare_parameter('gesture_transition_delay', 0.05)
        # Declare model path parameter with descriptor (if not already declared)
        try:
            # Check if parameter already exists
            if not self.has_parameter('model_path'):
                model_path_descriptor = ParameterDescriptor(
                    description='Path to the gesture recognition model file (.task). Can be absolute path or relative to package share directory.',
                    additional_constraints='Must be a valid path to a .task model file'
                )
                self.declare_parameter('model_path', 'models/gesture_recognizer.task', model_path_descriptor)
        except Exception as e:
            # Parameter may already be declared by launch file or parent class
            self.get_logger().debug(f"Model path parameter handling: {e}")

        # Declare visualization parameters
        try:
            if not self.has_parameter('publish_annotated_images'):
                self.declare_parameter('publish_annotated_images', False)
            if not self.has_parameter('show_landmark_indices'):
                self.declare_parameter('show_landmark_indices', False)
        except Exception as e:
            self.get_logger().debug(f"Visualization parameter handling: {e}")

        # Note: Parameters will be read when initializing controller (after launch file declares them)

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

        # Log the buffer configuration (using inherited buffered logger)
        buffer_stats = self.get_buffer_stats()
        self.get_logger().info(f"BufferedLogger initialized: {buffer_stats}")

        # Publishers
        self.gestures_publisher = self.create_publisher(
            HandGesture,
            '/vision/gestures',
            self.result_qos
        )

        # Conditional annotated image publisher
        self.annotated_image_publisher = None

        # Check if annotated image publishing is enabled
        try:
            publish_annotated = self.get_parameter('publish_annotated_images').value
        except Exception as e:
            self.get_logger().warn(f'Could not read publish_annotated_images parameter: {e}')
            publish_annotated = True  # Default to enabled (consistent with launch file default)

        if publish_annotated:
            from sensor_msgs.msg import Image
            self.annotated_image_publisher = self.create_publisher(
                Image,
                '/vision/gestures/annotated',
                self.image_qos
            )
            self.get_logger().info('Annotated gesture image publisher enabled')

        # Resolve model path using parameter and resource discovery
        self.model_path = self.resolve_model_path()

        # Initialize MediaPipe controller (composition)
        try:
            if self.model_path and self.model_path.exists():
                # Get parameters with fallbacks (launch file provides these)
                try:
                    confidence_threshold = float(self.get_parameter('confidence_threshold').value)
                except:
                    confidence_threshold = config.confidence_threshold
                    self.get_logger().debug(f'Using config confidence_threshold: {confidence_threshold}')

                try:
                    max_hands = int(self.get_parameter('max_hands').value)
                except:
                    max_hands = config.max_results
                    self.get_logger().debug(f'Using config max_hands: {max_hands}')

                try:
                    self.gesture_stability_threshold = float(self.get_parameter('gesture_stability_threshold').value)
                except:
                    self.gesture_stability_threshold = 0.1  # Maximum responsiveness
                    self.get_logger().debug(f'Using default gesture_stability_threshold: {self.gesture_stability_threshold}')

                try:
                    self.gesture_consistency_count = int(self.get_parameter('gesture_consistency_count').value)
                except:
                    self.gesture_consistency_count = 1  # Immediate response
                    self.get_logger().debug(f'Using default gesture_consistency_count: {self.gesture_consistency_count}')

                try:
                    self.gesture_transition_delay = float(self.get_parameter('gesture_transition_delay').value)
                except:
                    self.gesture_transition_delay = 0.05  # Fastest viable transitions
                    self.get_logger().debug(f'Using default gesture_transition_delay: {self.gesture_transition_delay}')

                # Update config with actual values for use in processing
                self.config.confidence_threshold = confidence_threshold
                self.config.max_results = max_hands

                self.controller = GestureRecognitionController(
                    model_path=str(self.model_path),
                    confidence_threshold=confidence_threshold,
                    max_hands=max_hands,
                    result_callback=self.create_callback('gesture')
                )
                self.get_logger().info('MediaPipe gesture recognizer initialized via controller')
            else:
                self.get_logger().error('Gesture model not found - controller not initialized')
                return
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MediaPipe controller: {e}')
            return

        # Enable callback processing
        self.enable_callback_processing()

        # Initialize visualization state
        self._current_rgb_frame = None
        self._current_frame_timestamp = None
        self._current_hand_landmarks = None

        # Visualization parameters
        try:
            self._show_landmark_indices = self.get_parameter('show_landmark_indices').value
        except:
            self._show_landmark_indices = False  # Default to disabled

        self.get_logger().info('Gesture Recognition Node initialized with callback architecture')

    def resolve_model_path(self) -> Path:
        """
        Resolve the model path using ROS 2 resource discovery and parameters.

        Returns:
            Path: Resolved path to the model file, or None if not found

        Raises:
            FileNotFoundError: If model file cannot be found in any location
        """
        # Get model path parameter
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value

        # If absolute path is provided, use it directly
        if os.path.isabs(model_path_param):
            model_path = Path(model_path_param)
            if model_path.exists():
                self.get_logger().info(f"Using absolute model path: {model_path}")
                return model_path
            else:
                self.get_logger().warn(f"Absolute model path does not exist: {model_path}")

        # Try to resolve relative path using package resource discovery
        try:
            package_share_dir = get_package_share_directory('gesturebot')
            package_model_path = Path(package_share_dir) / model_path_param
            if package_model_path.exists():
                self.get_logger().info(f"Using package model path: {package_model_path}")
                return package_model_path
            else:
                self.get_logger().warn(f"Package model path does not exist: {package_model_path}")
        except PackageNotFoundError:
            self.get_logger().warn("Package 'gesturebot' not found in ament index")

        # Fallback locations for development and alternative installations
        fallback_paths = [
            # Source location (development)
            Path(__file__).parent.parent.parent / 'models' / 'gesture_recognizer.task',
            # Alternative package locations
            Path('/opt/ros/jazzy/share/gesturebot') / model_path_param,
            # Legacy locations for backward compatibility
            Path.home() / 'GestureBot' / 'gesturebot_ws' / 'src' / 'gesturebot' / 'models' / 'gesture_recognizer.task',
            # Current working directory
            Path.cwd() / model_path_param,
        ]

        for path in fallback_paths:
            if path.exists():
                self.get_logger().info(f"Using fallback model path: {path}")
                return path

        # If no existing path found, log warning and return None for fallback
        try:
            package_share_dir = get_package_share_directory('gesturebot')
            preferred_path = Path(package_share_dir) / model_path_param
        except PackageNotFoundError:
            preferred_path = Path(model_path_param)

        # Log all attempted paths for debugging
        attempted_paths = [model_path_param] if os.path.isabs(model_path_param) else []
        try:
            attempted_paths.append(str(Path(get_package_share_directory('gesturebot')) / model_path_param))
        except PackageNotFoundError:
            pass
        attempted_paths.extend([str(p) for p in fallback_paths])

        warning_msg = f"Gesture model file not found. Will fall back to hand landmarks. Attempted paths:\n" + "\n".join(f"  - {p}" for p in attempted_paths)
        self.get_logger().warn(warning_msg)

        return None  # Return None to trigger fallback to hand landmarks

    def initialize_mediapipe(self) -> bool:
        """Deprecated in composition design; controller handles initialization."""
        self.get_logger().warn('initialize_mediapipe() is deprecated; controller handles initialization')
        return True

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """
        Process frame for gesture recognition using async callback architecture.
        This method only submits the frame to MediaPipe - results are published via callback.
        """
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
                # Fallback: Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.log_buffered_event(
                    'PREPROCESSING_CONVERSION',
                    'Converting BGR to RGB for MediaPipe',
                    frame_shape=str(frame.shape)
                )

            # Store RGB frame for callback access
            self._current_rgb_frame = rgb_frame
            self._current_frame_timestamp = timestamp

            # Use high-precision timestamp like the working sample
            timestamp_ms = time.time_ns() // 1_000_000

            # Use controller for gesture recognition
            if self.controller and self.controller.is_ready():
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.controller.detect_async(mp_image, timestamp_ms)
                self.log_buffered_event(
                    'ASYNC_RECOGNITION_SUBMITTED',
                    'Submitted frame for async gesture recognition',
                    timestamp_ms=timestamp_ms,
                    frame_shape=str(rgb_frame.shape)
                )
            else:
                self.get_logger().error('No controller available or not ready!')
                return None

            # Return None for callback-based nodes to prevent duplicate publishing
            # Actual results will be published via callback when ready
            return None

        except Exception as e:
            self.log_buffered_event(
                'PROCESSING_ERROR',
                f'Error processing frame: {str(e)}',
                timestamp=timestamp
            )
            return None



    def _process_callback_results(self, result, output_image, timestamp_ms: int, result_type: str) -> Optional[Dict]:
        """
        Process MediaPipe callback results into the format expected by publish_results().
        This method is called from the MediaPipe callback thread.

        Args:
            result: MediaPipe gesture recognition result
            output_image: MediaPipe output image (unused for gesture recognition)
            timestamp_ms: Timestamp in milliseconds
            result_type: Type of result being processed
        """
        # Note: output_image parameter is required by MediaPipe callback signature but unused
        try:
            # Debug logging like working sample
            self.log_buffered_event(
                'CALLBACK_RECEIVED',
                f'Callback received - gestures: {len(result.gestures) if result and result.gestures else 0}, hands: {len(result.hand_landmarks) if result and result.hand_landmarks else 0}',
                timestamp_ms=timestamp_ms
            )

            if result and (result.gestures or result.hand_landmarks):
                # Use stored RGB frame from process_frame
                rgb_frame = self._current_rgb_frame
                original_timestamp = self._current_frame_timestamp

                if rgb_frame is None:
                    self.log_buffered_event(
                        'CALLBACK_ERROR',
                        'No RGB frame available for callback processing',
                        timestamp_ms=timestamp_ms
                    )
                    return None

                # Store hand landmarks for visualization
                self._current_hand_landmarks = result.hand_landmarks

                # Process gesture recognition results
                gesture_info = self.analyze_gesture_results(
                    result.gestures,
                    result.hand_landmarks,
                    result.handedness,
                    original_timestamp
                )

                if gesture_info:
                    result_dict = {
                        'gesture_info': gesture_info,
                        'timestamp': original_timestamp,
                        'processing_time': (time.time() - original_timestamp) * 1000 if original_timestamp else 0,
                        'rgb_frame': rgb_frame  # Include original RGB frame for annotation
                    }

                    self.log_buffered_event(
                        'CALLBACK_RESULT_PROCESSED',
                        'Processed gesture callback results for publishing',
                        gesture_name=gesture_info.get('gesture_name', 'unknown'),
                        confidence=gesture_info.get('confidence', 0.0),
                        handedness=gesture_info.get('handedness', 'unknown'),
                        timestamp_ms=timestamp_ms
                    )

                    return result_dict
                else:
                    self.log_buffered_event(
                        'CALLBACK_NO_STABLE_GESTURE',
                        'No stable gesture detected in callback',
                        timestamp_ms=timestamp_ms
                    )
                    return None
            else:
                self.log_buffered_event(
                    'CALLBACK_NO_GESTURES',
                    'No gestures found in callback',
                    timestamp_ms=timestamp_ms
                )
                return None

        except Exception as e:
            self.log_buffered_event(
                'CALLBACK_PROCESSING_ERROR',
                f'Error processing callback results: {str(e)}',
                timestamp_ms=timestamp_ms,
                result_type=result_type
            )
            return None

    def extract_handedness(self, handedness_list, hand_index: int) -> str:
        """Extract handedness from MediaPipe results using standard category_name format."""
        if not handedness_list or hand_index >= len(handedness_list):
            return 'Unknown'

        try:
            handedness_data = handedness_list[hand_index]
            if hasattr(handedness_data, '__len__') and len(handedness_data) > 0:
                if hasattr(handedness_data[0], 'category_name'):
                    return handedness_data[0].category_name
        except (IndexError, AttributeError):
            pass

        return 'Unknown'

    def analyze_gesture_results(self, gestures, hand_landmarks_list, handedness_list, timestamp: float) -> Optional[Dict]:
        """Analyze MediaPipe gesture recognition results - simplified like working sample."""
        try:
            # Simple processing like the working sample code
            if gestures and len(gestures) > 0:
                # Process first hand with gestures (like sample code)
                for hand_index, hand_gestures in enumerate(gestures):
                    if hand_gestures and len(hand_gestures) > 0:
                        # Get the most confident gesture (like sample: gesture[0])
                        best_gesture = hand_gestures[0]
                        gesture_name = best_gesture.category_name
                        confidence = round(best_gesture.score, 2)  # Round like sample

                        # Get corresponding handedness - corrected MediaPipe access pattern
                        hand_label = self.extract_handedness(handedness_list, hand_index)

                        # Log raw detection
                        self.log_buffered_event(
                            'GESTURE_DETECTED_RAW',
                            f'Raw detection: {gesture_name} ({confidence}) on {hand_label} hand',
                            gesture=gesture_name,
                            confidence=confidence,
                            hand=hand_label,
                            timestamp=timestamp
                        )

                        # Apply enhanced stability filtering
                        if not self.check_gesture_stability(gesture_name, confidence, timestamp):
                            self.log_buffered_event(
                                'GESTURE_FILTERED_UNSTABLE',
                                f'Filtered unstable gesture: {gesture_name}',
                                gesture=gesture_name,
                                confidence=confidence,
                                timestamp=timestamp
                            )
                            return None  # Don't return unstable gestures

                        # Log stable detection
                        self.log_buffered_event(
                            'GESTURE_DETECTED_STABLE',
                            f'Stable gesture confirmed: {gesture_name} ({confidence}) on {hand_label} hand',
                            gesture=gesture_name,
                            confidence=confidence,
                            hand=hand_label,
                            timestamp=timestamp
                        )

                        hand_landmarks = hand_landmarks_list[hand_index] if hand_index < len(hand_landmarks_list) else None

                        # Calculate hand center safely
                        hand_center = Point()
                        try:
                            if hand_landmarks and hasattr(hand_landmarks, 'landmark') and hand_landmarks.landmark:
                                hand_center = self.calculate_hand_center(hand_landmarks.landmark)
                        except Exception as e:
                            self.log_buffered_event(
                                'HAND_CENTER_CALCULATION_ERROR',
                                f'Error calculating hand center: {str(e)} - using default Point()',
                                hand_index=hand_index,
                                has_landmarks=hand_landmarks is not None,
                                timestamp=timestamp
                            )

                        return {
                            'gesture_name': gesture_name,
                            'confidence': confidence,
                            'handedness': hand_label,
                            'nav_command': self.GESTURE_COMMANDS.get(gesture_name, ''),
                            'hand_center': hand_center,
                            'timestamp': timestamp
                        }

            # Log when no gestures found
            self.log_buffered_event(
                'NO_GESTURES_DETECTED',
                f'No gestures in results - gestures: {len(gestures) if gestures else 0}, hands: {len(hand_landmarks_list) if hand_landmarks_list else 0}',
                timestamp=timestamp
            )
            return None

        except Exception as e:
            self.log_buffered_event(
                'GESTURE_ANALYSIS_ERROR',
                f'Error analyzing gesture results: {str(e)}',
                timestamp=timestamp
            )
            return None

    def detect_simple_gesture(self, landmarks) -> Optional[str]:
        """Simple gesture detection based on finger positions (fallback method)."""
        try:
            # Get fingertip and pip (proximal interphalangeal) landmarks
            # Thumb: tip=4, pip=3
            # Index: tip=8, pip=6
            # Middle: tip=12, pip=10
            # Ring: tip=16, pip=14
            # Pinky: tip=20, pip=18

            # Check if fingers are extended
            thumb_up = landmarks[4].y < landmarks[3].y
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_up = landmarks[16].y < landmarks[14].y
            pinky_up = landmarks[20].y < landmarks[18].y

            fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
            total_fingers = sum(fingers_up)

            # Simple gesture classification
            if total_fingers == 0:
                return 'fist'
            elif total_fingers == 5:
                return 'open_palm'
            elif fingers_up == [True, False, False, False, False]:
                return 'thumbs_up'
            elif fingers_up == [False, True, False, False, False]:
                return 'pointing_up'
            elif fingers_up == [False, True, True, False, False]:
                return 'peace'
            elif total_fingers == 1 and index_up:
                # Check if pointing left or right based on hand orientation
                wrist_x = landmarks[0].x
                index_tip_x = landmarks[8].x
                if index_tip_x < wrist_x - 0.1:
                    return 'pointing_left'
                elif index_tip_x > wrist_x + 0.1:
                    return 'pointing_right'
                else:
                    return 'pointing_up'

            return None

        except Exception as e:
            self.log_buffered_event(
                'SIMPLE_GESTURE_ERROR',
                f'Error detecting simple gesture: {str(e)}'
            )
            return None

    def calculate_hand_center(self, landmarks) -> Point:
        """Calculate the center point of the hand."""
        try:
            # Calculate center as average of all landmarks
            x_sum = sum(lm.x for lm in landmarks)
            y_sum = sum(lm.y for lm in landmarks)
            z_sum = sum(lm.z for lm in landmarks)

            num_landmarks = len(landmarks)

            center = Point()
            center.x = x_sum / num_landmarks
            center.y = y_sum / num_landmarks
            center.z = z_sum / num_landmarks

            return center

        except Exception as e:
            self.log_buffered_event(
                'HAND_CENTER_ERROR',
                f'Error calculating hand center: {str(e)}'
            )
            return Point()

    def check_gesture_stability(self, gesture_name: str, confidence: float, timestamp: float) -> bool:
        """
        Enhanced stability checking with consistency and transition delay.
        Combines multiple stability criteria to prevent false positives.
        """
        try:
            # Add current detection to history
            self.gesture_detection_history.append({
                'gesture': gesture_name,
                'confidence': confidence,
                'timestamp': timestamp
            })

            # Keep only recent history
            if len(self.gesture_detection_history) > self.max_history_size:
                self.gesture_detection_history = self.gesture_detection_history[-self.max_history_size:]

            # Check 1: Consistency - same gesture detected N consecutive times
            if not self._check_gesture_consistency(gesture_name):
                return False

            # Check 2: Transition delay - minimum time between different gestures
            if not self._check_transition_delay(gesture_name, timestamp):
                return False

            # Check 3: Time-based stability - existing method
            if not self.is_gesture_stable(gesture_name, timestamp):
                return False

            # All checks passed - gesture is stable
            self.log_buffered_event(
                'GESTURE_STABILITY_CONFIRMED',
                f'Gesture {gesture_name} passed all stability checks',
                consistency_count=len([h for h in self.gesture_detection_history if h['gesture'] == gesture_name]),
                time_stable=timestamp - self.gesture_start_time if self.gesture_start_time else 0,
                timestamp=timestamp
            )
            return True

        except Exception as e:
            self.log_buffered_event(
                'STABILITY_CHECK_ERROR',
                f'Error in enhanced stability check: {str(e)}',
                gesture_name=gesture_name,
                timestamp=timestamp
            )
            return False

    def _check_gesture_consistency(self, gesture_name: str) -> bool:
        """Check if gesture appears consistently in recent history."""
        if len(self.gesture_detection_history) < self.gesture_consistency_count:
            return False

        # Check last N detections for consistency
        recent_gestures = [h['gesture'] for h in self.gesture_detection_history[-self.gesture_consistency_count:]]
        consistent_count = sum(1 for g in recent_gestures if g == gesture_name)

        return consistent_count >= self.gesture_consistency_count

    def _check_transition_delay(self, gesture_name: str, timestamp: float) -> bool:
        """Check if enough time has passed since last gesture change."""
        if self.current_gesture is None or self.current_gesture == gesture_name:
            return True  # No previous gesture or same gesture

        # Check if enough time has passed since last transition
        time_since_transition = timestamp - self.last_transition_time
        if time_since_transition < self.gesture_transition_delay:
            self.log_buffered_event(
                'GESTURE_TRANSITION_TOO_FAST',
                f'Gesture transition too fast: {time_since_transition:.2f}s < {self.gesture_transition_delay}s',
                from_gesture=self.current_gesture,
                to_gesture=gesture_name,
                timestamp=timestamp
            )
            return False

        return True

    def is_gesture_stable(self, gesture_name: str, timestamp: float) -> bool:
        """Check if gesture has been stable for minimum duration."""
        try:
            stability_threshold = self.gesture_stability_threshold

            if self.current_gesture != gesture_name:
                self.current_gesture = gesture_name
                self.gesture_start_time = timestamp
                self.last_transition_time = timestamp  # Track transition time
                self.log_buffered_event(
                    'GESTURE_CHANGE',
                    f'Gesture changed to: {gesture_name}',
                    timestamp=timestamp
                )
                return False

            if self.gesture_start_time and timestamp - self.gesture_start_time >= stability_threshold:
                self.log_buffered_event(
                    'GESTURE_STABLE',
                    f'Gesture {gesture_name} is stable',
                    duration=timestamp - self.gesture_start_time,
                    threshold=stability_threshold
                )
                return True

            return False

        except Exception as e:
            self.log_buffered_event(
                'STABILITY_CHECK_ERROR',
                f'Error checking gesture stability: {str(e)}',
                gesture_name=gesture_name,
                timestamp=timestamp
            )
            return False

    def publish_results(self, results: Dict, timestamp: float) -> None:
        """Publish gesture recognition results and optionally annotated images."""
        try:
            # Extract gesture info from results
            gesture_info = results['gesture_info']

            # Convert to ROS message
            msg = MessageConverter.create_gesture_message(
                gesture_info['gesture_name'],
                gesture_info['confidence'],
                gesture_info['handedness'],
                gesture_info['nav_command']
            )

            # Set header and additional fields
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            msg.hand_center = gesture_info['hand_center']

            if self.gesture_start_time:
                msg.gesture_duration = timestamp - self.gesture_start_time

            # Publish gesture results
            self.gestures_publisher.publish(msg)

            # Log navigation gestures
            if msg.is_nav_gesture:
                self.log_buffered_event(
                    'NAVIGATION_GESTURE_PUBLISHED',
                    f'Navigation gesture: {msg.gesture_name} -> {msg.nav_command}',
                    confidence=msg.confidence,
                    handedness=msg.handedness,
                    duration=msg.gesture_duration
                )

                # Also log to console for important navigation commands
                self.get_logger().info(
                    f'Navigation gesture: {msg.gesture_name} -> {msg.nav_command} '
                    f'(confidence: {msg.confidence:.2f}, {msg.handedness} hand)'
                )
            else:
                self.log_buffered_event(
                    'GESTURE_PUBLISHED',
                    f'Gesture detected: {msg.gesture_name}',
                    confidence=msg.confidence,
                    handedness=msg.handedness
                )

            # Publish annotated image if enabled
            if self.annotated_image_publisher and 'rgb_frame' in results:
                try:
                    annotated_frame = self.create_annotated_image(
                        results['rgb_frame'],
                        gesture_info
                    )
                    if annotated_frame is not None:
                        annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, 'rgb8')
                        annotated_msg.header = msg.header
                        self.annotated_image_publisher.publish(annotated_msg)

                        self.log_buffered_event(
                            'ANNOTATED_IMAGE_PUBLISHED',
                            'Published annotated gesture image',
                            gesture_name=gesture_info['gesture_name']
                        )
                except Exception as e:
                    self.log_buffered_event(
                        'ANNOTATION_ERROR',
                        f'Error creating annotated image: {str(e)}'
                    )

        except Exception as e:
            self.log_buffered_event(
                'PUBLISHING_ERROR',
                f'Error publishing results: {str(e)}',
                timestamp=timestamp
            )

    def create_annotated_image(self, rgb_frame: np.ndarray, gesture_info: Dict) -> Optional[np.ndarray]:
        """Create annotated image with comprehensive MediaPipe hand landmarks visualization."""
        try:
            # Create a copy for annotation (convert RGB to BGR for OpenCV drawing)
            annotated_frame = cv2.cvtColor(rgb_frame.copy(), cv2.COLOR_RGB2BGR)

            # Draw MediaPipe hand landmarks if available
            if hasattr(self, '_current_hand_landmarks') and self._current_hand_landmarks:
                self.draw_hand_landmarks(annotated_frame, self._current_hand_landmarks)

            # Add gesture text overlay
            gesture_text = f"{gesture_info['gesture_name']} ({gesture_info['confidence']:.2f})"
            nav_command = gesture_info.get('nav_command', '')
            if nav_command:
                gesture_text += f" -> {nav_command}"

            # Add text to image with background for better visibility
            self.draw_text_with_background(
                annotated_frame,
                gesture_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Standardized font scale
                (0, 255, 0),  # Green text
                1  # Standardized thickness
            )

            # Add handedness info
            handedness_text = f"{gesture_info['handedness']} hand"
            self.draw_text_with_background(
                annotated_frame,
                handedness_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )

            # Add hand center point if available
            hand_center = gesture_info.get('hand_center')
            if hand_center and (hand_center.x != 0.0 or hand_center.y != 0.0):
                center_x = int(hand_center.x * annotated_frame.shape[1])
                center_y = int(hand_center.y * annotated_frame.shape[0])
                cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 0, 255), -1)  # Magenta center
                cv2.circle(annotated_frame, (center_x, center_y), 12, (255, 255, 255), 2)  # White border

            # Convert back to RGB for ROS publishing
            return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            self.log_buffered_event(
                'ANNOTATION_CREATION_ERROR',
                f'Error creating annotation: {str(e)}'
            )
            return None

    def draw_hand_landmarks(self, image: np.ndarray, hand_landmarks_list) -> None:
        """Draw comprehensive MediaPipe hand landmarks and connections on the image."""
        try:
            # Import MediaPipe drawing utilities
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands

            # Draw landmarks for each detected hand
            for hand_index, hand_landmarks in enumerate(hand_landmarks_list):
                # Handle both MediaPipe NormalizedLandmarkList and Python list formats
                if hand_landmarks:
                    # Check if it's already a MediaPipe NormalizedLandmarkList
                    if hasattr(hand_landmarks, 'landmark'):
                        landmark_list = hand_landmarks
                    # Handle Python list format (convert to MediaPipe format)
                    elif isinstance(hand_landmarks, list):
                        # Create a MediaPipe NormalizedLandmarkList from the Python list
                        from mediapipe.framework.formats import landmark_pb2
                        landmark_list = landmark_pb2.NormalizedLandmarkList()
                        for landmark in hand_landmarks:
                            # Create a new landmark message and copy the values
                            new_landmark = landmark_list.landmark.add()
                            new_landmark.x = landmark.x
                            new_landmark.y = landmark.y
                            new_landmark.z = landmark.z
                            new_landmark.visibility = landmark.visibility
                            new_landmark.presence = landmark.presence
                    else:
                        continue

                    # Draw comprehensive hand landmarks visualization
                    try:
                        # Create custom drawing styles for optimal visibility
                        landmark_style = mp_drawing.DrawingSpec(
                            color=(0, 255, 0),  # Bright green
                            thickness=3,
                            circle_radius=4
                        )
                        connection_style = mp_drawing.DrawingSpec(
                            color=(255, 0, 255),  # Bright magenta
                            thickness=2
                        )

                        # Draw MediaPipe landmarks and connections
                        mp_drawing.draw_landmarks(
                            image,
                            landmark_list,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_style,
                            connection_style
                        )

                        # Add enhanced landmark visualization
                        height, width = image.shape[:2]

                        # Draw all landmarks with cyan circles for additional visibility
                        for i, landmark in enumerate(hand_landmarks):
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(image, (x, y), 3, (255, 255, 0), -1)  # Cyan circles
                            cv2.circle(image, (x, y), 5, (0, 255, 255), 1)   # Yellow border

                        # Draw hand skeleton connections manually to ensure visibility
                        hand_connections = [
                            # Thumb
                            (0, 1), (1, 2), (2, 3), (3, 4),
                            # Index finger
                            (0, 5), (5, 6), (6, 7), (7, 8),
                            # Middle finger
                            (0, 9), (9, 10), (10, 11), (11, 12),
                            # Ring finger
                            (0, 13), (13, 14), (14, 15), (15, 16),
                            # Pinky
                            (0, 17), (17, 18), (18, 19), (19, 20),
                            # Palm connections
                            (5, 9), (9, 13), (13, 17)
                        ]

                        for connection in hand_connections:
                            start_idx, end_idx = connection
                            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                                start_landmark = hand_landmarks[start_idx]
                                end_landmark = hand_landmarks[end_idx]
                                start_x = int(start_landmark.x * width)
                                start_y = int(start_landmark.y * height)
                                end_x = int(end_landmark.x * width)
                                end_y = int(end_landmark.y * height)
                                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)  # Bright magenta lines

                        # Highlight key landmarks (wrist and fingertips) with distinctive markers
                        for i, landmark in enumerate(hand_landmarks):
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            if i in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # Red circles
                                cv2.circle(image, (x, y), 10, (255, 255, 255), 2)  # White border
                                # Optional: Add landmark indices if enabled
                                if hasattr(self, '_show_landmark_indices') and self._show_landmark_indices:
                                    cv2.putText(image, str(i), (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Standardized thickness

                    except Exception as draw_error:
                        self.log_buffered_event(
                            'LANDMARK_DRAWING_ERROR',
                            f'Error drawing landmarks for hand {hand_index}: {str(draw_error)}'
                        )
                        # Simple fallback: draw basic landmark circles
                        try:
                            height, width = image.shape[:2]
                            for i, landmark in enumerate(hand_landmarks):
                                x = int(landmark.x * width)
                                y = int(landmark.y * height)
                                cv2.circle(image, (x, y), 4, (0, 255, 255), -1)  # Cyan fallback circles
                        except Exception as fallback_error:
                            self.log_buffered_event(
                                'FALLBACK_DRAWING_ERROR',
                                f'Fallback drawing failed: {str(fallback_error)}'
                            )

                    # Add landmark indices for debugging (optional - can be toggled)
                    if hasattr(self, '_show_landmark_indices') and self._show_landmark_indices:
                        self.draw_landmark_indices(image, hand_landmarks)

                    # Add hand bounding box
                    self.draw_hand_bounding_box(image, hand_landmarks, hand_index)

        except Exception as e:
            self.log_buffered_event(
                'LANDMARK_DRAWING_ERROR',
                f'Error drawing hand landmarks: {str(e)}'
            )

    def draw_landmark_indices(self, image: np.ndarray, hand_landmarks) -> None:
        """Draw landmark indices on each landmark point for debugging."""
        try:
            height, width = image.shape[:2]
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Draw small text with landmark index
                cv2.putText(
                    image,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Standardized font scale
                    (255, 255, 0),  # Yellow text
                    1  # Already using standardized thickness
                )
        except Exception as e:
            self.log_buffered_event(
                'LANDMARK_INDICES_ERROR',
                f'Error drawing landmark indices: {str(e)}'
            )

    def draw_hand_bounding_box(self, image: np.ndarray, hand_landmarks, hand_index: int) -> None:
        """Draw bounding box around detected hand."""
        try:
            height, width = image.shape[:2]

            # Handle both list format and MediaPipe landmark object format
            if hasattr(hand_landmarks, 'landmark'):
                # MediaPipe landmark object format
                landmarks = hand_landmarks.landmark
            else:
                # List format (already converted)
                landmarks = hand_landmarks

            # Calculate bounding box from landmarks
            x_coords = [landmark.x * width for landmark in landmarks]
            y_coords = [landmark.y * height for landmark in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)

            # Draw bounding box
            color = (0, 255, 255) if hand_index == 0 else (255, 0, 255)  # Cyan for first hand, Magenta for second
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # Add hand label
            label = f"Hand {hand_index + 1}"
            cv2.putText(
                image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Already using standardized font scale
                color,
                1  # Standardized thickness
            )

        except Exception as e:
            self.log_buffered_event(
                'BOUNDING_BOX_ERROR',
                f'Error drawing hand bounding box: {str(e)}'
            )

    def draw_text_with_background(self, image: np.ndarray, text: str, position: tuple,
                                font, font_scale: float, color: tuple, thickness: int) -> None:
        """Draw text with a semi-transparent background for better visibility."""
        try:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background rectangle
            x, y = position
            cv2.rectangle(
                image,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + baseline + 5),
                (0, 0, 0),  # Black background
                -1
            )

            # Draw text
            cv2.putText(image, text, position, font, font_scale, color, thickness)

        except Exception as e:
            self.log_buffered_event(
                'TEXT_DRAWING_ERROR',
                f'Error drawing text with background: {str(e)}'
            )

    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        try:
            # Disable callback processing
            self.disable_callback_processing()

            if self.controller:
                self.controller.close()
                self.log_buffered_event(
                    'CLEANUP_SUCCESS',
                    'MediaPipe gesture controller closed successfully'
                )

            super().cleanup()

        except Exception as e:
            self.log_buffered_event(
                'CLEANUP_ERROR',
                f'Error during cleanup: {str(e)}'
            )


def main(args=None):
    """Main function for gesture recognition node."""
    import rclpy

    rclpy.init(args=args)

    try:
        node = GestureRecognitionNode()

        # Log startup information
        node.get_logger().info('Gesture Recognition Node started with callback architecture')
        node.get_logger().info(f'Controller ready: {node.controller.is_ready() if node.controller else False}')
        node.get_logger().info(f'Callback processing active: {node.is_callback_active()}')

        rclpy.spin(node)

    except KeyboardInterrupt:
        print('Gesture recognition node interrupted by user')
    except Exception as e:
        print(f'Error in gesture recognition node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            try:
                node.cleanup()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
