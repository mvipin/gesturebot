#!/usr/bin/env python3
"""
Gesture Recognition Node for GestureBot Vision System
MediaPipe hand gesture recognition with navigation command mapping.
"""

import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import mediapipe as mp

from vision_core.base_node import MediaPipeBaseNode, ProcessingConfig
from vision_core.message_converter import MessageConverter
from gesturebot_vision.msg import HandGesture
from geometry_msgs.msg import Point


class GestureRecognitionNode(MediaPipeBaseNode):
    """
    ROS 2 node for real-time hand gesture recognition using MediaPipe.
    Maps recognized gestures to navigation commands for GestureBot.
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
        
        super().__init__('gesture_recognition_node', 'gesture_recognition', config)
        
        # MediaPipe components
        self.hands = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture state tracking
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_stability_threshold = 0.5  # seconds
        
        # Publishers
        self.gesture_publisher = self.create_publisher(
            HandGesture,
            '/vision/gestures',
            self.result_qos
        )
        
        # Parameters
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('max_hands', 2)
        self.declare_parameter('gesture_stability_threshold', 0.5)
        
        self.get_logger().info('Gesture Recognition Node initialized')
    
    def initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe hands detection."""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.get_parameter('max_hands').value,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.get_logger().info('MediaPipe hands initialized successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MediaPipe: {e}')
            return False
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """Process frame for gesture recognition."""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                # Analyze gestures from hand landmarks
                gesture_info = self.analyze_hand_gestures(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                    timestamp
                )
                
                if gesture_info:
                    return gesture_info
            
            return None
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            return None
    
    def analyze_hand_gestures(self, hand_landmarks_list, handedness_list, timestamp: float) -> Optional[Dict]:
        """Analyze hand landmarks to recognize gestures."""
        try:
            # Simple gesture recognition based on hand landmarks
            # This is a simplified implementation - you can enhance with more sophisticated algorithms
            
            for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
                hand_label = handedness.classification[0].label
                
                # Extract key landmarks
                landmarks = hand_landmarks.landmark
                
                # Simple gesture detection based on finger positions
                gesture_name = self.detect_simple_gesture(landmarks)
                
                if gesture_name and self.is_gesture_stable(gesture_name, timestamp):
                    return {
                        'gesture_name': gesture_name,
                        'confidence': 0.8,  # Simplified confidence
                        'handedness': hand_label,
                        'nav_command': self.GESTURE_COMMANDS.get(gesture_name, ''),
                        'hand_center': self.calculate_hand_center(landmarks),
                        'timestamp': timestamp
                    }
            
            return None
            
        except Exception as e:
            self.get_logger().error(f'Error analyzing gestures: {e}')
            return None
    
    def detect_simple_gesture(self, landmarks) -> Optional[str]:
        """Simple gesture detection based on finger positions."""
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
            self.get_logger().error(f'Error detecting gesture: {e}')
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
            self.get_logger().error(f'Error calculating hand center: {e}')
            return Point()
    
    def is_gesture_stable(self, gesture_name: str, timestamp: float) -> bool:
        """Check if gesture has been stable for minimum duration."""
        stability_threshold = self.get_parameter('gesture_stability_threshold').value
        
        if self.current_gesture != gesture_name:
            self.current_gesture = gesture_name
            self.gesture_start_time = timestamp
            return False
        
        if timestamp - self.gesture_start_time >= stability_threshold:
            return True
        
        return False
    
    def publish_results(self, results: Dict, timestamp: float) -> None:
        """Publish gesture recognition results."""
        try:
            msg = MessageConverter.create_gesture_message(
                results['gesture_name'],
                results['confidence'],
                results['handedness'],
                results['nav_command']
            )
            
            # Set header and additional fields
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            msg.hand_center = results['hand_center']
            
            if self.gesture_start_time:
                msg.gesture_duration = timestamp - self.gesture_start_time
            
            self.gesture_publisher.publish(msg)
            
            # Log navigation gestures
            if msg.is_nav_gesture:
                self.get_logger().info(
                    f'Navigation gesture: {msg.gesture_name} -> {msg.nav_command} '
                    f'(confidence: {msg.confidence:.2f})'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error publishing results: {e}')
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        try:
            if self.hands:
                self.hands.close()
            super().cleanup()
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')


def main(args=None):
    """Main function for gesture recognition node."""
    import rclpy
    
    rclpy.init(args=args)
    
    try:
        node = GestureRecognitionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in gesture recognition node: {e}')
    finally:
        if 'node' in locals():
            node.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
