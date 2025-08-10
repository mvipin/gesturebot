#!/usr/bin/env python3
"""
Message Converter for GestureBot Vision System
Handles conversion between OpenCV/MediaPipe data and ROS 2 messages.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from geometry_msgs.msg import Point, Vector3
from gesturebot.msg import (
    DetectedObject, DetectedObjects, HandGesture, HandLandmarks,
    PoseLandmarks, FaceDetection, VisionPerformance
)


class MessageConverter:
    """Utility class for converting between different message formats."""
    
    @staticmethod
    def mediapipe_detection_to_ros(detection, class_names: Dict[int, str]) -> DetectedObject:
        """Convert MediaPipe detection to ROS DetectedObject message."""
        # Create message with explicit field initialization
        msg = DetectedObject()

        # Initialize all fields with default values first
        msg.class_name = "unknown"
        msg.class_id = -1
        msg.bbox_x = 0
        msg.bbox_y = 0
        msg.bbox_width = 0
        msg.bbox_height = 0
        msg.has_3d_position = False
        msg.track_id = -1
        msg.is_tracked = False

        # Initialize confidence - try multiple approaches
        try:
            msg.confidence = 0.0
        except:
            # If property assignment fails, try direct attribute access
            object.__setattr__(msg, 'confidence', 0.0)

        try:
            # Get the best category
            if detection.categories and len(detection.categories) > 0:
                best_category = detection.categories[0]

                # Set class name with None check
                if hasattr(best_category, 'category_name') and best_category.category_name is not None:
                    msg.class_name = str(best_category.category_name)

                # Set class ID with None check
                if hasattr(best_category, 'index') and best_category.index is not None:
                    try:
                        msg.class_id = int(best_category.index)
                    except (TypeError, ValueError):
                        msg.class_id = -1  # Keep default

                # Handle confidence score with robust None checking
                if hasattr(best_category, 'score') and best_category.score is not None:
                    try:
                        score_val = best_category.score
                        # Ensure score_val is not None before conversion
                        if score_val is not None:
                            confidence_val = float(score_val)
                            # Try multiple assignment approaches
                            try:
                                msg.confidence = confidence_val
                            except:
                                object.__setattr__(msg, 'confidence', confidence_val)
                        else:
                            # Score exists but is None
                            try:
                                msg.confidence = 0.0
                            except:
                                object.__setattr__(msg, 'confidence', 0.0)
                    except (TypeError, ValueError, AttributeError):
                        try:
                            msg.confidence = 0.0
                        except:
                            object.__setattr__(msg, 'confidence', 0.0)

            # Bounding box with robust None handling
            if hasattr(detection, 'bounding_box') and detection.bounding_box is not None:
                bbox = detection.bounding_box
                try:
                    # Check for None values before conversion
                    origin_x = getattr(bbox, 'origin_x', None)
                    origin_y = getattr(bbox, 'origin_y', None)
                    width = getattr(bbox, 'width', None)
                    height = getattr(bbox, 'height', None)

                    msg.bbox_x = int(origin_x) if origin_x is not None else 0
                    msg.bbox_y = int(origin_y) if origin_y is not None else 0
                    msg.bbox_width = int(width) if width is not None else 0
                    msg.bbox_height = int(height) if height is not None else 0
                except (TypeError, ValueError, AttributeError):
                    pass  # Keep default values

        except Exception as e:
            # More detailed error logging for debugging
            import traceback
            print(f"Error in MediaPipe detection conversion: {e}")
            print(f"Detection object type: {type(detection)}")
            if hasattr(detection, 'categories'):
                print(f"Categories: {detection.categories}")
            if hasattr(detection, 'bounding_box'):
                print(f"Bounding box: {detection.bounding_box}")
            print(f"Traceback: {traceback.format_exc()}")
            # Keep default values already set

        return msg
    
    @staticmethod
    def mediapipe_detections_to_ros(detections, detector_name: str, 
                                   processing_time: float) -> DetectedObjects:
        """Convert list of MediaPipe detections to ROS DetectedObjects message."""
        msg = DetectedObjects()
        msg.detector_name = detector_name
        msg.processing_time = processing_time
        msg.total_detections = len(detections)
        
        for detection in detections:
            obj_msg = MessageConverter.mediapipe_detection_to_ros(detection, {})
            msg.objects.append(obj_msg)
        
        return msg
    
    @staticmethod
    def mediapipe_landmarks_to_points(landmarks) -> List[Point]:
        """Convert MediaPipe landmarks to ROS Point messages."""
        points = []
        for landmark in landmarks:
            point = Point()
            point.x = float(landmark.x)
            point.y = float(landmark.y)
            point.z = float(landmark.z) if hasattr(landmark, 'z') else 0.0
            points.append(point)
        return points
    
    @staticmethod
    def create_hand_landmarks_message(hand_landmarks, handedness: str, 
                                    confidence: float) -> HandLandmarks:
        """Create HandLandmarks message from MediaPipe hand landmarks."""
        msg = HandLandmarks()
        msg.handedness = handedness
        msg.confidence = confidence
        
        # Convert landmarks
        if hand_landmarks:
            msg.landmarks = MessageConverter.mediapipe_landmarks_to_points(hand_landmarks.landmark)
            
            # Extract key points
            if len(msg.landmarks) >= 21:  # Standard hand has 21 landmarks
                msg.wrist_position = msg.landmarks[0]  # Wrist is landmark 0
                
                # Fingertip positions (landmarks 4, 8, 12, 16, 20)
                fingertip_indices = [4, 8, 12, 16, 20]
                for i, idx in enumerate(fingertip_indices):
                    if idx < len(msg.landmarks):
                        msg.fingertip_positions[i] = msg.landmarks[idx]
        
        # Calculate hand size (distance from wrist to middle finger tip)
        if len(msg.landmarks) >= 12:
            wrist = msg.landmarks[0]
            middle_tip = msg.landmarks[12]
            msg.hand_size = np.sqrt(
                (middle_tip.x - wrist.x)**2 + 
                (middle_tip.y - wrist.y)**2 + 
                (middle_tip.z - wrist.z)**2
            )
        
        return msg
    
    @staticmethod
    def create_pose_landmarks_message(pose_landmarks, confidence: float) -> PoseLandmarks:
        """Create PoseLandmarks message from MediaPipe pose landmarks."""
        msg = PoseLandmarks()
        msg.confidence = confidence
        msg.is_present = pose_landmarks is not None
        
        if pose_landmarks:
            msg.landmarks = MessageConverter.mediapipe_landmarks_to_points(pose_landmarks.landmark)
            
            # Extract key body points (MediaPipe pose landmark indices)
            landmark_map = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            for body_part, idx in landmark_map.items():
                if idx < len(msg.landmarks):
                    setattr(msg, body_part, msg.landmarks[idx])
            
            # Calculate center of mass (average of hip positions)
            if len(msg.landmarks) > 24:
                left_hip = msg.landmarks[23]
                right_hip = msg.landmarks[24]
                msg.center_of_mass = Point()
                msg.center_of_mass.x = (left_hip.x + right_hip.x) / 2
                msg.center_of_mass.y = (left_hip.y + right_hip.y) / 2
                msg.center_of_mass.z = (left_hip.z + right_hip.z) / 2
        
        return msg
    
    @staticmethod
    def create_face_detection_message(face_detection, face_id: int = 0) -> FaceDetection:
        """Create FaceDetection message from MediaPipe face detection."""
        msg = FaceDetection()
        msg.face_id = face_id
        msg.is_present = face_detection is not None
        
        if face_detection:
            # Get detection confidence
            if hasattr(face_detection, 'score'):
                msg.confidence = face_detection.score[0] if isinstance(face_detection.score, list) else face_detection.score
            
            # Bounding box
            if hasattr(face_detection, 'location_data'):
                bbox = face_detection.location_data.relative_bounding_box
                msg.bbox_x = int(bbox.xmin * 640)  # Assuming 640px width
                msg.bbox_y = int(bbox.ymin * 480)  # Assuming 480px height
                msg.bbox_width = int(bbox.width * 640)
                msg.bbox_height = int(bbox.height * 480)
                
                # Face center
                msg.face_center = Point()
                msg.face_center.x = msg.bbox_x + msg.bbox_width / 2
                msg.face_center.y = msg.bbox_y + msg.bbox_height / 2
                
                # Face size (normalized)
                msg.face_size = bbox.width * bbox.height
        
        # Default values for optional fields
        msg.estimated_age = -1.0
        msg.estimated_emotion = ""
        msg.emotion_confidence = 0.0
        
        return msg
    
    @staticmethod
    def create_gesture_message(gesture_name: str, confidence: float, 
                             handedness: str, nav_command: str = "") -> HandGesture:
        """Create HandGesture message."""
        msg = HandGesture()
        msg.gesture_name = gesture_name
        msg.gesture_id = hash(gesture_name) % 1000  # Simple ID generation
        msg.confidence = confidence
        msg.handedness = handedness
        msg.is_present = True
        msg.nav_command = nav_command
        msg.is_nav_gesture = bool(nav_command)
        
        # Default values
        msg.hand_center = Point()
        msg.hand_size = 1.0
        msg.gesture_duration = 0.0
        
        return msg
    
    @staticmethod
    def opencv_contour_to_detected_object(contour, class_name: str, 
                                        confidence: float = 1.0) -> DetectedObject:
        """Convert OpenCV contour to DetectedObject message."""
        msg = DetectedObject()
        msg.class_name = class_name
        msg.class_id = -1
        msg.confidence = confidence
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        msg.bbox_x = x
        msg.bbox_y = y
        msg.bbox_width = w
        msg.bbox_height = h
        
        # Default values
        msg.has_3d_position = False
        msg.track_id = -1
        msg.is_tracked = False
        
        return msg
    
    @staticmethod
    def point_to_pixel_coords(point: Point, image_width: int, image_height: int) -> tuple:
        """Convert normalized point coordinates to pixel coordinates."""
        pixel_x = int(point.x * image_width)
        pixel_y = int(point.y * image_height)
        return (pixel_x, pixel_y)
    
    @staticmethod
    def pixel_coords_to_point(x: int, y: int, image_width: int, image_height: int) -> Point:
        """Convert pixel coordinates to normalized Point."""
        point = Point()
        point.x = float(x) / image_width
        point.y = float(y) / image_height
        point.z = 0.0
        return point
