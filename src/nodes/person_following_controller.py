#!/usr/bin/env python3
"""
Person Following Controller for GestureBot
Standalone person following system using object detection.
Phase 2A: Basic subscription and person filtering implementation.
"""

import time
from typing import Optional, List
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from gesturebot.msg import DetectedObjects, DetectedObject


class FollowingStatus(Enum):
    """Person following system states."""
    IDLE = "idle"
    SEARCHING = "searching"
    TRACKING = "tracking"
    FOLLOWING = "following"
    PERSON_LOST = "person_lost"
    SAFETY_STOP = "safety_stop"
    EMERGENCY_STOP = "emergency_stop"


class PersonFollowingController(Node):
    """
    Main controller for person following behavior.
    Subscribes to object detection and publishes motion commands.
    """
    
    def __init__(self):
        super().__init__('person_following_controller')
        
        # System state
        self.following_active = False
        self.current_status = FollowingStatus.IDLE
        self.last_detection_time = 0.0
        
        # Person tracking state
        self.target_person = None
        self.last_person_seen_time = 0.0
        self.person_lost_timeout = 3.0  # seconds

        # Control parameters
        self.target_distance = 1.5  # meters
        self.distance_tolerance = 0.1  # meters - reduced for more responsive control
        self.min_safe_distance = 0.8  # meters
        self.max_follow_distance = 5.0  # meters

        # Velocity smoothing state
        self.target_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.current_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.last_velocity_update = time.time()
        self.last_published_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.zero_velocity_logged = False

        # Control stability state
        self.last_control_calculation_time = 0.0
        self.control_hold_duration = 0.5  # Hold control commands for 500ms minimum
        self.control_active = False  # Track if we're in active control mode

        # Distance estimation smoothing
        self.distance_history = []
        self.distance_history_size = 3  # Average over 3 measurements
        
        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.objects_subscription = self.create_subscription(
            DetectedObjects,
            '/vision/objects',
            self.objects_callback,
            self.sensor_qos
        )
        
        self.emergency_subscription = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_callback,
            self.reliable_qos
        )
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            self.reliable_qos
        )
        
        # Services
        self.activate_service = self.create_service(
            SetBool,
            '/follow_mode/activate',
            self.activate_callback
        )
        
        # Control timer - reduced frequency for target calculation
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        # High-frequency velocity smoothing timer for stability
        velocity_update_rate = 25.0  # Hz - same as other navigation nodes
        self.velocity_smoothing_timer = self.create_timer(
            1.0 / velocity_update_rate,
            self.update_smoothed_velocity
        )
        
        self.get_logger().info('👤 Person Following Controller initialized')
        self.get_logger().info(f'📡 Subscribed to /vision/objects for person detection')
        self.get_logger().info(f'🎮 Control parameters: target_distance={self.target_distance}m, safe_distance={self.min_safe_distance}m')

    def objects_callback(self, msg: DetectedObjects) -> None:
        """
        Process incoming object detection messages to find and track people.
        This is the main entry point for person following logic.
        """
        try:
            current_time = time.time()
            self.last_detection_time = current_time
            
            # Only process if following is active
            if not self.following_active:
                return
            
            # Filter detected objects to find only people
            people = self.filter_people_from_objects(msg.objects)
            
            if not people:
                self.handle_no_people_detected(current_time)
                return
            
            # Process detected people
            self.process_detected_people(people, current_time)
            
        except Exception as e:
            self.get_logger().error(f'Error in objects callback: {e}')
            self.publish_stop_command("Error in object processing")

    def filter_people_from_objects(self, detected_objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Filter detected objects to find only 'person' class detections with sufficient confidence.
        
        Args:
            detected_objects: List of all detected objects
            
        Returns:
            List of person detections that meet our criteria
        """
        people = []
        
        for obj in detected_objects:
            # Check if this is a person detection
            if obj.class_name.lower() != 'person':
                continue
                
            # Check confidence threshold
            if obj.confidence < 0.6:  # Minimum confidence for person following
                continue
                
            # Check bounding box size (filter out very small detections)
            # Convert pixel coordinates to normalized coordinates (assuming 640x480 image)
            image_width = 640.0
            image_height = 480.0
            normalized_width = obj.bbox_width / image_width
            normalized_height = obj.bbox_height / image_height
            bbox_area = normalized_width * normalized_height

            if bbox_area < 0.02:  # Must be at least 2% of image
                continue
                
            people.append(obj)
        
        # Log detection info
        if people:
            self.get_logger().debug(
                f'👥 Found {len(people)} people: '
                f'confidences=[{", ".join([f"{p.confidence:.2f}" for p in people])}]'
            )
        
        return people

    def process_detected_people(self, people: List[DetectedObject], current_time: float) -> None:
        """
        Process the list of detected people and update tracking/following behavior.
        
        Args:
            people: List of person detections
            current_time: Current timestamp
        """
        # Select or update target person
        selected_person = self.select_target_person(people)
        
        if selected_person:
            self.target_person = selected_person
            self.last_person_seen_time = current_time
            self.current_status = FollowingStatus.TRACKING
            
            # Calculate normalized center coordinates
            image_width = 640.0
            image_height = 480.0
            center_x = (selected_person.bbox_x + selected_person.bbox_width / 2) / image_width
            center_y = (selected_person.bbox_y + selected_person.bbox_height / 2) / image_height
            norm_width = selected_person.bbox_width / image_width
            norm_height = selected_person.bbox_height / image_height

            self.get_logger().debug(
                f'🎯 Tracking person: confidence={selected_person.confidence:.2f}, '
                f'center=({center_x:.2f}, {center_y:.2f}), '
                f'size={norm_width:.2f}x{norm_height:.2f}'
            )
        else:
            self.handle_no_suitable_person(current_time)

    def select_target_person(self, people: List[DetectedObject]) -> Optional[DetectedObject]:
        """
        Select the best person to follow from the list of detected people.
        
        Args:
            people: List of person detections
            
        Returns:
            Selected person to follow, or None if no suitable person found
        """
        if not people:
            return None
        
        # If we don't have a current target, select the best initial target
        if self.target_person is None:
            return self.select_initial_target(people)
        
        # If we have a target, try to maintain continuity
        return self.maintain_target_continuity(people)

    def select_initial_target(self, people: List[DetectedObject]) -> DetectedObject:
        """
        Select the initial target person based on size and position criteria.
        
        Args:
            people: List of person detections
            
        Returns:
            Best person to start following
        """
        scored_people = []
        
        for person in people:
            # Calculate normalized coordinates
            image_width = 640.0
            image_height = 480.0
            norm_width = person.bbox_width / image_width
            norm_height = person.bbox_height / image_height
            center_x = (person.bbox_x + person.bbox_width / 2) / image_width

            # Size score (larger people are preferred)
            size_score = norm_width * norm_height

            # Center score (people closer to image center are preferred)
            center_score = 1.0 - abs(center_x - 0.5) * 2  # 0.5 is image center
            
            # Confidence score
            confidence_score = person.confidence
            
            # Combined score (weighted)
            total_score = (size_score * 0.4 + 
                          center_score * 0.3 + 
                          confidence_score * 0.3)
            
            scored_people.append((person, total_score))
        
        # Select person with highest score
        best_person = max(scored_people, key=lambda x: x[1])[0]
        
        # Calculate center for logging
        image_width = 640.0
        image_height = 480.0
        center_x = (best_person.bbox_x + best_person.bbox_width / 2) / image_width
        center_y = (best_person.bbox_y + best_person.bbox_height / 2) / image_height

        self.get_logger().info(
            f'🎯 Selected initial target: confidence={best_person.confidence:.2f}, '
            f'center=({center_x:.2f}, {center_y:.2f})'
        )
        
        return best_person

    def maintain_target_continuity(self, people: List[DetectedObject]) -> Optional[DetectedObject]:
        """
        Try to maintain continuity with the current target person.
        
        Args:
            people: List of current person detections
            
        Returns:
            Matched person or best alternative, or None if no suitable match
        """
        if not self.target_person:
            return self.select_initial_target(people)
        
        # Try to find the same person based on position similarity
        best_match = None
        min_distance = float('inf')

        # Calculate target person's normalized center
        image_width = 640.0
        image_height = 480.0
        target_center_x = (self.target_person.bbox_x + self.target_person.bbox_width / 2) / image_width
        target_center_y = (self.target_person.bbox_y + self.target_person.bbox_height / 2) / image_height
        target_center = (target_center_x, target_center_y)

        for person in people:
            # Calculate current person's normalized center
            person_center_x = (person.bbox_x + person.bbox_width / 2) / image_width
            person_center_y = (person.bbox_y + person.bbox_height / 2) / image_height
            person_center = (person_center_x, person_center_y)

            # Calculate distance between centers
            distance = ((target_center[0] - person_center[0]) ** 2 +
                       (target_center[1] - person_center[1]) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = person
        
        # If the best match is reasonably close, use it
        if best_match and min_distance < 0.3:  # Within 30% of image
            return best_match
        else:
            # No good match found - select new target
            self.get_logger().info('🔄 Target person lost, selecting new target')
            return self.select_initial_target(people)

    def handle_no_people_detected(self, current_time: float) -> None:
        """Handle the case when no people are detected."""
        time_since_last_person = current_time - self.last_person_seen_time
        
        if time_since_last_person < self.person_lost_timeout:
            self.current_status = FollowingStatus.SEARCHING
            self.publish_stop_command("Searching for person")
        else:
            self.current_status = FollowingStatus.PERSON_LOST
            self.following_active = False  # Auto-deactivate
            self.target_person = None
            self.publish_stop_command("Person lost - deactivating follow mode")
            self.get_logger().info('❌ Person lost for too long, deactivating follow mode')

    def handle_no_suitable_person(self, current_time: float) -> None:
        """Handle the case when people are detected but none are suitable for following."""
        self.current_status = FollowingStatus.SEARCHING
        self.publish_stop_command("No suitable person found")

    def emergency_callback(self, msg: Bool) -> None:
        """Handle emergency stop signals."""
        if msg.data:
            self.current_status = FollowingStatus.EMERGENCY_STOP
            self.following_active = False
            self.publish_stop_command("Emergency stop activated")
            self.get_logger().warn('🛑 Emergency stop activated - person following disabled')

    def activate_callback(self, request, response):
        """Service callback to activate/deactivate person following."""
        self.following_active = request.data
        
        if self.following_active:
            self.current_status = FollowingStatus.SEARCHING
            self.target_person = None
            response.message = "Person following activated"
            self.get_logger().info('✅ Person following mode activated')
        else:
            self.current_status = FollowingStatus.IDLE
            self.target_person = None
            self.publish_stop_command("Following deactivated")
            response.message = "Person following deactivated"
            self.get_logger().info('⏹️ Person following mode deactivated')
        
        response.success = True
        return response

    def control_loop(self) -> None:
        """
        Main control loop that calculates target velocities for following behavior.
        High-frequency smoothing is handled separately by update_smoothed_velocity().
        """
        current_time = time.time()

        if not self.following_active or self.current_status != FollowingStatus.TRACKING:
            # Set stop targets when not actively following
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            return

        if not self.target_person:
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            return

        # Check if we have recent person detection data
        time_since_last_person = current_time - self.last_person_seen_time
        if time_since_last_person > 1.0:  # No person data for 1000ms
            self.get_logger().info(f'⏰ Stale person data ({time_since_last_person:.2f}s), stopping')
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            self.control_active = False
            return

        try:
            # Check if we should update control (prevent rapid switching)
            time_since_last_control = current_time - self.last_control_calculation_time

            # If we're in active control mode, hold the current command longer
            if self.control_active and time_since_last_control < self.control_hold_duration:
                self.get_logger().debug(f'🔒 Holding control command for {self.control_hold_duration - time_since_last_control:.2f}s more')
                return  # Keep current targets

            # Update control calculation time
            self.last_control_calculation_time = current_time

            # Estimate distance to target person
            estimated_distance = self.estimate_distance(self.target_person)

            # Calculate control command
            control_cmd = self.calculate_following_command(self.target_person, estimated_distance)

            # Safety check and set target velocities
            if self.is_safe_to_move(control_cmd, estimated_distance):
                # Set new targets and mark control as active
                self.target_velocity['linear_x'] = control_cmd['linear_x']
                self.target_velocity['angular_z'] = control_cmd['angular_z']
                self.current_status = FollowingStatus.FOLLOWING
                self.control_active = True
                self.get_logger().info(f'🎯 NEW TARGETS: ({control_cmd["linear_x"]:.2f}, {control_cmd["angular_z"]:.2f}) - HOLDING for {self.control_hold_duration:.1f}s')
            else:
                # Safety stop - set zero targets
                self.target_velocity['linear_x'] = 0.0
                self.target_velocity['angular_z'] = 0.0
                self.current_status = FollowingStatus.SAFETY_STOP
                self.control_active = False
                self.get_logger().info('🛑 Safety stop - zero targets')

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            # Safety: set stop targets on error
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            self.control_active = False

    def estimate_distance(self, person: DetectedObject) -> float:
        """
        Estimate distance to person using bounding box size.
        Recalibrated thresholds for realistic camera geometry.

        Args:
            person: Detected person object

        Returns:
            Estimated distance in meters
        """
        # Use bounding box area as distance proxy
        # Convert to normalized coordinates
        image_width = 640.0
        image_height = 480.0
        norm_width = person.bbox_width / image_width
        norm_height = person.bbox_height / image_height
        bbox_area = norm_width * norm_height

        # Recalibrated empirical relationship for realistic distances
        # Based on typical person height (~1.7m) and camera field of view
        # Target: 50% frame occupation should estimate ~1.5-2.0m (near target distance)
        if bbox_area > 0.7:     # Extremely close (< 0.8m) - person fills >70% of frame
            estimated_distance = 0.6
        elif bbox_area > 0.55:  # Very close (0.8-1.2m) - person fills ~55-70% of frame
            estimated_distance = 1.0
        elif bbox_area > 0.35:  # Close (1.2-1.8m) - person fills ~35-55% of frame
            estimated_distance = 1.5  # Near target distance
        elif bbox_area > 0.20:  # Medium (1.8-2.5m) - person fills ~20-35% of frame
            estimated_distance = 2.2
        elif bbox_area > 0.10:  # Far (2.5-4.0m) - person fills ~10-20% of frame
            estimated_distance = 3.2
        elif bbox_area > 0.05:  # Very far (4.0-6.0m) - person fills ~5-10% of frame
            estimated_distance = 5.0
        else:                   # Extremely far (>6.0m) - person very small
            estimated_distance = 7.0

        # Apply distance smoothing to reduce noise
        self.distance_history.append(estimated_distance)
        if len(self.distance_history) > self.distance_history_size:
            self.distance_history.pop(0)

        # Use moving average for smoother distance estimation
        smoothed_distance = sum(self.distance_history) / len(self.distance_history)

        # Debug logging for distance calibration (reduced frequency)
        if len(self.distance_history) == 1:  # Only log on first measurement or significant changes
            self.get_logger().info(
                f'📏 Distance estimation: bbox_area={bbox_area:.3f} '
                f'({norm_width:.2f}×{norm_height:.2f}) → raw={estimated_distance:.1f}m, smoothed={smoothed_distance:.1f}m'
            )

        return smoothed_distance

    def calculate_following_command(self, person: DetectedObject, estimated_distance: float) -> dict:
        """
        Calculate velocity commands to follow the person.

        Args:
            person: Target person detection
            estimated_distance: Estimated distance to person

        Returns:
            Dictionary with linear_x, angular_z, and reason
        """
        # Distance control (linear velocity)
        distance_error = estimated_distance - self.target_distance
        linear_velocity = 0.0

        # Always apply proportional control (no dead zone for small errors)
        if abs(distance_error) > 0.05:  # Only ignore very small errors (<5cm)
            # Move forward/backward to maintain target distance
            linear_gain = 0.8  # Increased gain for more responsive control
            linear_velocity = distance_error * linear_gain

            # Clamp to safe limits
            max_linear = 0.3  # Increased max velocity
            linear_velocity = max(-max_linear, min(max_linear, linear_velocity))

        # Apply minimum velocity threshold to overcome static friction
        if abs(linear_velocity) > 0.0 and abs(linear_velocity) < 0.05:
            linear_velocity = 0.05 if linear_velocity > 0 else -0.05

        # Angular control (centering person in view)
        # Calculate normalized center coordinates
        image_width = 640.0
        image_height = 480.0
        center_x = (person.bbox_x + person.bbox_width / 2) / image_width
        center_error = center_x - 0.5  # 0.5 is image center

        # Only apply angular control if error is significant
        angular_velocity = 0.0
        if abs(center_error) > 0.05:  # 5% of image width
            angular_gain = 1.5  # Increased gain for more responsive turning
            angular_velocity = -center_error * angular_gain  # Negative for correct direction

            # Clamp angular velocity
            max_angular = 0.8  # Increased max angular velocity
            angular_velocity = max(-max_angular, min(max_angular, angular_velocity))

            # Apply minimum angular velocity threshold
            if abs(angular_velocity) > 0.0 and abs(angular_velocity) < 0.1:
                angular_velocity = 0.1 if angular_velocity > 0 else -0.1

        # Log the control calculation for debugging (reduced frequency)
        if abs(linear_velocity) > 0.1 or abs(angular_velocity) > 0.1:  # Only log significant commands
            self.get_logger().info(
                f'🎮 Control calc: dist={estimated_distance:.1f}m (target={self.target_distance:.1f}m), '
                f'dist_err={distance_error:.2f}m, center_err={center_error:.2f}, '
                f'cmd=({linear_velocity:.2f}, {angular_velocity:.2f})'
            )

        return {
            'linear_x': linear_velocity,
            'angular_z': angular_velocity,
            'reason': f'Following: dist={estimated_distance:.1f}m, center_err={center_error:.2f}'
        }

    def is_safe_to_move(self, control_cmd: dict, estimated_distance: float) -> bool:
        """
        Check if it's safe to execute the control command.

        Args:
            control_cmd: Velocity command dictionary
            estimated_distance: Estimated distance to person

        Returns:
            True if safe to move, False otherwise
        """
        # Check minimum safe distance - only prevent forward motion if too close
        if estimated_distance < self.min_safe_distance:
            if control_cmd['linear_x'] > 0:  # Don't move forward if too close
                self.get_logger().info(
                    f'🛑 Safety: Too close ({estimated_distance:.1f}m < {self.min_safe_distance}m), '
                    f'blocking forward motion (cmd: {control_cmd["linear_x"]:.2f})'
                )
                return False
            else:
                # Allow backward motion when too close
                self.get_logger().info(
                    f'✅ Safety: Too close but allowing backward motion (cmd: {control_cmd["linear_x"]:.2f})'
                )

        # Check maximum follow distance
        if estimated_distance > self.max_follow_distance:
            self.get_logger().info(
                f'🛑 Safety: Too far ({estimated_distance:.1f}m > {self.max_follow_distance}m), '
                f'stopping follow'
            )
            return False

        # Log that movement is allowed
        self.get_logger().debug(
            f'✅ Safety: Movement allowed - dist={estimated_distance:.1f}m, cmd=({control_cmd["linear_x"]:.2f}, {control_cmd["angular_z"]:.2f})'
        )

        # Check for emergency stop
        if self.current_status == FollowingStatus.EMERGENCY_STOP:
            return False

        return True

    # Note: Direct velocity publishing removed - now using smoothed velocity system

    def publish_stop_command(self, reason: str = "") -> None:
        """Publish a stop command to halt robot motion."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)
        
        if reason:
            self.get_logger().debug(f'🛑 Stop command: {reason}')

    def update_smoothed_velocity(self) -> None:
        """
        High-frequency velocity smoothing with acceleration limiting.
        Prevents abrupt velocity changes that cause jerky motion.
        """
        try:
            current_time = time.time()
            dt = current_time - self.last_velocity_update
            self.last_velocity_update = current_time

            # Skip if dt is too large (system lag) or too small
            if dt > 0.1 or dt < 0.001:
                return

            # Motion smoothing parameters (optimized for 25Hz update rate)
            motion_smoothing_enabled = True  # Always enabled for person following
            max_linear_accel = 1.0    # m/s² - increased for more responsive control
            max_angular_accel = 2.0   # rad/s² - increased for more responsive turning
            emergency_decel = 2.0     # m/s² for emergency stops

            # Check if motion smoothing is enabled
            if not motion_smoothing_enabled:
                # Direct velocity control (bypass smoothing)
                twist = Twist()
                twist.linear.x = self.target_velocity['linear_x']
                twist.angular.z = self.target_velocity['angular_z']
                self.cmd_vel_publisher.publish(twist)
                return

            # Use faster deceleration for emergency stops
            if self.current_status == FollowingStatus.EMERGENCY_STOP:
                max_linear_accel = emergency_decel
                max_angular_accel = emergency_decel

            # Apply acceleration limiting to linear velocity
            self.current_velocity['linear_x'] = self.apply_acceleration_limit(
                self.current_velocity['linear_x'],
                self.target_velocity['linear_x'],
                max_linear_accel,
                dt
            )

            # Apply acceleration limiting to angular velocity
            self.current_velocity['angular_z'] = self.apply_acceleration_limit(
                self.current_velocity['angular_z'],
                self.target_velocity['angular_z'],
                max_angular_accel,
                dt
            )

            # Create and publish smoothed Twist message
            twist = Twist()
            twist.linear.x = self.current_velocity['linear_x']
            twist.angular.z = self.current_velocity['angular_z']
            self.cmd_vel_publisher.publish(twist)

            # Smart logging: only log when velocity changes or is non-zero
            self.log_velocity_change(twist)

        except Exception as e:
            self.get_logger().error(f'Error in velocity smoothing: {e}')
            # Safety: publish stop command on error
            stop_twist = Twist()
            self.cmd_vel_publisher.publish(stop_twist)

    def apply_acceleration_limit(self, current_vel: float, target_vel: float, max_accel: float, dt: float) -> float:
        """
        Apply acceleration limiting to prevent abrupt velocity changes.

        Args:
            current_vel: Current velocity value
            target_vel: Desired target velocity
            max_accel: Maximum allowed acceleration (positive value)
            dt: Time step since last update

        Returns:
            New velocity value with acceleration limiting applied
        """
        velocity_diff = target_vel - current_vel
        max_change = max_accel * dt

        if abs(velocity_diff) <= max_change:
            return target_vel  # Can reach target this step
        else:
            # Limit the change to maximum allowed acceleration
            return current_vel + (max_change if velocity_diff > 0 else -max_change)

    def log_velocity_change(self, twist: Twist) -> None:
        """
        Smart logging that only logs when velocity actually changes or is significant.
        Reduces log spam from repeated zero velocity commands.
        """
        try:
            current_linear = twist.linear.x
            current_angular = twist.angular.z
            last_linear = self.last_published_velocity['linear_x']
            last_angular = self.last_published_velocity['angular_z']

            # Check if velocity is zero
            is_zero_velocity = abs(current_linear) < 0.001 and abs(current_angular) < 0.001
            was_zero_velocity = abs(last_linear) < 0.001 and abs(last_angular) < 0.001

            # Check if velocity has changed significantly (threshold to avoid noise)
            linear_changed = abs(current_linear - last_linear) > 0.01
            angular_changed = abs(current_angular - last_angular) > 0.01
            velocity_changed = linear_changed or angular_changed

            # Log conditions:
            # 1. Non-zero velocity commands (always log active motion)
            # 2. Velocity changed significantly from last command
            # 3. Transition from motion to stop (once)
            should_log = False
            log_reason = ""

            if not is_zero_velocity:
                # Always log non-zero velocities
                should_log = True
                log_reason = "FOLLOWING"
                self.zero_velocity_logged = False  # Reset zero velocity flag
            elif velocity_changed and not was_zero_velocity:
                # Log transition to stop (once)
                should_log = True
                log_reason = "STOP"
                self.zero_velocity_logged = True
            elif velocity_changed and not self.zero_velocity_logged:
                # Log first zero velocity command
                should_log = True
                log_reason = "INITIAL_STOP"
                self.zero_velocity_logged = True

            if should_log:
                self.get_logger().info(
                    f'🔄 Velocity [{log_reason}]: linear: {current_linear:.3f}, angular: {current_angular:.3f} '
                    f'(targets: {self.target_velocity["linear_x"]:.2f}, {self.target_velocity["angular_z"]:.2f})'
                )

            # Update last published velocity for next comparison
            self.last_published_velocity['linear_x'] = current_linear
            self.last_published_velocity['angular_z'] = current_angular

        except Exception as e:
            self.get_logger().error(f'Error in smart velocity logging: {e}')


def main(args=None):
    """Main function for person following controller."""
    rclpy.init(args=args)
    
    try:
        controller = PersonFollowingController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
