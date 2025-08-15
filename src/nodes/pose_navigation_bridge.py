#!/usr/bin/env python3
"""
Pose Navigation Bridge for GestureBot
Converts pose detection results into navigation commands for robot control.
Simplified 4-pose direct control system (no person following).
"""

import time
from typing import Dict, Optional
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from gesturebot.msg import PoseLandmarks


class NavigationState(Enum):
    """Navigation system states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


class PoseNavigationBridge(Node):
    """
    Bridge between pose detection and navigation system.
    Translates pose actions into robot navigation commands.
    Mirrors gesture navigation bridge architecture for consistency.
    """

    # Direct pose-to-motion mappings for stable pose control (simplified 4-pose system)
    POSE_MOTION_MAP = {
        'arms_raised': {'linear_x': 0.3, 'angular_z': 0.0},    # Move forward
        'pointing_left': {'linear_x': 0.0, 'angular_z': 0.8},  # Turn left
        'pointing_right': {'linear_x': 0.0, 'angular_z': -0.8}, # Turn right
        't_pose': {'linear_x': 0.0, 'angular_z': 0.0},         # Stop
        'no_pose': {'linear_x': 0.0, 'angular_z': 0.0}         # No pose - stop
    }

    # Emergency poses that immediately stop the robot
    EMERGENCY_POSES = ['t_pose', 'no_pose']

    def __init__(self):
        super().__init__('pose_navigation_bridge')

        # Motion control state with acceleration limiting (same as gesture bridge)
        self.nav_state = NavigationState.IDLE
        self.last_pose_time = 0.0
        self.last_motion_time = 0.0
        self.pose_timeout = 2.0  # seconds - faster stop for responsiveness

        # Velocity state tracking for smooth acceleration (same as gesture bridge)
        self.current_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.target_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.last_velocity_update = time.time()

        # Smart logging state to reduce noise (same as gesture bridge)
        self.last_published_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.zero_velocity_logged = False  # Track if we've already logged zero velocity

        # QoS profiles (same as gesture bridge)
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Parameters (identical to gesture system for consistency)
        self.declare_parameter('pose_confidence_threshold', 0.7)
        self.declare_parameter('max_linear_velocity', 0.3)
        self.declare_parameter('max_angular_velocity', 0.8)
        self.declare_parameter('max_linear_acceleration', 0.25)   # Same as gesture system
        self.declare_parameter('max_angular_acceleration', 0.5)   # Same as gesture system
        self.declare_parameter('velocity_update_rate', 25.0)      # Same as gesture system
        self.declare_parameter('emergency_deceleration', 1.2)     # Same as gesture system
        self.declare_parameter('motion_smoothing_enabled', True)  # Same as gesture system
        self.declare_parameter('pose_timeout', 2.0)               # Stop if no poses for 2s
        self.declare_parameter('pose_repeat_delay', 0.1)          # Minimum time between pose commands

        # Subscribers
        self.pose_subscription = self.create_subscription(
            PoseLandmarks,
            '/vision/poses',
            self.pose_callback,
            self.reliable_qos
        )

        # Publishers (same as gesture system)
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            self.reliable_qos
        )

        self.emergency_stop_publisher = self.create_publisher(
            Bool,
            '/emergency_stop',
            self.reliable_qos
        )

        # Timers (same pattern as gesture system)
        self.state_monitor_timer = self.create_timer(0.1, self.monitor_navigation_state)

        # High-frequency velocity smoothing timer (same as gesture system)
        velocity_update_rate = self.get_parameter('velocity_update_rate').value
        self.velocity_timer = self.create_timer(1.0 / velocity_update_rate, self.update_smoothed_velocity)

        self.get_logger().info('🎯 Pose Navigation Bridge initialized')
        self.get_logger().info(f'📋 4-Pose control system: {list(self.POSE_MOTION_MAP.keys())}')

    def pose_callback(self, msg: PoseLandmarks) -> None:
        """Handle incoming stable pose detection results and convert to motion."""
        try:
            # Check for pose repeat delay (same as gesture system)
            current_time = time.time()
            repeat_delay = self.get_parameter('pose_repeat_delay').value
            if current_time - self.last_pose_time < repeat_delay:
                return

            self.last_pose_time = current_time

            # Convert pose to motion command
            pose_action = msg.pose_action

            if pose_action in self.POSE_MOTION_MAP:
                motion_cmd = self.POSE_MOTION_MAP[pose_action]
                self.execute_motion_command(motion_cmd, pose_action)
            else:
                self.get_logger().warn(f'Unknown pose for motion control: {pose_action}')

        except Exception as e:
            self.get_logger().error(f'Error processing pose: {e}')

    def execute_motion_command(self, motion_cmd: Dict, pose_action: str) -> None:
        """Set target velocities from pose mapping - smoothing handled by velocity controller."""
        try:
            # Get velocity limits
            max_linear = self.get_parameter('max_linear_velocity').value
            max_angular = self.get_parameter('max_angular_velocity').value

            # Set target velocities with safe limits (don't publish directly)
            target_linear = max(-max_linear, min(max_linear, motion_cmd['linear_x']))
            target_angular = max(-max_angular, min(max_angular, motion_cmd['angular_z']))

            # Update target velocity for smoothing system
            self.target_velocity = {
                'linear_x': target_linear,
                'angular_z': target_angular
            }

            # Handle emergency poses
            if pose_action in self.EMERGENCY_POSES:
                self.nav_state = NavigationState.EMERGENCY_STOP
                # Publish emergency stop signal
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_publisher.publish(emergency_msg)
                self.get_logger().info(f'🛑 Emergency stop triggered by pose: {pose_action}')
            else:
                self.nav_state = NavigationState.NAVIGATING
                self.get_logger().debug(
                    f'🎯 Pose motion: {pose_action} -> linear={target_linear:.2f}, angular={target_angular:.2f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error executing motion command: {e}')

    def monitor_navigation_state(self) -> None:
        """Monitor navigation state and handle pose timeouts."""
        current_time = time.time()
        pose_timeout = self.get_parameter('pose_timeout').value

        # Handle pose timeout - smoothly stop robot if no recent poses
        if (self.nav_state == NavigationState.NAVIGATING and
            current_time - self.last_pose_time > pose_timeout):

            # Set target velocities to zero for smooth deceleration
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            self.nav_state = NavigationState.IDLE

            self.get_logger().info(
                f'⏰ Pose timeout ({pose_timeout:.1f}s) - smoothly stopping robot for safety'
            )

    def update_smoothed_velocity(self) -> None:
        """
        High-frequency velocity smoothing with acceleration limiting.
        Prevents abrupt velocity changes that cause wobbling in tall robots.
        """
        try:
            current_time = time.time()
            dt = current_time - self.last_velocity_update
            self.last_velocity_update = current_time

            # Skip if dt is too large (system lag) or too small
            if dt > 0.1 or dt < 0.001:
                return

            # Check if motion smoothing is enabled
            if not self.get_parameter('motion_smoothing_enabled').value:
                # Direct velocity control (bypass smoothing)
                twist = Twist()
                twist.linear.x = self.target_velocity['linear_x']
                twist.angular.z = self.target_velocity['angular_z']
                self.cmd_vel_publisher.publish(twist)
                return

            # Get acceleration limits
            max_linear_accel = self.get_parameter('max_linear_acceleration').value
            max_angular_accel = self.get_parameter('max_angular_acceleration').value

            # Use faster deceleration for emergency stops
            if self.nav_state == NavigationState.EMERGENCY_STOP:
                emergency_decel = self.get_parameter('emergency_deceleration').value
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
                log_reason = "MOTION"
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
    """Main function for pose navigation bridge."""
    rclpy.init(args=args)

    try:
        bridge = PoseNavigationBridge()
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

    def apply_acceleration_limit(self, current_vel: float, target_vel: float, max_accel: float, dt: float) -> float:
        """
        Apply acceleration limiting to prevent abrupt velocity changes.
        Same implementation as gesture navigation bridge.
        """
        velocity_diff = target_vel - current_vel
        max_change = max_accel * dt

        if abs(velocity_diff) <= max_change:
            return target_vel  # Can reach target this step
        else:
            # Limit the change to maximum allowed acceleration
            return current_vel + (max_change if velocity_diff > 0 else -max_change)

    def log_velocity_change(self, twist: Twist) -> None:
        """Log velocity changes for debugging (smart logging to reduce noise)."""
        # Only log if velocity is non-zero or has changed significantly
        linear_vel = twist.linear.x
        angular_vel = twist.angular.z

        if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
            self.get_logger().debug(
                f'Pose motion: linear={linear_vel:.2f} m/s, angular={angular_vel:.2f} rad/s, '
                f'state={self.nav_state.value}, following={self.following_active}'
            )


def main(args=None):
    """Main function for pose navigation bridge."""
    rclpy.init(args=args)

    try:
        node = PoseNavigationBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
