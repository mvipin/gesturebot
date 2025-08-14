#!/usr/bin/env python3
"""
Gesture Navigation Bridge for GestureBot
Converts gesture recognition results into navigation commands for Nav2.
"""

import time
from typing import Dict, Optional
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from gesturebot.msg import HandGesture


class NavigationState(Enum):
    """Navigation system states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    FOLLOWING = "following"


class GestureNavigationBridge(Node):
    """
    Bridge between gesture recognition and navigation system.
    Translates hand gestures into robot navigation commands.
    """
    
    # Direct gesture-to-motion mappings for stable gesture control
    GESTURE_MOTION_MAP = {
        'Thumb_Up': {'linear_x': 0.3, 'angular_z': 0.0},      # Move forward
        'Thumb_Down': {'linear_x': -0.2, 'angular_z': 0.0},   # Move backward
        'Open_Palm': {'linear_x': 0.0, 'angular_z': 0.0},     # Stop all movement
        'Pointing_Up': {'linear_x': 0.3, 'angular_z': 0.0},   # Move forward (alternative)
        'Victory': {'linear_x': 0.0, 'angular_z': 0.8},       # Turn left
        'ILoveYou': {'linear_x': 0.0, 'angular_z': -0.8},     # Turn right
        'Closed_Fist': {'linear_x': 0.0, 'angular_z': 0.0},   # Emergency stop
        'None': {'linear_x': 0.0, 'angular_z': 0.0}           # No gesture - stop
    }

    # Emergency gestures that immediately stop the robot
    EMERGENCY_GESTURES = ['Closed_Fist', 'Open_Palm']
    
    def __init__(self):
        super().__init__('gesture_navigation_bridge')
        
        # Motion control state with acceleration limiting
        self.nav_state = NavigationState.IDLE
        self.last_gesture_time = 0.0
        self.last_motion_time = 0.0
        self.gesture_timeout = 2.0  # seconds - faster stop for responsiveness

        # Velocity state tracking for smooth acceleration
        self.current_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.target_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.last_velocity_update = time.time()

        # Smart logging state to reduce noise
        self.last_published_velocity = {'linear_x': 0.0, 'angular_z': 0.0}
        self.zero_velocity_logged = False  # Track if we've already logged zero velocity
        
        # QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.gesture_subscription = self.create_subscription(
            HandGesture,
            '/vision/gestures',
            self.gesture_callback,
            self.reliable_qos
        )
        
        # Publishers
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
        
        # Action clients
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )
        
        # Parameters for motion control
        self.declare_parameter('gesture_confidence_threshold', 0.6)  # Lower for more responsive
        self.declare_parameter('emergency_stop_enabled', True)
        self.declare_parameter('max_linear_velocity', 0.3)   # Safe testing speed
        self.declare_parameter('max_angular_velocity', 0.8)  # Safe turning speed
        self.declare_parameter('gesture_timeout', 2.0)      # Balanced timeout for responsive stopping

        # Acceleration limiting parameters (balanced for responsiveness + stability)
        self.declare_parameter('max_linear_acceleration', 0.25)   # m/s² - balanced: responsive yet stable
        self.declare_parameter('max_angular_acceleration', 0.5)   # rad/s² - balanced turning
        self.declare_parameter('velocity_update_rate', 25.0)      # Hz - higher frequency for smoother control
        self.declare_parameter('emergency_deceleration', 1.2)     # m/s² - faster emergency stops
        self.declare_parameter('motion_smoothing_enabled', True)  # Enable/disable smoothing
        
        # Timers
        self.state_monitor_timer = self.create_timer(0.1, self.monitor_navigation_state)

        # High-frequency velocity smoothing timer for stability
        velocity_update_rate = self.get_parameter('velocity_update_rate').value
        self.velocity_smoothing_timer = self.create_timer(
            1.0 / velocity_update_rate,
            self.update_smoothed_velocity
        )
        
        self.get_logger().info('Gesture Navigation Bridge initialized')
        self.get_logger().info(f'Supported gestures: {list(self.GESTURE_MOTION_MAP.keys())}')
        max_linear = self.get_parameter('max_linear_velocity').value
        max_angular = self.get_parameter('max_angular_velocity').value
        self.get_logger().info(f'Motion control: max_linear={max_linear:.1f} m/s, max_angular={max_angular:.1f} rad/s')
    
    def gesture_callback(self, msg: HandGesture) -> None:
        """Handle incoming stable gesture recognition results and convert to motion."""
        try:
            # Check confidence threshold
            confidence_threshold = self.get_parameter('gesture_confidence_threshold').value
            if msg.confidence < confidence_threshold:
                return

            current_time = time.time()
            self.last_gesture_time = current_time

            # Convert gesture to motion command
            gesture_name = msg.gesture_name

            if gesture_name in self.GESTURE_MOTION_MAP:
                motion_cmd = self.GESTURE_MOTION_MAP[gesture_name]
                self.execute_motion_command(motion_cmd, gesture_name, msg.confidence)
            else:
                self.get_logger().warn(f'Unknown gesture for motion control: {gesture_name}')

        except Exception as e:
            self.get_logger().error(f'Error processing gesture: {e}')
    
    def execute_motion_command(self, motion_cmd: Dict, gesture_name: str, confidence: float) -> None:
        """Set target velocities from gesture mapping - smoothing handled by velocity controller."""
        try:
            # Get velocity limits
            max_linear = self.get_parameter('max_linear_velocity').value
            max_angular = self.get_parameter('max_angular_velocity').value

            # Set target velocities with safe limits (don't publish directly)
            target_linear = max(-max_linear, min(max_linear, motion_cmd['linear_x']))
            target_angular = max(-max_angular, min(max_angular, motion_cmd['angular_z']))

            # Handle emergency stop gestures immediately
            if gesture_name in self.EMERGENCY_GESTURES:
                self.target_velocity['linear_x'] = 0.0
                self.target_velocity['angular_z'] = 0.0
                self.nav_state = NavigationState.EMERGENCY_STOP
                self.get_logger().warn(f'🛑 EMERGENCY STOP triggered by {gesture_name}')

                # For emergency stops, bypass smoothing and stop immediately
                emergency_twist = Twist()
                self.current_velocity['linear_x'] = 0.0
                self.current_velocity['angular_z'] = 0.0
                self.cmd_vel_publisher.publish(emergency_twist)

                # Publish emergency stop signal
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_publisher.publish(emergency_msg)
            else:
                # Set target velocities for smooth acceleration
                self.target_velocity['linear_x'] = target_linear
                self.target_velocity['angular_z'] = target_angular
                self.nav_state = NavigationState.NAVIGATING

            self.last_motion_time = time.time()

            # Log target velocity command (only if targets changed significantly)
            if (abs(target_linear - self.target_velocity.get('linear_x', 0)) > 0.01 or
                abs(target_angular - self.target_velocity.get('angular_z', 0)) > 0.01):
                self.get_logger().info(
                    f'🎯 Target: {gesture_name} → linear: {target_linear:.2f}, angular: {target_angular:.2f} '
                    f'(confidence: {confidence:.2f})'
                )

        except Exception as e:
            self.get_logger().error(f'Error setting motion target: {e}')
            # Safety: set stop targets on error
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0

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

    def start_navigation(self) -> None:
        """Start or resume navigation."""
        if self.nav_state == NavigationState.PAUSED:
            self.nav_state = NavigationState.NAVIGATING
            self.get_logger().info('Navigation resumed')
        elif self.nav_state == NavigationState.IDLE:
            # Could start a predefined route or wait for specific target
            self.nav_state = NavigationState.NAVIGATING
            self.get_logger().info('Navigation started')
    
    def stop_navigation(self) -> None:
        """Stop current navigation."""
        if self.nav_state in [NavigationState.NAVIGATING, NavigationState.FOLLOWING]:
            self.cancel_current_goal()
            self.nav_state = NavigationState.IDLE
            self.get_logger().info('Navigation stopped')
    
    def pause_navigation(self) -> None:
        """Pause current navigation."""
        if self.nav_state == NavigationState.NAVIGATING:
            self.nav_state = NavigationState.PAUSED
            self.publish_stop_command()
            self.get_logger().info('Navigation paused')
    
    def execute_emergency_stop(self) -> None:
        """Execute emergency stop."""
        self.nav_state = NavigationState.EMERGENCY_STOP
        self.cancel_current_goal()
        self.publish_stop_command()
        
        # Publish emergency stop signal
        if self.get_parameter('emergency_stop_enabled').value:
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_stop_publisher.publish(emergency_msg)
        
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')
    
    def start_person_following(self) -> None:
        """Start person following mode."""
        self.nav_state = NavigationState.FOLLOWING
        self.get_logger().info('Person following mode activated')
        # Implementation would depend on person detection integration
    
    def execute_movement_command(self, direction: str) -> None:
        """Execute direct movement commands."""
        if self.nav_state == NavigationState.EMERGENCY_STOP:
            self.get_logger().warn('Cannot move: Emergency stop active')
            return
        
        twist = Twist()
        max_linear = self.get_parameter('max_linear_velocity').value
        max_angular = self.get_parameter('max_angular_velocity').value
        
        if direction == 'forward':
            twist.linear.x = max_linear * 0.5  # 50% of max speed
        elif direction == 'left':
            twist.angular.z = max_angular * 0.5
        elif direction == 'right':
            twist.angular.z = -max_angular * 0.5
        
        # Publish movement command for short duration
        self.cmd_vel_publisher.publish(twist)
        
        # Schedule stop after brief movement
        self.create_timer(0.5, lambda: self.publish_stop_command(), clock=self.get_clock())
        
        self.get_logger().info(f'Executing movement: {direction}')
    
    def return_to_home(self) -> None:
        """Return robot to home position."""
        # This would typically involve sending a goal to a predefined home position
        self.get_logger().info('Returning to home position')
        # Implementation depends on your specific home position setup
    
    def cancel_current_goal(self) -> None:
        """Cancel current navigation goal."""
        if self.current_goal and self.nav_to_pose_client.server_is_ready():
            self.nav_to_pose_client.cancel_goal_async(self.current_goal)
            self.current_goal = None
    
    def publish_stop_command(self) -> None:
        """Publish stop command to cmd_vel."""
        stop_twist = Twist()  # All zeros
        self.cmd_vel_publisher.publish(stop_twist)
    
    def monitor_navigation_state(self) -> None:
        """Monitor navigation state and handle gesture timeouts."""
        current_time = time.time()
        gesture_timeout = self.get_parameter('gesture_timeout').value

        # Handle gesture timeout - smoothly stop robot if no recent gestures
        if (self.nav_state == NavigationState.NAVIGATING and
            current_time - self.last_gesture_time > gesture_timeout):

            # Set target velocities to zero for smooth deceleration
            self.target_velocity['linear_x'] = 0.0
            self.target_velocity['angular_z'] = 0.0
            self.nav_state = NavigationState.IDLE

            self.get_logger().info(
                f'⏰ Gesture timeout ({gesture_timeout:.1f}s) - smoothly stopping robot for safety'
            )

        # Reset emergency stop after extended timeout
        if (self.nav_state == NavigationState.EMERGENCY_STOP and
            current_time - self.last_gesture_time > 10.0):  # 10 second timeout for emergency
            self.nav_state = NavigationState.IDLE
            self.get_logger().info('🔓 Emergency stop cleared - returning to idle')
    
    def get_navigation_state(self) -> NavigationState:
        """Get current navigation state."""
        return self.nav_state
    
    def is_emergency_stopped(self) -> bool:
        """Check if robot is in emergency stop state."""
        return self.nav_state == NavigationState.EMERGENCY_STOP


def main(args=None):
    """Main function for gesture navigation bridge."""
    rclpy.init(args=args)
    
    try:
        node = GestureNavigationBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in gesture navigation bridge: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
