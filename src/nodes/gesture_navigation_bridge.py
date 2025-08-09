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
    
    # Gesture command mappings
    NAVIGATION_GESTURES = {
        'thumbs_up': 'start_navigation',
        'thumbs_down': 'stop_navigation',
        'open_palm': 'pause_navigation',
        'fist': 'emergency_stop',
        'peace': 'follow_person',
        'pointing_up': 'move_forward',
        'pointing_left': 'turn_left',
        'pointing_right': 'turn_right',
        'wave': 'return_home'
    }
    
    # Emergency gestures that immediately stop the robot
    EMERGENCY_GESTURES = ['fist', 'stop_sign', 'crossed_arms']
    
    def __init__(self):
        super().__init__('gesture_navigation_bridge')
        
        # Navigation state
        self.nav_state = NavigationState.IDLE
        self.last_gesture_time = 0.0
        self.gesture_timeout = 2.0  # seconds
        self.current_goal = None
        
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
        
        # Parameters
        self.declare_parameter('gesture_confidence_threshold', 0.8)
        self.declare_parameter('emergency_stop_enabled', True)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('gesture_repeat_delay', 1.0)
        
        # Timers
        self.state_monitor_timer = self.create_timer(0.1, self.monitor_navigation_state)
        
        self.get_logger().info('Gesture Navigation Bridge initialized')
        self.get_logger().info(f'Supported gestures: {list(self.NAVIGATION_GESTURES.keys())}')
    
    def gesture_callback(self, msg: HandGesture) -> None:
        """Handle incoming gesture recognition results."""
        try:
            # Check confidence threshold
            confidence_threshold = self.get_parameter('gesture_confidence_threshold').value
            if msg.confidence < confidence_threshold:
                return
            
            # Check for gesture repeat delay
            current_time = time.time()
            repeat_delay = self.get_parameter('gesture_repeat_delay').value
            if current_time - self.last_gesture_time < repeat_delay:
                return
            
            # Process navigation gesture
            if msg.is_nav_gesture and msg.nav_command:
                self.process_navigation_command(msg)
                self.last_gesture_time = current_time
                
                self.get_logger().info(
                    f'Processing gesture: {msg.gesture_name} -> {msg.nav_command} '
                    f'(confidence: {msg.confidence:.2f})'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error processing gesture: {e}')
    
    def process_navigation_command(self, gesture_msg: HandGesture) -> None:
        """Process a navigation command from gesture recognition."""
        command = gesture_msg.nav_command
        gesture_name = gesture_msg.gesture_name
        
        # Handle emergency stop gestures immediately
        if gesture_name in self.EMERGENCY_GESTURES:
            self.execute_emergency_stop()
            return
        
        # Process other navigation commands
        if command == 'start_navigation':
            self.start_navigation()
        elif command == 'stop_navigation':
            self.stop_navigation()
        elif command == 'pause_navigation':
            self.pause_navigation()
        elif command == 'emergency_stop':
            self.execute_emergency_stop()
        elif command == 'follow_person':
            self.start_person_following()
        elif command == 'move_forward':
            self.execute_movement_command('forward')
        elif command == 'turn_left':
            self.execute_movement_command('left')
        elif command == 'turn_right':
            self.execute_movement_command('right')
        elif command == 'return_home':
            self.return_to_home()
        else:
            self.get_logger().warn(f'Unknown navigation command: {command}')
    
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
        """Monitor navigation state and handle timeouts."""
        current_time = time.time()
        
        # Reset emergency stop after timeout (if no recent emergency gestures)
        if (self.nav_state == NavigationState.EMERGENCY_STOP and 
            current_time - self.last_gesture_time > 5.0):  # 5 second timeout
            self.nav_state = NavigationState.IDLE
            self.get_logger().info('Emergency stop cleared - returning to idle')
        
        # Handle gesture timeout for movement commands
        if (self.nav_state in [NavigationState.NAVIGATING, NavigationState.FOLLOWING] and
            current_time - self.last_gesture_time > self.gesture_timeout):
            # Could implement timeout behavior here
            pass
    
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
