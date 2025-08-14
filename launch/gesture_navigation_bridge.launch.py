#!/usr/bin/env python3
"""
Gesture Navigation Bridge Launch File
Launches the gesture-to-motion control bridge for robot navigation.

This modular launch file enables gesture-controlled robot motion with:
- Acceleration limiting for tall robot stability
- Velocity smoothing for mechanical safety
- Emergency stop functionality
- Configurable motion parameters

Designed for Phase 4 multi-modal navigation integration where multiple
detection modes (gesture, object, pose) can feed into unified navigation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for gesture navigation bridge."""
    
    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================
    
    # Core navigation parameters
    declare_enable_navigation_bridge = DeclareLaunchArgument(
        'enable_navigation_bridge',
        default_value='true',
        description='Enable gesture navigation bridge for motion control'
    )
    
    declare_gesture_confidence_threshold = DeclareLaunchArgument(
        'gesture_confidence_threshold',
        default_value='0.6',
        description='Minimum confidence threshold for gesture commands'
    )
    
    declare_emergency_stop_enabled = DeclareLaunchArgument(
        'emergency_stop_enabled',
        default_value='true',
        description='Enable emergency stop functionality'
    )
    
    # Velocity limits for safe operation
    declare_max_linear_velocity = DeclareLaunchArgument(
        'max_linear_velocity',
        default_value='0.3',
        description='Maximum linear velocity (m/s) for safe robot operation'
    )
    
    declare_max_angular_velocity = DeclareLaunchArgument(
        'max_angular_velocity',
        default_value='0.8',
        description='Maximum angular velocity (rad/s) for safe robot turning'
    )
    
    declare_gesture_timeout = DeclareLaunchArgument(
        'gesture_timeout',
        default_value='2.0',
        description='Timeout (seconds) after which robot stops if no gestures received'
    )
    
    # Acceleration limiting parameters for tall robot stability
    declare_max_linear_acceleration = DeclareLaunchArgument(
        'max_linear_acceleration',
        default_value='0.25',
        description='Maximum linear acceleration (m/s²) - balanced for responsiveness and stability'
    )
    
    declare_max_angular_acceleration = DeclareLaunchArgument(
        'max_angular_acceleration',
        default_value='0.5',
        description='Maximum angular acceleration (rad/s²) - balanced for responsive turning'
    )
    
    declare_velocity_update_rate = DeclareLaunchArgument(
        'velocity_update_rate',
        default_value='25.0',
        description='Velocity smoothing update rate (Hz) for smooth motion control'
    )
    
    declare_emergency_deceleration = DeclareLaunchArgument(
        'emergency_deceleration',
        default_value='1.2',
        description='Emergency deceleration rate (m/s²) for faster emergency stops'
    )
    
    declare_motion_smoothing_enabled = DeclareLaunchArgument(
        'motion_smoothing_enabled',
        default_value='true',
        description='Enable velocity smoothing and acceleration limiting'
    )
    
    # Debug and logging parameters
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode with verbose logging'
    )
    
    # ========================================
    # GESTURE NAVIGATION BRIDGE NODE
    # ========================================
    
    gesture_navigation_bridge_node = Node(
        package='gesturebot',
        executable='gesture_navigation_bridge.py',
        name='gesture_navigation_bridge',
        parameters=[{
            # Core navigation parameters
            'gesture_confidence_threshold': LaunchConfiguration('gesture_confidence_threshold'),
            'emergency_stop_enabled': LaunchConfiguration('emergency_stop_enabled'),
            'max_linear_velocity': LaunchConfiguration('max_linear_velocity'),
            'max_angular_velocity': LaunchConfiguration('max_angular_velocity'),
            'gesture_timeout': LaunchConfiguration('gesture_timeout'),
            
            # Acceleration limiting for tall robot stability
            'max_linear_acceleration': LaunchConfiguration('max_linear_acceleration'),
            'max_angular_acceleration': LaunchConfiguration('max_angular_acceleration'),
            'velocity_update_rate': LaunchConfiguration('velocity_update_rate'),
            'emergency_deceleration': LaunchConfiguration('emergency_deceleration'),
            'motion_smoothing_enabled': LaunchConfiguration('motion_smoothing_enabled'),
            
            # Debug configuration
            'debug_mode': LaunchConfiguration('debug_mode'),
        }],
        remappings=[
            # Subscribe to gesture recognition results
            ('gestures', '/vision/gestures'),
            # Publish motion commands
            ('cmd_vel', '/cmd_vel'),
            ('emergency_stop', '/emergency_stop'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_navigation_bridge')),
        output='screen'
    )
    
    # ========================================
    # NODE GROUPING
    # ========================================
    
    # Group navigation bridge for organized launch
    navigation_bridge_group = GroupAction([
        gesture_navigation_bridge_node,
    ])
    
    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================
    
    return LaunchDescription([
        # Launch arguments
        declare_enable_navigation_bridge,
        declare_gesture_confidence_threshold,
        declare_emergency_stop_enabled,
        declare_max_linear_velocity,
        declare_max_angular_velocity,
        declare_gesture_timeout,
        declare_max_linear_acceleration,
        declare_max_angular_acceleration,
        declare_velocity_update_rate,
        declare_emergency_deceleration,
        declare_motion_smoothing_enabled,
        declare_debug_mode,
        
        # Navigation bridge system
        navigation_bridge_group
    ])
