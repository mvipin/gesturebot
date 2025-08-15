#!/usr/bin/env python3
"""
Pose Navigation Bridge Launch File for GestureBot
Launches the simplified 4-pose navigation control system.
Converts pose classifications to robot motion commands.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for pose navigation bridge."""
    
    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================
    
    # Pose processing parameters
    declare_pose_confidence_threshold = DeclareLaunchArgument(
        'pose_confidence_threshold',
        default_value='0.7',
        description='Minimum confidence threshold for pose actions'
    )

    declare_pose_repeat_delay = DeclareLaunchArgument(
        'pose_repeat_delay',
        default_value='0.1',
        description='Minimum time between pose commands (seconds)'
    )
    
    # Motion control parameters (same as gesture system for consistency)
    declare_max_linear_velocity = DeclareLaunchArgument(
        'max_linear_velocity',
        default_value='0.3',
        description='Maximum linear velocity (m/s)'
    )
    
    declare_max_angular_velocity = DeclareLaunchArgument(
        'max_angular_velocity',
        default_value='0.8',
        description='Maximum angular velocity (rad/s)'
    )
    
    # Acceleration limiting parameters (same as gesture system)
    declare_max_linear_acceleration = DeclareLaunchArgument(
        'max_linear_acceleration',
        default_value='0.25',
        description='Maximum linear acceleration (m/s²) - balanced for tall robot stability'
    )
    
    declare_max_angular_acceleration = DeclareLaunchArgument(
        'max_angular_acceleration',
        default_value='0.5',
        description='Maximum angular acceleration (rad/s²) - balanced turning'
    )
    
    declare_velocity_update_rate = DeclareLaunchArgument(
        'velocity_update_rate',
        default_value='25.0',
        description='Velocity smoothing update rate (Hz) - higher for smoother control'
    )
    
    declare_emergency_deceleration = DeclareLaunchArgument(
        'emergency_deceleration',
        default_value='1.2',
        description='Emergency deceleration rate (m/s²) - faster stops'
    )
    
    declare_motion_smoothing_enabled = DeclareLaunchArgument(
        'motion_smoothing_enabled',
        default_value='true',
        description='Enable velocity smoothing and acceleration limiting'
    )
    
    # Timeout parameters
    declare_pose_timeout = DeclareLaunchArgument(
        'pose_timeout',
        default_value='2.0',
        description='Timeout for pose commands (seconds) - stop robot if no poses detected'
    )
    
    # ========================================
    # POSE NAVIGATION BRIDGE NODE
    # ========================================
    
    pose_navigation_bridge_node = Node(
        package='gesturebot',
        executable='pose_navigation_bridge.py',
        name='pose_navigation_bridge',
        parameters=[{
            # Pose processing parameters
            'pose_confidence_threshold': LaunchConfiguration('pose_confidence_threshold'),
            'pose_repeat_delay': LaunchConfiguration('pose_repeat_delay'),

            # Motion control parameters
            'max_linear_velocity': LaunchConfiguration('max_linear_velocity'),
            'max_angular_velocity': LaunchConfiguration('max_angular_velocity'),

            # Acceleration limiting parameters
            'max_linear_acceleration': LaunchConfiguration('max_linear_acceleration'),
            'max_angular_acceleration': LaunchConfiguration('max_angular_acceleration'),
            'velocity_update_rate': LaunchConfiguration('velocity_update_rate'),
            'emergency_deceleration': LaunchConfiguration('emergency_deceleration'),
            'motion_smoothing_enabled': LaunchConfiguration('motion_smoothing_enabled'),

            # Timeout parameters
            'pose_timeout': LaunchConfiguration('pose_timeout'),
        }],
        output='screen'
    )
    
    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================
    
    return LaunchDescription([
        # Launch arguments
        declare_pose_confidence_threshold,
        declare_pose_repeat_delay,
        declare_max_linear_velocity,
        declare_max_angular_velocity,
        declare_max_linear_acceleration,
        declare_max_angular_acceleration,
        declare_velocity_update_rate,
        declare_emergency_deceleration,
        declare_motion_smoothing_enabled,
        declare_pose_timeout,

        # Nodes
        pose_navigation_bridge_node
    ])
