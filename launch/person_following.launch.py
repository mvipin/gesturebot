#!/usr/bin/env python3
"""
Person Following Launch File for GestureBot
Launches the standalone person following system.
Phase 2A: Basic person following with object detection integration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for person following system."""
    
    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================
    
    # Following behavior parameters
    declare_target_distance = DeclareLaunchArgument(
        'target_distance',
        default_value='1.5',
        description='Target distance to maintain when following person (meters)'
    )
    
    declare_distance_tolerance = DeclareLaunchArgument(
        'distance_tolerance',
        default_value='0.3',
        description='Distance tolerance for following behavior (meters)'
    )
    
    declare_min_safe_distance = DeclareLaunchArgument(
        'min_safe_distance',
        default_value='0.8',
        description='Minimum safe distance to person (meters)'
    )
    
    declare_max_follow_distance = DeclareLaunchArgument(
        'max_follow_distance',
        default_value='5.0',
        description='Maximum distance to follow person (meters) - stop following if exceeded'
    )
    
    declare_person_lost_timeout = DeclareLaunchArgument(
        'person_lost_timeout',
        default_value='3.0',
        description='Timeout before deactivating follow mode when person is lost (seconds)'
    )
    
    # Control parameters
    declare_max_linear_velocity = DeclareLaunchArgument(
        'max_linear_velocity',
        default_value='0.25',
        description='Maximum linear velocity for following (m/s)'
    )
    
    declare_max_angular_velocity = DeclareLaunchArgument(
        'max_angular_velocity',
        default_value='0.6',
        description='Maximum angular velocity for following (rad/s)'
    )
    
    # Detection parameters
    declare_person_confidence_threshold = DeclareLaunchArgument(
        'person_confidence_threshold',
        default_value='0.6',
        description='Minimum confidence threshold for person detections'
    )
    
    declare_min_person_size = DeclareLaunchArgument(
        'min_person_size',
        default_value='0.02',
        description='Minimum person bounding box area (normalized)'
    )

    # Velocity smoothing parameters
    declare_max_linear_acceleration = DeclareLaunchArgument(
        'max_linear_acceleration',
        default_value='0.25',
        description='Maximum linear acceleration (m/s²) - balanced for smooth following'
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
    
    # ========================================
    # PERSON FOLLOWING CONTROLLER NODE
    # ========================================
    
    person_following_controller_node = Node(
        package='gesturebot',
        executable='person_following_controller.py',
        name='person_following_controller',
        parameters=[{
            # Following behavior parameters
            'target_distance': LaunchConfiguration('target_distance'),
            'distance_tolerance': LaunchConfiguration('distance_tolerance'),
            'min_safe_distance': LaunchConfiguration('min_safe_distance'),
            'max_follow_distance': LaunchConfiguration('max_follow_distance'),
            'person_lost_timeout': LaunchConfiguration('person_lost_timeout'),
            
            # Control parameters
            'max_linear_velocity': LaunchConfiguration('max_linear_velocity'),
            'max_angular_velocity': LaunchConfiguration('max_angular_velocity'),
            
            # Detection parameters
            'person_confidence_threshold': LaunchConfiguration('person_confidence_threshold'),
            'min_person_size': LaunchConfiguration('min_person_size'),

            # Velocity smoothing parameters
            'max_linear_acceleration': LaunchConfiguration('max_linear_acceleration'),
            'max_angular_acceleration': LaunchConfiguration('max_angular_acceleration'),
            'velocity_update_rate': LaunchConfiguration('velocity_update_rate'),
            'emergency_deceleration': LaunchConfiguration('emergency_deceleration'),
            'motion_smoothing_enabled': LaunchConfiguration('motion_smoothing_enabled'),
        }],
        output='screen'
    )
    
    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================
    
    return LaunchDescription([
        # Launch arguments
        declare_target_distance,
        declare_distance_tolerance,
        declare_min_safe_distance,
        declare_max_follow_distance,
        declare_person_lost_timeout,
        declare_max_linear_velocity,
        declare_max_angular_velocity,
        declare_person_confidence_threshold,
        declare_min_person_size,
        declare_max_linear_acceleration,
        declare_max_angular_acceleration,
        declare_velocity_update_rate,
        declare_emergency_deceleration,
        declare_motion_smoothing_enabled,
        
        # Nodes
        person_following_controller_node
    ])
