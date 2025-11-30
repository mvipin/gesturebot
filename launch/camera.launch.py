#!/usr/bin/env python3
"""
Camera Launch File for GestureBot
Simple camera node launcher for testing and development.

Usage:
    ros2 launch gesturebot camera.launch.py
    ros2 launch gesturebot camera.launch.py camera_id:=0 width:=640 height:=480
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================

    declare_camera_id = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera ID or device path'
    )

    declare_camera_format = DeclareLaunchArgument(
        'camera_format',
        default_value='BGR888',
        description='Camera pixel format'
    )

    declare_camera_width = DeclareLaunchArgument(
        'camera_width',
        default_value='640',
        description='Camera image width'
    )

    declare_camera_height = DeclareLaunchArgument(
        'camera_height',
        default_value='480',
        description='Camera image height'
    )

    declare_camera_fps = DeclareLaunchArgument(
        'camera_fps',
        default_value='15.0',
        description='Camera frame rate'
    )

    # ========================================
    # CAMERA NODE
    # ========================================

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            "camera": LaunchConfiguration('camera_id'),
            "width": LaunchConfiguration('camera_width'),
            "height": LaunchConfiguration('camera_height'),
            "format": LaunchConfiguration('camera_format'),
            # Performance parameters
            "buffer_queue_size": 2,
            # Camera controls for 15fps
            "FrameDurationLimits": [66667, 66667],  # 15 FPS
            "ExposureTime": 20000,  # 1/50s in microseconds
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            # Quality settings
            "jpeg_quality": 80,
            # Sensor mode
            "sensor_mode": "640:480",
            # Use sim time setting
            'use_sim_time': False,
        }],
        remappings=[
            ('~/image_raw', '/camera/image_raw'),
            ('~/image_raw/compressed', '/camera/image_raw/compressed'),
            ('~/camera_info', '/camera/camera_info'),
        ],
        output='screen'
    )

    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================

    return LaunchDescription([
        # Launch arguments
        declare_camera_id,
        declare_camera_format,
        declare_camera_width,
        declare_camera_height,
        declare_camera_fps,
        
        # Nodes
        camera_node,
    ])
