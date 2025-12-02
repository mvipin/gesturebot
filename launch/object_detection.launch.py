#!/usr/bin/env python3
"""
Object Detection Launch File for GestureBot Vision System
Launches camera + ros2_mediapipe object detection with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for object detection system."""

    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================

    # Camera configuration
    declare_enable_camera = DeclareLaunchArgument(
        'enable_camera',
        default_value='true',
        description='Enable camera node for image capture'
    )

    declare_camera_id = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera ID or device path'
    )

    declare_camera_format = DeclareLaunchArgument(
        'camera_format',
        default_value='BGR888',
        description='Camera pixel format (BGR888 for optimal object detection performance)'
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

    # Object detection configuration
    declare_enable_object_detection = DeclareLaunchArgument(
        'enable_object_detection',
        default_value='true',
        description='Enable object detection node'
    )

    declare_confidence_threshold = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Object detection confidence threshold'
    )

    declare_max_results = DeclareLaunchArgument(
        'max_results',
        default_value='5',
        description='Maximum number of detection results'
    )

    declare_frame_skip = DeclareLaunchArgument(
        'frame_skip',
        default_value='0',
        description='Process every Nth frame (0 = process all frames)'
    )

    # Debug configuration
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug output and logging'
    )

    # Model configuration
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/efficientdet.tflite',
        description='Path to the EfficientDet model file'
    )

    # ========================================
    # CAMERA NODE (High-Performance Configuration)
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
            "buffer_queue_size": 2,
            "FrameDurationLimits": [100000, 100000],  # 10 FPS - optimal for Pi 5 (see README)
            "ExposureTime": 20000,
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            "jpeg_quality": 80,
            "sensor_mode": "640:480",
            'use_sim_time': False,
        }],
        remappings=[
            ('~/image_raw', '/camera/image_raw'),
            ('~/image_raw/compressed', '/camera/image_raw/compressed'),
            ('~/camera_info', '/camera/camera_info'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_camera')),
        output='screen'
    )

    # ========================================
    # OBJECT DETECTION NODE (ros2_mediapipe)
    # ========================================

    object_detection_node = Node(
        package='ros2_mediapipe',
        executable='object_detection_node.py',
        name='object_detection_node',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_results': LaunchConfiguration('max_results'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'model_path': LaunchConfiguration('model_path'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_object_detection')),
        output='screen'
    )

    # ========================================
    # NODE GROUPING
    # ========================================

    object_detection_group = GroupAction([
        camera_node,
        object_detection_node,
    ])

    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================

    return LaunchDescription([
        # Launch arguments
        declare_enable_camera,
        declare_camera_id,
        declare_camera_format,
        declare_camera_width,
        declare_camera_height,
        declare_enable_object_detection,
        declare_confidence_threshold,
        declare_max_results,
        declare_frame_skip,
        declare_debug_mode,
        declare_model_path,

        # Object detection system nodes
        object_detection_group
    ])
