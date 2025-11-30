#!/usr/bin/env python3
"""
Gesture Recognition Launch File for GestureBot
Launches camera + ros2_mediapipe gesture recognition with configurable parameters.

Usage:
    ros2 launch gesturebot gesture_recognition.launch.py

For gesture-controlled robot motion:
    # Terminal 1: Gesture detection
    ros2 launch gesturebot gesture_recognition.launch.py

    # Terminal 2: Motion control
    ros2 launch gesturebot gesture_navigation_bridge.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for gesture recognition."""

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
        description='Camera pixel format (BGR888 for optimal gesture recognition performance)'
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
        description='Camera frame rate (optimized for gesture recognition)'
    )

    # Gesture recognition configuration
    declare_enable_gesture_recognition = DeclareLaunchArgument(
        'enable_gesture_recognition',
        default_value='true',
        description='Enable gesture recognition node'
    )

    declare_confidence_threshold = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Gesture recognition confidence threshold'
    )

    declare_max_hands = DeclareLaunchArgument(
        'max_hands',
        default_value='2',
        description='Maximum number of hands to detect'
    )

    declare_frame_skip = DeclareLaunchArgument(
        'frame_skip',
        default_value='1',
        description='Process every Nth frame (1 = process all frames)'
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
        default_value='models/gesture_recognizer.task',
        description='Path to the gesture recognition model file'
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
            "FrameDurationLimits": [66667, 66667],  # 15 FPS
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
    # GESTURE RECOGNITION NODE (ros2_mediapipe)
    # ========================================

    gesture_recognition_node = Node(
        package='ros2_mediapipe',
        executable='gesture_recognition_node.py',
        name='gesture_recognition_node',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_hands': LaunchConfiguration('max_hands'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'model_path': LaunchConfiguration('model_path'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_gesture_recognition')),
        output='screen'
    )

    # ========================================
    # NODE GROUPING
    # ========================================

    gesture_recognition_group = GroupAction([
        camera_node,
        gesture_recognition_node,
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
        declare_camera_fps,
        declare_enable_gesture_recognition,
        declare_confidence_threshold,
        declare_max_hands,
        declare_frame_skip,
        declare_debug_mode,
        declare_model_path,

        # Gesture recognition system nodes
        gesture_recognition_group
    ])
