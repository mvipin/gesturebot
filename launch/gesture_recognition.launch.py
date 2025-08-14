#!/usr/bin/env python3
"""
Gesture Recognition Launch File for GestureBot
Launches the gesture detection system (camera + gesture recognition).

This modular launch file provides:
- Camera node for image capture
- Gesture recognition node for MediaPipe-based hand gesture detection
- Publishes gesture results to /vision/gestures topic

For robot motion control, launch the separate navigation bridge:
    ros2 launch gesturebot gesture_navigation_bridge.launch.py

Usage:
    ros2 launch gesturebot gesture_recognition.launch.py

For complete gesture-controlled robot motion:
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

    # Camera configuration (matching object detection patterns)
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

    declare_gesture_stability_threshold = DeclareLaunchArgument(
        'gesture_stability_threshold',
        default_value='0.1',
        description='Minimum duration (seconds) for gesture stability (maximum responsiveness)'
    )

    declare_publish_annotated_images = DeclareLaunchArgument(
        'publish_annotated_images',
        default_value='true',
        description='Enable publishing of annotated images with gesture overlays'
    )

    declare_show_landmark_indices = DeclareLaunchArgument(
        'show_landmark_indices',
        default_value='false',
        description='Show landmark indices on each landmark point for debugging'
    )

    # Debug configuration (matching object detection patterns)
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug output and logging'
    )

    # Buffered logging configuration (matching object detection patterns)
    declare_unlimited_buffer_mode = DeclareLaunchArgument(
        'unlimited_buffer_mode',
        default_value='false',
        description='Enable unlimited buffer mode (timer-only flushing for comprehensive diagnostics). When false, uses circular buffer mode (auto-drop when full).'
    )

    declare_buffer_logging_enabled = DeclareLaunchArgument(
        'buffer_logging_enabled',
        default_value='false',
        description='Enable buffered logging system. When false, disables all buffering and only logs critical errors directly.'
    )

    declare_enable_performance_tracking = DeclareLaunchArgument(
        'enable_performance_tracking',
        default_value='false',
        description='Enable performance tracking and publishing to /vision/performance topic. When true, publishes detailed pipeline timing metrics every 5 seconds.'
    )

    # Model configuration
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/gesture_recognizer.task',
        description='Path to the gesture recognition model file. Can be absolute path or relative to package share directory.'
    )
    
    # ========================================
    # CAMERA NODE (High-Performance Configuration for Gesture Recognition)
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
            # High-performance parameters (adapted from object detection launch)
            "buffer_queue_size": 2,  # Reduced buffer for lower latency
            # Camera controls optimized for gesture recognition (15fps)
            "FrameDurationLimits": [66667, 66667],  # 15 FPS = 66.67ms = 66,667 microseconds
            "ExposureTime": 20000,  # 1/50s in microseconds
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            # Quality settings
            "jpeg_quality": 80,  # Reduced for better performance
            # Sensor mode for IMX219
            "sensor_mode": "640:480",
            # Use sim time setting
            'use_sim_time': False,
        }],
        remappings=[
            # Ensure proper topic naming (using relative remapping like object detection)
            ('~/image_raw', '/camera/image_raw'),
            ('~/image_raw/compressed', '/camera/image_raw/compressed'),
            ('~/camera_info', '/camera/camera_info'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_camera')),
        output='screen'
    )
    
    # ========================================
    # GESTURE RECOGNITION NODE
    # ========================================

    gesture_recognition_node = Node(
        package='gesturebot',
        executable='gesture_recognition_node.py',
        name='gesture_recognition_node',
        parameters=[{
            'enabled': True,
            'max_fps': 15.0,
            'frame_skip': 1,
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_hands': LaunchConfiguration('max_hands'),
            'gesture_stability_threshold': LaunchConfiguration('gesture_stability_threshold'),
            'priority': 1,
            'publish_annotated_images': LaunchConfiguration('publish_annotated_images'),
            'show_landmark_indices': LaunchConfiguration('show_landmark_indices'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'unlimited_buffer_mode': LaunchConfiguration('unlimited_buffer_mode'),
            'buffer_logging_enabled': LaunchConfiguration('buffer_logging_enabled'),
            'enable_performance_tracking': LaunchConfiguration('enable_performance_tracking'),
            'model_path': LaunchConfiguration('model_path'),
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_gesture_recognition')),
        output='screen'
    )



    # ========================================
    # NODE GROUPING
    # ========================================

    # Group all nodes for organized launch
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
        declare_gesture_stability_threshold,
        declare_publish_annotated_images,
        declare_show_landmark_indices,
        declare_debug_mode,
        declare_unlimited_buffer_mode,
        declare_buffer_logging_enabled,
        declare_enable_performance_tracking,
        declare_model_path,

        # Gesture recognition system nodes
        gesture_recognition_group
    ])
