#!/usr/bin/env python3
"""
Object Detection Launch File for GestureBot Vision System
Dedicated launch file for camera + object detection with configurable parameters.
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
    
    declare_camera_fps = DeclareLaunchArgument(
        'camera_fps',
        default_value='5.0',
        description='Camera frame rate'
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
    
    declare_publish_annotated_images = DeclareLaunchArgument(
        'publish_annotated_images',
        default_value='true',
        description='Enable publishing of annotated images for visual debugging'
    )
    
    # Debug configuration
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug output and logging'
    )

    # Buffered logging configuration
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
        default_value='models/efficientdet.tflite',
        description='Path to the EfficientDet model file. Can be absolute path or relative to package share directory.'
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
            # High-performance parameters (from camera_high_fps.launch.py)
            "buffer_queue_size": 2,  # Reduced buffer for lower latency
            # Camera controls optimized for 5fps
            "FrameDurationLimits": [200000, 200000],  # 5 FPS = 200ms = 200,000 microseconds
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
            # Ensure proper topic naming
            ('~/image_raw', '/camera/image_raw'),
            ('~/image_raw/compressed', '/camera/image_raw/compressed'),
            ('~/camera_info', '/camera/camera_info'),
        ],
        condition=IfCondition(LaunchConfiguration('enable_camera')),
        output='screen'
    )
    
    # ========================================
    # OBJECT DETECTION NODE
    # ========================================
    
    object_detection_node = Node(
        package='gesturebot',
        executable='object_detection_node.py',
        name='object_detection_node',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'max_results': LaunchConfiguration('max_results'),
            'publish_annotated_images': LaunchConfiguration('publish_annotated_images'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'unlimited_buffer_mode': LaunchConfiguration('unlimited_buffer_mode'),
            'buffer_logging_enabled': LaunchConfiguration('buffer_logging_enabled'),
            'enable_performance_tracking': LaunchConfiguration('enable_performance_tracking'),
            'model_path': LaunchConfiguration('model_path'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_object_detection')),
        output='screen'
    )
    
    # ========================================
    # NODE GROUPING
    # ========================================
    
    # Group all nodes for organized launch
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
        declare_camera_fps,
        declare_enable_object_detection,
        declare_confidence_threshold,
        declare_max_results,
        declare_publish_annotated_images,
        declare_debug_mode,
        declare_unlimited_buffer_mode,
        declare_buffer_logging_enabled,
        declare_enable_performance_tracking,
        declare_model_path,

        # Object detection system nodes
        object_detection_group
    ])
