#!/usr/bin/env python3
"""
Pose Detection Launch File for GestureBot Vision System
Launches camera + ros2_mediapipe pose detection with 33-point pose landmark tracking.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for pose detection system."""

    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================

    # Camera configuration
    declare_camera_format = DeclareLaunchArgument(
        'camera_format',
        default_value='RGB888',
        description='Camera pixel format (RGB888, BGR888, etc.)'
    )

    declare_camera_width = DeclareLaunchArgument(
        'camera_width',
        default_value='640',
        description='Camera frame width'
    )

    declare_camera_height = DeclareLaunchArgument(
        'camera_height',
        default_value='480',
        description='Camera frame height'
    )

    declare_camera_fps = DeclareLaunchArgument(
        'camera_fps',
        default_value='15',
        description='Camera frame rate'
    )

    # Pose detection parameters
    declare_num_poses = DeclareLaunchArgument(
        'num_poses',
        default_value='2',
        description='Maximum number of poses to detect'
    )

    declare_min_pose_detection_confidence = DeclareLaunchArgument(
        'min_pose_detection_confidence',
        default_value='0.5',
        description='Minimum confidence for pose detection'
    )

    declare_min_pose_presence_confidence = DeclareLaunchArgument(
        'min_pose_presence_confidence',
        default_value='0.5',
        description='Minimum confidence for pose presence'
    )

    declare_min_tracking_confidence = DeclareLaunchArgument(
        'min_tracking_confidence',
        default_value='0.5',
        description='Minimum confidence for pose tracking'
    )

    # Model configuration
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/pose_landmarker.task',
        description='Path to pose landmarker model file'
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

    # ========================================
    # CAMERA NODE
    # ========================================

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'format': LaunchConfiguration('camera_format'),
            'width': LaunchConfiguration('camera_width'),
            'height': LaunchConfiguration('camera_height'),
            'fps': LaunchConfiguration('camera_fps'),
        }],
        output='screen'
    )

    # ========================================
    # POSE DETECTION NODE (ros2_mediapipe)
    # ========================================

    pose_detection_node = Node(
        package='ros2_mediapipe',
        executable='pose_detection_node.py',
        name='pose_detection_node',
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'num_poses': LaunchConfiguration('num_poses'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_presence_confidence': LaunchConfiguration('min_pose_presence_confidence'),
            'min_tracking_confidence': LaunchConfiguration('min_tracking_confidence'),
            'model_path': LaunchConfiguration('model_path'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'debug_mode': LaunchConfiguration('debug_mode'),
        }],
        output='screen'
    )

    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================

    return LaunchDescription([
        # Launch arguments
        declare_camera_format,
        declare_camera_width,
        declare_camera_height,
        declare_camera_fps,
        declare_num_poses,
        declare_min_pose_detection_confidence,
        declare_min_pose_presence_confidence,
        declare_min_tracking_confidence,
        declare_model_path,
        declare_frame_skip,
        declare_debug_mode,

        # Nodes
        camera_node,
        pose_detection_node
    ])
