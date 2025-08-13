#!/usr/bin/env python3
"""
Pose Detection Launch File for GestureBot Vision System
Launches MediaPipe-based pose detection with 33-point pose landmark tracking.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    
    declare_output_segmentation_masks = DeclareLaunchArgument(
        'output_segmentation_masks',
        default_value='false',
        description='Enable pose segmentation mask output'
    )
    
    # Model configuration
    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='models/pose_landmarker.task',
        description='Path to pose landmarker model file'
    )
    
    # Visualization parameters
    declare_publish_annotated_images = DeclareLaunchArgument(
        'publish_annotated_images',
        default_value='true',
        description='Enable annotated image publishing'
    )
    
    declare_show_landmark_indices = DeclareLaunchArgument(
        'show_landmark_indices',
        default_value='false',
        description='Show landmark indices on annotated images (debug mode)'
    )
    
    # Performance and logging parameters
    declare_buffer_logging_enabled = DeclareLaunchArgument(
        'buffer_logging_enabled',
        default_value='false',
        description='Enable buffered logging for performance analysis'
    )
    
    declare_unlimited_buffer_mode = DeclareLaunchArgument(
        'unlimited_buffer_mode',
        default_value='false',
        description='Enable unlimited buffer mode for extensive logging'
    )
    
    declare_enable_performance_tracking = DeclareLaunchArgument(
        'enable_performance_tracking',
        default_value='false',
        description='Enable detailed performance tracking and metrics'
    )
    
    # ========================================
    # CAMERA NODE (Direct launch without image viewer)
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
            'camera_calibration_file': PathJoinSubstitution([
                FindPackageShare('gesturebot'),
                'config',
                'camera_calibration.yaml'
            ]),
        }],
        output='screen'
    )
    
    # ========================================
    # POSE DETECTION NODE
    # ========================================
    
    pose_detection_node = Node(
        package='gesturebot',
        executable='pose_detection_node.py',
        name='pose_detection_node',
        parameters=[{
            # Pose detection parameters
            'num_poses': LaunchConfiguration('num_poses'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_presence_confidence': LaunchConfiguration('min_pose_presence_confidence'),
            'min_tracking_confidence': LaunchConfiguration('min_tracking_confidence'),
            'output_segmentation_masks': LaunchConfiguration('output_segmentation_masks'),
            
            # Model configuration
            'model_path': LaunchConfiguration('model_path'),
            
            # Visualization parameters
            'publish_annotated_images': LaunchConfiguration('publish_annotated_images'),
            'show_landmark_indices': LaunchConfiguration('show_landmark_indices'),
            
            # Performance and logging
            'buffer_logging_enabled': LaunchConfiguration('buffer_logging_enabled'),
            'unlimited_buffer_mode': LaunchConfiguration('unlimited_buffer_mode'),
            'enable_performance_tracking': LaunchConfiguration('enable_performance_tracking'),
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
        declare_output_segmentation_masks,
        declare_model_path,
        declare_publish_annotated_images,
        declare_show_landmark_indices,
        declare_buffer_logging_enabled,
        declare_unlimited_buffer_mode,
        declare_enable_performance_tracking,
        
        # Nodes
        camera_node,
        pose_detection_node
    ])
