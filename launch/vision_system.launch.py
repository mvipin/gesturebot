#!/usr/bin/env python3
"""
Launch file for GestureBot Vision System
Configurable launch of MediaPipe features with performance monitoring.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for vision system."""
    
    # Package directories
    vision_pkg_dir = FindPackageShare('gesturebot_vision')
    
    # Launch arguments for feature enable/disable
    declare_object_detection = DeclareLaunchArgument(
        'enable_object_detection',
        default_value='true',
        description='Enable object detection node'
    )
    
    declare_gesture_recognition = DeclareLaunchArgument(
        'enable_gesture_recognition',
        default_value='true',
        description='Enable gesture recognition node'
    )
    
    declare_hand_landmarks = DeclareLaunchArgument(
        'enable_hand_landmarks',
        default_value='false',
        description='Enable hand landmark detection node'
    )
    
    declare_pose_landmarks = DeclareLaunchArgument(
        'enable_pose_landmarks',
        default_value='false',
        description='Enable pose landmark detection node'
    )
    
    declare_face_detection = DeclareLaunchArgument(
        'enable_face_detection',
        default_value='false',
        description='Enable face detection node'
    )
    
    declare_ball_tracking = DeclareLaunchArgument(
        'enable_ball_tracking',
        default_value='false',
        description='Enable OpenCV ball tracking node'
    )
    
    declare_performance_monitor = DeclareLaunchArgument(
        'enable_performance_monitor',
        default_value='true',
        description='Enable performance monitoring'
    )
    
    declare_camera_source = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera topic for vision processing'
    )
    
    declare_debug_mode = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug output and visualization'
    )
    
    declare_navigation_bridge = DeclareLaunchArgument(
        'enable_navigation_bridge',
        default_value='true',
        description='Enable gesture navigation bridge'
    )
    
    # Core vision nodes
    object_detection_node = Node(
        package='gesturebot_vision',
        executable='object_detection_node.py',
        name='object_detection_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'confidence_threshold': 0.5,
            'max_results': 5,
        }],
        condition=IfCondition(LaunchConfiguration('enable_object_detection')),
        output='screen'
    )
    
    gesture_recognition_node = Node(
        package='gesturebot_vision',
        executable='gesture_recognition_node.py',
        name='gesture_recognition_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'confidence_threshold': 0.7,
            'gesture_stability_threshold': 0.5,
            'max_hands': 2
        }],
        condition=IfCondition(LaunchConfiguration('enable_gesture_recognition')),
        output='screen'
    )
    
    hand_landmarks_node = Node(
        package='gesturebot_vision',
        executable='hand_landmarks_node.py',
        name='hand_landmarks_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'max_hands': 2
        }],
        condition=IfCondition(LaunchConfiguration('enable_hand_landmarks')),
        output='screen'
    )
    
    pose_landmarks_node = Node(
        package='gesturebot_vision',
        executable='pose_landmarks_node.py',
        name='pose_landmarks_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'model_complexity': 1
        }],
        condition=IfCondition(LaunchConfiguration('enable_pose_landmarks')),
        output='screen'
    )
    
    face_detection_node = Node(
        package='gesturebot_vision',
        executable='face_detection_node.py',
        name='face_detection_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'min_detection_confidence': 0.5
        }],
        condition=IfCondition(LaunchConfiguration('enable_face_detection')),
        output='screen'
    )
    
    ball_tracking_node = Node(
        package='gesturebot_vision',
        executable='ball_tracking_node.py',
        name='ball_tracking_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            'min_radius': 10,
            'max_radius': 100,
            'color_lower': [0, 100, 100],  # HSV lower bound
            'color_upper': [10, 255, 255]  # HSV upper bound
        }],
        condition=IfCondition(LaunchConfiguration('enable_ball_tracking')),
        output='screen'
    )
    
    # Performance monitoring node
    performance_monitor_node = Node(
        package='gesturebot_vision',
        executable='performance_monitor_node.py',
        name='performance_monitor_node',
        parameters=[{
            'monitor_interval': 5.0,
            'log_performance': True,
            'alert_threshold_cpu': 80.0,
            'alert_threshold_memory': 1500.0  # MB
        }],
        condition=IfCondition(LaunchConfiguration('enable_performance_monitor')),
        output='screen'
    )
    
    # Navigation bridge node
    navigation_bridge_node = Node(
        package='gesturebot_vision',
        executable='gesture_navigation_bridge.py',
        name='gesture_navigation_bridge',
        parameters=[{
            'gesture_confidence_threshold': 0.8,
            'emergency_stop_enabled': True,
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 1.0,
            'gesture_repeat_delay': 1.0
        }],
        condition=IfCondition(LaunchConfiguration('enable_navigation_bridge')),
        output='screen'
    )
    
    # Group all vision nodes
    vision_group = GroupAction([
        object_detection_node,
        gesture_recognition_node,
        hand_landmarks_node,
        pose_landmarks_node,
        face_detection_node,
        ball_tracking_node,
        performance_monitor_node,
        navigation_bridge_node
    ])
    
    return LaunchDescription([
        # Launch arguments
        declare_object_detection,
        declare_gesture_recognition,
        declare_hand_landmarks,
        declare_pose_landmarks,
        declare_face_detection,
        declare_ball_tracking,
        declare_performance_monitor,
        declare_camera_source,
        declare_debug_mode,
        declare_navigation_bridge,
        
        # Vision system nodes
        vision_group
    ])
