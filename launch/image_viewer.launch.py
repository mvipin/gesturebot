#!/usr/bin/env python3
"""
Unified Image Viewer Launch File for GestureBot Vision System
Launches OpenCV-based image viewer for multiple vision topics (object detection, gesture recognition, etc.).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Generate launch description for unified image viewer."""

    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================

    declare_window_name = DeclareLaunchArgument(
        'window_name',
        default_value='GestureBot Vision System',
        description='Base OpenCV window name for display'
    )

    declare_display_fps = DeclareLaunchArgument(
        'display_fps',
        default_value='10.0',
        description='Maximum display refresh rate (FPS) to limit CPU usage'
    )

    declare_window_width = DeclareLaunchArgument(
        'window_width',
        default_value='640',
        description='Display window width (0 = original size)'
    )

    declare_window_height = DeclareLaunchArgument(
        'window_height',
        default_value='480',
        description='Display window height (0 = original size)'
    )

    declare_show_fps_overlay = DeclareLaunchArgument(
        'show_fps_overlay',
        default_value='true',
        description='Show FPS overlay on displayed image'
    )

    declare_image_topics = DeclareLaunchArgument(
        'image_topics',
        default_value='["/vision/objects/annotated"]',
        description='List of ROS image topics to subscribe to (JSON array format)'
    )

    declare_topic_window_names = DeclareLaunchArgument(
        'topic_window_names',
        default_value='{}',
        description='Optional mapping of topic names to custom window names (JSON object format)'
    )

    # ========================================
    # UNIFIED IMAGE VIEWER NODE
    # ========================================

    unified_image_viewer_node = Node(
        package='gesturebot',
        executable='image_viewer_node.py',
        name='unified_image_viewer',
        parameters=[{
            'window_name': LaunchConfiguration('window_name'),
            'display_fps': LaunchConfiguration('display_fps'),
            'window_width': LaunchConfiguration('window_width'),
            'window_height': LaunchConfiguration('window_height'),
            'show_fps_overlay': LaunchConfiguration('show_fps_overlay'),
            'image_topics': ParameterValue(LaunchConfiguration('image_topics'), value_type=str),
            'topic_window_names': ParameterValue(LaunchConfiguration('topic_window_names'), value_type=str),
        }],
        output='screen'
    )

    # ========================================
    # LAUNCH DESCRIPTION
    # ========================================

    return LaunchDescription([
        # Launch arguments
        declare_window_name,
        declare_display_fps,
        declare_window_width,
        declare_window_height,
        declare_show_fps_overlay,
        declare_image_topics,
        declare_topic_window_names,

        # Unified image viewer node
        unified_image_viewer_node
    ])
