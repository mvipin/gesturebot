#!/usr/bin/env python3
"""
Image Viewer Launch File for GestureBot Vision System
Launches OpenCV-based image viewer for annotated object detection images.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for image viewer."""
    
    # ========================================
    # LAUNCH ARGUMENTS
    # ========================================
    
    declare_window_name = DeclareLaunchArgument(
        'window_name',
        default_value='GestureBot Object Detection',
        description='OpenCV window name for display'
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
    
    # ========================================
    # IMAGE VIEWER NODE
    # ========================================
    
    image_viewer_node = Node(
        package='gesturebot',
        executable='image_viewer_node.py',
        name='image_viewer_node',
        parameters=[{
            'window_name': LaunchConfiguration('window_name'),
            'display_fps': LaunchConfiguration('display_fps'),
            'window_width': LaunchConfiguration('window_width'),
            'window_height': LaunchConfiguration('window_height'),
            'show_fps_overlay': LaunchConfiguration('show_fps_overlay'),
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
        
        # Image viewer node
        image_viewer_node
    ])
