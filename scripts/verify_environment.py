#!/usr/bin/env python3
"""
GestureBot Environment Verification Script
Verifies that ROS 2 Jazzy and virtual environment packages work together correctly.
"""

import os
import sys
import importlib
from pathlib import Path


def print_section(title):
    """Print section header."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print('='*50)


def check_python_environment():
    """Check Python environment details."""
    print_section("PYTHON ENVIRONMENT")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if virtual environment is active
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print(f"✅ Virtual environment active: {venv_path}")
    else:
        print("⚠️  No virtual environment detected (using system Python)")
    
    # Print Python path
    print(f"\nPython path ({len(sys.path)} entries):")
    for i, path in enumerate(sys.path):
        if 'gesturebot_env' in path:
            print(f"  {i:2d}. 🐍 {path}")  # Virtual env
        elif 'ros' in path.lower():
            print(f"  {i:2d}. 🤖 {path}")  # ROS 2
        else:
            print(f"  {i:2d}.    {path}")  # Other


def check_ros2_packages():
    """Check ROS 2 package availability."""
    print_section("ROS 2 PACKAGES")
    
    # Check ROS_DISTRO environment variable
    ros_distro = os.environ.get('ROS_DISTRO')
    if ros_distro:
        print(f"✅ ROS_DISTRO: {ros_distro}")
    else:
        print("❌ ROS_DISTRO not set - source ROS 2 setup.bash first")
        return False
    
    # Test ROS 2 Python packages
    ros2_packages = [
        ('rclpy', 'ROS 2 Python client library'),
        ('sensor_msgs.msg', 'ROS 2 sensor messages'),
        ('geometry_msgs.msg', 'ROS 2 geometry messages'),
        ('cv_bridge', 'OpenCV-ROS bridge'),
        ('nav2_msgs.msg', 'Navigation 2 messages'),
    ]
    
    ros2_success = True
    for package, description in ros2_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} - {description}")
        except ImportError as e:
            print(f"  ❌ {package} - {description}: {e}")
            ros2_success = False
    
    return ros2_success


def check_vision_packages():
    """Check computer vision package availability."""
    print_section("COMPUTER VISION PACKAGES")
    
    vision_packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('mediapipe', 'MediaPipe'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('psutil', 'System monitoring'),
    ]
    
    vision_success = True
    for package, description in vision_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            
            # Check installation location
            module_file = getattr(module, '__file__', 'unknown')
            if 'gesturebot_env' in module_file:
                location = "🐍 venv"
            elif 'site-packages' in module_file:
                location = "🌐 system"
            else:
                location = "❓ other"
            
            print(f"  ✅ {package} v{version} - {description} ({location})")
            
        except ImportError as e:
            print(f"  ❌ {package} - {description}: {e}")
            vision_success = False
    
    return vision_success


def check_gesturebot_packages():
    """Check GestureBot package imports."""
    print_section("GESTUREBOT PACKAGES")
    
    # Add GestureBot source to path
    gesturebot_src = Path(__file__).parent.parent / 'src'
    if gesturebot_src.exists():
        sys.path.insert(0, str(gesturebot_src))
        print(f"Added to Python path: {gesturebot_src}")
    
    gesturebot_packages = [
        ('vision_core.base_node', 'Vision core base classes'),
        ('vision_core.message_converter', 'Message conversion utilities'),
    ]
    
    gesturebot_success = True
    for package, description in gesturebot_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} - {description}")
        except ImportError as e:
            print(f"  ❌ {package} - {description}: {e}")
            gesturebot_success = False
    
    return gesturebot_success


def check_camera_functionality():
    """Check camera functionality and rpicam tools."""
    print_section("CAMERA FUNCTIONALITY")

    import subprocess
    import os

    # Check if rpicam tools are available
    rpicam_tools = ['rpicam-still', 'rpicam-hello', 'rpicam-vid']
    available_tools = []

    for tool in rpicam_tools:
        try:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            if result.returncode == 0:
                tool_path = result.stdout.strip()
                print(f"  ✅ {tool} available at {tool_path}")
                available_tools.append(tool)
            else:
                print(f"  ❌ {tool} not found")
        except Exception as e:
            print(f"  ❌ {tool} check failed: {e}")

    # Test rpicam-still functionality if available
    if 'rpicam-still' in available_tools:
        try:
            print("  🔍 Testing rpicam-still capture...")
            result = subprocess.run([
                'rpicam-still', '--output', '/tmp/test_capture.jpg',
                '--timeout', '1000', '--nopreview'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and os.path.exists('/tmp/test_capture.jpg'):
                file_size = os.path.getsize('/tmp/test_capture.jpg')
                print(f"  ✅ Camera capture successful ({file_size} bytes)")
                os.unlink('/tmp/test_capture.jpg')
                return True
            else:
                print(f"  ⚠️  Camera capture failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("  ⚠️  Camera capture timed out")
            return False
        except Exception as e:
            print(f"  ❌ Camera test error: {e}")
            return False
    else:
        print("  ❌ rpicam-still not available - camera functionality limited")
        return False


def check_ros2_integration():
    """Test ROS 2 integration functionality."""
    print_section("ROS 2 INTEGRATION TEST")

    try:
        import rclpy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import cv2
        import numpy as np

        # Test basic ROS 2 functionality
        rclpy.init()

        # Test CV Bridge
        cv_bridge = CvBridge()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        ros_image = cv_bridge.cv2_to_imgmsg(test_image, 'bgr8')
        converted_back = cv_bridge.imgmsg_to_cv2(ros_image, 'bgr8')

        print("  ✅ ROS 2 initialization successful")
        print("  ✅ CV Bridge conversion successful")
        print("  ✅ Image message handling successful")

        rclpy.shutdown()
        return True

    except Exception as e:
        print(f"  ❌ ROS 2 integration test failed: {e}")
        return False


def main():
    """Main verification function."""
    print("🤖 GestureBot Environment Verification")
    print("Checking ROS 2 Jazzy + Virtual Environment compatibility...")
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_environment),
        ("ROS 2 Packages", check_ros2_packages),
        ("Vision Packages", check_vision_packages),
        ("GestureBot Packages", check_gesturebot_packages),
        ("Camera Functionality", check_camera_functionality),
        ("ROS 2 Integration", check_ros2_integration),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            if callable(check_func):
                result = check_func()
                results.append((check_name, result if result is not None else True))
            else:
                check_func()
                results.append((check_name, True))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed_checks = sum(1 for _, success in results if success)
    total_checks = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 Environment verification successful!")
        print("Your system is ready for GestureBot development.")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_checks} check(s) failed.")
        print("Please address the issues above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
