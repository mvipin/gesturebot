#!/usr/bin/env python3
"""
GestureBot Quick Test
Fast validation test for basic system functionality.
"""

import os
import sys
import time
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('mediapipe', 'MediaPipe'),
        ('rclpy', 'ROS 2 Python'),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def test_ros2_environment():
    """Test ROS 2 environment setup."""
    print("\n🔍 Testing ROS 2 environment...")
    
    # Check ROS_DISTRO
    ros_distro = os.environ.get('ROS_DISTRO')
    if ros_distro:
        print(f"  ✅ ROS_DISTRO: {ros_distro}")
    else:
        print("  ❌ ROS_DISTRO not set")
        return False
    
    # Check if we can initialize rclpy
    try:
        import rclpy
        rclpy.init()
        print("  ✅ ROS 2 initialization successful")
        rclpy.shutdown()
        return True
    except Exception as e:
        print(f"  ❌ ROS 2 initialization failed: {e}")
        return False

def test_mediapipe_basic():
    """Test basic MediaPipe functionality."""
    print("\n🔍 Testing MediaPipe basic functionality...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Test MediaPipe version
        print(f"  ✅ MediaPipe version: {mp.__version__}")
        
        # Test basic MediaPipe operations
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Try to initialize hands solution
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(test_image)
            print("  ✅ MediaPipe hands processing successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MediaPipe test failed: {e}")
        return False

def test_opencv_basic():
    """Test basic OpenCV functionality."""
    print("\n🔍 Testing OpenCV basic functionality...")
    
    try:
        import cv2
        import numpy as np
        
        # Test OpenCV version
        print(f"  ✅ OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Color conversion
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        
        # Basic filtering
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print("  ✅ OpenCV image processing successful")
        return True
        
    except Exception as e:
        print(f"  ❌ OpenCV test failed: {e}")
        return False

def test_package_structure():
    """Test GestureBot package structure."""
    print("\n🔍 Testing package structure...")
    
    package_dir = Path(__file__).parent.parent
    
    required_dirs = [
        'src',
        'msg', 
        'srv',
        'launch',
        'config',
        'test'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = package_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ directory")
        else:
            print(f"  ❌ {dir_name}/ directory missing")
            missing_dirs.append(dir_name)
    
    # Check for key files
    key_files = [
        'package.xml',
        'CMakeLists.txt',
        'README.md'
    ]
    
    for file_name in key_files:
        file_path = package_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} missing")
    
    return len(missing_dirs) == 0

def test_models_availability():
    """Test MediaPipe models availability."""
    print("\n🔍 Testing MediaPipe models...")
    
    package_dir = Path(__file__).parent.parent
    models_dir = package_dir / 'models'
    
    if not models_dir.exists():
        print(f"  ⚠️  Models directory not found: {models_dir}")
        return False
    
    expected_models = [
        'efficientdet.tflite'
    ]
    
    available_models = []
    missing_models = []
    
    for model_name in expected_models:
        model_path = models_dir / model_name
        if model_path.exists():
            file_size = model_path.stat().st_size
            print(f"  ✅ {model_name} ({file_size / 1024 / 1024:.1f}MB)")
            available_models.append(model_name)
        else:
            print(f"  ❌ {model_name} missing")
            missing_models.append(model_name)
    
    if missing_models:
        print(f"  ⚠️  Missing models can be downloaded using setup script")
    
    return len(available_models) > 0

def main():
    """Run quick validation tests."""
    print("=== GestureBot Quick Test ===")
    print("Fast validation of basic system functionality\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("ROS 2 Environment", test_ros2_environment), 
        ("MediaPipe Basic", test_mediapipe_basic),
        ("OpenCV Basic", test_opencv_basic),
        ("Package Structure", test_package_structure),
        ("MediaPipe Models", test_models_availability)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            success, *details = test_func() if test_func() is not None else (test_func(), [])
            if isinstance(success, tuple):
                success, details = success
            results.append((test_name, success, details))
        except Exception as e:
            print(f"  ❌ Test error: {e}")
            results.append((test_name, False, [str(e)]))
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, success, _ in results if success)
    total_tests = len(results)
    
    print("\n" + "="*50)
    print("QUICK TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Execution time: {total_time:.1f}s")
    
    print("\nDETAILED RESULTS:")
    for test_name, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not success and details:
            for detail in details[:3]:  # Show first 3 details
                print(f"    └─ {detail}")
    
    if passed_tests == total_tests:
        print("\n🎉 All quick tests passed! System appears ready for full testing.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check details above.")
        print("💡 Try running the setup script or installing missing dependencies.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
