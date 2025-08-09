# GestureBot Vision System

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.18-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Comprehensive MediaPipe-based computer vision system for robotics applications, specifically designed for the GestureBot platform running on Raspberry Pi 5. Features real-time object detection, gesture recognition, hand/pose tracking, and seamless integration with ROS 2 Navigation stack for autonomous robot control through intuitive hand gestures.

> **📖 Complete Project Setup**: For full project setup instructions including virtual environment and camera system configuration, see the [main project README](../../../README.md).

![GestureBot Vision System Overview](media/system_overview.png)
<!-- TODO: Capture system overview photo showing Pi 5, camera, and robot platform -->

## 🚀 Quick Start

> **⚠️ Important**: This package requires proper virtual environment setup. See the [main project README](../../../README.md) for complete setup instructions.

### For Existing Setup
```bash
# Activate GestureBot environment (includes virtual env + ROS 2)
cd ~/GestureBot/gesturebot_ws
source activate_gesturebot.sh

# Test package functionality
python3 -c "import rclpy, mediapipe; print('✅ gesturebot package ready!')"

# Run package tests
cd src/gesturebot/test
python3 quick_test.py

# Test MediaPipe features
python3 mediapipe_feature_tester.py --feature hands
```

### For New Setup
```bash
# 1. Create and activate virtual environment
cd ~/GestureBot
python3 -m venv gesturebot_env
source gesturebot_env/bin/activate

# 2. Install Python dependencies from requirements file
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Build this package specifically
cd gesturebot_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select gesturebot
source install/setup.bash

# 4. Verify installation
python3 -c "import rclpy, mediapipe; print('✅ gesturebot package ready!')"
```

![Quick Start Demo](media/quick_start_demo.gif)
<!-- TODO: Record GIF showing complete setup and launch process -->

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)

### **1. [Vision System Overview](#1-vision-system-overview)**
  - [Hardware Components](#hardware-components)
  - [Software Architecture](#software-architecture)
  - [Performance Specifications](#performance-specifications)

### **2. [MediaPipe Features](#2-mediapipe-features)**
  - [Object Detection](#object-detection)
  - [Gesture Recognition](#gesture-recognition)
  - [Hand Landmark Detection](#hand-landmark-detection)
  - [Pose Estimation](#pose-estimation)
  - [Face Detection](#face-detection)

### **3. [OpenCV Integration](#3-opencv-integration)**
  - [Ball Tracking](#ball-tracking)
  - [Blob Detection](#blob-detection)
  - [Color-based Tracking](#color-based-tracking)

### **4. [Navigation Integration](#4-navigation-integration)**
  - [Gesture-based Robot Control](#gesture-based-robot-control)
  - [Safety Systems](#safety-systems)
  - [Emergency Stop Features](#emergency-stop-features)

### **5. [Performance & Optimization](#5-performance--optimization)**
  - [Resource Management](#resource-management)
  - [Adaptive Processing](#adaptive-processing)
  - [Benchmarking Tools](#benchmarking-tools)

### **6. [Installation & Setup](#6-installation--setup)**
  - [Prerequisites](#prerequisites)
  - [Automated Setup](#automated-setup)
  - [Manual Configuration](#manual-configuration)

### **7. [Configuration & Usage](#7-configuration--usage)**
  - [Launch Files](#launch-files)
  - [Parameter Tuning](#parameter-tuning)
  - [Topic Monitoring](#topic-monitoring)

### **8. [Development & Testing](#8-development--testing)**
  - [Adding New Features](#adding-new-features)
  - [Testing Framework](#testing-framework)
  - [Performance Benchmarking](#performance-benchmarking)

### **9. [Troubleshooting](#9-troubleshooting)**
  - [Common Issues](#common-issues)
  - [Performance Problems](#performance-problems)
  - [Hardware Debugging](#hardware-debugging)

### **Additional Resources**
  - [API Documentation](#api-documentation)
  - [Contributing](#contributing)
  - [License](#license)

---

## 1. Vision System Overview

![System Architecture Diagram](media/architecture_diagram.png)
<!-- TODO: Create comprehensive system architecture diagram showing all components -->

### Hardware Components

**Compute Platform:**
- **Raspberry Pi 5**: 8GB RAM, ARM Cortex-A76 quad-core processor
- **Pi Camera Module 3**: 12MP sensor with autofocus, 30 FPS capability
- **MicroSD Card**: 64GB+ Class 10 for optimal performance
- **Cooling**: Active cooling recommended for sustained processing

![Hardware Setup](media/hardware_setup.jpg)
<!-- TODO: Photograph complete hardware setup with labeled components -->

**Physical Specifications:**
- Camera resolution: 640x480 @ 30 FPS (configurable)
- Processing capability: ~15 FPS with multiple MediaPipe features
- Power consumption: ~8W total system power
- Operating temperature: 0°C to 70°C

### Software Architecture

**Core ROS 2 Packages:**
- `gesturebot.vision_core`: Shared utilities and base classes
- `gesturebot`: Main package with nodes and messages
- `cv_bridge`: OpenCV-ROS 2 image conversion
- `image_transport`: Efficient image streaming

**Processing Pipeline:**
```
Pi Camera → rpicam-still → ROS 2 Image → MediaPipe/OpenCV → Vision Results → Navigation Commands
```

![Processing Pipeline Visualization](media/processing_pipeline.png)
<!-- TODO: Create visual flowchart of the complete processing pipeline -->

### Performance Specifications

**MediaPipe Performance (Pi 5):**
- **Object Detection**: ~15 FPS, 97ms processing time
- **Gesture Recognition**: ~12 FPS, 80ms processing time  
- **Hand Landmarks**: ~10 FPS, 100ms processing time
- **Combined Features**: ~8-12 FPS with adaptive processing

![Performance Benchmarks](media/performance_charts.png)
<!-- TODO: Generate performance benchmark charts for different feature combinations -->

---

## 2. MediaPipe Features

### Object Detection

**Implementation:**
- **Model**: EfficientDet Lite (TensorFlow Lite optimized)
- **Classes**: 80 COCO dataset objects (person, car, bottle, etc.)
- **Confidence Threshold**: 0.5 (configurable)
- **Max Results**: 5 objects per frame

![Object Detection Demo](media/object_detection_demo.gif)
<!-- TODO: Record object detection working on various objects -->

**Key Capabilities:**
- Real-time object recognition and bounding box detection
- Confidence scoring for detection reliability
- Integration with navigation costmap for obstacle avoidance
- Support for custom object classes

**Configuration:**
```yaml
object_detection_node:
  ros__parameters:
    confidence_threshold: 0.5
    max_results: 5
    model_path: "models/efficientdet.tflite"
```

### Gesture Recognition

**Supported Gestures:**
- 👍 **Thumbs Up**: Start navigation
- 👎 **Thumbs Down**: Stop navigation  
- ✋ **Open Palm**: Pause navigation
- ✊ **Fist**: Emergency stop
- ✌️ **Peace Sign**: Follow person mode
- 👆 **Pointing**: Directional movement

![Gesture Recognition Demo](media/gesture_recognition_demo.gif)
<!-- TODO: Record all supported gestures being recognized -->

**Navigation Integration:**
```
Hand Gesture → Gesture Recognition → Navigation Command → Robot Movement
```

![Gesture Navigation Flow](media/gesture_navigation_flow.png)
<!-- TODO: Create diagram showing gesture-to-navigation command mapping -->

### Hand Landmark Detection

**Features:**
- **21 Landmark Points**: Complete hand skeleton tracking
- **Dual Hand Support**: Track both hands simultaneously
- **3D Coordinates**: X, Y, Z position data
- **Confidence Scoring**: Per-landmark reliability metrics

![Hand Landmarks Visualization](media/hand_landmarks_demo.gif)
<!-- TODO: Record hand landmark detection with overlay visualization -->

**Applications:**
- Precise gesture analysis
- Hand pose estimation
- Fine motor control interfaces
- Sign language recognition (future)

### Pose Estimation

**Capabilities:**
- **33 Body Landmarks**: Full body pose detection
- **Real-time Tracking**: 30Hz pose estimation
- **3D Pose Data**: Complete skeletal information
- **Multi-person Support**: Track multiple people

![Pose Estimation Demo](media/pose_estimation_demo.gif)
<!-- TODO: Record full body pose estimation in action -->

### Face Detection

**Features:**
- **Face Bounding Boxes**: Precise face localization
- **Confidence Scoring**: Detection reliability metrics
- **Multi-face Support**: Detect multiple faces
- **Age/Emotion Estimation**: Extended analysis capabilities

![Face Detection Demo](media/face_detection_demo.gif)
<!-- TODO: Record face detection with multiple people -->

---

## 3. OpenCV Integration

### Ball Tracking

**Implementation:**
- **Color-based Detection**: HSV color space filtering
- **Contour Analysis**: Shape and size validation
- **Kalman Filtering**: Smooth trajectory prediction
- **Multi-ball Support**: Track multiple objects

![Ball Tracking Demo](media/ball_tracking_demo.gif)
<!-- TODO: Record ball tracking with colored balls -->

**Configuration:**
```yaml
ball_tracking_node:
  ros__parameters:
    color_lower: [0, 100, 100]    # HSV lower bound
    color_upper: [10, 255, 255]   # HSV upper bound
    min_radius: 10
    max_radius: 100
```

### Blob Detection

**Features:**
- **SimpleBlobDetector**: OpenCV's optimized blob detection
- **Size Filtering**: Configurable blob size ranges
- **Circularity**: Shape-based filtering
- **Real-time Performance**: Optimized for Pi 5

### Color-based Tracking

**Capabilities:**
- **HSV Color Space**: Robust color detection
- **Dynamic Thresholding**: Adaptive color ranges
- **Multiple Color Support**: Track different colored objects
- **Lighting Compensation**: Automatic exposure adjustment

---

## 4. Navigation Integration

![Navigation Integration Overview](media/navigation_integration.png)
<!-- TODO: Create diagram showing vision-navigation integration -->

### Gesture-based Robot Control

**Control Mapping:**
| Gesture | Navigation Command | Robot Action |
|---------|-------------------|--------------|
| 👍 Thumbs Up | `start_navigation` | Begin autonomous navigation |
| 👎 Thumbs Down | `stop_navigation` | Stop all movement |
| ✋ Open Palm | `pause_navigation` | Pause current navigation |
| 👆 Pointing Up | `move_forward` | Move forward briefly |
| 👈 Pointing Left | `turn_left` | Turn left |
| 👉 Pointing Right | `turn_right` | Turn right |
| ✌️ Peace Sign | `follow_person` | Enter person-following mode |
| ✊ Fist | `emergency_stop` | Immediate emergency stop |
| 👋 Wave | `return_home` | Return to home position |

![Gesture Control Demo](media/gesture_control_demo.gif)
<!-- TODO: Record complete gesture control sequence -->

### Safety Systems

**Multi-layered Safety:**
- **Confidence Thresholds**: High confidence required for navigation commands
- **Gesture Stability**: Commands require stable gesture for 0.5 seconds
- **Emergency Override**: Fist gesture immediately stops robot
- **Collision Avoidance**: Object detection feeds into navigation costmap

![Safety Systems Visualization](media/safety_systems.png)
<!-- TODO: Create diagram showing all safety layers -->

### Emergency Stop Features

**Emergency Triggers:**
- **Gesture-based**: Fist gesture for immediate stop
- **Object Detection**: Large obstacle detection
- **System Monitoring**: CPU/memory overload protection
- **Manual Override**: ROS 2 service call emergency stop

---

## 5. Performance & Optimization

### Resource Management

**Adaptive Processing System:**
- **CPU < 60%**: All features enabled
- **CPU 60-75%**: Disable low priority features  
- **CPU 75-90%**: Only high priority features
- **CPU > 90%**: Critical features only

![Resource Management Chart](media/resource_management.png)
<!-- TODO: Create chart showing adaptive processing behavior -->

**Priority Levels:**
- **Priority 0 (Critical)**: Object detection, safety systems
- **Priority 1 (High)**: Gesture recognition, navigation
- **Priority 2 (Medium)**: Hand/pose landmarks, face detection
- **Priority 3 (Low)**: Analysis features, classification

### Adaptive Processing

**Dynamic Feature Control:**
```python
# Automatic feature management based on system load
if cpu_usage > 75:
    disable_low_priority_features()
elif cpu_usage > 60:
    disable_medium_priority_features()
```

### Benchmarking Tools

**Performance Monitoring:**
- **Real-time Metrics**: FPS, processing time, memory usage
- **Historical Analysis**: Performance trends over time
- **Comparative Testing**: Feature-by-feature performance analysis
- **Hardware Profiling**: CPU, GPU, memory utilization

![Performance Dashboard](media/performance_dashboard.png)
<!-- TODO: Screenshot of performance monitoring dashboard -->

---

## 6. Installation & Setup

> **📖 Complete Setup Guide**: For full project setup instructions, see the [main project README](../../../README.md)

### Prerequisites

**System Requirements:**
- **Operating System**: Ubuntu 24.04 LTS (Noble) for Raspberry Pi
- **ROS 2 Distribution**: Jazzy Jalopy
- **Hardware**: Raspberry Pi 5 (8GB recommended) with Pi Camera Module
- **Storage**: 64GB+ MicroSD card (Class 10)

### Package-Specific Setup

**This package requires the standardized ROS 2 + virtual environment workflow:**

#### **Environment Activation (Required)**
```bash
# Option 1: Use convenience script (recommended)
cd ~/GestureBot/gesturebot_ws
source activate_gesturebot.sh

# Option 2: Manual activation
source ~/GestureBot/gesturebot_env/bin/activate  # Virtual env FIRST
source /opt/ros/jazzy/setup.bash                 # ROS 2 SECOND
source ~/GestureBot/gesturebot_ws/install/setup.bash  # Workspace THIRD
```

#### **Build This Package**
```bash
# Ensure environment is activated (see above)
cd ~/GestureBot/gesturebot_ws

# Build gesturebot package
colcon build --packages-select gesturebot

# Source the built package
source install/setup.bash
```

#### **Verify Package Installation**
```bash
# Test package availability
ros2 pkg list | grep gesturebot

# Test Python integration (virtual environment must be active!)
python3 -c "
import rclpy
import mediapipe as mp
from gesturebot.vision_core import MediaPipeProcessor
print('✅ gesturebot package ready!')
"
```

### Dependencies

**This package depends on:**
- **ROS 2 System Packages**: Installed via `rosdep install`
- **Virtual Environment Packages**: MediaPipe, OpenCV (installed via `pip`)
- **Source-Built Tools**: libcamera, rpicam-apps, camera_ros

**Note**: Package.xml has been configured to exclude conflicting camera and MediaPipe system dependencies.
colcon build --packages-select gesturebot

# 4. Source environment
source install/setup.bash
```

---

## 7. Configuration & Usage

### Environment Activation (Required)

**Before using this package, ensure proper environment activation:**
```bash
# Activate virtual environment + ROS 2 + workspace
cd ~/GestureBot/gesturebot_ws
source activate_gesturebot.sh

# Verify environment
python3 -c "import rclpy, mediapipe; print('✅ Environment ready!')"
```

### Launch Files

**Primary Launch Commands:**
```bash
# Full vision system
ros2 launch gesturebot vision_system.launch.py

# Specific features only
ros2 launch gesturebot vision_system.launch.py \
    enable_object_detection:=true \
    enable_gesture_recognition:=true \
    enable_navigation_bridge:=true

# Debug mode with visualization
ros2 launch gesturebot vision_system.launch.py \
    debug_mode:=true
```

### Parameter Tuning

**Configuration File:**
```yaml
# config/vision_params.yaml
vision_system:
  camera_topic: "/camera/image_raw"
  debug_mode: false
  performance_monitoring: true

object_detection_node:
  ros__parameters:
    confidence_threshold: 0.5
    max_results: 5
    enabled: true

gesture_recognition_node:
  ros__parameters:
    confidence_threshold: 0.7
    max_hands: 2
    enabled: true
```

### Topic Monitoring

**Key Topics:**
```bash
# Vision results
ros2 topic echo /vision/objects
ros2 topic echo /vision/gestures
ros2 topic echo /vision/hand_landmarks

# Performance monitoring
ros2 topic echo /vision/*/performance

# Navigation commands
ros2 topic echo /cmd_vel
ros2 topic echo /emergency_stop
```

---

## 8. Development & Testing

### Adding New Features

**Development Workflow:**
1. Create new node inheriting from `MediaPipeBaseNode`
2. Implement required abstract methods
3. Add message definitions if needed
4. Update launch files and configuration
5. Add comprehensive tests

**Example Node Structure:**
```python
from vision_core.base_node import MediaPipeBaseNode

class CustomVisionNode(MediaPipeBaseNode):
    def initialize_mediapipe(self) -> bool:
        # Initialize MediaPipe components
        pass
    
    def process_frame(self, frame, timestamp) -> Any:
        # Process frame with custom algorithm
        pass
    
    def publish_results(self, results, timestamp) -> None:
        # Publish results to ROS 2 topics
        pass
```

### Testing Framework

**Comprehensive Testing:**
```bash
# Ensure environment is activated first
source ~/GestureBot/gesturebot_ws/activate_gesturebot.sh

# Run all MediaPipe feature tests
cd ~/GestureBot/gesturebot_ws/src/gesturebot/test
python3 mediapipe_feature_tester.py

# Quick functionality test
python3 quick_test.py

# Run all tests
python3 run_all_tests.py

# Integration tests
cd ~/GestureBot/gesturebot_ws
colcon test --packages-select gesturebot
```

![Testing Results](media/testing_results.png)
<!-- TODO: Screenshot of test results and performance metrics -->

### Performance Benchmarking

**Benchmark Categories:**
- **Feature Performance**: Individual MediaPipe feature testing
- **System Integration**: Combined feature performance
- **Hardware Utilization**: CPU, memory, and thermal analysis
- **Comparative Analysis**: Before/after optimization comparisons

---

## 9. Troubleshooting

### Common Issues

**MediaPipe Import Error:**
```bash
# Problem: ModuleNotFoundError: No module named 'mediapipe'
# Solution: Ensure virtual environment is activated BEFORE sourcing ROS 2

# ❌ Wrong order (causes error):
source install/setup.bash && python3 -c "import mediapipe"

# ✅ Correct order:
source ~/GestureBot/gesturebot_env/bin/activate  # Virtual env FIRST
source install/setup.bash && python3 -c "import mediapipe"

# ✅ Or use convenience script:
source ~/GestureBot/gesturebot_ws/activate_gesturebot.sh
```

**Camera Not Found:**
```bash
# Check camera hardware (source-built tools)
rpicam-still --list-cameras
which rpicam-still  # Should show /usr/local/bin/rpicam-still

# Verify camera topic (requires camera_ros running)
ros2 run camera_ros camera_node &
ros2 topic list | grep camera
ros2 topic echo /camera/image_raw --once
pkill -f camera_node
```

**Package Import Errors:**
```bash
# Problem: Cannot import gesturebot modules
# Solution: Ensure package is built and workspace is sourced

# Check package installation
ros2 pkg list | grep gesturebot

# Rebuild if necessary
cd ~/GestureBot/gesturebot_ws
colcon build --packages-select gesturebot
source install/setup.bash
```

**MediaPipe Model Missing:**
```bash
# Download missing models
cd ~/GestureBot/gesturebot_ws/src/gesturebot/models/
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite -O efficientdet.tflite
```

### Performance Problems

**Poor FPS Performance:**
```bash
# Check system resources
htop

# Monitor vision performance
ros2 topic echo /vision/object_detection/performance

# Enable adaptive processing
ros2 param set /object_detection_node adaptive_processing true
```

![Performance Troubleshooting](media/performance_troubleshooting.png)
<!-- TODO: Screenshot showing performance monitoring tools -->

### Hardware Debugging

**Pi 5 Specific Issues:**
- **Thermal Throttling**: Ensure adequate cooling
- **Power Supply**: Use official Pi 5 power adapter
- **SD Card Speed**: Use Class 10 or better
- **Camera Connection**: Verify ribbon cable connection

**Debug Commands:**
```bash
# Check system temperature
vcgencmd measure_temp

# Monitor CPU frequency
watch -n 1 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Check memory usage
free -h
```

---

## API Documentation

**Core Classes:**
- `MediaPipeBaseNode`: Base class for all vision nodes
- `MessageConverter`: Utility for ROS 2 message conversion
- `PerformanceMonitor`: System performance tracking

**Message Types:**
- `DetectedObjects`: Object detection results
- `HandGesture`: Gesture recognition output
- `VisionPerformance`: Performance metrics

**Services:**
- `EnableFeature`: Enable/disable vision features
- `ConfigureFeature`: Runtime parameter configuration

---

## Contributing

We welcome contributions from the robotics and computer vision community! 

**Areas for Contribution:**
- Additional MediaPipe features
- Custom ML model integration
- Performance optimizations
- Hardware support expansion
- Documentation improvements

**Development Guidelines:**
- Follow ROS 2 Jazzy best practices
- Maintain modular architecture
- Include comprehensive tests
- Document all new features
- Use virtual environment for ML/CV dependencies
- Preserve source-built camera tool compatibility

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repository if you found it helpful! ⭐

**Built with ❤️ for the robotics community**

**Powered by MediaPipe | Enhanced with ROS 2 Jazzy | Optimized for Pi 5**

---

*GestureBot Vision - Advancing human-robot interaction through computer vision*

![Footer Image](media/footer_banner.png)
<!-- TODO: Create attractive footer banner with project logos -->

</div>
