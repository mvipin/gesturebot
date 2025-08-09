# GestureBot Scripts Directory

This directory contains utility scripts for setting up, configuring, and managing the GestureBot vision system.

## 📁 Scripts Overview

### `setup_workspace.sh`
**Main workspace setup script** - Comprehensive setup for the entire GestureBot vision system.

**Features:**
- ✅ System dependency installation (ROS 2, OpenCV, MediaPipe)
- ✅ Python package installation and configuration
- ✅ MediaPipe model downloads (EfficientDet, Hand Landmarker, Gesture Recognizer)
- ✅ ROS 2 workspace building and configuration
- ✅ Environment setup with convenient aliases
- ✅ Installation validation and testing
- ✅ Raspberry Pi 5 optimization

**Usage:**
```bash
# From workspace root
cd camera_ws
./setup.sh

# Or directly from scripts directory
cd camera_ws/src/gesturebot/scripts
./setup_workspace.sh

# With comprehensive testing
./setup_workspace.sh --with-tests
```

## 🚀 Quick Setup Guide

### 1. Initial Setup
```bash
# Navigate to workspace
cd ~/GestureBot/camera_ws

# Run setup script
./setup.sh
```

### 2. Environment Activation
```bash
# Source environment (run after setup)
source setup_env.sh

# Load convenient aliases
source aliases.sh
```

### 3. Verify Installation
```bash
# Quick system validation
quick_test

# Full test suite
test_all
```

### 4. Launch System
```bash
# Launch complete vision system
launch_vision

# Launch specific features
launch_object_detection
launch_gesture_recognition
```

## 🔧 Setup Script Details

### System Dependencies Installed
- **ROS 2 Humble**: Core robotics framework
- **OpenCV**: Computer vision library
- **MediaPipe**: Google's ML framework for perception
- **NumPy, SciPy**: Scientific computing
- **psutil**: System monitoring
- **rpicam-apps**: Raspberry Pi camera tools (Pi only)

### Python Dependencies
- **mediapipe**: Core MediaPipe functionality
- **dataclasses**: Data structure utilities
- **typing-extensions**: Type hint extensions

### MediaPipe Models Downloaded
- **EfficientDet Lite**: Object detection (6.2MB)
- **Hand Landmarker**: Hand pose estimation (26.8MB)
- **Gesture Recognizer**: Hand gesture classification (13.3MB)

### Directory Structure Created
```
camera_ws/
├── src/gesturebot/
│   ├── models/              # MediaPipe models
│   ├── src/
│   │   ├── vision_core/     # Shared utilities
│   │   └── nodes/           # Individual processing nodes
│   ├── test/                # Test suite
│   ├── config/              # Configuration files
│   ├── launch/              # ROS 2 launch files
│   └── scripts/             # Utility scripts (this directory)
├── build/                   # Build artifacts
├── install/                 # Installation files
└── log/                     # Build logs
```

## 🎯 Environment Variables Set

After running the setup script and sourcing the environment:

- **`GESTUREBOT_WORKSPACE`**: Path to workspace root
- **`GESTUREBOT_MODELS`**: Path to MediaPipe models directory
- **`GESTUREBOT_CONFIG`**: Path to configuration files
- **`PYTHONPATH`**: Updated to include vision_core and nodes

## 🔗 Convenient Aliases Created

### Launch Commands
- `launch_vision` - Launch complete vision system
- `launch_object_detection` - Object detection only
- `launch_gesture_recognition` - Gesture recognition only

### Testing Commands
- `quick_test` - Fast system validation
- `test_mediapipe` - Test MediaPipe models
- `test_camera` - Test camera integration
- `test_vision_nodes` - Test vision processing nodes
- `test_all` - Run complete test suite

### Monitoring Commands
- `monitor_performance` - Monitor system performance
- `monitor_gestures` - Monitor gesture recognition
- `monitor_objects` - Monitor object detection

### Development Commands
- `build_gesturebot` - Build the gesturebot package
- `test_gesturebot` - Run ROS 2 package tests
- `start_nav_bridge` - Start navigation bridge

## 🛠️ Troubleshooting

### Common Setup Issues

**Permission Denied**
```bash
chmod +x camera_ws/src/gesturebot/scripts/setup_workspace.sh
```

**ROS 2 Not Found**
```bash
# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop
```

**MediaPipe Installation Failed**
```bash
# Install MediaPipe manually
pip3 install --user mediapipe

# Check installation
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

**Model Download Failed**
```bash
# Check internet connection
ping google.com

# Download models manually
cd camera_ws/src/gesturebot/models
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite -O efficientdet.tflite
```

**Build Failed**
```bash
# Check ROS 2 environment
echo $ROS_DISTRO

# Clean and rebuild
cd camera_ws
rm -rf build install log
colcon build --packages-select gesturebot
```

### Raspberry Pi Specific Issues

**Camera Not Detected**
```bash
# Check camera connection
rpicam-hello --list-cameras

# Install camera tools
sudo apt install rpicam-apps
```

**Insufficient Memory**
```bash
# Check available memory
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**Thermal Throttling**
```bash
# Check temperature
vcgencmd measure_temp

# Monitor throttling
vcgencmd get_throttled
```

## 📝 Script Customization

### Modifying Setup Parameters
Edit the setup script to customize:
- Model download URLs
- Dependency versions
- Directory structure
- Environment variables

### Adding Custom Models
```bash
# Add to download_models() function
if [[ ! -f "$MODELS_DIR/custom_model.tflite" ]]; then
    print_status "Downloading custom model..."
    wget -q --show-progress \
        "https://example.com/custom_model.tflite" \
        -O "$MODELS_DIR/custom_model.tflite"
fi
```

### Platform-Specific Optimizations
```bash
# Add to check_raspberry_pi() function
if [[ "$RPI_PLATFORM" == "true" ]]; then
    # Pi-specific optimizations
    export OPENCV_THREAD_COUNT=4
    export MEDIAPIPE_DISABLE_GPU=1
fi
```

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Re-run setup to update dependencies
cd camera_ws
./setup.sh

# Update models only
cd src/gesturebot/scripts
./setup_workspace.sh --models-only  # (if implemented)
```

### Backup Configuration
```bash
# Backup important files
tar -czf gesturebot_backup.tar.gz \
    src/gesturebot/config/ \
    src/gesturebot/models/ \
    setup_env.sh \
    aliases.sh
```

---

**Last Updated**: Current Date  
**Script Version**: 2.0.0  
**Compatible with**: GestureBot v2.0.0, ROS 2 Humble
