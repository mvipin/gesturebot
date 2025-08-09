# GestureBot Vision System Test Suite

Comprehensive test suite for validating the GestureBot vision system functionality, performance, and integration.

## 🧪 Test Overview

The test suite validates the complete vision pipeline from camera input to ROS 2 message output, ensuring reliable operation of all MediaPipe features and system integration.

### Test Categories

1. **MediaPipe Model Tests** (`test_mediapipe_models.py`)
   - Model file availability and loading
   - MediaPipe initialization and basic functionality
   - Performance benchmarking of model inference

2. **Camera Integration Tests** (`test_camera_integration.py`)
   - Camera image capture and publishing
   - ROS 2 image message format validation
   - Mock camera functionality for hardware-independent testing

3. **Vision Node Tests** (`test_vision_nodes.py`)
   - End-to-end vision processing pipeline
   - ROS 2 node integration and communication
   - Message publishing and topic validation

4. **Performance Benchmarks** (`test_performance_benchmarks.py`)
   - System resource usage monitoring
   - Processing time and FPS measurements
   - Memory usage and stability testing

## 🚀 Quick Start

### Run Quick Validation
```bash
# Fast system check (30 seconds)
cd camera_ws/src/gesturebot/test
python3 quick_test.py
```

### Run Individual Tests
```bash
# Test MediaPipe models
python3 test_mediapipe_models.py

# Test camera integration
python3 test_camera_integration.py

# Test vision nodes
python3 test_vision_nodes.py
```

### Run Complete Test Suite
```bash
# Run all tests with detailed reporting
python3 run_all_tests.py
```

### Run with colcon (ROS 2 standard)
```bash
# From workspace root
cd camera_ws
colcon test --packages-select gesturebot
colcon test-result --verbose
```

## 📋 Test Requirements

### System Prerequisites
- **ROS 2 Humble** or later
- **Python 3.8+** with required packages
- **MediaPipe** (`pip install mediapipe`)
- **OpenCV** (`sudo apt install python3-opencv`)
- **NumPy, psutil** (for performance monitoring)

### Hardware Requirements
- **Raspberry Pi 5** (recommended) or compatible system
- **8GB RAM** minimum for full test suite
- **Camera** (optional - tests use mock camera by default)

### Model Files
The following MediaPipe models should be present in `models/` directory:
- `efficientdet.tflite` (required for object detection tests)
- `hand_landmarker.task` (optional for hand tracking tests)
- `gesture_recognizer.task` (optional for gesture recognition tests)

## 🔧 Test Configuration

### Configuration File
Edit `test_config.yaml` to customize test parameters:

```yaml
performance_thresholds:
  max_processing_time: 200.0  # milliseconds
  min_fps: 5.0
  max_cpu_usage: 80.0  # percentage
  max_memory_usage: 2048.0  # MB

test_images:
  resolution: [640, 480]
  count: 5
  complexity_levels: ["simple", "medium", "complex"]
```

### Environment Variables
```bash
# Optional: Set test-specific parameters
export GESTUREBOT_TEST_TIMEOUT=300
export GESTUREBOT_TEST_VERBOSE=true
export GESTUREBOT_MODELS_DIR=/path/to/models
```

## 📊 Test Results

### Success Criteria
Tests pass when:
- ✅ All required models load successfully
- ✅ Camera images publish at expected rate (>5 FPS)
- ✅ Vision processing completes within time limits (<200ms)
- ✅ ROS 2 messages publish correctly
- ✅ Memory usage remains stable
- ✅ No critical errors or exceptions

### Result Files
Test results are saved to `test/results/`:
- `test_results_<timestamp>.json` - Detailed test results
- `performance_data_<timestamp>.json` - Performance metrics
- `error_logs_<timestamp>.txt` - Error details (if any)

### Example Output
```
=== GESTUREBOT VISION SYSTEM TEST SUMMARY ===
Test Modules: 3
Successful Modules: 3
Total Tests Run: 15
Failures: 0
Errors: 0
Success Rate: 100.0%

✅ PASS test_mediapipe_models: 6 tests in 12.3s
✅ PASS test_camera_integration: 4 tests in 8.7s  
✅ PASS test_vision_nodes: 5 tests in 15.2s

🎉 ALL TESTS PASSED! 🎉
```

## 🐛 Troubleshooting

### Common Issues

**MediaPipe Import Error**
```bash
# Install MediaPipe
pip3 install mediapipe

# Check installation
python3 -c "import mediapipe; print(mediapipe.__version__)"
```

**ROS 2 Environment Not Found**
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Verify environment
echo $ROS_DISTRO
```

**Model Files Missing**
```bash
# Download models using setup script
cd camera_ws
./setup_workspace.sh

# Or manually download EfficientDet
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite -O models/efficientdet.tflite
```

**Performance Tests Failing**
- Check system resources: `htop`
- Reduce test complexity in `test_config.yaml`
- Ensure adequate cooling for Pi 5
- Close unnecessary applications

### Debug Mode
Run tests with additional debugging:
```bash
# Enable verbose output
python3 test_mediapipe_models.py -v

# Run with Python debugger
python3 -m pdb test_vision_nodes.py
```

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
# .github/workflows/test.yml
name: GestureBot Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup ROS 2
        uses: ros-tooling/setup-ros@v0.3
      - name: Run Tests
        run: |
          cd camera_ws/src/gesturebot/test
          python3 run_all_tests.py
```

### Local Pre-commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
cd camera_ws/src/gesturebot/test
python3 quick_test.py
```

## 📈 Performance Benchmarks

### Expected Performance (Pi 5)
- **Object Detection**: ~15 FPS, 97ms processing
- **Gesture Recognition**: ~12 FPS, 80ms processing
- **Hand Landmarks**: ~10 FPS, 100ms processing
- **Memory Usage**: <1.5GB peak
- **CPU Usage**: <70% average

### Benchmark Commands
```bash
# Run performance benchmarks
python3 test_performance_benchmarks.py

# Monitor system resources during tests
htop &
python3 run_all_tests.py
```

## 🤝 Contributing

### Adding New Tests
1. Create test file following naming convention: `test_<feature>.py`
2. Inherit from `unittest.TestCase`
3. Add test to `run_all_tests.py` module list
4. Update this README with test description

### Test Guidelines
- Use descriptive test method names: `test_<component>_<functionality>`
- Include both positive and negative test cases
- Mock external dependencies when possible
- Add performance assertions for critical paths
- Document expected behavior in docstrings

---

**Last Updated**: Current Date  
**Test Suite Version**: 1.0.0  
**Compatible with**: GestureBot v1.0.0, ROS 2 Humble
