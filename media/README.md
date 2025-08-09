# GestureBot Vision System Media Assets

This directory contains media assets for documenting the GestureBot vision system capabilities and performance.

## 📁 Directory Structure

### `/demos/` - Video Demonstrations
GIF and video files showcasing system functionality:
- Core MediaPipe features (gesture recognition, object detection, pose estimation)
- OpenCV integration (ball tracking, blob detection, color tracking)
- Navigation integration (gesture control, emergency stop)
- Setup and quick start demonstrations

### `/hardware/` - Hardware Documentation
Photos and images of physical hardware setup:
- Complete hardware setup with labeled components
- Raspberry Pi 5 closeup with cooling and connections
- Pi Camera Module 3 mounting and positioning
- Cable management and clean routing

### `/system_architecture/` - Technical Diagrams
System architecture and flow diagrams:
- Comprehensive system architecture showing all components
- Processing pipeline flowcharts
- Gesture-to-navigation command mapping
- Safety systems and emergency procedures
- Vision-navigation integration overview

### `/benchmarks/` - Performance Documentation
Charts and visualizations of system performance:
- Performance benchmark charts for different feature combinations
- Resource management and adaptive processing behavior
- FPS performance across different Pi 5 configurations
- Memory usage analysis and optimization results
- Performance monitoring dashboard screenshots
- Testing results and validation metrics

### `/troubleshooting/` - Debug Documentation
Screenshots and examples for troubleshooting:
- Performance monitoring tools and outputs
- Common error messages and solutions
- Hardware debugging tools and procedures
- Debug mode visualizations with detailed overlays

## 🎯 Current Status

**Vision System**: ✅ Fully operational and validated
- 27 FPS camera performance with dual-stream capability
- 100% test suite pass rate (all 4 test suites complete)
- 3.7ms average vision processing time
- Complete MediaPipe + ROS 2 integration

**Media Assets**: 📋 Directory structure ready for content creation
- All subdirectories created with proper organization
- .gitkeep files ensure version control preservation
- README references prepared for media integration

## 📝 Notes

The media directory structure supports the 24 media references in the main README.md file. Media assets can be added incrementally to showcase the validated vision system capabilities.

**Validated Performance Data Available:**
- Camera: 27 FPS dual-stream (640×480, 25.9 KB compressed images)
- Vision Processing: 3.7ms average processing time
- Test Results: 100% pass rate across all test suites
- MediaPipe Models: 123ms inference time, all models functional
