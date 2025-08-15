cmake_minimum_required(VERSION 3.8)
project(gesturebot)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/DetectedObjects.msg"
  "msg/HandGesture.msg"
  "msg/HandLandmarks.msg"
  "msg/PoseLandmarks.msg"
  "msg/FaceDetection.msg"
  "msg/VisionPerformance.msg"
  "msg/GestureCommand.msg"
  "srv/EnableFeature.srv"
  "srv/ConfigureFeature.srv"
  "action/TrackObject.action"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs nav2_msgs action_msgs
)

# Include directories
include_directories(include)

# Install Python modules
ament_python_install_package(vision_core PACKAGE_DIR src/vision_core)

# Install Python executables
install(PROGRAMS
  src/nodes/object_detection_node.py
  src/nodes/gesture_recognition_node.py
  src/nodes/gesture_navigation_bridge.py
  src/nodes/image_viewer_node.py
  src/nodes/pose_detection_node.py
  src/nodes/pose_navigation_bridge.py
  src/nodes/person_following_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install test scripts
install(PROGRAMS
  test/test_mediapipe_models.py
  test/test_camera_integration.py
  test/test_vision_nodes.py
  test/quick_test.py
  test/run_all_tests.py
  test/mediapipe_feature_tester.py
  DESTINATION lib/${PROJECT_NAME}/test
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install models directory (for MediaPipe models)
install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  
  # Linting
  ament_lint_auto_find_test_dependencies()
  
  # Python tests (using actual test files)
  ament_add_pytest_test(test_mediapipe_models test/test_mediapipe_models.py)
  ament_add_pytest_test(test_camera_integration test/test_camera_integration.py)
  ament_add_pytest_test(test_vision_nodes test/test_vision_nodes.py)
endif()

ament_package()
