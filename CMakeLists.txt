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
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# Messages now provided by ros2_mediapipe package

# Include directories
include_directories(include)

# Install Python modules
ament_python_install_package(vision_core PACKAGE_DIR src/vision_core)

# Install Python executables (vision nodes now provided by ros2_mediapipe)
install(PROGRAMS
  src/nodes/gesture_navigation_bridge.py
  src/nodes/image_viewer_node.py
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
