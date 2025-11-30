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

# Install Python executables (vision nodes now provided by ros2_mediapipe)
install(PROGRAMS
  src/nodes/gesture_navigation_bridge.py
  src/nodes/image_viewer_node.py
  src/nodes/pose_navigation_bridge.py
  src/nodes/person_following_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install docs directory
install(DIRECTORY
  docs/
  DESTINATION share/${PROJECT_NAME}/docs
  OPTIONAL
)

# Export dependencies
ament_export_include_directories(include)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
