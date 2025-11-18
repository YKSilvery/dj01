cmake_minimum_required(VERSION 3.8)
project(aim_auto)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(aimbot_msgs REQUIRED)
find_package(Ceres REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)

#配置选项：调试可视化
option(DEBUG_VISUALIZATION "Enable debug visualization" ON)
if(DEBUG_VISUALIZATION)
  add_definitions(-DDEBUG_VISUALIZATION)
endif()

# Component library
# add_library(detector_component SHARED
#   src/detector_node.cpp
#   src/armor_pnp_solver.cpp
# )
# ament_target_dependencies(detector_component rclcpp rclcpp_components sensor_msgs cv_bridge geometry_msgs OpenCV aimbot_msgs std_msgs)
# target_link_libraries(detector_component openvino::runtime Ceres::ceres Eigen3::Eigen)
# target_include_directories(detector_component PUBLIC include)
# rclcpp_components_register_nodes(detector_component "aim_auto::DetectorNode")

add_executable(detector_node
  src/detector_node.cpp
  src/armor_pnp_solver.cpp
)
ament_target_dependencies(detector_node rclcpp sensor_msgs cv_bridge geometry_msgs OpenCV aimbot_msgs std_msgs)
target_link_libraries(detector_node openvino::runtime Ceres::ceres Eigen3::Eigen)
target_include_directories(detector_node PUBLIC include)

add_executable(detector_ref_node
  src/detector_ref_node.cpp
  src/ref_detection.cpp
  src/armor_pnp_solver.cpp
)
ament_target_dependencies(detector_ref_node rclcpp sensor_msgs cv_bridge geometry_msgs OpenCV aimbot_msgs std_msgs)
target_link_libraries(detector_ref_node Ceres::ceres Eigen3::Eigen)
target_include_directories(detector_ref_node PUBLIC include)

add_executable(tracker_node
  src/tracker_node.cpp
  src/ekf_tracker.cpp
)
ament_target_dependencies(tracker_node rclcpp geometry_msgs OpenCV aimbot_msgs visualization_msgs)
target_link_libraries(tracker_node Eigen3::Eigen)
target_include_directories(tracker_node PUBLIC include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  detector_node
  detector_ref_node
  tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

# install(TARGETS
#   detector_component
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

ament_package()
