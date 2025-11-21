cmake_minimum_required(VERSION 3.8)
project(camera)

# 配置选项(使用视频文件或硬件相机)
option(CAMERA_USE_VIDEO_SOURCE "Use prerecorded video files instead of the hardware camera" ON)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(aimbot_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(console_bridge REQUIRED)

if(CAMERA_USE_VIDEO_SOURCE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
endif()

# 海康相机SDK路径
set(MVS_SDK_DIR /opt/MVS)
set(MVS_INCLUDE_DIR ${MVS_SDK_DIR}/include)
set(MVS_LIB_DIR ${MVS_SDK_DIR}/lib/64)

# 创建组件库
add_library(camera_component SHARED src/camera.cpp)
target_compile_features(camera_component PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(camera_component rclcpp rclcpp_components sensor_msgs aimbot_msgs ament_index_cpp)
target_include_directories(camera_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
rclcpp_components_register_nodes(camera_component "camera::CameraNode")

# 创建可执行文件（用于独立运行）
add_executable(camera_node src/camera.cpp)

# 设置编译特性
target_compile_features(camera_node PUBLIC c_std_99 cxx_std_17)

# 添加依赖和链接库
ament_target_dependencies(camera_node rclcpp rclcpp_components sensor_msgs aimbot_msgs ament_index_cpp)

# 设置包含目录
target_include_directories(camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(CAMERA_USE_VIDEO_SOURCE)
  target_compile_definitions(camera_component PUBLIC CAMERA_USE_VIDEO_STREAM)
  target_compile_definitions(camera_node PUBLIC CAMERA_USE_VIDEO_STREAM)
  target_link_libraries(camera_component ${OpenCV_LIBS})
  target_link_libraries(camera_node ${OpenCV_LIBS})
else()
  target_link_libraries(camera_component ${MVS_LIB_DIR}/libMvCameraControl.so)
  target_link_libraries(camera_node ${MVS_LIB_DIR}/libMvCameraControl.so)
  target_include_directories(camera_component SYSTEM PRIVATE ${MVS_INCLUDE_DIR})
  target_include_directories(camera_node SYSTEM PRIVATE ${MVS_INCLUDE_DIR})
endif()

# 设置运行时库路径
set_target_properties(camera_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)

# 安装规则
install(TARGETS camera_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装launch文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY video/
  DESTINATION share/${PROJECT_NAME}/video
)

# 安装SDK库文件
install(DIRECTORY ${MVS_LIB_DIR}/
  DESTINATION lib
  FILES_MATCHING
    PATTERN "*.so*"
    PATTERN "ThirdParty" EXCLUDE
)

# 安装SDK配置文件
install(DIRECTORY ${MVS_LIB_DIR}/
  DESTINATION lib
  FILES_MATCHING
    PATTERN "*.ini"
    PATTERN "*.cti"
    PATTERN "ThirdParty" EXCLUDE
)

# 测试配置
# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   set(ament_cmake_copyright_FOUND TRUE)
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

ament_package()
