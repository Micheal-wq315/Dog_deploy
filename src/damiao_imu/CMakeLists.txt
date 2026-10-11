# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2026 DAMIAO IMU Driver
cmake_minimum_required(VERSION 3.12)

file(READ "package.xml" PACKAGE_XML_CONTENT)
string(REGEX MATCH "<version>([0-9\\.]+)</version>" _ ${PACKAGE_XML_CONTENT})
project(damiao_imu VERSION ${CMAKE_MATCH_1})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ── Custom message ──────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DamiaoImuData.msg"
  DEPENDENCIES std_msgs
)

# ── Common dependencies ─────────────────────────────────────────────
set(PUBLIC_DEPENDENCIES pthread)

# ── Subdirectories ──────────────────────────────────────────────────
add_subdirectory(src)

# ── IMU library (static) ────────────────────────────────────────────
add_library(damiao_imu_lib STATIC
  src/imu_driver.cpp
)

target_include_directories(damiao_imu_lib
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(damiao_imu_lib PUBLIC
  ${PUBLIC_DEPENDENCIES}
  damiao_imu_driver
  damiao_imu_protocol
)

# ── ROS2 publisher node ─────────────────────────────────────────────
add_executable(damiao_imu_node
  src_node/damiao_imu_node.cpp
)

target_include_directories(damiao_imu_node
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

ament_target_dependencies(damiao_imu_node
  rclcpp
  sensor_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(damiao_imu_node
  damiao_imu_lib
  ${PUBLIC_DEPENDENCIES}
  "${cpp_typesupport_target}"
)

# ── Install ─────────────────────────────────────────────────────────
install(DIRECTORY include/ DESTINATION include)

install(TARGETS
  damiao_imu_lib
  damiao_imu_driver
  damiao_imu_protocol
  damiao_imu_node
  EXPORT damiao_imu-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_export_libraries(damiao_imu_lib damiao_imu_driver damiao_imu_protocol)
ament_export_include_directories(include)
ament_export_dependencies()
ament_export_targets(damiao_imu-targets)
ament_package()
