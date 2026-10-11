# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2026 DAMIAO IMU Driver
file(GLOB_RECURSE PROTOCOL_SRCS *.cpp)

add_library(damiao_imu_protocol
        STATIC
        ${PROTOCOL_SRCS}
)
target_include_directories(damiao_imu_protocol
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(damiao_imu_protocol PUBLIC pthread)
