# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2026 DAMIAO IMU Driver
AUX_SOURCE_DIRECTORY(. SOURCE_LIST_DAMIAO_IMU)
add_library(damiao_imu_driver
        STATIC
        ${SOURCE_LIST_DAMIAO_IMU}
)
target_include_directories(damiao_imu_driver
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(damiao_imu_driver PUBLIC pthread damiao_imu_protocol)
