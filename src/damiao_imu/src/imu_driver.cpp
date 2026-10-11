// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file imu_driver.cpp
 * @brief Factory implementation for creating IMU driver instances.
 * @details Provides IMUDriver::create_imu() to instantiate the appropriate
 *          driver backend based on configuration.
 */

#include "damiao_imu/imu_driver.hpp"
#include "drivers/damiao/damiao_imu_driver.hpp"

#include <stdexcept>

std::shared_ptr<IMUDriver> IMUDriver::create_imu(
    uint16_t imu_id,
    const std::string& interface_type,
    const std::string& interface,
    const std::string& imu_type,
    const int baudrate) {

    if (imu_type == "DAMIAO") {
        return std::make_shared<DamiaoIMUDriver>(
            imu_id, interface_type, interface, baudrate);
    } else {
        throw std::runtime_error("IMU type '" + imu_type + "' not supported");
    }
}
