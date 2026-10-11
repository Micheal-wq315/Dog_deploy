// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file imu_driver.hpp
 * @brief Abstract IMU driver interface and sensor data structures.
 * @details Defines the IMUDriver base class and packed sensor data structs.
 *          Supports multiple IMU backends via factory pattern.
 */

#pragma once

#include <stdint.h>
#include <string.h>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/// @brief Packed sensor data shared across all driver backends.
struct SensorData {
    float acc_x  = 0.f;
    float acc_y  = 0.f;
    float acc_z  = 0.f;
    float gyr_x  = 0.f;
    float gyr_y  = 0.f;
    float gyr_z  = 0.f;
    float roll   = 0.f;
    float pitch  = 0.f;
    float yaw    = 0.f;
    float quat_w = 0.f;
    float quat_x = 0.f;
    float quat_y = 0.f;
    float quat_z = 0.f;
    float temperature = 0.f;
    uint8_t device_id = 0;
};

class IMUDriver {
   public:
    IMUDriver() = default;
    virtual ~IMUDriver() = default;

    /// @brief Factory: create an IMU driver instance by type.
    /// @param imu_id        Logical ID for this IMU (used for multi-IMU setups).
    /// @param interface_type "serial" or "can".
    /// @param interface      Device path (e.g., /dev/ttyUSB0) or CAN iface name.
    /// @param imu_type       Backend identifier (e.g., "DAMIAO").
    /// @param baudrate       Baud rate for serial (ignored for CAN).
    static std::shared_ptr<IMUDriver> create_imu(
        uint16_t imu_id,
        const std::string& interface_type,
        const std::string& interface,
        const std::string& imu_type,
        const int baudrate = 0);

    virtual uint16_t get_imu_id() { return imu_id_; }

    virtual std::vector<float> get_ang_vel() { return ang_vel_; }
    virtual std::vector<float> get_quat()    { return quat_; }
    virtual std::vector<float> get_lin_acc() { return lin_acc_; }
    virtual std::vector<float> get_euler()   { return euler_; }
    virtual float get_temperature()          { return temperature_; }

    /// @brief Set a callback invoked when new sensor data arrives.
    virtual void set_data_callback(std::function<void(const SensorData&)> cb) {
        data_callback_ = std::move(cb);
    }

   protected:
    uint16_t imu_id_ = 0;

    std::vector<float> quat_{0.f, 0.f, 0.f, 0.f};  // w, x, y, z
    std::vector<float> ang_vel_{0.f, 0.f, 0.f};    // x, y, z  (rad/s)
    std::vector<float> lin_acc_{0.f, 0.f, 0.f};    // x, y, z  (m/s²)
    std::vector<float> euler_{0.f, 0.f, 0.f};      // roll, pitch, yaw (rad)
    float temperature_{0.f};

    SensorData sensor_data_;
    std::function<void(const SensorData&)> data_callback_;
};
