// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_imu_driver.hpp
 * @brief DAMIAO IMU driver — USB serial backend.
 * @details Implements the IMUDriver interface for DAMIAO IMUs connected
 *          via USB virtual COM port. Handles device configuration
 *          (enter settings → enable outputs → save → enter normal mode)
 *          and parses the binary frame protocol in real time.
 */

#pragma once

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <cmath>

#include "damiao_imu/imu_driver.hpp"
#include "protocol/damiao/damiao_serial_port.hpp"
#include "damiao_protocol.h"

#define GRA_ACC     (9.80665)
#define DEG_TO_RAD  (0.017453292519943295)
#define RAD_TO_DEG  (57.29577951308232)

class DamiaoIMUDriver : public IMUDriver {
   public:
    DamiaoIMUDriver(uint16_t imu_id,
                    const std::string& interface_type,
                    const std::string& interface,
                    const int baudrate = 460800);
    ~DamiaoIMUDriver();

    // ── Overridden accessors (thread-safe) ──────────────────────────
    std::vector<float> get_ang_vel() override;
    std::vector<float> get_quat() override;
    std::vector<float> get_lin_acc() override;
    std::vector<float> get_euler() override;
    float get_temperature() override;

   private:
    // ── Serial RX callback (runs on the serial RX thread) ───────────
    void serial_rx_cbk(const uint8_t* data, size_t length);

    // ── Protocol frame parser (byte-by-byte state machine) ──────────
    void parse_byte(uint8_t byte);

    // ── Process a complete frame ────────────────────────────────────
    void process_frame();

    // ── Device configuration sequence ───────────────────────────────
    bool send_command(const uint8_t* cmd, size_t len);
    bool configure_device();

    // ── Fields ──────────────────────────────────────────────────────
    int baudrate_;
    std::string interface_type_;
    std::string interface_;

    mutable std::shared_mutex imu_mutex_;

    std::shared_ptr<DamiaoSerialPort> serial_;
    damiao_parser_t parser_;

    // Per-message-type temp buffers (accumulated into sensor_data_)
    float tmp_accel_[3]  = {0.f, 0.f, 0.f};
    float tmp_gyro_[3]   = {0.f, 0.f, 0.f};
    float tmp_euler_[3]  = {0.f, 0.f, 0.f};
    float tmp_quat_[4]   = {0.f, 0.f, 0.f, 0.f};

    bool accel_valid_  = false;
    bool gyro_valid_   = false;
    bool euler_valid_  = false;
    bool quat_valid_   = false;
};
