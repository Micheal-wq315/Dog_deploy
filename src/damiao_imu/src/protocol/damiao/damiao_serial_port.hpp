// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_serial_port.hpp
 * @brief USB serial port interface for DAMIAO IMU communication.
 * @details Provides a UART serial port abstraction with asynchronous read
 *          callback support for receiving IMU data streams over USB.
 */

#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <iostream>
#include <cerrno>
#include <termios.h>
#include <pthread.h>

#define DAMIAO_BUF_SIZE 4096

class DamiaoSerialPort {
public:
    using SerialCbkFunc = std::function<void(const uint8_t*, size_t)>;

    DamiaoSerialPort(const DamiaoSerialPort &) = delete;
    DamiaoSerialPort &operator=(const DamiaoSerialPort &) = delete;
    ~DamiaoSerialPort();

    static std::shared_ptr<DamiaoSerialPort> open(
        const std::string& interface, int baudrate);

    void set_serial_callback(SerialCbkFunc callback);
    void close();

    /// @brief Write raw bytes to the serial port.
    /// @return Number of bytes written, or -1 on error.
    int write(const uint8_t *data, size_t len);

    /// @brief Read raw bytes from the serial port (non-blocking).
    /// @return Number of bytes read, or -1 on error.
    int read(uint8_t *buf, size_t max_len);

    /// @brief Flush the serial port buffers.
    void flush();

    /// @brief Check if the port is open and the read thread is running.
    bool is_open() const { return fd_ >= 0 && running_; }

private:
    DamiaoSerialPort(const std::string& interface, int baudrate);
    void init();

    std::string interface_;
    int baudrate_;
    int fd_;
    std::atomic<bool> running_;
    std::thread rx_thread_;
    SerialCbkFunc callback_;
};
