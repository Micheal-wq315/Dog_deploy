// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_serial_port.cpp
 * @brief USB serial port implementation for DAMIAO IMU communication.
 * @details Implements UART serial port opening, configuration, and
 *          asynchronous reading with callback-based data delivery.
 */

#include "damiao_serial_port.hpp"

DamiaoSerialPort::DamiaoSerialPort(const std::string& interface, int baudrate)
    : interface_(interface), baudrate_(baudrate), fd_(-1), running_(false) {
    init();
}

void DamiaoSerialPort::init() {
    fd_ = ::open(interface_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        std::cerr << "Failed to open serial port: " << interface_
                  << " (" << strerror(errno) << ")" << std::endl;
        throw std::runtime_error("Failed to open serial port: " + interface_);
    }

    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "Failed to get serial attributes: " << interface_
                  << " (" << strerror(errno) << ")" << std::endl;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to get serial attributes: " + interface_);
    }

    // ── Configure baud rate ──────────────────────────────────────
    speed_t speed;
    switch (baudrate_) {
        case 9600:    speed = B9600;    break;
        case 19200:   speed = B19200;   break;
        case 38400:   speed = B38400;   break;
        case 57600:   speed = B57600;   break;
        case 115200:  speed = B115200;  break;
        case 230400:  speed = B230400;  break;
        case 460800:  speed = B460800;  break;
        case 921600:  speed = B921600;  break;
        default:      speed = B460800;  break;  // DAMIAO default
    }

    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    // ── 8N1, no flow control ─────────────────────────────────────
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CRTSCTS;

    // ── Raw mode ─────────────────────────────────────────────────
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "Failed to set serial attributes: " << interface_
                  << " (" << strerror(errno) << ")" << std::endl;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to set serial attributes: " + interface_);
    }

    // ── Start async read thread ──────────────────────────────────
    running_ = true;
    rx_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "damiao_imu_rx");
        struct sched_param sp{};
        sp.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            std::cerr << "Warning: Failed to set realtime priority for IMU serial RX" << std::endl;
        }
        uint8_t buf[DAMIAO_BUF_SIZE] = {0};

        while (running_) {
            fd_set readfds;
            struct timeval tv;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);
            tv.tv_sec  = 0;
            tv.tv_usec = 1000;  // 1 ms timeout

            int ret = select(fd_ + 1, &readfds, NULL, NULL, &tv);
            if (ret < 0) {
                if (errno == EINTR) continue;
                if (errno == EBADF) break;  // fd was closed by close()
                std::cerr << "select error: " << strerror(errno) << std::endl;
                break;
            } else if (ret == 0) {
                continue;
            }

            int n = ::read(fd_, buf, DAMIAO_BUF_SIZE);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                std::cerr << "Serial read error: " << strerror(errno) << std::endl;
                break;
            } else if (n > 0) {
                if (callback_) {
                    callback_(buf, static_cast<size_t>(n));
                }
            }
        }
    });
}

DamiaoSerialPort::~DamiaoSerialPort() {
    close();
}

std::shared_ptr<DamiaoSerialPort> DamiaoSerialPort::open(
    const std::string& interface, int baudrate) {
    return std::shared_ptr<DamiaoSerialPort>(new DamiaoSerialPort(interface, baudrate));
}

int DamiaoSerialPort::write(const uint8_t *data, size_t len) {
    if (fd_ < 0) return -1;
    int n = ::write(fd_, data, len);
    if (n < 0) {
        std::cerr << "Serial write error: " << strerror(errno) << std::endl;
    }
    return n;
}

int DamiaoSerialPort::read(uint8_t *buf, size_t max_len) {
    if (fd_ < 0) return -1;
    int n = ::read(fd_, buf, max_len);
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        std::cerr << "Serial read error: " << strerror(errno) << std::endl;
    }
    return n;
}

void DamiaoSerialPort::flush() {
    if (fd_ >= 0) {
        tcflush(fd_, TCIOFLUSH);
    }
}

void DamiaoSerialPort::close() {
    running_ = false;

    // Close the file descriptor FIRST to unblock select() in the
    // RX thread.  This guarantees rx_thread_ exits promptly rather
    // than waiting for a timeout or a stuck publish() call.
    if (fd_ >= 0) {
        if (::close(fd_) < 0) {
            std::cerr << "Warning: Failed to close serial port " << interface_
                      << ": " << strerror(errno) << std::endl;
        }
        fd_ = -1;
    }

    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }
}

void DamiaoSerialPort::set_serial_callback(SerialCbkFunc callback) {
    callback_ = std::move(callback);
}
