// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_imu_driver.cpp
 * @brief DAMIAO IMU driver implementation.
 * @details Full implementation of the DAMIAO USB-serial IMU driver:
 *          - CRC-16 routines
 *          - Byte-by-byte frame parser (state machine)
 *          - Device configuration sequence (enter settings → enable → save → normal)
 *          - Thread-safe sensor data access
 */

#include "damiao_imu_driver.hpp"

#include <cstdio>
#include <chrono>

/* ═══════════════════════════════════════════════════════════════════
   CRC-16 routines
   ═══════════════════════════════════════════════════════════════════ */

// CRC-16/CCITT lookup table (polynomial 0x1021) — matches DAMIAO IMU
static const uint16_t crc16_table[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
    0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
    0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
    0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
    0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
    0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
    0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
    0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
    0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
    0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
    0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
    0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
    0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
    0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
    0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
    0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
    0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
    0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
    0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
    0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
    0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
};

/// DAMIAO CRC-16 (polynomial 0x1021, lookup-table).
/// @param shift  1 (matches official Get_CRC16) or 8 (standard table usage).
/// @param init   Initial CRC value (0x0000 or 0xFFFF).
static uint16_t damiao_crc16_ex(const uint8_t *ptr, uint16_t len,
                                 int shift, uint16_t init) {
    uint16_t crc = init;
    for (uint16_t i = 0; i < len; i++) {
        uint8_t index = (crc >> 8) ^ ptr[i];
        crc = (crc << shift) ^ crc16_table[index];
    }
    return crc;
}

// Legacy byte-wise functions kept for reference / fallback
uint16_t crc16_compute(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
        crc ^= (uint16_t)(data[i] << 8);
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

uint16_t crc16_v1(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0x0000;
    for (uint16_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

/* ═══════════════════════════════════════════════════════════════════
   Helper: copy floats from raw bytes (little-endian IEEE 754)
   ═══════════════════════════════════════════════════════════════════ */

static inline void unpack_float3(const uint8_t *src, float *dst) {
    memcpy(dst, src, 3 * sizeof(float));
}

static inline void unpack_float4(const uint8_t *src, float *dst) {
    memcpy(dst, src, 4 * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════
   Construction / Destruction
   ═══════════════════════════════════════════════════════════════════ */

DamiaoIMUDriver::DamiaoIMUDriver(uint16_t imu_id,
                                 const std::string& interface_type,
                                 const std::string& interface,
                                 const int baudrate)
    : IMUDriver(), interface_type_(interface_type), interface_(interface) {
    imu_id_   = imu_id;
    baudrate_ = (baudrate > 0) ? baudrate : 460800;

    damiao_parser_init(&parser_);

    if (interface_type_ == "serial") {
        serial_ = DamiaoSerialPort::open(interface_, baudrate_);
        auto callback = std::bind(&DamiaoIMUDriver::serial_rx_cbk,
                                  this,
                                  std::placeholders::_1,
                                  std::placeholders::_2);
        serial_->set_serial_callback(callback);

        // ── Configure the IMU on startup ──────────────────────────
        if (!configure_device()) {
            fprintf(stderr, "DAMIAO IMU: device configuration failed!\n");
        }
    } else {
        throw std::runtime_error("DAMIAO driver currently supports only 'serial' interface");
    }
}

DamiaoIMUDriver::~DamiaoIMUDriver() {
    // Clear callback FIRST so that any frames still being processed
    // by the RX thread don't try to call into the (possibly destroyed)
    // ROS2 node during teardown.
    {
        std::unique_lock<std::shared_mutex> lock(imu_mutex_);
        data_callback_ = nullptr;
    }
    if (serial_) {
        serial_->close();
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Device configuration
   ═══════════════════════════════════════════════════════════════════ */

bool DamiaoIMUDriver::send_command(const uint8_t* cmd, size_t len) {
    if (!serial_ || !serial_->is_open()) return false;
    int written = serial_->write(cmd, len);
    return (written == static_cast<int>(len));
}

bool DamiaoIMUDriver::configure_device() {
    fprintf(stderr, "DAMIAO IMU: configuring device...\n");

    // Wait for device to be ready after open
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Flush any stale data
    serial_->flush();

    // 1) Enter settings mode
    {
        const uint8_t cmd[] = DAMIAO_CMD_ENTER_SETTINGS;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enter settings mode failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 2) Enable outputs
    {
        const uint8_t cmd[] = DAMIAO_CMD_ENABLE_ACCEL;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enable accel failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        const uint8_t cmd[] = DAMIAO_CMD_ENABLE_GYRO;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enable gyro failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        const uint8_t cmd[] = DAMIAO_CMD_ENABLE_EULER;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enable euler failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        const uint8_t cmd[] = DAMIAO_CMD_ENABLE_QUAT;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enable quat failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 3) Set USB as output interface
    {
        const uint8_t cmd[] = DAMIAO_CMD_SET_USB_OUTPUT;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: set USB output failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 4) Save parameters to flash
    {
        const uint8_t cmd[] = DAMIAO_CMD_SAVE_PARAMS;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: save parameters failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // 5) Enter normal mode (start streaming)
    {
        const uint8_t cmd[] = DAMIAO_CMD_EXIT_SETTINGS;
        if (!send_command(cmd, DAMIAO_CONFIG_CMD_LEN)) {
            fprintf(stderr, "IMUReader: enter normal mode failed\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Flush again after config
    serial_->flush();
    damiao_parser_init(&parser_);

    fprintf(stderr, "DAMIAO IMU: device configured successfully\n");
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
   Serial RX callback  →  byte-by-byte parser
   ═══════════════════════════════════════════════════════════════════ */

void DamiaoIMUDriver::serial_rx_cbk(const uint8_t* data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        parse_byte(data[i]);
    }
}

void DamiaoIMUDriver::parse_byte(uint8_t byte) {
    switch (parser_.state) {

    case STATE_WAIT_HEADER0:
        if (byte == DAMIAO_FRAME_HEADER0) {
            parser_.state = STATE_WAIT_HEADER1;
        }
        break;

    case STATE_WAIT_HEADER1:
        if (byte == DAMIAO_FRAME_HEADER1) {
            parser_.state = STATE_GET_MSG_TYPE;
        } else if (byte == DAMIAO_FRAME_HEADER0) {
            // Stay in STATE_WAIT_HEADER1 (keep syncing)
        } else {
            parser_.state = STATE_WAIT_HEADER0; // Lost sync
        }
        break;

    case STATE_GET_MSG_TYPE:
        parser_.msg_type = byte;
        parser_.state = STATE_GET_MSG_ID;
        break;

    case STATE_GET_MSG_ID:
        parser_.msg_id   = byte;
        parser_.data_len = damiao_expected_data_len(parser_.msg_id);
        if (parser_.data_len <= 0 || parser_.data_len > DAMIAO_MAX_PAYLOAD) {
            parser_.state = STATE_WAIT_HEADER0;
        } else {
            parser_.data_idx = 0;
            parser_.state    = STATE_COLLECT_DATA;
        }
        break;

    case STATE_COLLECT_DATA:
        parser_.data[parser_.data_idx++] = byte;
        if (parser_.data_idx >= parser_.data_len) {
            parser_.state = STATE_CRC_BYTE0;
        }
        break;

    case STATE_CRC_BYTE0:
        parser_.crc_lo = byte;
        parser_.state   = STATE_CRC_BYTE1;
        break;

    case STATE_CRC_BYTE1:
        parser_.crc_hi    = byte;
        parser_.received_crc = (static_cast<uint16_t>(parser_.crc_hi) << 8)
                               | parser_.crc_lo;
        parser_.state = STATE_WAIT_FOOTER;
        break;

    case STATE_WAIT_FOOTER:
        if (byte == DAMIAO_FRAME_FOOTER) {
            process_frame();
        }
        // Always reset to look for next frame
        parser_.state = STATE_WAIT_HEADER0;
        break;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Frame processing
   ═══════════════════════════════════════════════════════════════════ */

void DamiaoIMUDriver::process_frame() {
    // ── CRC-16 verification ────────────────────────────────────────
    //
    // Device CRC covers: 0x55 0xAA + ID + type + data
    // Algorithm: shift=1, init=0xFFFF (manual Appendix 4 Get_CRC16)
    //
    // Frame format:  55 AA [ID] [type] [data] CRC16(LE) 0A
    // Parser:  parser_.msg_type = ID (first byte after 55 AA)
    //          parser_.msg_id   = type (second byte after 55 AA)

    uint8_t crc_buf[DAMIAO_MAX_PAYLOAD + 4];
    crc_buf[0] = 0x55;
    crc_buf[1] = 0xAA;
    crc_buf[2] = parser_.msg_type;   // device ID
    crc_buf[3] = parser_.msg_id;     // frame type
    memcpy(crc_buf + 4, parser_.data, parser_.data_len);
    uint16_t crc_len = static_cast<uint16_t>(parser_.data_len + 4);

    uint16_t computed = damiao_crc16_ex(crc_buf, crc_len, 1, 0xFFFF);

    if (computed != parser_.received_crc) {
        return;  // Drop corrupted frame
    }

    // ── Parse payload by message ID (frame subtype) ──────────────────
    std::unique_lock<std::shared_mutex> lock(imu_mutex_);

    sensor_data_.device_id = parser_.msg_type;

    switch (parser_.msg_id) {
    case DAMIAO_MSG_ACCEL:  // 3 × float → accel (m/s², already SI)
        unpack_float3(parser_.data, tmp_accel_);
        sensor_data_.acc_x = tmp_accel_[0];
        sensor_data_.acc_y = tmp_accel_[1];
        sensor_data_.acc_z = tmp_accel_[2];
        accel_valid_ = true;
        break;

    case DAMIAO_MSG_GYRO:  // 3 × float → gyro (rad/s, already SI)
        unpack_float3(parser_.data, tmp_gyro_);
        sensor_data_.gyr_x = tmp_gyro_[0];
        sensor_data_.gyr_y = tmp_gyro_[1];
        sensor_data_.gyr_z = tmp_gyro_[2];
        gyro_valid_ = true;
        break;

    case DAMIAO_MSG_EULER:  // 3 × float → roll, pitch, yaw (deg → rad)
        unpack_float3(parser_.data, tmp_euler_);
        sensor_data_.roll  = tmp_euler_[0] * DEG_TO_RAD;
        sensor_data_.pitch = tmp_euler_[1] * DEG_TO_RAD;
        sensor_data_.yaw   = tmp_euler_[2] * DEG_TO_RAD;
        euler_valid_ = true;
        break;

    case DAMIAO_MSG_QUAT:  // 4 × float → quat (w,x,y,z)
        unpack_float4(parser_.data, tmp_quat_);
        sensor_data_.quat_w = tmp_quat_[0];
        sensor_data_.quat_x = tmp_quat_[1];
        sensor_data_.quat_y = tmp_quat_[2];
        sensor_data_.quat_z = tmp_quat_[3];
        quat_valid_ = true;
        break;
    }

    // ── Update base-class vectors ──────────────────────────────────
    lin_acc_ = {sensor_data_.acc_x, sensor_data_.acc_y, sensor_data_.acc_z};
    ang_vel_ = {sensor_data_.gyr_x, sensor_data_.gyr_y, sensor_data_.gyr_z};
    euler_   = {sensor_data_.roll, sensor_data_.pitch, sensor_data_.yaw};
    quat_    = {sensor_data_.quat_w, sensor_data_.quat_x,
                sensor_data_.quat_y, sensor_data_.quat_z};
    temperature_ = sensor_data_.temperature;

    // ── Notify callback (hold lock to prevent destructor from
    //     nulling data_callback_ while we're about to call it) ───
    if (data_callback_) {
        data_callback_(sensor_data_);
    }
    lock.unlock();
}

/* ═══════════════════════════════════════════════════════════════════
   Thread-safe accessors
   ═══════════════════════════════════════════════════════════════════ */

std::vector<float> DamiaoIMUDriver::get_ang_vel() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.gyr_x, sensor_data_.gyr_y, sensor_data_.gyr_z};
}

std::vector<float> DamiaoIMUDriver::get_quat() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.quat_w, sensor_data_.quat_x,
            sensor_data_.quat_y, sensor_data_.quat_z};
}

std::vector<float> DamiaoIMUDriver::get_lin_acc() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.acc_x, sensor_data_.acc_y, sensor_data_.acc_z};
}

std::vector<float> DamiaoIMUDriver::get_euler() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.roll, sensor_data_.pitch, sensor_data_.yaw};
}

float DamiaoIMUDriver::get_temperature() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return sensor_data_.temperature;
}
