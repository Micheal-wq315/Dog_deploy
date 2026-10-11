// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_protocol.h
 * @brief DAMIAO IMU USB serial protocol constants and frame definitions.
 * @details Defines the binary frame format, command bytes, and CRC routines
 *          used by the DAMIAO IMU over USB virtual COM port.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Frame delimiters ─────────────────────────────────────────────── */
#define DAMIAO_FRAME_HEADER0   0x55   ///< First sync byte
#define DAMIAO_FRAME_HEADER1   0xAA   ///< Second sync byte
#define DAMIAO_FRAME_FOOTER    0x0A   ///< Frame terminator

/* ── Message type → data payload length (bytes) ──────────────────── */
#define DAMIAO_MSG_ACCEL       0x01   ///< Accelerometer data  (12 B → 3 floats)
#define DAMIAO_MSG_GYRO        0x02   ///< Gyroscope data      (12 B → 3 floats)
#define DAMIAO_MSG_EULER       0x03   ///< Euler angles        (12 B → 3 floats)
#define DAMIAO_MSG_QUAT        0x04   ///< Quaternion + temp   (16 B → 4 floats)

static inline int damiao_expected_data_len(uint8_t msg_type) {
    switch (msg_type) {
        case DAMIAO_MSG_ACCEL:  return 12;   // 3 × float
        case DAMIAO_MSG_GYRO:   return 12;
        case DAMIAO_MSG_EULER:  return 12;
        case DAMIAO_MSG_QUAT:   return 16;   // 4 × float
        default:                return 0;
    }
}

/* ── Configuration command bytes ──────────────────────────────────── */
/* All config commands use the same 4-byte format:  AA [CMD] [PARAM] 0D */

#define DAMIAO_CMD_ENTER_SETTINGS   {0xAA, 0x06, 0x01, 0x0D}
#define DAMIAO_CMD_EXIT_SETTINGS    {0xAA, 0x06, 0x00, 0x0D}
#define DAMIAO_CMD_ENABLE_ACCEL     {0xAA, 0x01, 0x14, 0x0D}
#define DAMIAO_CMD_ENABLE_GYRO      {0xAA, 0x01, 0x15, 0x0D}
#define DAMIAO_CMD_ENABLE_EULER     {0xAA, 0x01, 0x16, 0x0D}
#define DAMIAO_CMD_ENABLE_QUAT      {0xAA, 0x01, 0x17, 0x0D}
#define DAMIAO_CMD_SET_USB_OUTPUT   {0xAA, 0x0A, 0x00, 0x0D}
#define DAMIAO_CMD_SAVE_PARAMS      {0xAA, 0x03, 0x01, 0x0D}

#define DAMIAO_CONFIG_CMD_LEN       4

/* ── Frame parser states ──────────────────────────────────────────── */
enum DamiaoParserState {
    STATE_WAIT_HEADER0 = 0,   ///< Waiting for 0x55
    STATE_WAIT_HEADER1 = 1,   ///< Waiting for 0xAA
    STATE_GET_MSG_TYPE = 2,   ///< Reading message type
    STATE_GET_MSG_ID   = 3,   ///< Reading message ID, computing expected len
    STATE_COLLECT_DATA = 4,   ///< Collecting payload bytes
    STATE_CRC_BYTE0    = 5,   ///< Reading CRC low byte
    STATE_CRC_BYTE1    = 6,   ///< Reading CRC high byte
    STATE_WAIT_FOOTER  = 7,   ///< Waiting for 0x0A terminator
};

/* ── Frame parser runtime context ─────────────────────────────────── */
#define DAMIAO_MAX_PAYLOAD  64

typedef struct {
    enum DamiaoParserState state;
    uint8_t  msg_type;                         ///< Frame message type
    uint8_t  msg_id;                           ///< Frame message ID (device ID)
    uint8_t  data[DAMIAO_MAX_PAYLOAD];         ///< Accumulated payload
    int      data_len;                         ///< Expected payload length
    int      data_idx;                         ///< Current write position
    uint8_t  crc_lo;                           ///< Received CRC low byte
    uint8_t  crc_hi;                           ///< Received CRC high byte
    uint16_t received_crc;                     ///< Combined received CRC-16
} damiao_parser_t;

static inline void damiao_parser_init(damiao_parser_t *p) {
    memset(p, 0, sizeof(*p));
    p->state = STATE_WAIT_HEADER0;
}

/* ── CRC-16 routines (matches the two variants found in the binary) ─ */

/**
 * @brief CRC-16 variant #1 used by DAMIAO IMU frames.
 */
uint16_t crc16_compute(const uint8_t *data, uint16_t len);

/**
 * @brief CRC-16 variant #2 (byte-swapped) used as fallback check.
 */
uint16_t crc16_v1(const uint8_t *data, uint16_t len);

#ifdef __cplusplus
}
#endif
