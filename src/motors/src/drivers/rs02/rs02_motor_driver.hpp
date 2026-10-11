#pragma once

#include <atomic>
#include <string>

#include "motor_driver.hpp"
#include "protocol/can_iso.hpp"
#include "utils.hpp"

// RS02 default host (master) CAN ID — normal feedback arrives with this ID.
constexpr uint16_t RS02_DEFAULT_HOST_ID = 0xFD;

// RS02 command magic bytes (Byte7 in 8-byte data)
enum RS02Cmd : uint8_t {
    RS02_CMD_ENABLE      = 0xFC,  // 指令1: motor enable
    RS02_CMD_DISABLE     = 0xFD,  // 指令2: motor stop
    RS02_CMD_SET_ZERO    = 0xFE,  // 指令4: set zero position
    RS02_CMD_CLEAR_ERROR = 0xFB,  // 指令5: clear error / read error status
    RS02_CMD_SET_MODE    = 0xFC,  // 指令6: set operation mode (distinguished by data[6])
    RS02_CMD_SET_ID      = 0xFA,  // 指令7: modify motor CAN ID
    RS02_CMD_SET_PROTOCOL = 0xFD, // 指令8: switch motor protocol
    RS02_CMD_SET_HOST_ID  = 0x01, // 指令9: modify host CAN ID
    RS02_CMD_SAVE        = 0xF8,  // 指令12: save motor data
    RS02_CMD_ACTIVE_REPORT = 0xF9 // 指令13: active report switch
};

// RS02 operation modes (used with 指令6)
enum RS02Mode : uint8_t {
    RS02_MODE_MIT = 0,  // MIT impedance control (default)
    RS02_MODE_POS = 1,  // Position mode (CSP)
    RS02_MODE_SPD = 2,  // Speed mode
};

enum RS02Protocol : uint8_t {
    RS02_PROTOCOL_PRIVATE = 0,
    RS02_PROTOCOL_CANOPEN = 1,
    RS02_PROTOCOL_MIT = 2,
};

enum RS02MotorState : uint8_t {
    RS02_STATE_RESET = 0,
    RS02_STATE_CALI = 1,
    RS02_STATE_MOTOR = 2,
};

// RS02 CAN ID bit layout for control commands (standard 11-bit frame)
// bits 10-8: mode type (0=MIT, 1=POS, 2=SPD, 3=READ, 4=WRITE)
// bits 7-0:  motor CAN ID
constexpr uint8_t RS02_MODE_MIT_CAN  = 0;  // MIT mode bits[10:8]
constexpr uint8_t RS02_MODE_POS_CAN  = 1;  // Position mode bits[10:8]
constexpr uint8_t RS02_MODE_SPD_CAN  = 2;  // Speed mode bits[10:8]
constexpr uint8_t RS02_MODE_READ_CAN = 3;  // Read param bits[10:8]
constexpr uint8_t RS02_MODE_WRITE_CAN = 4; // Write param bits[10:8]

// RS02 motor parameter limits
typedef struct {
    float PosMax;   // Maximum position (rad), default 12.57
    float SpdMax;   // Maximum velocity (rad/s), default 44
    float TauMax;   // Maximum torque (Nm), default 17
    float OKpMax;   // Maximum Kp, default 500
    float OKdMax;   // Maximum Kd, default 5
} RS02_Limit_Param;

class Rs02MotorDriver : public MotorDriver {
   public:
    Rs02MotorDriver(uint16_t motor_id, const std::string& can_interface,
                    double motor_zero_offset = 0.0);
    Rs02MotorDriver(uint16_t motor_id, const std::string& can_interface,
                    uint16_t host_can_id, double motor_zero_offset);
    ~Rs02MotorDriver();

    virtual void lock_motor() override;
    virtual void unlock_motor() override;
    virtual uint8_t init_motor() override;
    virtual void deinit_motor() override;
    virtual bool set_motor_zero() override;
    virtual bool write_motor_flash() override;
    virtual void get_motor_param(uint8_t param_cmd) override;

    virtual void motor_pos_cmd(float pos, float spd, bool ignore_limit) override;
    virtual void motor_spd_cmd(float spd) override;
    virtual void motor_mit_cmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) override;
    virtual void motor_mit_cmd(float* f_p, float* f_v, float* f_kp, float* f_kd, float* f_t) override;
    virtual void set_motor_control_mode(uint8_t motor_control_mode) override;
    virtual int get_response_count() const override {
        return response_count_;
    }
    virtual void set_motor_id(uint8_t old_id, uint8_t new_id) override;
    virtual void reset_motor_id() override;
    virtual void refresh_motor_status() override;
    virtual void clear_motor_error() override;

    void set_motor_protocol(uint8_t protocol);
    void set_host_can_id(uint8_t host_id);
    uint16_t get_host_can_id() const { return host_can_id_.load(); }
    void set_active_report(bool enable);
    void read_motor_error();
    uint32_t get_fault_bits() const { return fault_bits_; }
    uint8_t get_motor_state() const { return motor_state_; }
    bool get_motor_warning() const { return warning_; }

   private:
    std::atomic<int> response_count_{0};
    RS02_Limit_Param limit_param_;
    std::atomic<uint8_t> motor_state_{0};
    std::atomic<bool> warning_{false};
    std::atomic<uint32_t> fault_bits_{0};
    std::atomic<bool> expect_fault_response_{false};
    std::atomic<uint16_t> host_can_id_{RS02_DEFAULT_HOST_ID};

    void set_motor_zero_rs02();
    void clear_motor_error_rs02();
    void read_motor_error_rs02();
    void send_special_cmd(uint8_t f_cmd, uint8_t cmd);
    void set_local_host_can_id(uint16_t host_can_id);
    void write_register_rs02(uint16_t index, float value);
    void read_register_rs02(uint16_t index);
    bool map_control_mode(uint8_t motor_control_mode, uint8_t& rs02_mode) const;

    virtual void can_rx_cbk(const can_frame& rx_frame);
    std::shared_ptr<MotorsCAN> can_;
};
