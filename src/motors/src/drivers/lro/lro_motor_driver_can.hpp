#pragma once

#include <atomic>
#include <string>

#include "motor_driver.hpp"
#include "protocol/can_iso.hpp"
#include "utils.hpp"

// LeadRobot error codes (from type-1 feedback, 5-bit error field)
enum LROCanError : uint8_t {
    LRO_CAN_NO_ERROR        = 0x00,
    LRO_CAN_MOTOR_OVERHEAT  = 0x01,
    LRO_CAN_OVER_CURRENT    = 0x02,
    LRO_CAN_UNDER_VOLTAGE   = 0x03,
    LRO_CAN_ENCODER_ERROR   = 0x04,
    LRO_CAN_BRAKE_OVERVOLT  = 0x06,
    LRO_CAN_DRV_ERROR       = 0x07,
};

enum LRO_CAN_Motor_Model {
    LRO_CAN_5550,
    LRO_CAN_6562,
    LRO_CAN_8462,
    LRO_CAN_10062,
    LRO_CAN_Num_Of_Motor
};

// LeadRobot motor mode byte (upper 3 bits of Byte0 in control frames)
enum LROCanMotorMode : uint8_t {
    LRO_CAN_MODE_MIT     = 0x00,
    LRO_CAN_MODE_POS     = 0x01,
    LRO_CAN_MODE_SPD     = 0x02,
    LRO_CAN_MODE_CUR     = 0x03,
    LRO_CAN_MODE_CONFIG  = 0x06,
    LRO_CAN_MODE_QUERY   = 0x07,
};

// LeadRobot 0x7FF setup command codes (Byte3)
enum LROCanSetupCmd : uint8_t {
    LRO_CAN_CMD_SET_ZERO   = 0x03,
    LRO_CAN_CMD_SET_ID     = 0x04,
    LRO_CAN_CMD_RESET_ID   = 0x05,
    LRO_CAN_CMD_ENABLE     = 0x06,
    LRO_CAN_CMD_DISABLE    = 0x07,
    LRO_CAN_CMD_QUERY_MODE = 0x81,
    LRO_CAN_CMD_QUERY_ID   = 0x82,
};

// LeadRobot feedback message type (upper 3 bits of Byte0 in response)
enum LROCanFeedbackType : uint8_t {
    LRO_CAN_FB_TYPE1 = 0x01,
    LRO_CAN_FB_TYPE2 = 0x02,
    LRO_CAN_FB_TYPE3 = 0x03,
    LRO_CAN_FB_TYPE4 = 0x04,
    LRO_CAN_FB_TYPE5 = 0x05,
};

// LeadRobot config codes (used with mode 0x06)
enum LROCanConfigCode : uint8_t {
    LRO_CAN_CFG_ACCEL       = 0x01,
    LRO_CAN_CFG_DECEL       = 0x02,
    LRO_CAN_CFG_MAX_SPD     = 0x03,
    LRO_CAN_CFG_TORQUE_SENS = 0x04,
    LRO_CAN_CFG_KP_MAX      = 0x05,
    LRO_CAN_CFG_KD_MAX      = 0x06,
    LRO_CAN_CFG_POS_MAX     = 0x07,
    LRO_CAN_CFG_SPD_MAX     = 0x08,
    LRO_CAN_CFG_TOR_MAX     = 0x09,
    LRO_CAN_CFG_CUR_MAX     = 0x0A,
    LRO_CAN_CFG_TIMEOUT     = 0x0B,
    LRO_CAN_CFG_CUR_PI      = 0x0C,
    LRO_CAN_CFG_SPD_PI      = 0x0D,
    LRO_CAN_CFG_POS_PD      = 0x0E,
    LRO_CAN_CFG_KT_CALIB    = 0x0F
};

// Query codes for Mode 0x07
enum LROCanQueryCode : uint8_t {
    LRO_CAN_QRY_MANUFACTURER = 0x00,
    LRO_CAN_QRY_POS          = 0x01,
    LRO_CAN_QRY_SPD          = 0x02,
    LRO_CAN_QRY_CUR          = 0x03,
    LRO_CAN_QRY_POWER        = 0x04,
    LRO_CAN_QRY_ACCEL        = 0x05,
    LRO_CAN_QRY_FLUX_GAIN    = 0x06,
    LRO_CAN_QRY_DIST_COMP    = 0x07,
    LRO_CAN_QRY_FB_COMP      = 0x08,
    LRO_CAN_QRY_DAMPING      = 0x09,
    LRO_CAN_QRY_KT           = 0x16,
    LRO_CAN_QRY_KP_RANGE     = 0x17,
    LRO_CAN_QRY_KD_RANGE     = 0x18,
    LRO_CAN_QRY_POS_RANGE    = 0x19,
    LRO_CAN_QRY_SPD_RANGE    = 0x1A,
    LRO_CAN_QRY_TOR_RANGE    = 0x1B,
    LRO_CAN_QRY_CUR_RANGE    = 0x1C,
    LRO_CAN_QRY_MCU_UUID     = 0x1D,
    LRO_CAN_QRY_VERSION      = 0x1E,
    LRO_CAN_QRY_CAN_TIMEOUT  = 0x1F,
    LRO_CAN_QRY_CUR_PI       = 0x20,
    LRO_CAN_QRY_SPD_PI       = 0x21,
    LRO_CAN_QRY_POS_PD       = 0x22,
    LRO_CAN_QRY_KT_CALIB     = 0x23,
};

typedef struct {
    float PosMax;
    float SpdMax;
    float TauMax;
    float OKpMax;
    float OKdMax;
} LRO_Can_Limit_Param;

class LroMotorDriverCAN : public MotorDriver {
   public:
    LroMotorDriverCAN(uint16_t motor_id, const std::string& can_interface,
                      LRO_CAN_Motor_Model motor_model, double motor_zero_offset = 0.0);
    ~LroMotorDriverCAN();

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

   private:
    std::atomic<int> response_count_{0};
    LRO_CAN_Motor_Model motor_model_;
    LRO_Can_Limit_Param limit_param_;
    std::atomic<uint8_t> mos_temperature_{0};
    void set_motor_zero_lro();
    void clear_motor_error_lro();
    void write_register_lro(uint8_t rid, float value);
    void write_register_lro(uint8_t index, int32_t value);
    void save_register_lro();

    virtual void can_rx_cbk(const can_frame& rx_frame);
    std::shared_ptr<MotorsCAN> can_;
};
