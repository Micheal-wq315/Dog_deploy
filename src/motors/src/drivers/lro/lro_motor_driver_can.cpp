/**
 * @file
 * LRO 系列电机 CAN 协议驱动实现。
 *
 * 协议概述：
 *   - MIT 模式帧：8 字节 packed（模式 | KP | KD | 位置 | 速度 | 扭矩）
 *   - 广播帧：can_id=0x7FF（使能/失能/设零/设ID）
 *   - 回包：电机自动回复，can_id=电机ID，包含位置/速度/电流/温度/错误码
 *
 * 注意：
 *   - refresh_motor_status() 实际发送零 MIT 指令等回包，不是读寄存器
 *   - response_count_ 跟踪未应答指令数，RX 回调中清零
 *   - 每个 motor_mit_cmd 调用一次 transmit → 一个 CAN 帧
 */

#include "lro_motor_driver_can.hpp"

// ── 电机型号参数表（位置/速度/扭矩限制、KP/KD 上限） ──
LRO_Can_Limit_Param lro_can_limit_param[LRO_CAN_Num_Of_Motor] = {
    {12.5, 45.0, 40.0, 500.0, 5.0},       // LRO_CAN_PJ2_55_5550（小型）
    {12.5, 45.0, 40.0, 500.0, 5.0},       // LRO_CAN_PJ3_60_6562
    {12.5, 18.0, 90.0, 500.0, 5.0},       // LRO_CAN_PJ3_75_8462
    {12.5, 25.0, 80.0, 500.0, 5.0}        // LRO_CAN_PJ3_97_10062（大型）
};

LroMotorDriverCAN::LroMotorDriverCAN(uint16_t motor_id, const std::string& can_interface,
                                     LRO_CAN_Motor_Model motor_model, double motor_zero_offset)
    : MotorDriver(), motor_model_(motor_model) {
    comm_type_ = CommType::CAN;
    motor_id_ = motor_id;
    limit_param_ = lro_can_limit_param[motor_model_];
    can_interface_ = can_interface;
    motor_zero_offset_ = motor_zero_offset;

    // 获取该 CAN 口的共享实例（每个接口一个 SocketCAN 实例）
    can_ = MotorsCAN::get(can_interface);

    // 注册回包回调：用 motor_id 做键，电机会回复 can_id=motor_id 的帧
    CanCbkFunc can_callback = std::bind(&LroMotorDriverCAN::can_rx_cbk, this, std::placeholders::_1);
    can_->add_can_callback(can_callback, motor_id_);
}

LroMotorDriverCAN::~LroMotorDriverCAN() {
    can_->remove_can_callback(motor_id_);
}

// ── 电机使能：广播 CMD_ENABLE（0x06）到 0x7FF ──
void LroMotorDriverCAN::lock_motor() {
    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;           // 广播地址
    tx_frame.can_dlc = 0x04;

    tx_frame.data[0] = (motor_id_ >> 8) & 0xFF;   // ID 高字节
    tx_frame.data[1] = motor_id_ & 0xFF;           // ID 低字节
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_ENABLE;         // 0x06

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 电机失能：广播 CMD_DISABLE（0x07） → 卸力 ──
void LroMotorDriverCAN::unlock_motor() {
    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x04;

    tx_frame.data[0] = (motor_id_ >> 8) & 0xFF;
    tx_frame.data[1] = motor_id_ & 0xFF;
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_DISABLE;        // 0x07

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 电机初始化序列：失能 → 设 MIT 模式 → 使能 → 读状态 → 检查错误 ──
uint8_t LroMotorDriverCAN::init_motor() {
    LroMotorDriverCAN::unlock_motor();
    Timer::sleep_for(normal_sleep_time);            // 等待电机响应
    LroMotorDriverCAN::set_motor_control_mode(MIT);
    Timer::sleep_for(normal_sleep_time);
    LroMotorDriverCAN::lock_motor();
    Timer::sleep_for(normal_sleep_time);
    LroMotorDriverCAN::refresh_motor_status();      // 发送零 MIT 指令获取回包
    Timer::sleep_for(normal_sleep_time);

    // 检查回包中的错误码
    switch (error_id_) {
        case LROCanError::LRO_CAN_MOTOR_OVERHEAT:   return LROCanError::LRO_CAN_MOTOR_OVERHEAT;
        case LROCanError::LRO_CAN_OVER_CURRENT:     return LROCanError::LRO_CAN_OVER_CURRENT;
        case LROCanError::LRO_CAN_UNDER_VOLTAGE:    return LROCanError::LRO_CAN_UNDER_VOLTAGE;
        case LROCanError::LRO_CAN_ENCODER_ERROR:    return LROCanError::LRO_CAN_ENCODER_ERROR;
        case LROCanError::LRO_CAN_BRAKE_OVERVOLT:   return LROCanError::LRO_CAN_BRAKE_OVERVOLT;
        case LROCanError::LRO_CAN_DRV_ERROR:        return LROCanError::LRO_CAN_DRV_ERROR;
        default:                                    return error_id_;
    }
}

// ── 去初始化：失能即可 ──
void LroMotorDriverCAN::deinit_motor() {
    LroMotorDriverCAN::unlock_motor();
    Timer::sleep_for(normal_sleep_time);
}

bool LroMotorDriverCAN::write_motor_flash() { return true; }

// ── 设置机械零点：发设零指令 → 读回位置验证 ──
bool LroMotorDriverCAN::set_motor_zero() {
    LroMotorDriverCAN::set_motor_zero_lro();
    Timer::sleep_for(setup_sleep_time);
    LroMotorDriverCAN::refresh_motor_status();
    Timer::sleep_for(setup_sleep_time);
    logger_->info("motor_id: {0}\tposition: {1}", motor_id_, get_motor_pos());

    LroMotorDriverCAN::unlock_motor();
    if (get_motor_pos() > judgment_accuracy_threshold || get_motor_pos() < -judgment_accuracy_threshold) {
        logger_->warn("set zero error");
        return false;
    }
    logger_->info("set zero success");
    return true;
}

// ═══════════════════════════════════════════
// CAN 帧接收回调（在 RX 线程中执行）
// ═══════════════════════════════════════════
void LroMotorDriverCAN::can_rx_cbk(const can_frame& rx_frame) {
    response_count_ = 0;  // 收到回包，清零未应答计数

    if (rx_frame.can_dlc < 0x08) return;  // 非数据帧，跳过

    uint16_t pos_int, spd_int, t_int;
    uint8_t fb_type = (rx_frame.data[0] >> 5) & 0x07;  // 反馈类型（位置/速度/电流）
    error_id_ = rx_frame.data[0] & 0x1F;                // 低 5 位：错误码

    if (error_id_ > 0) {
        if (logger_) {
            logger_->error("can_interface: {0}\tmotor_id: {1}\terror_id: 0x{2:x}",
                can_interface_, motor_id_, static_cast<uint32_t>(error_id_));
        }
    }

    // 解析位置（16bit）、速度（12bit）、电流（12bit）
    pos_int = (static_cast<uint16_t>(rx_frame.data[1]) << 8) | rx_frame.data[2];
    spd_int = (static_cast<uint16_t>(rx_frame.data[3]) << 4) | ((rx_frame.data[4] >> 4) & 0x0F);
    t_int = (static_cast<uint16_t>(rx_frame.data[4] & 0x0F) << 8) | rx_frame.data[5];

    // 量程映射：原始值 → 物理量
    motor_pos_ = range_map(pos_int, uint16_t(0), bitmax<uint16_t>(16),
                           -limit_param_.PosMax, limit_param_.PosMax) + static_cast<float>(motor_zero_offset_);
    motor_spd_ = range_map(spd_int, uint16_t(0), bitmax<uint16_t>(12),
                           -limit_param_.SpdMax, limit_param_.SpdMax);
    motor_current_ = range_map(t_int, uint16_t(0), bitmax<uint16_t>(12),
                               -limit_param_.TauMax, limit_param_.TauMax);

    // 温度：MOS 管温度（data[7]） + 电机线圈温度（data[6]-25）
    mos_temperature_ = rx_frame.data[7];
    motor_temperature_ = static_cast<float>(static_cast<int>(rx_frame.data[6]) - 25);
}

// ── 查询电机参数（如 PID 值） ──
void LroMotorDriverCAN::get_motor_param(uint8_t param_cmd) {
    can_frame tx_frame{};
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x02;

    tx_frame.data[0] = (uint8_t)(LRO_CAN_MODE_QUERY << 5);
    tx_frame.data[1] = param_cmd;

    can_->transmit(tx_frame);
    response_count_++;
}

// ═══════════════════════════════════════════
// 位置控制模式（POS）
// ═══════════════════════════════════════════
void LroMotorDriverCAN::motor_pos_cmd(float pos, float spd, bool ignore_limit) {
    if (motor_control_mode_ != POS) {
        set_motor_control_mode(POS);
        return;
    }
    float pos_deg = (pos - static_cast<float>(motor_zero_offset_)) * 180.0f / static_cast<float>(M_PI);
    float spd_rpm = std::abs(spd) * 60.0f / (2.0f * static_cast<float>(M_PI));
    uint16_t spd_val = static_cast<uint16_t>(limit(spd_rpm * 10.0f, 0.0f, 32767.0f));
    uint16_t cur_limit = 4095;
    uint8_t ack = 1;
    union32_t rv_type_convert;
    rv_type_convert.f = pos_deg;

    can_frame tx_frame;
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x08;

    // 打包：模式(3b) | 角度(float32, 22b) | 速度(15b) | 电流限制(12b) | ACK(2b)
    uint64_t packed = 0;
    packed |= (static_cast<uint64_t>(LRO_CAN_MODE_POS & 0x07)) << 61;
    packed |= (static_cast<uint64_t>(rv_type_convert.buf[3]) << 29) | (static_cast<uint64_t>(rv_type_convert.buf[2]) << 21)
             | (static_cast<uint64_t>(rv_type_convert.buf[1]) << 13) | (static_cast<uint64_t>(rv_type_convert.buf[0]) << 5);
    packed |= (static_cast<uint64_t>(spd_val & 0x7FFF)) << 14;
    packed |= (static_cast<uint64_t>(cur_limit & 0x0FFF)) << 2;
    packed |= (static_cast<uint64_t>(ack & 0x03));

    for (int i = 0; i < 8; ++i)
        tx_frame.data[i] = (packed >> (56 - i * 8)) & 0xFF;

    can_->transmit(tx_frame);
    response_count_++;
}

// ═══════════════════════════════════════════
// 速度控制模式（SPD）
// ═══════════════════════════════════════════
void LroMotorDriverCAN::motor_spd_cmd(float spd) {
    if (motor_control_mode_ != SPD) {
        set_motor_control_mode(SPD);
        return;
    }
    float spd_rpm = spd * 60.0f / (2.0f * static_cast<float>(M_PI));
    uint16_t cur_limit = 65535;
    uint8_t ack = 1;
    union32_t rv_type_convert;
    rv_type_convert.f = spd_rpm;

    can_frame tx_frame;
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x07;

    tx_frame.data[0] = ((LRO_CAN_MODE_SPD & 0x07) << 5) | (ack & 0x03);
    tx_frame.data[1] = rv_type_convert.buf[3];
    tx_frame.data[2] = rv_type_convert.buf[2];
    tx_frame.data[3] = rv_type_convert.buf[1];
    tx_frame.data[4] = rv_type_convert.buf[0];
    tx_frame.data[5] = (cur_limit >> 8) & 0xFF;
    tx_frame.data[6] = cur_limit & 0xFF;

    can_->transmit(tx_frame);
    response_count_++;
}

// ═══════════════════════════════════════════
// MIT 模式（位置 + 速度 + KP + KD + 前馈扭矩）
//
// 8 字节 packed 格式：
//   [模式 3b] [KP 12b] [KD 9b] [位置 16b] [速度 12b] [扭矩 12b]
// ═══════════════════════════════════════════
void LroMotorDriverCAN::motor_mit_cmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
    if (motor_control_mode_ != MIT) {
        set_motor_control_mode(MIT);
        Timer::sleep_for(normal_sleep_time);
    }

    uint16_t p, v, kp, kd, t;

    // 限制到电机允许范围
    f_p -= static_cast<float>(motor_zero_offset_);
    f_p = limit(f_p, -limit_param_.PosMax, limit_param_.PosMax);
    f_v = limit(f_v, -limit_param_.SpdMax, limit_param_.SpdMax);
    f_kp = limit(f_kp, 0.0f, limit_param_.OKpMax);
    f_kd = limit(f_kd, 0.0f, limit_param_.OKdMax);
    f_t = limit(f_t, -limit_param_.TauMax, limit_param_.TauMax);

    // 物理量 → 原始值映射
    p  = range_map(f_p,  -limit_param_.PosMax, limit_param_.PosMax, uint16_t(0), bitmax<uint16_t>(16));
    v  = range_map(f_v,  -limit_param_.SpdMax, limit_param_.SpdMax, uint16_t(0), bitmax<uint16_t>(12));
    kp = range_map(f_kp,  0.0f, limit_param_.OKpMax, uint16_t(0), bitmax<uint16_t>(12));
    kd = range_map(f_kd,  0.0f, limit_param_.OKdMax, uint16_t(0), bitmax<uint16_t>(9));
    t  = range_map(f_t,   -limit_param_.TauMax, limit_param_.TauMax, uint16_t(0), bitmax<uint16_t>(12));

    // 组装 64 位 packed 帧
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x08;

    uint64_t packed = 0;
    packed |= (static_cast<uint64_t>(LRO_CAN_MODE_MIT & 0x07)) << 61;  // bits 61-63
    packed |= (static_cast<uint64_t>(kp & 0x0FFF)) << 49;              // bits 49-60
    packed |= (static_cast<uint64_t>(kd & 0x01FF)) << 40;              // bits 40-48
    packed |= (static_cast<uint64_t>(p & 0xFFFF)) << 24;               // bits 24-39
    packed |= (static_cast<uint64_t>(v & 0x0FFF)) << 12;               // bits 12-23
    packed |= static_cast<uint64_t>(t & 0x0FFF);                        // bits 0-11

    for (int i = 0; i < 8; ++i)
        tx_frame.data[i] = (packed >> (56 - i * 8)) & 0xFF;

    can_->transmit(tx_frame);  // 推入 TX 队列
    response_count_++;
}

// ── 多槽位 MIT 命令（8 个电机批量发送，本机未用） ──
void LroMotorDriverCAN::motor_mit_cmd(float* f_p, float* f_v, float* f_kp, float* f_kd, float* f_t) {
    if (!f_p || !f_v || !f_kp || !f_kd || !f_t) return;

    if (motor_control_mode_ != MIT) {
        set_motor_control_mode(MIT);
        Timer::sleep_for(normal_sleep_time);
    }

    for (uint8_t slot = 0; slot < 8; ++slot) {
        float p_f = limit(f_p[slot] - static_cast<float>(motor_zero_offset_), -limit_param_.PosMax, limit_param_.PosMax);
        float v_f = limit(f_v[slot], -limit_param_.SpdMax, limit_param_.SpdMax);
        float kp_f = limit(f_kp[slot], 0.0f, limit_param_.OKpMax);
        float kd_f = limit(f_kd[slot], 0.0f, limit_param_.OKdMax);
        float t_f = limit(f_t[slot], -limit_param_.TauMax, limit_param_.TauMax);

        uint16_t kp = range_map(kp_f, 0.0f, limit_param_.OKpMax, uint16_t(0), uint16_t(0x0FFF));
        uint16_t kd = range_map(kd_f, 0.0f, limit_param_.OKdMax, uint16_t(0), uint16_t(0x01FF));
        uint16_t p  = range_map(p_f,  -limit_param_.PosMax, limit_param_.PosMax, uint16_t(0), uint16_t(0xFFFF));
        uint16_t v  = range_map(v_f,  -limit_param_.SpdMax, limit_param_.SpdMax, uint16_t(0), uint16_t(0x0FFF));
        uint16_t t  = range_map(t_f,  -limit_param_.TauMax, limit_param_.TauMax, uint16_t(0), uint16_t(0x0FFF));

        uint64_t packed = 0;
        packed |= (static_cast<uint64_t>(LRO_CAN_MODE_MIT & 0x07)) << 61;
        packed |= (static_cast<uint64_t>(kp & 0x0FFF)) << 49;
        packed |= (static_cast<uint64_t>(kd & 0x01FF)) << 40;
        packed |= (static_cast<uint64_t>(p & 0xFFFF)) << 24;
        packed |= (static_cast<uint64_t>(v & 0x0FFF)) << 12;
        packed |= static_cast<uint64_t>(t & 0x0FFF);

        can_frame tx_frame;
        tx_frame.can_id = motor_id_;
        tx_frame.can_dlc = 0x08;
        for (int i = 0; i < 8; ++i)
            tx_frame.data[i] = (packed >> (56 - i * 8)) & 0xFF;

        can_->transmit(tx_frame);
    }
    response_count_++;
}

// ── 切换控制模式（POS/SPD/MIT/CUR） ──
void LroMotorDriverCAN::set_motor_control_mode(uint8_t motor_control_mode) {
    if (motor_control_mode > LRO_CAN_MODE_CUR) {
        logger_->error("Invalid motor control mode: {} (ID: {})", motor_control_mode, motor_id_);
        return;
    }

    uint8_t old_mode = motor_control_mode_;
    logger_->info("Switching motor control mode: {} -> {} (ID: {})", old_mode, motor_control_mode, motor_id_);

    // 切换到 MIT 时先发一帧零指令同步模式
    if (motor_control_mode == LRO_CAN_MODE_MIT && old_mode != LRO_CAN_MODE_MIT) {
        logger_->debug("Sending zero MIT command to synchronize mode (ID: {})", motor_id_);
        motor_mit_cmd(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        Timer::sleep_for(normal_sleep_time);
    }

    motor_control_mode_ = motor_control_mode;
}

// ── 设置电机 CAN ID（通过广播地址 0x7FF） ──
void LroMotorDriverCAN::set_motor_id(uint8_t old_id, uint8_t new_id) {
    if (old_id < 1 || old_id > 0x7FF || new_id < 1 || new_id > 0x7FF) {
        logger_->error("Invalid ID range: old={}, new={}", old_id, new_id);
        return;
    }
    if (old_id == new_id) {
        logger_->warn("Skipping ID set: Old and New ID are identical ({})", old_id);
        return;
    }

    logger_->info("Changing Motor ID: {} -> {} (Interface: {})", old_id, new_id, can_interface_);

    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x06;
    tx_frame.data[0] = (old_id >> 8) & 0xFF;
    tx_frame.data[1] = old_id & 0xFF;
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_SET_ID;
    tx_frame.data[4] = (new_id >> 8) & 0xFF;
    tx_frame.data[5] = new_id & 0xFF;

    can_->transmit(tx_frame);
    response_count_++;
    Timer::sleep_for(setup_sleep_time);
    logger_->info("Set ID command sent. Verify the new ID via bus query.");
}

// ── 重置电机 ID 为默认值 ──
void LroMotorDriverCAN::reset_motor_id() {
    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x06;
    tx_frame.data[0] = 0x7F; tx_frame.data[1] = 0x7F;
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_RESET_ID;
    tx_frame.data[4] = 0x7F; tx_frame.data[5] = 0x7F;

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 内部设零（LRO 广播指令） ──
void LroMotorDriverCAN::set_motor_zero_lro() {
    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x04;
    tx_frame.data[0] = (motor_id_ >> 8) & 0xFF;
    tx_frame.data[1] = motor_id_ & 0xFF;
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_SET_ZERO;

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 清除电机错误：失能 → 延时 → 使能 ──
void LroMotorDriverCAN::clear_motor_error_lro() {
    // 失能
    can_frame tx_frame{};
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x04;
    tx_frame.data[0] = (motor_id_ >> 8) & 0xFF;
    tx_frame.data[1] = motor_id_ & 0xFF;
    tx_frame.data[2] = 0x00;
    tx_frame.data[3] = LRO_CAN_CMD_DISABLE;
    can_->transmit(tx_frame);
    response_count_++;

    Timer::sleep_for(normal_sleep_time);

    // 使能
    tx_frame.data[3] = LRO_CAN_CMD_ENABLE;
    can_->transmit(tx_frame);
    response_count_++;
}

// ── 写寄存器（float 值） ──
void LroMotorDriverCAN::write_register_lro(uint8_t rid, float value) {
    uint8_t* vbuf = reinterpret_cast<uint8_t*>(&value);

    can_frame tx_frame{};
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x06;
    tx_frame.data[0] = (LRO_CAN_MODE_CONFIG << 5);
    tx_frame.data[1] = rid;
    tx_frame.data[2] = vbuf[0]; tx_frame.data[3] = vbuf[1];
    tx_frame.data[4] = vbuf[2]; tx_frame.data[5] = vbuf[3];

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 写寄存器（int32 值） ──
void LroMotorDriverCAN::write_register_lro(uint8_t index, int32_t value) {
    can_frame tx_frame{};
    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x06;
    tx_frame.data[0] = (LRO_CAN_MODE_CONFIG << 5);
    tx_frame.data[1] = index;
    tx_frame.data[2] = (value >> 24) & 0xFF;
    tx_frame.data[3] = (value >> 16) & 0xFF;
    tx_frame.data[4] = (value >> 8) & 0xFF;
    tx_frame.data[5] = value & 0xFF;

    can_->transmit(tx_frame);
    response_count_++;
}

// ── 保存寄存器（LRO 协议自动保存，无需显式指令） ──
void LroMotorDriverCAN::save_register_lro() {
    logger_->warn("save_register_lro: LRO protocol has no explicit save command, parameters are auto-saved on write");
}

// ── 刷新电机状态：发零 MIT 帧触发电机回包 ──
void LroMotorDriverCAN::refresh_motor_status() {
    motor_mit_cmd(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

void LroMotorDriverCAN::clear_motor_error() {
    clear_motor_error_lro();
}
