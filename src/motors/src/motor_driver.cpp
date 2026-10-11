#include "motor_driver.hpp"
#include "lro_motor_driver.hpp"
#include "lro_motor_driver_can.hpp"
#include "rs02_motor_driver.hpp"

// #include "dm_motor_driver.hpp"
// #include "evo_motor_driver.hpp"
// #include "stw_motor_driver.hpp"
// #include "xyn_motor_driver.hpp"

MotorDriver::MotorDriver() { logger_ = setup_logger("motors"); }
std::shared_ptr<MotorDriver> MotorDriver::create_motor(uint16_t motor_id, const std::string& interface_type, const std::string& interface,
                                                      const std::string& motor_type, int motor_model, uint16_t master_id_offset, double motor_zero_offset) {
    // if (motor_type == "DM") {
    //     return std::make_shared<DmMotorDriver>(motor_id, interface_type, interface, master_id_offset,
    //                                            static_cast<DM_Motor_Model>(motor_model), motor_zero_offset);
    // } else if (motor_type == "EVO") {
    //     return std::make_shared<EvoMotorDriver>(motor_id, interface_type, interface,
    //                                             static_cast<EVO_Motor_Model>(motor_model), motor_zero_offset);
    /* } else */ if (motor_type == "LRO") {
        return std::make_shared<LroMotorDriver>(motor_id, interface_type, interface,
                                                static_cast<LRO_Motor_Model>(motor_model), motor_zero_offset);
    } else if (motor_type == "LRO_CAN") {
        return std::make_shared<LroMotorDriverCAN>(motor_id, interface,
                                                    static_cast<LRO_CAN_Motor_Model>(motor_model), motor_zero_offset);
    } else if (motor_type == "RS02") {
        uint16_t host_can_id = master_id_offset == 0 ? RS02_DEFAULT_HOST_ID : master_id_offset;
        return std::make_shared<Rs02MotorDriver>(motor_id, interface, host_can_id, motor_zero_offset);
    // } else if (motor_type == "STW") {
    //     return std::make_shared<StwMotorDriver>(motor_id, interface_type, interface,
    //                                             static_cast<STW_Motor_Model>(motor_model), motor_zero_offset);
    // } else if (motor_type == "XYN") {
    //     return std::make_shared<XynMotorDriver>(motor_id, interface_type, interface,
    //                                             static_cast<XYN_Motor_Model>(motor_model), motor_zero_offset);
    } else {
        throw std::runtime_error("Motor type not supported");
    }
}
