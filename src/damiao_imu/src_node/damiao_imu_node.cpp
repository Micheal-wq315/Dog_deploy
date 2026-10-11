// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 DAMIAO IMU Driver

/**
 * @file damiao_imu_node.cpp
 * @brief ROS2 node that publishes DAMIAO IMU data.
 * @details Opens a DAMIAO IMU over USB serial, configures it,
 *          and publishes DamiaoImuData and sensor_msgs/Imu messages.
 *
 * Usage:
 *   ros2 run damiao_imu damiao_imu_node <device> [baud]
 *
 * Example:
 *   ros2 run damiao_imu damiao_imu_node /dev/ttyUSB0 460800
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "damiao_imu/imu_driver.hpp"
#include "damiao_imu/msg/damiao_imu_data.hpp"

class DamiaoIMUNode : public rclcpp::Node {
   public:
    DamiaoIMUNode(const std::string& device)
        : Node("damiao_imu_node") {

        // ── Publishers ─────────────────────────────────────────────
        pub_raw_ = this->create_publisher<damiao_imu::msg::DamiaoImuData>(
            "imu/data_raw", 10);

        pub_imu_ = this->create_publisher<sensor_msgs::msg::Imu>(
            "imu/data", 10);

        // ── Create driver ──────────────────────────────────────────
        RCLCPP_INFO(this->get_logger(),
                    "Opening DAMIAO IMU on %s", device.c_str());

        try {
            driver_ = IMUDriver::create_imu(0, "serial", device, "DAMIAO");

            driver_->set_data_callback(
                [this](const SensorData& data) {
                    this->on_sensor_data(data);
                });

            RCLCPP_INFO(this->get_logger(), "DAMIAO IMU initialized successfully");
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to open IMU: %s", e.what());
            rclcpp::shutdown();
            throw;
        }

    }

    ~DamiaoIMUNode() override = default;

   private:
    void on_sensor_data(const SensorData& data) {
        auto now = this->now();

        // ── Custom DAMIAO message ──────────────────────────────────
        {
            auto msg = damiao_imu::msg::DamiaoImuData();
            msg.device_id   = data.device_id;
            msg.accel_x     = data.acc_x;
            msg.accel_y     = data.acc_y;
            msg.accel_z     = data.acc_z;
            msg.gyro_x      = data.gyr_x;
            msg.gyro_y      = data.gyr_y;
            msg.gyro_z      = data.gyr_z;
            msg.roll        = data.roll;
            msg.pitch       = data.pitch;
            msg.yaw         = data.yaw;
            msg.q_w         = data.quat_w;
            msg.q_x         = data.quat_x;
            msg.q_y         = data.quat_y;
            msg.q_z         = data.quat_z;
            msg.temperature = data.temperature;
            pub_raw_->publish(msg);
        }

        // ── Standard sensor_msgs/Imu ───────────────────────────────
        {
            auto msg = sensor_msgs::msg::Imu();
            msg.header.stamp    = now;
            msg.header.frame_id = "imu_link";

            msg.linear_acceleration.x = data.acc_x;
            msg.linear_acceleration.y = data.acc_y;
            msg.linear_acceleration.z = data.acc_z;

            msg.angular_velocity.x = data.gyr_x;
            msg.angular_velocity.y = data.gyr_y;
            msg.angular_velocity.z = data.gyr_z;

            msg.orientation.w = data.quat_w;
            msg.orientation.x = data.quat_x;
            msg.orientation.y = data.quat_y;
            msg.orientation.z = data.quat_z;

            // Covariances: set to -1 (unknown) by default
            msg.orientation_covariance[0]       = -1.0;
            msg.angular_velocity_covariance[0]  = -1.0;
            msg.linear_acceleration_covariance[0] = -1.0;

            pub_imu_->publish(msg);
        }
    }

    std::shared_ptr<IMUDriver> driver_;
    rclcpp::Publisher<damiao_imu::msg::DamiaoImuData>::SharedPtr pub_raw_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device>\n", argv[0]);
        fprintf(stderr, "Example: %s /dev/ttyACM0\n", argv[0]);
        return 1;
    }

    std::string device = argv[1];

    std::shared_ptr<DamiaoIMUNode> node;
    try {
        node = std::make_shared<DamiaoIMUNode>(device);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::spin(node);

    // ── Shut down ROS2 context BEFORE destroying the node, so the
    //     serial RX thread does not try to publish during teardown.
    rclcpp::shutdown();

    // Explicitly destroy the node (and driver, which joins the RX
    // thread) after the ROS2 context is fully shut down.
    node.reset();

    return 0;
}
