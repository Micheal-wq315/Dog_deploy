// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "damiao_imu/imu_driver.hpp"

// ============================================================================
// RobotInterface — 机器人硬件抽象层 (Hardware Abstraction Layer)
//
// 职责定位:
//   这是推理系统与真实机器人硬件之间唯一的接口层。上层 InferenceNode 不直接
//   接触任何电机驱动协议或 IMU 通信细节, 一切硬件操作都通过本类完成。
//
// 核心功能 (五大模块):
//   1. 硬件初始化  — 从 robot.yaml 加载配置, 创建 MotorDriver + IMUDriver 实例
//   2. 电机控制    — MIT 模式 (位置+速度+KP+KD+前馈力矩) 的 PD 控制器
//   3. 传感器读取  — 关节位置/速度/力矩 + IMU 四元数/角速度
//   4. IMU 外参处理 — 将 IMU 坐标系的数据变换到机体坐标系
//   5. 安全机制    — 电机离线检测、未初始化守卫、闭链解耦保护
//
// 线程安全设计:
//   - joint_mutex_:  保护 joint_q_/joint_vel_/joint_tau_ (control 写, inference 读)
//   - motors_mutex_: 保护 motor_*_target_ 和 CAN 总线 (control 独占)
//   - is_init_:      atomic<bool>, 无锁, 标记电机是否已初始化
//
// 设计模式:
//   - 外观模式 (Facade):    将电机和 IMU 两个独立子系统封装为统一接口
//   - 工厂模式 (Factory):   MotorDriver::create_motor() / IMUDriver::create_imu()
//   - 策略模式 (Strategy):  ankle_decouple_ (闭链解耦算法可替换)
// ============================================================================
class RobotInterface {
   public:
    // =========================================================================
    // 构造函数 — 加载 robot.yaml 并初始化所有硬件
    //
    // config_file: robot.yaml 的绝对路径 (如 /path/to/robots/wheel_quad/robot.yaml)
    //
    // 初始化顺序 (严格):
    //   ① 解析 IMU 配置 → setup_imu() (可选, 无 IMU 字段则跳过)
    //   ② 解析 motors 配置 → setup_motors() (必须, 否则 throw)
    //   ③ 解析 robot 配置 → 构建 motor2urdf_ 映射 + 外参矩阵 + 解耦器
    //   ④ 创建线程池 (线程数 = CAN 口数)
    //   ⑤ 所有状态 buffer 初始化为零
    // =========================================================================
    RobotInterface(const std::string& config_file);

    // =========================================================================
    // 析构函数 — 安全释放硬件资源
    //
    // 顺序: 先停机 (deinit_motors) → 清空电机列表 → 释放 IMU
    // 不 join 任何线程 — 线程池的 join 在 ThreadPool 析构中完成
    // =========================================================================
    ~RobotInterface() {
        deinit_motors();     // 发送停机指令到所有电机 (失能使能)
        motors_.clear();     // 释放所有 MotorDriver 智能指针
        imu_.reset();        // 释放 IMU 驱动
    }

    // =========================================================================
    // IMUCfg — IMU 配置结构体 (从 robot.yaml 的 imu: 段解析)
    // =========================================================================
    struct IMUCfg {
        int imu_id_;                             // IMU 设备 ID (CAN ID 或串口号)
        int baudrate_;                           // 串口波特率 (CAN 模式忽略)
        std::string imu_type_;                   // IMU 型号 (如 "Wit", "HIPNUC")
        std::string imu_interface_type_;         // 接口类型 ("canfd" / "serial")
        std::string imu_interface_;              // 接口名 ("can0" / "/dev/ttyUSB0")
    };

    // =========================================================================
    // MotorsCfg — 电机配置结构体 (从 robot.yaml 的 motors: 段解析)
    //
    // 四条平行数组的设计原因:
    //   电机按 CAN 总线分组, 每个 CAN 口挂载不同数量的电机。
    //   motor_interface_[i] = "can0"          ← 第 i 个 CAN 口的接口名
    //   motor_num_[i]       = 3               ← 第 i 个 CAN 口上的电机数
    //   motor_id_[start..start+3] = {1,2,3}   ← 这些电机的物理 ID
    //   motor_type_[k]      = "RPO"           ← 第 k 个电机的类型 (每电机一个)
    //   所以 motor_interface_.size() == CAN 口数量, motor_id_.size() == 总电机数
    // =========================================================================
    struct MotorsCfg {
        int master_id_offset_;                                    // 主机 ID 偏移 (CAN 帧标识)
        std::vector<std::string> motor_type_;                     // 电机类型 (每电机一个, 如 "RPO")
        std::vector<std::string> motor_interface_type_;           // 接口类型 (每 CAN 口一个, 如 "canfd")
        std::vector<std::string> motor_interface_;                // 接口名 (每 CAN 口一个, 如 "can0")
        std::vector<long int> motor_id_;                          // 电机物理 ID (每电机一个)
        std::vector<long int> motor_model_;                       // 电机型号 (每电机一个, 不同关节可能用不同型号)
        std::vector<long int> motor_num_;                         // 每个 CAN 口挂载的电机数量
        std::vector<double> motor_zero_offset_;                   // 电机零点偏移量 (编码器零位修正)
    };

    // =========================================================================
    // RobotCfg — 机器人运动学/控制配置 (从 robot.yaml 的 robot: 段解析)
    // =========================================================================
    struct RobotCfg {
        // 闭链关节电机索引: 哪些电机属于踝关节并联连杆机构
        // 例如 [10, 11, 14, 15] 表示左踝和右踝各有 2 个电机
        std::vector<long int> close_chain_motor_idx_;

        // 电机方向符号: +1 或 -1, 决定了电机正向旋转对应关节正转还是反转
        // 物理安装方向可能导致电机与 URDF 定义的关节正方向相反
        std::vector<long int> motor_sign_;

        // URDF 关节索引 → CAN 电机索引 的映射 (usd2urdf 的反向映射)
        // urdf2motor_[urdf_joint_id] = motor_array_index
        std::vector<long int> urdf2motor_;

        // PD 控制器的默认增益 (每个电机的 kp 和 kd)
        // 训练时网络输出的是关节位置偏移, 乘以 kp 得到力矩
        std::vector<double> kp_, kd_;

        // IMU 外参: 3×3 旋转矩阵, 按行主序存储的 9 个元素
        // 将 IMU 坐标系下的矢量转换到机体坐标系
        // R_body = R_imu2body × v_imu
        std::vector<double> extrinsic_R_;
    };

    // =========================================================================
    // apply_action — 施加动作到机器人硬件 (250Hz, control 线程调用)
    //
    // 这是最核心的硬件控制接口。将推理输出的关节目标值转换为电机 MIT 命令并发送。
    //
    // MIT 模式 = 位置 + 速度前馈 + KP + KD + 前馈力矩
    //   τ = kp × (p_des - p_actual) + kd × (v_des - v_actual) + τ_ff
    //
    // 参数 (均为 URDF 关节顺序):
    //   p:   目标关节位置 [rad], 默认取 joint_default_angle_
    //   v:   目标关节速度 [rad/s], 默认 0
    //   kp:  位置增益, 默认取 robot.yaml 的 kp
    //   kd:  速度阻尼, 默认取 robot.yaml 的 kd
    //   tau: 前馈力矩 [Nm], 默认 0
    //
    // 执行流程:
    //   ① (锁 joint_mutex_) 反向读出电机实际位置/速度/电流
    //   ② (可选) 踝关节闭链解耦: 电机读数 → 真实关节角度, 真实力矩 → 电机电流
    //   ③ (锁 motors_mutex_) 组装 motor_*_target_ 数组
    //   ④ motors_mit_cmd() 通过线程池并行发送到各 CAN 口
    //
    // 注意: 如果电机未初始化 (is_init_==false), 函数直接返回, 不执行任何操作
    // =========================================================================
    void apply_action(std::vector<float> p,
                      std::vector<float> v  = {},
                      std::vector<float> kp = {},
                      std::vector<float> kd = {},
                      std::vector<float> tau = {});

    // =========================================================================
    // 电机生命周期管理
    // =========================================================================
    void init_motors();         // 初始化所有电机 (上电, 使能)
    void deinit_motors();       // 取消初始化所有电机 (下电, 失能)
    void reset_joints(std::vector<double> joint_default_angle);  // 缓慢回到默认姿态
    void stand_up(std::vector<double> joint_default_angle, double duration_sec = 3.0);  // 插值过渡到站立姿态
    void set_zeros();           // 将当前编码器位置设为零点
    void clear_errors();        // 清除所有电机故障码

    // =========================================================================
    // refresh_joints — 刷新关节状态 (初始化阶段使用, main 线程)
    //
    // 两阶段刷新流程:
    //   ① exec_motors_parallel(refresh_motor_status): 发送状态请求到所有电机
    //   ② sleep(1000ms):                               等待电机回复
    //   ③ exec_motors_parallel(get_motor_pos/spd/current): 读取编码器数据
    //   ④ (可选) forward_close_chain():                  踝关节数值解耦
    //
    // 与 apply_action 中读回的区别:
    //   - refresh_joints 需要 1000ms 等待 (电机回复慢)
    //   - apply_action 是快速读回 (电机刚发完命令, 编码器数据已就绪)
    //   - refresh_joints 只在初始化阶段调用一次, 确认所有电机在线
    // =========================================================================
    void refresh_joints();

    // =========================================================================
    // 传感器读取接口 (getter)
    //
    // 设计约定:
    //   - 前三个 getter 返回拷贝 (非引用), 因为返回后锁即释放, 引用会悬垂
    //   - get_quat/get_ang_vel 返回 const 引用, 因为 quat_buf_/ang_vel_buf_ 是
    //     对象成员, 生命周期与对象相同, 引用在下一次调用前有效
    //   - 所有 getter 都有前置条件守卫: is_init_ (电机) 或 imu_ (IMU)
    //     条件不满足时 throw std::runtime_error
    // =========================================================================

    // -------------------------------------------------------------------------
    // get_joint_q — 读取关节位置 [rad] (URDF 顺序, 23 维)
    //
    // 数据来源: joint_q_, 在 apply_action() 中由电机编码器刷新 (250Hz)
    // 坐标系:   已通过 motor2urdf_ 从 CAN 顺序转为 URDF 顺序
    //           已通过 forward_close_chain 从电机角度解耦为关节角度
    // -------------------------------------------------------------------------
    std::vector<float> get_joint_q() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_q_;                       // 返回拷贝 — 锁在函数退出时释放
    }

    // -------------------------------------------------------------------------
    // get_joint_vel — 读取关节速度 [rad/s] (URDF 顺序, 23 维)
    // -------------------------------------------------------------------------
    std::vector<float> get_joint_vel() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_vel_;
    }

    // -------------------------------------------------------------------------
    // get_joint_tau — 读取关节力矩 [Nm] (URDF 顺序, 23 维)
    //
    // 注意: 这是电机电流 × 力矩常数 估算的力矩, 不是直接力矩传感器
    // -------------------------------------------------------------------------
    std::vector<float> get_joint_tau() {
        if (!is_init_.load()) {
            throw std::runtime_error("Motors not initialized");
        }
        std::unique_lock<std::mutex> lock(joint_mutex_);
        return joint_tau_;
    }

    // -------------------------------------------------------------------------
    // get_quat — 读取 IMU 姿态四元数 (机体坐标系, [w, x, y, z])
    //
    // 变换链路:
    //   IMU 原始四元数 (IMU 坐标系 → 世界坐标系)
    //   → q_body = q_imu × extrinsic_q_inv_
    //      其中 extrinsic_q_inv_ 是外参旋转矩阵的逆四元数
    //        (Body → IMU 的旋转取逆 = IMU → Body 的旋转)
    //   → normalize() → 返回 quat_buf_ 的引用
    //
    // 存储约定: [w, x, y, z], w 为标量部分
    // -------------------------------------------------------------------------
    const std::vector<float>& get_quat() {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        auto raw = imu_->get_quat();           // 原始 IMU 四元数 (w, x, y, z)
        // 外参变换: 取消 IMU 安装偏差
        q_body_ = Eigen::Quaternionf(raw[0], raw[1], raw[2], raw[3]) * extrinsic_q_inv_;
        q_body_.normalize();                   // 保证单位四元数 (避免浮点累积误差)
        quat_buf_[0] = q_body_.w();
        quat_buf_[1] = q_body_.x();
        quat_buf_[2] = q_body_.y();
        quat_buf_[3] = q_body_.z();
        return quat_buf_;                      // 返回引用: 生命周期与 this 相同
    }

    // -------------------------------------------------------------------------
    // get_ang_vel — 读取 IMU 角速度 (机体坐标系, [ωx, ωy, ωz] rad/s)
    //
    // 变换: ω_body = extrinsic_R_mat_ × ω_imu
    //       其中 extrinsic_R_mat_ 是 3×3 外参旋转矩阵 (IMU → Body)
    // -------------------------------------------------------------------------
    const std::vector<float>& get_ang_vel() {
        if (!imu_) {
            throw std::runtime_error("IMU not initialized");
        }
        auto raw = imu_->get_ang_vel();        // IMU 原始角速度 (IMU 坐标系)
        // Eigen::Map 零拷贝: 将 raw vector 的内存解释为 Eigen::Vector3f
        Eigen::Map<const Eigen::Vector3f> omega_imu(raw.data());
        // 外参旋转后写入缓冲区
        Eigen::Map<Eigen::Vector3f>(ang_vel_buf_.data()) = extrinsic_R_mat_ * omega_imu;
        return ang_vel_buf_;
    }

    // =========================================================================
    // is_init_ — 电机初始化状态 (原子标志, 无锁跨线程可见)
    //
    // true:  所有电机已完成上电使能 (init_motors 成功)
    // false: 电机未初始化或已下电 (deinit_motors 调用后)
    //
    // 作用:
    //   - get_joint_*() 的前置条件: 防止读取未初始化的垃圾数据
    //   - apply_action() 的前置条件: 防止向未上电的电机发命令
    //   - control 线程判断是否可以执行控制循环
    // =========================================================================
    std::atomic<bool> is_init_{false};

   private:
    // =========================================================================
    // 配置对象 (从 robot.yaml 加载, 智能指针管理, 构造后只读不修改)
    // =========================================================================
    std::shared_ptr<IMUCfg> imu_cfg_;                // IMU 配置
    std::shared_ptr<MotorsCfg> motors_cfg_;          // 电机配置
    std::shared_ptr<RobotCfg> robot_cfg_;            // 机器人控制配置

    // =========================================================================
    // 离线检测阈值
    //
    // 电机驱动返回的 response_count 记录连续丢帧次数。
    // 超过此阈值 (25 次) 判定电机离线 → throw runtime_error → control 线程捕获
    // → 暂停发送命令, 防止电机在无反馈状态下继续执行已失效的指令
    // =========================================================================
    int offline_threshold_ = 25;

    // =========================================================================
    // 硬件驱动实例
    // =========================================================================
    std::shared_ptr<IMUDriver> imu_;                       // IMU 驱动 (具体型号由工厂创建)

    // =========================================================================
    // IMU 外参变换矩阵
    //
    // extrinsic_R_mat_:  3×3 旋转矩阵, 将 IMU 坐标系矢量转到机体坐标系
    // extrinsic_q_inv_:  对应四元数的逆, 用于旋转 IMU 姿态四元数
    //
    // 为什么需要外参?
    //   - IMU 安装在机器人躯干上, 但不在机体几何中心
    //   - IMU 坐标系和 URDF 定义的机体坐标系之间存在旋转偏差
    //   - 训练数据是在 Isaac Sim 的机体坐标系下采集的
    //   - 部署时必须将 IMU 数据变换到同样的坐标系, 否则模型输入不一致
    //
    // 外参标定方法: 将机器人摆出已知姿态, 对比 IMU 读数和理论值求 R
    // =========================================================================
    Eigen::Matrix3f extrinsic_R_mat_ = Eigen::Matrix3f::Identity();     // 默认: 无旋转
    Eigen::Quaternionf extrinsic_q_inv_ = Eigen::Quaternionf::Identity(); // 默认: 无旋转

    // =========================================================================
    // IMU 数据缓存 (成员变量, 避免每次读取 malloc)
    // =========================================================================
    Eigen::Quaternionf q_body_;                      // 机体坐标系四元数 (中间变量)
    std::vector<float> quat_buf_{0.f, 0.f, 0.f, 0.f};    // 四元数缓存 [w, x, y, z]
    std::vector<float> ang_vel_buf_{0.f, 0.f, 0.f};      // 角速度缓存 [ωx, ωy, ωz]

    // =========================================================================
    // 电机驱动实例列表 (按 CAN 顺序, 索引为电机在数组中的位置)
    // =========================================================================
    std::vector<std::shared_ptr<MotorDriver>> motors_;

    // =========================================================================
    // 线程池 — CAN 总线并行通信
    //
    // 每个 CAN 口需要一个线程独立收发。线程池大小 = CAN 口数量,
    // 构造时创建, 析构时回收。
    //
    // 使用场景:
    //   - exec_motors_parallel(): 通过 run_parallel 批量提交任务
    //   - motors_mit_cmd():       发送 MIT 命令 (高频, 250Hz)
    // =========================================================================
    std::unique_ptr<ThreadPool> thread_pool_;

    // =========================================================================
    // =========================================================================
    // 互斥锁
    //
    // motors_mutex_: 保护 CAN 总线通信的互斥
    //   - exec_motors_parallel 和 motors_mit_cmd 都需要独占 CAN 口
    //   - 两者可能被不同线程调用 (control 和 main), 需要互斥
    //
    // joint_mutex_: 保护 joint_q_/joint_vel_/joint_tau_
    //   - 写入: apply_action() 中的 exec_motors_parallel (control 线程, 250Hz)
    //   - 读取: get_joint_*() (inference 线程, 50Hz)
    // =========================================================================
    std::mutex motors_mutex_, joint_mutex_;

    // =========================================================================
    // 关节状态缓冲区 (URDF 顺序)
    //
    // 更新频率: 250Hz (在 apply_action 中由电机编码器刷新)
    // 读取频率: 50Hz (inference 线程通过 get_joint_*() 读取)
    //
    // motor2urdf_ 映射:
    //   写入时: joint_q_[motor2urdf_[motor_idx]] = motor_value
    //           CAN 顺序的电机数据映射到 URDF 顺序的关节数据
    //   读取时: 直接返回 joint_q_, 因为已经是 URDF 顺序
    //
    // 踝关节闭链解耦:
    //   在 forward_close_chain() 中原地修改 joint_q_, joint_vel_, joint_tau_
    //   将两个耦合的电机读数解耦为两个独立的关节角度
    // =========================================================================
    std::vector<float> joint_q_, joint_vel_, joint_tau_;

    // =========================================================================
    // 电机控制目标缓冲区 (CAN 顺序, 索引 = 电机在 motors_ 列表中的位置)
    //
    // motor2urdf_ 映射 (反向):
    //   写入时: motor_pos_target_[i] = p[motor2urdf_[i]]
    //           URDF 顺序的关节目标映射到 CAN 顺序的电机目标
    //
    // 五个目标量对应 MIT 控制模式:
    //   pos_target: 位置目标 (电机编码器单位)
    //   vel_target: 速度目标 (rad/s)
    //   kp_target:  位置增益 (Nm/rad)
    //   kd_target:  速度阻尼 (Nm·s/rad)
    //   tau_target: 前馈力矩 (Nm)
    //   最终输出: τ = kp × (pos_target - pos_actual) + kd × (vel_target - vel_actual) + tau_target
    // =========================================================================
    std::vector<float> motor_pos_target_, motor_vel_target_, motor_kp_target_,
                       motor_kd_target_, motor_tau_target_;

    // =========================================================================
    // 索引映射表 (构造时从 robot.yaml 的 urdf2motor 计算)
    //
    // close_chain_joint_idx_: 踝关节在 URDF 中的索引
    //   从 close_chain_motor_idx_ 通过 urdf2motor_ 映射而来
    //   例如: close_chain_motor_idx_ = [10, 11, 14, 15]
    //         urdf2motor_ = [3,0,5,1,2,4,6,7,9,8,10,11,14,15,12,13,16,17,18,19,20,21,22]
    //         → close_chain_joint_idx_ = [10, 11, 12, 13]  (这些是 URDF 中的踝关节)
    //
    // motor2urdf_: CAN 电机索引 → URDF 关节索引
    //   由 urdf2motor_ 反推: motor2urdf_[urdf2motor_[i]] = i
    //   例如: motor2urdf_ = [1, 3, 4, 0, 5, 2, 6, 7, 8, 9, 10, 11, 14, 15, 12, 13, 16, ...]
    //         motor2urdf_[0] = 1 → 电机数组中第 0 个电机对应 URDF 第 1 个关节
    // =========================================================================
    std::vector<int> close_chain_joint_idx_, motor2urdf_;

    // =========================================================================
    // 私有方法
    // =========================================================================

    // -------------------------------------------------------------------------
    // setup_motors — 根据 motors 配置创建所有 MotorDriver 实例
    //
    // 双重循环:
    //   for each CAN 口:
    //     for each 该 CAN 口的电机:
    //       MotorDriver::create_motor(id, interface_type, interface, type, model, ...)
    //
    // 调用 factory 创建具体电机驱动 (如 RpoMotor, UnitreeMotor 等)
    // -------------------------------------------------------------------------
    void setup_motors();

    // -------------------------------------------------------------------------
    // setup_imu — 根据 imu 配置创建 IMUDriver 实例
    //
    // IMUDriver::create_imu(id, interface_type, interface, type, baudrate)
    // 调用 factory 创建具体 IMU 驱动 (如 WitIMU, HipnucIMU 等)
    // -------------------------------------------------------------------------
    void setup_imu();

    // -------------------------------------------------------------------------
    // exec_motors_parallel — 在所有 CAN 口上并行执行同一个命令函数
    //
    // 将一个回调函数 cmd_func 分发到所有电机, 按 CAN 口分组并行执行。
    // 内部加 motors_mutex_ 锁, 保证 CAN 总线通信的原子性。
    //
    // 任务分组逻辑:
    //   motor_interface_[0] = "can0", motor_num_[0] = 3 → task0: 处理电机 0,1,2
    //   motor_interface_[1] = "can1", motor_num_[1] = 2 → task1: 处理电机 3,4
    //   → thread_pool_->run_parallel({task0, task1}): task0 和 task1 并行
    //
    // cmd_func 签名: void(std::shared_ptr<MotorDriver>& motor, int idx)
    //   motor: 电机驱动实例引用
    //   idx:   电机在 motors_ 数组中的索引
    // -------------------------------------------------------------------------
    void exec_motors_parallel(const std::function<void(std::shared_ptr<MotorDriver>&, int)>& cmd_func);

    // -------------------------------------------------------------------------
    // motors_mit_cmd — 通过线程池并行发送 MIT 命令到所有 CAN 口
    //
    // MIT 命令格式因接口类型而异:
    //   - CAN-FD (canfd): 8 个电机打包一帧 (MIT 批量模式, 减少总线负载)
    //   - 普通 CAN:       逐电机发送 (单电机 MIT 模式)
    //
    // motor_sign_ 方向处理:
    //   电机正向与关节正向可能不一致, 发送前乘以 motor_sign_[idx] 翻转方向
    // =========================================================================
    void motors_mit_cmd();

    // forward_close_chain removed — no closed-chain ankle mechanism in use
};
