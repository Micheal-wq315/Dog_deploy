// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "robot_interface.hpp"

// ============================================================================
// 构造函数 — 加载 robot.yaml 并初始化所有硬件子系统
//
// 从 YAML 配置文件顺序解析三段配置:
//   ① imu:    (可选) IMU 配置 → setup_imu() 创建 IMU 驱动
//   ② motors: (必须) 电机配置 → setup_motors() 创建所有电机驱动
//   ③ robot:  (必须) 机器人控制配置 → 索引映射 + IMU 外参 + 闭链解耦器
//
// 配置对象用 shared_ptr 管理 (可传递到外部), 解析完毕创建线程池和初始化状态 buffer。
//
// 参数:
//   config_file: robot.yaml 的绝对路径 (由 InferenceNode 构造函数传入,
//                路径来自 ROS2 参数 robot_config)
//
// 错误处理:
//   文件不存在或 YAML 格式错误 → YAML::LoadFile 抛出异常
//   缺少 motors 段            → throw runtime_error
//   缺少 robot 段             → throw runtime_error
// ============================================================================
RobotInterface::RobotInterface(const std::string& config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    // ── ① IMU 配置 (可选) ──────────────────────────────────────────────
    // 如果 robot.yaml 中没有 imu: 段 → 跳过, 无 IMU 的配置仍然可用
    imu_cfg_ = std::make_shared<IMUCfg>();
    if (config["imu"]) {
        YAML::Node imu_node = config["imu"];
        if (imu_node["imu_id"]) imu_cfg_->imu_id_ = imu_node["imu_id"].as<int>();
        if (imu_node["baudrate"]) imu_cfg_->baudrate_ = imu_node["baudrate"].as<int>();
        if (imu_node["imu_type"]) imu_cfg_->imu_type_ = imu_node["imu_type"].as<std::string>();
        if (imu_node["imu_interface_type"]) imu_cfg_->imu_interface_type_ = imu_node["imu_interface_type"].as<std::string>();
        if (imu_node["imu_interface"]) imu_cfg_->imu_interface_ = imu_node["imu_interface"].as<std::string>();
        setup_imu();   // 工厂创建具体 IMU 驱动 (如 WitIMU / HipnucIMU)
    }

    // ── ② 电机配置 (必须) ──────────────────────────────────────────────
    // 每个字段前用 if 做宽容解析 — 缺失字段用默认值, 兼容新旧 YAML 版本
    motors_cfg_ = std::make_shared<MotorsCfg>();
    if (config["motors"]) {
        YAML::Node motors_node = config["motors"];
        if (motors_node["motor_zero_offset"]) motors_cfg_->motor_zero_offset_ = motors_node["motor_zero_offset"].as<std::vector<double>>();
        if (motors_node["master_id_offset"]) motors_cfg_->master_id_offset_ = motors_node["master_id_offset"].as<int>();
        if (motors_node["motor_type"]) motors_cfg_->motor_type_ = motors_node["motor_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface_type"]) motors_cfg_->motor_interface_type_ = motors_node["motor_interface_type"].as<std::vector<std::string>>();
        if (motors_node["motor_interface"]) motors_cfg_->motor_interface_ = motors_node["motor_interface"].as<std::vector<std::string>>();
        if (motors_node["motor_id"]) motors_cfg_->motor_id_ = motors_node["motor_id"].as<std::vector<long int>>();
        if (motors_node["motor_model"]) motors_cfg_->motor_model_ = motors_node["motor_model"].as<std::vector<long int>>();
        if (motors_node["motor_num"]) motors_cfg_->motor_num_ = motors_node["motor_num"].as<std::vector<long int>>();
        setup_motors();  // 工厂创建所有 MotorDriver 实例
    } else {
        throw std::runtime_error("Motors configuration not found in " + config_file);
    }

    // ── ③ 机器人运动学/控制配置 (必须) ─────────────────────────────────
    robot_cfg_ = std::make_shared<RobotCfg>();
    if (config["robot"]) {
        YAML::Node robot_node = config["robot"];
        if (robot_node["kp"]) robot_cfg_->kp_ = robot_node["kp"].as<std::vector<double>>();
        if (robot_node["kd"]) robot_cfg_->kd_ = robot_node["kd"].as<std::vector<double>>();
        if (robot_node["close_chain_motor_idx"]) robot_cfg_->close_chain_motor_idx_ = robot_node["close_chain_motor_idx"].as<std::vector<long int>>();
        if (robot_node["motor_sign"]) robot_cfg_->motor_sign_ = robot_node["motor_sign"].as<std::vector<long int>>();
        if (robot_node["urdf2motor"]) robot_cfg_->urdf2motor_ = robot_node["urdf2motor"].as<std::vector<long int>>();

        // ── 构建 motor2urdf_ 反向索引映射 ───────────────────────────
        // urdf2motor_[urdf_idx] = motor_idx  →  motor2urdf_[motor_idx] = urdf_idx
        // 例: urdf2motor_ = [3, 0, 5, 1, ...]
        //      → motor2urdf_[3] = 0, motor2urdf_[0] = 1, ...
        motor2urdf_ = std::vector<int>(motors_cfg_->motor_id_.size(), -1);  // -1 标记未映射
        for (size_t i = 0; i < robot_cfg_->urdf2motor_.size(); ++i) {
            motor2urdf_[robot_cfg_->urdf2motor_[i]] = i;
        }

        // ── IMU 外参旋转矩阵 ────────────────────────────────────────
        if (robot_node["extrinsic_R"]) {
            robot_cfg_->extrinsic_R_ = robot_node["extrinsic_R"].as<std::vector<double>>();
            if (robot_cfg_->extrinsic_R_.size() == 9) {
                // 按行主序 (Row-major) 的 9 个元素:
                //   [r00, r01, r02, r10, r11, r12, r20, r21, r22]
                // R 将 Body→IMU 坐标系的旋转矩阵
                extrinsic_R_mat_ << robot_cfg_->extrinsic_R_[0], robot_cfg_->extrinsic_R_[1], robot_cfg_->extrinsic_R_[2],
                                    robot_cfg_->extrinsic_R_[3], robot_cfg_->extrinsic_R_[4], robot_cfg_->extrinsic_R_[5],
                                    robot_cfg_->extrinsic_R_[6], robot_cfg_->extrinsic_R_[7], robot_cfg_->extrinsic_R_[8];
                Eigen::Quaternionf q_R(extrinsic_R_mat_);   // R → 四元数 (Body→IMU)
                extrinsic_q_inv_ = q_R.inverse();            // 逆 = IMU→Body (get_quat 中用到)
            }
        }

        // ── 构建 close_chain_joint_idx_ — 踝关节的 URDF 索引 ───────
        // close_chain_motor_idx_ 存的是 CAN 总线电机索引,
        // 通过 urdf2motor_ 转换为 URDF 关节索引供 forward_close_chain 使用
        // close_chain_motor_idx_[i] → (在 urdf2motor_ 中查找) → URDF 索引
        for (auto idx : robot_cfg_->close_chain_motor_idx_) {
            auto it = std::find(robot_cfg_->urdf2motor_.begin(), robot_cfg_->urdf2motor_.end(), idx);
            if (it != robot_cfg_->urdf2motor_.end()) {
                close_chain_joint_idx_.push_back(std::distance(robot_cfg_->urdf2motor_.begin(), it));
            }
        }

        // 闭链解耦功能已移除
    } else {
        throw std::runtime_error("Robot configuration not found in " + config_file);
    }

    // ── 创建线程池 ──────────────────────────────────────────────────────
    // 线程数 = CAN 口数 (motor_interface_ 的元素个数)
    // 每个 CAN 口一个专用 worker, 实现 CAN 总线间并行
    thread_pool_ = std::make_unique<ThreadPool>(motors_cfg_->motor_interface_.size());

    // ── 状态缓冲区初始化为全零 ──────────────────────────────────────────
    // 大小 = 电机总数 (wheel_quad: 16)
    joint_q_   = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_vel_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_tau_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_pos_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_vel_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kp_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_kd_target_  = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    motor_tau_target_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
}

// ============================================================================
// setup_motors — 创建所有 MotorDriver 实例
//
// 配置结构: motor_interface_ + motor_num_ 描述 CAN 总线拓扑
//   motor_interface_[0]="can0"  motor_num_[0]=3  → CAN0 上挂 3 个电机
//   motor_interface_[1]="can1"  motor_num_[1]=2  → CAN1 上挂 2 个电机
//
// MotorDriver::create_motor 是工厂函数:
//   根据 motor_type + motor_interface_type 创建具体驱动
//   如 type="RPO" + interface="canfd" → RpoCanFdMotor
// ============================================================================
void RobotInterface::setup_motors(){
    size_t count = 0;
    motors_.resize(motors_cfg_->motor_id_.size());   // 预分配, 避免 push_back 的扩容开销

    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i) {
        for (size_t j = 0; j < motors_cfg_->motor_num_[i]; ++j) {
            motors_[count] = MotorDriver::create_motor(
                motors_cfg_->motor_id_[count],           // 电机硬件 ID (CAN 帧地址)
                motors_cfg_->motor_interface_type_[i],   // "canfd" 或 "can"
                motors_cfg_->motor_interface_[i],        // "can0", "can1"...
                motors_cfg_->motor_type_[count],         // 电机类型 (每电机一个, 如 "LRO_CAN"/"RS02")
                motors_cfg_->motor_model_[count],        // 电机子型号
                motors_cfg_->master_id_offset_,          // 主机 CAN ID 偏移
                motors_cfg_->motor_zero_offset_[count]   // 编码器零位修正
            );
            count += 1;
        }
    }
}

// ============================================================================
// setup_imu — 创建 IMU 驱动实例
// ============================================================================
void RobotInterface::setup_imu(){
    imu_ = IMUDriver::create_imu(imu_cfg_->imu_id_,
                                  imu_cfg_->imu_interface_type_,
                                  imu_cfg_->imu_interface_,
                                  imu_cfg_->imu_type_,
                                  imu_cfg_->baudrate_);
}

// forward_close_chain removed — no closed-chain ankle mechanism in use

// ============================================================================
// apply_action — 施加动作到机器人硬件 (最核心接口, control 线程 250Hz)
//
// MIT 控制公式 (在电机驱动层执行):
//   τ = kp × (p_des - p_actual) + kd × (v_des - v_actual) + τ_ff
//
// 参数 (均为 URDF 关节顺序, 默认值 = 空 vector):
//   p:   目标关节位置 [rad]
//   v:   目标关节速度 [rad/s]
//   kp:  位置增益 [Nm/rad]
//   kd:  速度阻尼 [Nm·s/rad]
//   tau: 前馈力矩 [Nm]
//
// 六步流程:
//   ① 前置检查 — 电机未初始化 → 直接返回
//   ② 反向读出 — 电机编码器 → joint_q_, joint_vel_, joint_tau_ (加 joint_mutex_)
//   ③ 正解耦   — (可选) forward_close_chain: 耦合电机值 → 独立关节值
//   ④ 逆解耦   — (可选) get_decoupleQVT: 独立关节力矩 → 耦合电机力矩
//   ⑤ 组装目标 — URDF 顺序 → CAN 顺序 (加 motors_mutex_)
//   ⑥ 发送命令 — motors_mit_cmd() 并行发送到各 CAN 口
// ============================================================================
void RobotInterface::apply_action(std::vector<float> p,
                                  std::vector<float> v,
                                  std::vector<float> kp,
                                  std::vector<float> kd,
                                  std::vector<float> tau) {
    // ① 前置检查: 电机未初始化 → 不执行, 防止向未上电电机发命令
    if(!is_init_.load()){
        return;
    }

    {
        // ② 反向读出: 从电机编码器读取实际状态
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            joint_q_[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];

            if (motor->get_response_count() > offline_threshold_) {
                throw std::runtime_error(
                    "Motor id " + std::to_string(motors_cfg_->motor_id_[idx]) + " offline");
            }
        });
    }  // ← joint_mutex_ 释放

    // ③ 组装电机目标值 (motors_mutex_ 保护)
    // URDF 顺序 → CAN 顺序, MIT 位置控制
    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_pos_target_.size(); i++){
            const size_t ji = motor2urdf_[i];
            motor_pos_target_[i] = p[ji];
            motor_vel_target_[i] = v.empty()  ? 0.0f : v[ji] * robot_cfg_->motor_sign_[i];
            motor_kp_target_[i]  = kp.empty() ? static_cast<float>(robot_cfg_->kp_[i]) : kp[ji];
            motor_kd_target_[i]  = kd.empty() ? static_cast<float>(robot_cfg_->kd_[i]) : kd[ji];
            motor_tau_target_[i] = tau.empty() ? 0.0f : tau[ji] * robot_cfg_->motor_sign_[i];
        }
    }

    // ⑥ 并行发送 MIT 命令
    motors_mit_cmd();
}

// ============================================================================
// reset_joints — 缓慢将机器人复位到默认姿态 (关节零位)
//
// 两阶段 KP 缩放防冲击策略:
//   阶段1: KP = 原始 × 0.4  →  电机力矩小, 缓慢趋近目标 (不伤减速器)
//          等待 1000ms     →  机器人到达目标附近
//   阶段2: KP = 原始值      →  恢复正常工作刚度, 保持姿态
//
// 闭链关节需要先逆解耦: 独立关节角度 → 并联电机目标位置
// ============================================================================
void RobotInterface::reset_joints(std::vector<double> joint_default_angle) {
    // ① 阶段1: 低 KP 缓慢复位
    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_pos_target_.size(); i++){
            motor_pos_target_[i] = static_cast<float>(joint_default_angle[motor2urdf_[i]]);
            motor_vel_target_[i] = 0.0f;
            motor_kp_target_[i]  = static_cast<float>(robot_cfg_->kp_[i]) * (1.0f / 2.5f);  // KP → 40%
            motor_kd_target_[i]  = static_cast<float>(robot_cfg_->kd_[i]);
            motor_tau_target_[i] = 0.0f;
        }
    }
    motors_mit_cmd();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));   // 等待完成运动

    // ③ 阶段2: 恢复满 KP, 保持姿态
    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_kp_target_.size(); i++){
            motor_kp_target_[i] = static_cast<float>(robot_cfg_->kp_[i]);   // 恢复 100%
        }
    }
    motors_mit_cmd();
}

// ============================================================================
// stand_up — 从当前姿态平滑插值过渡到默认站立姿态
//
// 与 reset_joints 的区别:
//   reset_joints 直接跳到目标位置 (仅靠降 KP 防冲击),
//   stand_up     从当前位置线性插值到目标, 分多步渐进过渡。
//
// 流程:
//   ① refresh_joints() 读取当前所有关节位置作为起点
//   ② 计算步数 = duration / dt (如 3s / 0.004 = 750 步)
//   ③ 每步: lerp(起点, 终点, t) → MIT 命令 → sleep(dt)
//   ④ 到达终点后, 恢复满 KP 保持姿态
//
// 参数:
//   joint_default_angle: 目标站立姿态 (URDF 顺序, 弧度)
//   duration_sec:        过渡时长 (秒), 默认 3.0
//
// 线程安全:
//   此函数应由 main 线程在推理停止期间调用 (类似 reset_joints),
//   不与 control 线程的 apply_action 并发。
// ============================================================================
void RobotInterface::stand_up(std::vector<double> joint_default_angle, double duration_sec) {
    if (!is_init_.load()) {
        throw std::runtime_error("Motors not initialized, cannot stand up");
    }
    const int num_joints = motors_cfg_->motor_id_.size();
    const double dt = 0.004;                        // 控制周期 250Hz
    const int steps = static_cast<int>(duration_sec / dt);
    if (steps < 1) {
        throw std::runtime_error("Stand up duration too short");
    }

    // ① 读取当前关节位置作为插值起点 (URDF 顺序)
    std::vector<double> start_pos(num_joints, 0.0);
    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([&](std::shared_ptr<MotorDriver>& motor, int idx) {
            start_pos[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
        });
    }

    // ② 逐步插值 + 低 KP 渐进过渡
    for (int step = 0; step <= steps; step++) {
        const double t = static_cast<double>(step) / steps;  // 0 → 1
        {
            std::unique_lock<std::mutex> lock(motors_mutex_);
            for (size_t i = 0; i < motor_pos_target_.size(); i++) {
                const size_t ji = motor2urdf_[i];
                // 线性插值: pos = start + (target - start) × t
                motor_pos_target_[i] = static_cast<float>(
                    start_pos[ji] + (joint_default_angle[ji] - start_pos[ji]) * t);
                motor_vel_target_[i] = 0.0f;
                motor_kp_target_[i]  = static_cast<float>(robot_cfg_->kp_[i]) * (1.0f / 2.5f);  // 40% KP
                motor_kd_target_[i]  = static_cast<float>(robot_cfg_->kd_[i]);
                motor_tau_target_[i] = 0.0f;
            }
        }
        motors_mit_cmd();
        std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(dt * 1e6)));
    }

    // ③ 到达终点, 恢复满 KP 保持姿态
    {
        std::unique_lock<std::mutex> lock(motors_mutex_);
        for (size_t i = 0; i < motor_kp_target_.size(); i++) {
            motor_kp_target_[i] = static_cast<float>(robot_cfg_->kp_[i]);
        }
    }
    motors_mit_cmd();
}

// ============================================================================
// refresh_joints — 刷新关节状态 (初始化阶段, main 线程)
//
// 三阶段:
//   ① 主动请求 (refresh_motor_status) → ② 等待 1000ms → ③ 读取数据
//   (可选) ④ 踝关节解耦
//
// 与 apply_action 中读回的区别: refresh 需要主动请求 + 等待回复;
// apply_action 是读回刚发送命令后已就绪的编码器数据, 不需要额外交互。
// ============================================================================
void RobotInterface::refresh_joints() {
    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            motor->refresh_motor_status();   // ① 向电机发送状态请求
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // ② 等待回复

        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            // ③ 读取编码器数据
            joint_q_[motor2urdf_[idx]] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[motor2urdf_[idx]] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[motor2urdf_[idx]] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
        });

    }
}

// ============================================================================
// 以下四个函数是电机的管理接口, 都通过 exec_motors_parallel 并行执行
// ============================================================================

void RobotInterface::set_zeros() {
    // 将当前编码器位置记录为零点偏置, 后续 get_motor_pos() 返回相对位置
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->set_motor_zero();
    });
}

void RobotInterface::clear_errors() {
    // 清除电机故障码 (过流/过温/通信超时/跟踪误差)
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->clear_motor_error();
    });
}

void RobotInterface::init_motors() {
    // 电机上电使能 + 清除故障 + 配置 MIT 模式
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->init_motor();
    });
    is_init_.store(true);   // 原子写入, control 线程立即可见
}

void RobotInterface::deinit_motors() {
    // 电机失能, 自由转动 (无保持力矩)
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->deinit_motor();
    });
    is_init_.store(false);  // 原子写入, 阻止 control 线程继续发送命令
}

// ============================================================================
// motors_mit_cmd — 通过线程池并行发送 MIT 命令到所有 CAN 口
//
// CAN-FD vs 普通 CAN:
//   CAN-FD: 8 个电机打包一帧 MIT 批量命令 (减少总线负载, 单帧搞定)
//   普通 CAN: 逐电机单独发送 (每个电机一帧)
//
// 电机 ID → 槽位映射 (CAN-FD 模式):
//   motor_id ∈ [1,8] → slot = motor_id - 1
//   否则 → slot = j (按序排列)
// ============================================================================
void RobotInterface::motors_mit_cmd() {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;

    for (size_t bus = 0; bus < motors_cfg_->motor_interface_.size(); ++bus) {
        const size_t num_motors = motors_cfg_->motor_num_[bus];
        const size_t start_count = count;

        if (motors_cfg_->motor_interface_type_[bus] == "canfd") {
            // ── CAN-FD 批量模式 ──────────────────────────────────────
            tasks.push_back([this, start_count, num_motors]() {
                float pos[8] = {}, vel[8] = {}, kp[8] = {}, kd[8] = {}, tau[8] = {};
                for (size_t j = 0; j < num_motors; ++j) {
                    const size_t idx = start_count + j;
                    const long int motor_id = motors_cfg_->motor_id_[idx];
                    const size_t slot = (motor_id > 0 && motor_id <= 8)
                        ? static_cast<size_t>(motor_id - 1) : j;   // ID→槽位
                    if (slot >= 8) continue;
                    pos[slot] = motor_pos_target_[idx] * robot_cfg_->motor_sign_[idx];
                    vel[slot] = motor_vel_target_[idx];
                    kp[slot]  = motor_kp_target_[idx];
                    kd[slot]  = motor_kd_target_[idx];
                    tau[slot] = motor_tau_target_[idx];
                }
                motors_[start_count]->motor_mit_cmd(pos, vel, kp, kd, tau);  // 批量发送
            });
        } else {
            // ── 普通 CAN 逐电机模式 ──────────────────────────────────
            tasks.push_back([this, start_count, num_motors]() {
                for (size_t j = 0; j < num_motors; ++j) {
                    const size_t idx = start_count + j;
                    motors_[idx]->motor_mit_cmd(
                        motor_pos_target_[idx] * robot_cfg_->motor_sign_[idx],
                        motor_vel_target_[idx],
                        motor_kp_target_[idx],
                        motor_kd_target_[idx],
                        motor_tau_target_[idx]
                    );
                }
            });
        }
        count += num_motors;
    }

    thread_pool_->run_parallel(tasks);   // 所有 CAN 口并行发送
}

// ============================================================================
// exec_motors_parallel — 在所有 CAN 口上并行执行同一命令函数
//
// 核心逻辑: 按 CAN 口分组 → 组内串行 (共享物理总线) → 组间并行 (独立 CAN 口)
//
// 参数 cmd_func: void(shared_ptr<MotorDriver>& motor, int idx)
//   motor: 电机驱动引用
//   idx:   电机在 motors_ 数组中的全局索引
//
// motors_mutex_ 锁: 防止 main 线程 (如 init_motors) 和 control 线程
//   (如 apply_action→motors_mit_cmd) 同时操作同一条 CAN 总线 → 总线冲突
// ============================================================================
void RobotInterface::exec_motors_parallel(const std::function<void(std::shared_ptr<MotorDriver>&, int)>& cmd_func) {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;

    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i) {
        size_t num_motors = motors_cfg_->motor_num_[i];     // 该 CAN 口的电机数
        size_t start_idx = count;                            // 该口第一个电机的全局索引

        // 每个 CAN 口 = 一个 task, task 内电机串行
        tasks.push_back([this, start_idx, num_motors, cmd_func]() {
            for (size_t j = 0; j < num_motors; ++j) {
                size_t idx = start_idx + j;
                cmd_func(motors_[idx], idx);
            }
        });
        count += num_motors;
    }

    thread_pool_->run_parallel(tasks);   // 等待所有 task 完成
}
