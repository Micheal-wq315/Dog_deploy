// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

// ============================================================================
// 匿名命名空间 — 文件内部辅助函数 (仅本 .cpp 可见, 类似 C 的 static)
// ============================================================================
namespace {

// ---------------------------------------------------------------------------
// trim_copy — 去除字符串首尾空白字符
//
// 例如 " ang_vel:3 " → "ang_vel:3"
// 用于解析 YAML 配置中的 obs_layout 字符串, 容错空格
// ---------------------------------------------------------------------------
std::string trim_copy(const std::string& value) {
    // find_if_not: 从前往后找第一个非空白字符
    const auto first = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    // 从后往前找第一个非空白字符
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
                                       [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) {
        return "";                         // 全是空白 → 返回空串
    }
    return std::string(first, last);       // 截取 [first, last)
}

// ---------------------------------------------------------------------------
// make_source_spec — 构造一个 ObsSourceSpec 对象
//
// 输入:
//   name:   观测分量名 (如 "ang_vel")
//   source: 对应的采集函数指针 (如 &InferenceNode::get_ang_vel_obs)
//   size:   该分量的维度 (如 3)
// 返回:
//   填充完整的 ObsSourceSpec 结构体
// ---------------------------------------------------------------------------
ObsSourceSpec make_source_spec(const std::string& name,
                                const ObsSourceDefinition& source, int size) {
    ObsSourceSpec spec;
    spec.name = name;
    spec.source = &source;                 // 存指针, 指向静态注册表中的条目
    spec.size = size;
    return spec;
}

// ---------------------------------------------------------------------------
// split_obs_layout_spec — 将逗号分隔的布局字符串拆分为独立的分量描述
//
// 输入:  "ang_vel:3, gravity_b:3, dof_pos:23, dof_vel:23"
// 输出:  ["ang_vel:3", "gravity_b:3", "dof_pos:23", "dof_vel:23"]
//
// 处理细节:
//   - 用逗号作为分隔符
//   - 每个 token 会过一遍 trim_copy 去除空格
//   - 空 token (连续逗号、首尾逗号) 被跳过
// ---------------------------------------------------------------------------
std::vector<std::string> split_obs_layout_spec(const std::string& layout_spec) {
    std::vector<std::string> layout_specs;
    size_t start = 0;
    while (start < layout_spec.size()) {
        const size_t end = layout_spec.find(',', start);   // 找下一个逗号
        // 截取 [start, end) 子串并 trim
        const std::string token = trim_copy(layout_spec.substr(
            start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty()) {
            layout_specs.push_back(token);                 // 非空才加入
        }
        if (end == std::string::npos) {
            break;                                         // 最后一个 token, 结束
        }
        start = end + 1;                                   // 跳过逗号, 继续
    }
    return layout_specs;
}

}  // namespace

// ============================================================================
// obs_source_definitions — 返回所有可用观测源的静态注册表
//
// 这是一个"名字 → 函数指针"的映射表, 编译期确定, 运行时不变。
// 设计为 static 局部变量 → 首次调用时初始化一次, 之后直接返回引用, 零开销。
//
// 注册表的作用:
//   YAML 里的 obs_layout: "ang_vel:3, dof_pos:23, ..."
//        ↓ parse_obs_layout() 解析
//        ↓ 用 "ang_vel" 查这张表 → 找到 &InferenceNode::get_ang_vel_obs
//        ↓ 存入 ObsSourceSpec.source
//        ↓ update_obs_segments() 通过成员函数指针调用采集函数
//
// 添加新观测分量的步骤:
//   1. 在这里加一行注册项 {"new_obs", &InferenceNode::get_new_obs}
//   2. 在头文件声明 get_new_obs(std::vector<float>& segment)
//   3. 在本文件实现 get_new_obs 的采集逻辑
//   4. YAML 的 obs_layout 中加入 "new_obs:维度"
// ============================================================================
const std::vector<ObsSourceDefinition>& InferenceNode::obs_source_definitions() {
    static const std::vector<ObsSourceDefinition> definitions{
        // 传感器数据: IMU 角速度 (机体坐标系, 3 维)
        {"ang_vel", &InferenceNode::get_ang_vel_obs},

        // 传感器数据: 机体坐标系下的重力方向 (由 IMU 四元数计算, 3 维)
        {"gravity_b", &InferenceNode::get_gravity_b_obs},

        // 指令数据: 速度指令 [vx, vy, ωz] (来自手柄摇杆或 /cmd_vel 话题)
        {"cmd_vel", &InferenceNode::get_cmd_vel_obs},

        // 本体感知: 关节位置 (电机编码器读数, 维度由 obs_layout 指定, 经 usd2urdf 重排)
        {"dof_pos", &InferenceNode::get_dof_pos_obs},

        // 本体感知: 关节速度 (电机转速, 维度由 obs_layout 指定, 经 usd2urdf 重排)
        {"dof_vel", &InferenceNode::get_dof_vel_obs},

        // 历史数据: 上一帧 ONNX 模型的输出动作 (维度由 obs_layout 指定)
        // 这是训练/推理一致性的关键: 训练时网络也接收上一帧输出作为输入,
        // 部署时必须复现同样的输入格式
        {"last_action", &InferenceNode::get_last_action_obs},

        // 模式标志: 中断模式标志 (1 维, 0.0=正常推理 / 1.0=中断模式)
        // 让网络知道当前处于中断状态, 可以调整行为 (例如不输出大幅度动作)
        {"interrupt", &InferenceNode::get_interrupt_obs},

    };
    return definitions;
}

// ============================================================================
// parse_obs_layout — 解析 YAML 的观测布局字符串 → 可执行的 ObsSourceSpec 列表
//
// 这是整个观测系统的核心入口。YAML 中的 obs_layout 是一行纯文本:
//   "ang_vel:3, gravity_b:3, cmd_vel:3, dof_pos:23, dof_vel:23, last_action:23"
//
// 解析过程:
//   1. 按逗号拆分成独立的分量描述 ("ang_vel:3", "gravity_b:3", ...)
//   2. 每个分量描述按冒号拆分为 name 和 size
//   3. 用 name 查 obs_source_definitions() 注册表, 找到对应的采集函数指针
//   4. 构造 ObsSourceSpec {name, function_pointer, size}
//
// 参数:
//   layout_spec: YAML 中的布局字符串 (如 "ang_vel:3, gravity_b:3, ...")
//   layout_name: 布局的名称 ("obs_layout" 或 "extra_obs_layout"), 用于错误信息
//
// 返回:
//   vector<ObsSourceSpec> — 顺序与 YAML 中一致, 推理时按此顺序逐分量采集
//
// 格式要求:
//   - 必须是 "name:size" 格式, size 为纯数字
//   - name 必须在 obs_source_definitions() 注册表中存在
//   - layout 中分量的排列顺序即 ONNX 模型输入中观测的顺序
//   - 顺序必须与训练时的 obs_layout 完全一致, 否则输入错位 → 推理输出错误
// ============================================================================
std::vector<ObsSourceSpec> InferenceNode::parse_obs_layout(
    const std::string& layout_spec,
    const std::string& layout_name) {

    // 第一步: 按逗号拆分为独立的分量描述字符串
    const std::vector<std::string> layout_specs = split_obs_layout_spec(layout_spec);
    if (layout_specs.empty()) {
        throw std::runtime_error(layout_name + " must be explicitly configured");
    }

    std::vector<ObsSourceSpec> layout;
    layout.reserve(layout_specs.size());

    for (const std::string& raw_spec : layout_specs) {
        const std::string spec = trim_copy(raw_spec);

        // 第二步: 按冒号拆分为 name 和 size
        const size_t separator = spec.find(':');
        if (separator == std::string::npos || separator == 0 || separator == spec.size() - 1) {
            throw std::runtime_error(
                layout_name + " entry must use 'name:size' format: " + raw_spec);
        }

        const std::string name = trim_copy(spec.substr(0, separator));
        const std::string size_text = trim_copy(spec.substr(separator + 1));

        if (name.empty() || size_text.empty()) {
            throw std::runtime_error(
                layout_name + " entry must use 'name:size' format: " + raw_spec);
        }

        // 第三步: 验证 size 是纯数字
        if (!std::all_of(size_text.begin(), size_text.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            throw std::runtime_error(
                layout_name + " field size must be a positive integer: " + raw_spec);
        }

        // 第四步: 在注册表中查找对应的采集函数指针
        const auto& definitions = obs_source_definitions();
        const auto source = std::find_if(
            definitions.begin(), definitions.end(),
            [&name](const ObsSourceDefinition& definition) {
                return name == definition.name;
            });

        // 注册表中找不到 → 配置错误, 报错
        if (source == definitions.end()) {
            throw std::runtime_error("Unsupported obs source: " + name);
        }

        // 第五步: 构造 ObsSourceSpec 并加入列表
        layout.push_back(make_source_spec(name, *source, std::stoi(size_text)));
    }
    return layout;
}

// ============================================================================
// has_obs_source — 检查任意策略是否使用指定的观测分量
//
// 遍历所有策略的 obs_layout 和 extra_obs_layout,
// 检查是否有名为 source_name 的观测分量。
//
// 用途:
//   - supports_interrupt(): 检查是否有策略包含了 "interrupt" 分量
//     如果有 → 中断模式可用 (因为模型期望接收中断标志)
//     如果没有 → 中断模式不可用 (模型不知道中断状态, 强行中断可能出错)
// ============================================================================
bool InferenceNode::has_obs_source(const std::string& source_name) const {
    return std::any_of(policies_.begin(), policies_.end(),
        [this, &source_name](const PolicyRuntime& policy) {
            // lambda: 在指定 layout 中查找 source_name
            const auto source_matches = [&source_name](const ObsSourceSpec& spec) {
                return spec.name == source_name;
            };
            // 检查主观测布局或额外观测布局中是否包含该分量
            return std::any_of(policy.obs_layout.begin(), policy.obs_layout.end(), source_matches) ||
                   std::any_of(policy.extra_obs_layout.begin(), policy.extra_obs_layout.end(),
                               source_matches);
        });
}

// ============================================================================
// update_obs_segments — 逐分量采集观测: 调用每个分量的采集函数
//
// 遍历 ObsSourceSpec 列表, 对每个分量:
//   1. 取出 source->get (成员函数指针, 指向 get_xxx_obs)
//   2. 调用 (this->*get)(segments[i]) → 将采集结果写入对应的 segment
//
// 成员函数指针调用语法:
//   (this->*(layout[i].source->get))(segments[i])
//     ↑          ↑          ↑            ↑
//     对象指针    解引用运算符  成员函数指针      实参: segment 引用
//
//   等价于: this->get_ang_vel_obs(segments[0])
//          this->get_gravity_b_obs(segments[1])
//          ...
//   但不需要硬编码调用顺序 — 顺序由 YAML 的 obs_layout 动态决定
//
// 参数:
//   segments: 输出, vector<vector<float>>, segments[i] 是第 i 个分量的缓存
//   layout:   输入, ObsSourceSpec 列表, 定义采集什么、按什么顺序
//
// 注意: segments 和 layout 的长度必须一致, 由调用方保证
//       (初始化时 policy.obs_segments 根据 policy.obs_layout 构造)
// ============================================================================
void InferenceNode::update_obs_segments(std::vector<std::vector<float>>& segments,
                                         const std::vector<ObsSourceSpec>& layout) {
    for (size_t i = 0; i < layout.size(); i++) {
        // 通过成员函数指针调用采集函数
        // layout[i].source 指向 ObsSourceDefinition
        // layout[i].source->get 是成员函数指针 (void (InferenceNode::*)(vector<float>&))
        // (this->*ptr)(arg) 是 C++ 调用成员函数指针的标准语法
        (this->*(layout[i].source->get))(segments[i]);
    }
}

// ============================================================================
// flatten_obs_segments — 将多个分量缓存拼接为一维观测向量
//
// 输入:  segments = [[ang_vel:3], [gravity_b:3], [cmd_vel:3],
//                    [dof_pos:23], [dof_vel:23], [last_action:23]]
//
// 输出:  output = [ang_vel₀, ang_vel₁, ang_vel₂,     ← offset 0
//                  gravity_b₀, gravity_b₁, gravity_b₂, ← offset 3
//                  cmd_vel₀, cmd_vel₁, cmd_vel₂,       ← offset 6
//                  dof_pos₀...dof_pos₂₂,               ← offset 9
//                  dof_vel₀...dof_vel₂₂,               ← offset 32
//                  last_action₀...last_action₂₂]        ← offset 55
//                                                        total: 78 维
//
// 这是观测构建的最终一步: 各分量按 layout 顺序首尾相接,
// 形成一个与训练时完全一致的扁平的观测向量, 然后送入帧栈 (update_stacked_obs)
// 和 ONNX 模型。
// ============================================================================
void InferenceNode::flatten_obs_segments(const std::vector<std::vector<float>>& segments,
                                          std::vector<float>::iterator output_begin) {
    int offset = 0;
    for (size_t i = 0; i < segments.size(); i++) {
        // 将第 i 个分量拷贝到输出缓冲区的 offset 位置
        std::copy(segments[i].begin(), segments[i].end(), output_begin + offset);
        offset += static_cast<int>(segments[i].size());  // 偏移递增 → 下一个分量紧挨着
    }
}


// ============================================================================
// 以下是 10 个观测分量采集函数 (getter)
//
// 每个函数遵循相同的接口约定:
//   void get_xxx_obs(std::vector<float>& segment)
//
// 参数 segment:
//   - 大小已由调用方在初始化时 resize 为分量维度 (如 3 或 23)
//   - 函数直接写入 segment[i], 不改变 segment 的大小
//
// 数据来源:
//   - 传感器 (RobotInterface): ang_vel, gravity_b, dof_pos, dof_vel
//   - 共享缓冲区:            cmd_vel (手柄/话题), perception (外部)
//   - 模型输出:              last_action (上一帧 ONNX 输出)
//   - 运动文件:              motion_pos, motion_vel (NPZ 轨迹)
//   - 原子标志位:            interrupt (is_interrupt_)
//
// 缩放 (scaling):
//   训练时观测会做归一化 (除以标准差或以范围缩放), 部署时必须复现完全相同的缩放。
//   缩放系数从 YAML 读取 (obs_scales_ang_vel_, obs_scales_dof_pos_ 等)。
// ============================================================================

// ---------------------------------------------------------------------------
// get_ang_vel_obs — IMU 角速度观测
//
// 数据流:
//   RobotInterface::get_ang_vel() → IMU 原始角速度
//   → 外参旋转 (extrinsic_R_mat_) 从 IMU 坐标系转到机体坐标系
//   → × obs_scales_ang_vel_ 缩放
//   → 写入 segment[0:2] = [ωx, ωy, ωz]
// ---------------------------------------------------------------------------
void InferenceNode::get_ang_vel_obs(std::vector<float>& segment) {
    ang_vel_buffer_ = robot_->get_ang_vel();   // IMU 角速度 (已做外参变换)
    for (int i = 0; i < 3; i++) {
        segment[i] = ang_vel_buffer_[i] * obs_scales_ang_vel_;
    }
}

// ---------------------------------------------------------------------------
// get_gravity_b_obs — 机体坐标系下的重力方向观测
//
// 从 IMU 姿态四元数计算重力矢量在机体坐标系下的投影。
//
// 计算过程:
//   1. IMU 输出的是 q_body→world (机体 → 世界坐标系的旋转)
//   2. 世界坐标系下重力恒为 [0, 0, -1] (Z 轴向下)
//   3. 机体坐标系下重力 = q_world→body × [0, 0, -1]
//   4. q_world→body = q_body→world.inverse()
//
// 夹带安全检测:
//   重力 Z 分量 > gravity_z_upper_ → 机器人倒地 → 紧急关机
//   当机器人倒地时, 机体 Z 轴不再指向天空,
//   重力在机体 Z 轴的分量变小 (甚至为正), 这是跌倒的特征信号。
// ---------------------------------------------------------------------------
void InferenceNode::get_gravity_b_obs(std::vector<float>& segment) {
    quat_buffer_ = robot_->get_quat();         // IMU 四元数 [w, x, y, z] (已做外参补偿)
    // 构造四元数: w 为标量, xyz 为向量
    Eigen::Quaternionf q_b2w(quat_buffer_[0], quat_buffer_[1],
                              quat_buffer_[2], quat_buffer_[3]);
    // 世界坐标系下的重力方向 (指向地心)
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
    // 求逆得到 q_world→body, 将世界系重力转到机体坐标系
    Eigen::Quaternionf q_w2b = q_b2w.inverse();
    Eigen::Vector3f gravity_b = q_w2b * gravity_w;

    // 跌落检测: 机器人倒地时重力 Z 分量异常
    //   正常站立: gravity_b.z ≈ -1 (重力几乎全部在机体 Z 轴负方向)
    //   倒地:     gravity_b.z ≈ 0 或正值
    //   阈值 gravity_z_upper_ 如 -0.7 → Z 分量大于此值即判定为倒地
    if (gravity_b.z() > gravity_z_upper_) {
        RCLCPP_FATAL(this->get_logger(), "Robot fell down! Shutting down...");
        rclcpp::shutdown();
        throw std::runtime_error("Robot fell down");
    }

    // 写入缩放后的重力矢量
    segment[0] = gravity_b.x() * obs_scales_gravity_b_;
    segment[1] = gravity_b.y() * obs_scales_gravity_b_;
    segment[2] = gravity_b.z() * obs_scales_gravity_b_;
}

// ---------------------------------------------------------------------------
// get_cmd_vel_obs — 速度指令观测
//
// 数据源:
//   - 手柄模式 (is_joy_control_=true):  /joy 话题 → subs_joy_callback → cmd_vel_
//   - 话题模式 (is_joy_control_=false): /cmd_vel 话题 → subs_cmd_callback → cmd_vel_
//
// cmd_vel_ = [vx, vy, ωz] 其中 v 为线速度 (m/s), ω 为偏航角速度 (rad/s)
// 缩放: 线速度分量 × obs_scales_lin_vel_, 角速度分量 × obs_scales_ang_vel_
//
// 加锁原因: cmd_vel_ 由 main 线程 (ROS2 回调) 写入, 由 inference 线程读取
// ---------------------------------------------------------------------------
void InferenceNode::get_cmd_vel_obs(std::vector<float>& segment) {
    std::unique_lock<std::mutex> lock(cmd_mutex_);   // 保护 cmd_vel_ 的读写互斥
    segment[0] = cmd_vel_[0] * obs_scales_lin_vel_;  // vx 缩放
    segment[1] = cmd_vel_[1] * obs_scales_lin_vel_;  // vy 缩放
    segment[2] = cmd_vel_[2] * obs_scales_ang_vel_;  // ωz 缩放
}

// ---------------------------------------------------------------------------
// get_dof_pos_obs — 关节位置观测 (本体感知)
//
// 数据流:
//   电机编码器 (绝对位置) → RobotInterface::get_joint_q()
//   → usd2urdf_ 重排 (从 Isaac Sim USD 关节顺序 → URDF 顺序)
//   → 减去默认角度 (zero offset, 使零位对应站姿)
//   → × obs_scales_dof_pos_ 缩放
//
// usd2urdf_ 映射:
//   训练在 Isaac Sim 中运行, 关节顺序是 USD 文件定义的;
//   部署在真实机器人上, 关节顺序是 URDF 文件定义的。
//   两个顺序可能不同 — usd2urdf_[i] 表示 USD 第 i 个关节对应 URDF 第几个关节。
//   例如 usd2urdf_[0]=5 表示推理中索引 0 取的是 URDF 索引 5 的关节数据。
//
// 夹带安全检测:
//   任意关节位置超出 joint_limits_ 范围 → 关节限位触发 → 紧急关机
//   joint_limits_ 格式: [lower₀, upper₀, lower₁, upper₁, ...]
// ---------------------------------------------------------------------------
void InferenceNode::get_dof_pos_obs(std::vector<float>& segment) {
    joint_pos_buffer_ = robot_->get_joint_q();     // 从硬件读取关节位置 (URDF 顺序)

    // 按 USD 顺序重排并做偏移+缩放
    // 注意: dof_pos 只含腿部关节 (segment.size() = 12), 轮子不在此观测内
    const int n = static_cast<int>(segment.size());
    for (int i = 0; i < n; i++) {
        segment[i] = (joint_pos_buffer_[usd2urdf_[i]]            // 从 URDF 索引取数据
                      - joint_default_angle_[usd2urdf_[i]])      // 减去默认角度 (归零)
                     * obs_scales_dof_pos_;                      // 缩放
    }

    // 关节限位检测 (用 URDF 顺序的原始值, 不缩放)
    // joint_limits_ 是 [lower₀, upper₀, lower₁, upper₁, ...] 交错格式
    for (size_t i = 0; i < joint_limits_.size() / 2; i++) {
        if (joint_pos_buffer_[i] < joint_limits_[i * 2] ||          // 低于下限
            joint_pos_buffer_[i] > joint_limits_[i * 2 + 1]) {      // 高于上限
            RCLCPP_FATAL(this->get_logger(), "Joint %zu out of limit! Shutting down...", i + 1);
            rclcpp::shutdown();
            throw std::runtime_error("Joint out of limit");
        }
    }
}

// ---------------------------------------------------------------------------
// get_dof_vel_obs — 关节速度观测 (本体感知)
//
// 同 dof_pos 的处理流程: 读取 → usd2urdf 重排 → 缩放。
// 速度缩放系数与位置不同 (obs_scales_dof_vel_ vs obs_scales_dof_pos_),
// 因为速度和位置的值域不同 (速度通常 ±10 rad/s, 位置通常 ±3.14 rad)。
// ---------------------------------------------------------------------------
void InferenceNode::get_dof_vel_obs(std::vector<float>& segment) {
    joint_vel_buffer_ = robot_->get_joint_vel();   // 从硬件读取关节速度 (URDF 顺序)
    const int n = static_cast<int>(segment.size());
    for (int i = 0; i < n; i++) {
        segment[i] = joint_vel_buffer_[usd2urdf_[i]] * obs_scales_dof_vel_;
    }
}

// ---------------------------------------------------------------------------
// get_last_action_obs — 上一帧动作观测 (自回归输入)
//
// 这是训练/推理一致性的关键设计点之一:
//   训练时网络接收上一帧的输出动作作为输入 (类似于自回归),
//   推理时必须使用完全相同的信号 — 即上一帧 ONNX "输出" 而非"执行到电机的值"。
//
// 为什么用 output_buffer 而不是 last_act_?
//   - last_act_ 经过了 EMA 平滑, 与训练时的信号不一致
//   - output_buffer 是 ONNX 模型的原始输出, 与训练时 behavior cloning 的目标一致
//   - 如果用 EMA 平滑后的值, 会累积分布偏移 (distribution shift)
//
// 注意: 首帧时 output_buffer 还是初始化时的全零 → 网络收到的 last_action 是全零,
//       这与训练时的 initial observation 一致 (训练首帧也没有上一帧动作)。
// ---------------------------------------------------------------------------
void InferenceNode::get_last_action_obs(std::vector<float>& segment) {
    const auto& policy = active_policy();
    const int n = static_cast<int>(segment.size());
    for (int i = 0; i < n; i++) {
        segment[i] = policy.ctx->output_buffer[i];    // 直接从 ONNX 输出缓冲区取
    }
}

// ---------------------------------------------------------------------------
// get_interrupt_obs — 中断模式标志观测
//
// 将原子布尔标志转换为 1 维浮点数观测:
//   0.0 → 正常推理模式 (网络按策略输出动作)
//   1.0 → 中断模式 (外部直接控制关节, 网络知道自己在中断)
//
// 为什么需要这个观测?
//   中断模式下控制线程发送的是 interrupt_action_ 而不是网络输出,
//   但网络还在运行。如果网络不知道自己处于中断状态,
//   它的 last_action 会偏离实际机器人正在执行的动作,
//   会导致观测分布偏移。有了这个标志, 网络可以调整内部状态
//   (例如 AMP 的判别器网络在中断时可能会切换行为)。
//
// 原子读取: is_interrupt_ 由 main 线程写入 (服务回调), 由 inference 线程读取
// ---------------------------------------------------------------------------
void InferenceNode::get_interrupt_obs(std::vector<float>& segment) {
    segment[0] = is_interrupt_.load() ? 1.0f : 0.0f;
}

// get_perception_obs removed — not used by wheel_quad
