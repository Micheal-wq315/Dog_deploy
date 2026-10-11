// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <sys/mman.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include "robot_interface.hpp"

// ============================================================================
// ObsStackOrder — 观测帧栈的排列方式枚举
//
// 控制多帧观测历史在 ONNX 输入的 input_buffer 中如何排列:
//   FrameMajor: 整帧为单位 [frm₀(78) | frm₁(78) | ... | frm₉(78)]
//   ObsMajor:   分量为单位 [ang_vel₀ ang_vel₁ ang_vel₂ | grav₀ grav₁ grav₂ | ...]
// ============================================================================
enum class ObsStackOrder {
    FrameMajor,  // 帧优先排列 (RPO-Flat 默认, 10 帧历史)
    ObsMajor,    // 观测分量优先排列 (AMP 使用, 3 帧历史)
};

class InferenceNode;

// ============================================================================
// ObsSourceDefinition — 观测源定义 (静态注册表)
//
// 将观测分量名称和其采集函数指针绑定在一起，构成一个"观测源注册表"。
// 每个观测分量 (ang_vel, gravity_b, dof_pos 等) 都有一条注册项。
// ============================================================================
struct ObsSourceDefinition {
    const char* name;                          // 观测源名称 (如 "ang_vel")
    void (InferenceNode::*get)(std::vector<float>& segment);  // 指向采集函数的成员函数指针
};

// ============================================================================
// ObsSourceSpec — 观测源规格 (运行时实例)
//
// 从 YAML 解析出的 obs_layout 中的一项，记录某个观测分量的:
//   name:   观测分量名称 (对应 ObsSourceDefinition::name)
//   source: 指向该分量采集函数的指针
//   size:   该分量的维度 (如 ang_vel=3, dof_pos=23)
// ============================================================================
struct ObsSourceSpec {
    std::string name;                          // 分量名
    const ObsSourceDefinition* source;         // 采集函数指针
    int size;                                  // 分量维度
};

// ============================================================================
// InferenceNode — 机器人推理节点主类
//
// 继承自 rclcpp::Node (ROS2 节点)，负责:
//   1. 加载 YAML 配置 → 初始化多个策略 (PolicyRuntime)
//   2. 创建 ONNX Runtime 推理会话 (ModelContext)
//   3. 订阅传感器话题 (手柄/JointState/IMU/高程) → 构建观测 → 运行推理 → 发布动作
//   4. 提供 10 个 ROS2 Service 用于电机控制和推理管理
//
// 线程模型 (三线程):
//   main 线程:      ROS2 回调 (话题订阅、Service 处理)
//   inference 线程: 50Hz, SCHED_FIFO priority=70, ONNX 推理 + 观测构建
//   control 线程:   250Hz, SCHED_FIFO priority=70, 动作发布
//
// 运行模式:
//   手柄模式 (默认):   摇杆 → cmd_vel
//   /cmd_vel 模式:     外部速度指令
//   运动策略模式:       播放预录运动轨迹 (NPZ)
//   中断模式:           外部直接控制关节位置
// ============================================================================
class InferenceNode : public rclcpp::Node {
   public:
    // =========================================================================
    // ModelContext — 单个 ONNX 模型的推理上下文
    //
    // 持有 ONNX Runtime Session、内存分配器、输入/输出张量的 buffer 和 shape。
    // 一个 ModelContext 对应一个 .onnx 文件，每个策略 (PolicyRuntime) 持有一个。
    // =========================================================================
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;       // ONNX Runtime 推理会话
        std::unique_ptr<Ort::MemoryInfo> memory_info; // CPU 内存分配器
        std::unique_ptr<Ort::Value> input_tensor;     // 输入张量 (绑定到 input_buffer)
        std::unique_ptr<Ort::Value> output_tensor;    // 输出张量 (绑定到 output_buffer)
        std::vector<std::string> input_names;         // 模型输入节点名列表
        std::vector<std::string> output_names;        // 模型输出节点名列表
        std::vector<const char*> input_names_raw;     // 输入节点名 C 字符串 (ORT API 要求)
        std::vector<const char*> output_names_raw;    // 输出节点名 C 字符串
        std::vector<int64_t> input_shape;             // 输入 shape (如 [1, 159])
        std::vector<int64_t> output_shape;            // 输出 shape (如 [1, 16])
        std::vector<float> input_buffer;              // 输入数据缓冲区 (帧栈展开后写入此)
        std::vector<float> output_buffer;             // 输出数据缓冲区 (推理结果读取此)
        size_t num_inputs;                            // 输入节点数
        size_t num_outputs;                           // 输出节点数
    };

    // =========================================================================
    // PolicyRuntime — 单个策略的运行时状态
    //
    // 每个策略对应一个 ONNX 模型，有独立的:
    //   - 观测布局 (obs_layout) 和帧栈参数 (frame_stack, stack_order)
    //   - ONNX 推理上下文 (ModelContext)
    //   - 可选的运动轨迹文件 (motion_loader)
    //   - 首帧标志 (is_first_frame) 用于观测历史初始化
    // =========================================================================
    struct PolicyRuntime {
        std::string name;                            // 策略/模型名 (如 "policy.onnx")
        std::string model_path;                      // ONNX 模型文件绝对路径
        std::vector<ObsSourceSpec> obs_layout;       // 主观测布局规范列表
        std::vector<int> obs_layout_sizes;           // 各分量维度 (如 [3,3,3,12,16,16])
        std::vector<std::vector<float>> obs_segments; // 各分量临时缓存 (采集时逐分量填入)
        std::vector<float> obs;                      // 单帧观测向量 (所有分量拼接后)
        std::vector<ObsSourceSpec> extra_obs_layout; // 额外观测布局 (如 perception)
        std::vector<std::vector<float>> extra_obs_segments; // 额外观测分量缓存
        int obs_num = 0;                             // 单帧观测总维度 (wheel_quad: 53)
        int extra_obs_num = 0;                       // 额外观测总维度 (RPO 感知用, wheel_quad: 0)
        int frame_stack = 1;                         // 保留多少帧历史 (wheel_quad: 3)
        ObsStackOrder stack_order = ObsStackOrder::FrameMajor; // 帧排列方式
        std::unique_ptr<ModelContext> ctx;           // ONNX 推理上下文
        bool is_first_frame = true;                  // 是否首帧 (用于初始化历史缓冲区)
    };

    // =========================================================================
    // 构造函数 — 加载配置、初始化硬件、创建 ONNX 环境、启动推理/控制线程
    // =========================================================================
    InferenceNode() : Node("inference_node") {
        load_config();                                        // 加载 YAML 配置

        robot_ = std::make_shared<RobotInterface>(robot_config_path_);

        // 配置 ONNX Runtime 线程数
        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");

        if (policies_.empty()) {
            throw std::runtime_error("At least one policy must be configured");
        }

        // 为每个策略创建 ONNX 推理会话并初始化观测缓存
        for (size_t i = 0; i < policies_.size(); i++) {
            PolicyRuntime& policy = policies_[i];
            policy.obs.resize(policy.obs_num, 0.0f);
            policy.obs_segments.resize(policy.obs_layout.size());
            for (size_t j = 0; j < policy.obs_layout.size(); j++) {
                policy.obs_segments[j].resize(policy.obs_layout[j].size, 0.0f);
            }
            policy.extra_obs_segments.resize(policy.extra_obs_layout.size());
            for (size_t j = 0; j < policy.extra_obs_layout.size(); j++) {
                policy.extra_obs_segments[j].resize(policy.extra_obs_layout[j].size, 0.0f);
            }
            // Motion policy support removed (not used by wheel_quad)
            // input_size = obs_num × frame_stack + extra_obs_num
            // 如: 53 × 3 + 0 = 159 (wheel_quad)
            setup_model(policy.ctx, policy.model_path,
                        policy.obs_num * policy.frame_stack + policy.extra_obs_num);
        }
        initialize_runtime_state();
        reset_runtime_state();

        // ── 创建 ROS2 订阅者 ──────────────────────────────────────────────
        // 使用 best_effort + KeepLast(1) 避免延迟堆积
        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", data_qos,
            std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", data_qos,
            std::bind(&InferenceNode::subs_cmd_callback, this, std::placeholders::_1));
        elevation_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            perception_obs_topic_, data_qos,
            std::bind(&InferenceNode::subs_elevation_callback, this, std::placeholders::_1));
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_ref_states", data_qos,
            std::bind(&InferenceNode::subs_joint_state_callback, this, std::placeholders::_1));

        // ── 创建发布者 ────────────────────────────────────────────────────
        action_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/action", data_qos);
        imu_publisher_ =
            this->create_publisher<sensor_msgs::msg::Imu>("/imu", data_qos);
        joint_state_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", data_qos);

        // ── 启动工作线程 ──────────────────────────────────────────────────
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        control_thread_ = std::thread(&InferenceNode::control, this);

        // ── 创建 ROS2 Service ─────────────────────────────────────────────
        reset_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_joints",
            std::bind(&InferenceNode::reset_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        stand_up_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stand_up",
            std::bind(&InferenceNode::stand_up_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<std_srvs::srv::Trigger>(
            "set_zeros",
            std::bind(&InferenceNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "clear_errors",
            std::bind(&InferenceNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
        refresh_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "refresh_joints",
            std::bind(&InferenceNode::refresh_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_joints",
            std::bind(&InferenceNode::read_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_imu_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_imu",
            std::bind(&InferenceNode::read_imu_srv, this, std::placeholders::_1, std::placeholders::_2));
        init_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "init_motors",
            std::bind(&InferenceNode::init_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        deinit_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "deinit_motors",
            std::bind(&InferenceNode::deinit_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        start_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_inference",
            std::bind(&InferenceNode::start_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        stop_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stop_inference",
            std::bind(&InferenceNode::stop_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
    }

    ~InferenceNode() {
        // 析构顺序: 先等线程结束，再释放硬件和 ONNX 资源
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        reset_runtime_state();
        if (robot_) {
            robot_.reset();
        }
    }

    // 查询是否有中断观测源 (ang_vel, cmd_vel 等之外的 interrupt 分量)
    bool supports_interrupt() const;
    // has_motion_policy removed — not used by wheel_quad

   private:
    // =========================================================================
    // 硬件接口与运行时状态
    // =========================================================================
    std::shared_ptr<RobotInterface> robot_;                 // 机器人硬件抽象层 (读写电机/IMU)
    std::atomic<bool> is_running_{false};                   // 推理是否运行中
    std::atomic<bool> is_joy_control_{true};                // true=手柄控制 / false=/cmd_vel 控制
    std::atomic<bool> is_interrupt_{false};                 // 是否处于中断模式 (外部控制关节)
    std::atomic<bool> is_motion_policy_{false};             // 是否处于运动策略模式 (播放 NPZ 轨迹)
    std::string robot_config_path_;                         // robot.yaml 路径
    std::string perception_obs_topic_;                      // 感知观测话题名 (如 "elevation_data")
    size_t current_motion_policy_idx_ = 0;                  // 当前选中的运动策略在 motion_policy_indices_ 中的序号
    int active_policy_idx_ = 0;                             // 当前活跃策略在 policies_ 中的索引
    int perception_obs_num_;                                // 感知观测维度 (高度图)
    int joint_num_;                                         // 关节数量 (wheel_quad: 16)
    int decimation_;                                        // 控制频率降采样因子 (decimation × dt = 控制周期)

    // =========================================================================
    // ONNX Runtime 环境
    // =========================================================================
    std::unique_ptr<Ort::Env> env_;                         // ONNX Runtime 全局环境
    int intra_threads_;                                     // ONNX Runtime 内部线程数 (-1=自动)
    Ort::AllocatorWithDefaultOptions allocator_;            // ONNX Runtime 内存分配器

    // =========================================================================
    // ROS2 通信 (订阅者/发布者/Service)
    // =========================================================================
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr elevation_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr action_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

    // =========================================================================
    // 工作线程 (三线程模型)
    //
    // main 线程:       ROS2 回调 — 话题订阅和 Service 响应 (由 executor 驱动)
    // inference 线程:  50Hz, ONNX 推理 — 采集传感器数据 → 构建观测 → 帧栈更新 → 模型推理
    // control 线程:    250Hz, 动作发布 — 将最新推理结果 act_ 发布到 /action
    // =========================================================================
    std::thread inference_thread_;
    std::thread control_thread_;

    // =========================================================================
    // 配置参数 (从 YAML 加载)
    // =========================================================================
    float act_alpha_;                                      // 动作平滑系数 (EMA: act = α × new_act + (1-α) × old_act)
    float dt_;                                             // 物理时间步长 (s)
    float obs_scales_lin_vel_;                             // 线速度观测缩放
    float obs_scales_ang_vel_;                             // 角速度观测缩放
    float obs_scales_dof_pos_;                             // 关节位置观测缩放
    float obs_scales_dof_vel_;                             // 关节速度观测缩放
    float obs_scales_gravity_b_;                           // 重力方向观测缩放
    float clip_observations_;                              // 观测截断阈值
    std::vector<double> action_scales_;                    // 每关节动作缩放 (USD 顺序, [joint_num])
    std::vector<long int> wheel_joint_indices_;            // 速度控制关节的 USD 索引 (轮子)
    float clip_actions_;                                   // 动作截断阈值
    std::vector<double> clip_cmd_;                         // 速度指令限幅 [x_min, x_max, y_min, y_max, yaw_min, yaw_max]
    std::vector<double> joint_default_angle_;              // 关节默认角度 (零点位置)
    std::vector<double> joint_limits_;                     // 关节软限位 [lower, upper, lower, upper, ...]
    std::vector<long int> usd2urdf_;                       // USD→URDF 关节索引映射
    float gravity_z_upper_;                                // 重力 Z 分量上限 (用于跌落检测)

    // =========================================================================
    // 手柄按钮状态 (用于边沿检测)
    // =========================================================================
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0,
        last_button3_ = 0, last_button4_ = 0, last_button5_ = 0;

    // =========================================================================
    // 策略列表
    // =========================================================================
    std::vector<PolicyRuntime> policies_;                   // 所有策略 (至少 1 个)
    std::vector<int> motion_policy_indices_;                // 运动策略在 policies_ 中的索引

    // =========================================================================
    // ROS2 Service 实例 (11 个)
    // =========================================================================
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_joints_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stand_up_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr set_zeros_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_errors_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr refresh_joints_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr read_joints_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr read_imu_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr init_motors_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr deinit_motors_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_inference_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_inference_service_;

    // =========================================================================
    // 互斥锁 — 保护各线程共享的数据
    //
    // act_mutex_:        act_ / last_act_ (inference 写, control 读)
    // perception_mutex_: perception_obs_buffer_ (回调写, inference 读)
    // interrupt_mutex_:  interrupt_action_ (回调写, control 读)
    // cmd_mutex_:        cmd_vel_ (joy/cmd_vel 回调写, inference 读)
    // mode_mutex_:       is_interrupt_, is_motion_policy_, active_policy_idx_ (回调写, inference 读)
    // lb_switch_mutex_:  策略切换瞬态保护 (防止 LB 按钮快速连按导致状态混乱)
    // =========================================================================
    std::mutex act_mutex_, perception_mutex_, interrupt_mutex_,
               cmd_mutex_, mode_mutex_, lb_switch_mutex_;

    // =========================================================================
    // 数据缓冲区 — 线程间共享
    // =========================================================================
    std::vector<float> act_;                                // 当前策略输出的目标关节位置 [joint_num]
    std::vector<float> last_act_;                           // 上一帧的输出 (用于 EMA 平滑)
    std::vector<float> act_vel_;                            // 当前策略输出的目标关节速度 (轮子) [joint_num]
    std::vector<float> last_act_vel_;                       // 上一帧的速度输出 (用于 EMA 平滑)
    std::vector<float> cmd_vel_;                            // 速度指令 [vx, vy, ωz] (3 维)
    std::vector<float> interrupt_action_;                   // 中断模式下外部发送的关节目标
    std::vector<float> perception_obs_buffer_;              // 感知观测缓存 (高度图等)
    std::vector<float> joint_pos_buffer_;                   // 关节位置缓存 (从硬件读取)
    std::vector<float> joint_vel_buffer_;                   // 关节速度缓存
    std::vector<float> joint_torques_buffer_;               // 关节力矩缓存
    std::vector<float> quat_buffer_;                        // IMU 四元数缓存 [w, x, y, z]
    std::vector<float> ang_vel_buffer_;                     // IMU 角速度缓存 [ωx, ωy, ωz]
    sensor_msgs::msg::JointState joint_state_msg_;         // 预分配的 JointState 消息 (复用避免 malloc)
    sensor_msgs::msg::JointState action_msg_;              // 预分配的动作消息

    // =========================================================================
    // ROS2 回调函数 (main 线程)
    // =========================================================================
    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);

    // =========================================================================
    // 工作线程函数
    // =========================================================================
    void inference();   // 推理主循环: 采集观测 → 帧栈更新 → ONNX Run → 写 act_
    void control();     // 控制主循环: 读 act_ → 发布 /action

    // 将 act_ 写入机器人硬件 (中断模式下使用 interrupt_action_, 否则使用策略输出)
    void apply_action();

    // 获取当前活跃策略的引用 (可读写 / 只读)
    PolicyRuntime& active_policy();
    const PolicyRuntime& active_policy() const;

    // =========================================================================
    // 配置与模型初始化
    // =========================================================================
    void load_config();   // 加载 YAML 配置 → 构建 policies_ 列表
    void setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size);

    // =========================================================================
    // 运行时状态管理
    // =========================================================================
    void initialize_runtime_state();                        // 初始化所有 buffer 和消息结构
    void reset_runtime_state();                             // 重置当前策略的状态 (首帧标志 + 运动帧)
    void reset_policy_runtime(PolicyRuntime& policy);       // 重置指定策略的观测历史
    void step_motion_frame();                               // 运动策略模式: 推进一帧

    // =========================================================================
    // 观测注册与解析 (静态/半静态)
    // =========================================================================
    // 返回所有可用观测源的静态注册表 (编译期确定，运行时不变)
    static const std::vector<ObsSourceDefinition>& obs_source_definitions();
    // 解析 YAML 的 obs_layout 字符串，如 "ang_vel:3, gravity_b:3, ..." → vector<ObsSourceSpec>
    std::vector<ObsSourceSpec> parse_obs_layout(const std::string& layout_spec,
                                                const std::string& layout_name);
    // 检查 policies_ 中是否包含名为 source_name 的观测分量
    bool has_obs_source(const std::string& source_name) const;
    // 解析 YAML 的 obs_stack_orders 字符串 → ObsStackOrder 枚举
    ObsStackOrder parse_obs_stack_order(const std::string& stack_order_name);

    // =========================================================================
    // 观测运行时辅助函数
    // =========================================================================
    // 逐个采集观测分量 → 写入对应 obs_segments (调用 ObsSourceDefinition::get)
    void update_obs_segments(std::vector<std::vector<float>>& segments,
                             const std::vector<ObsSourceSpec>& layout);
    // 将所有分量缓存拼接为单帧观测向量 obs (各分量首尾相接)
    void flatten_obs_segments(const std::vector<std::vector<float>>& segments,
                              std::vector<float>::iterator output_begin);
    // 将单帧观测 obs 写入 input_buffer 的滑动窗口 (FrameMajor 或 ObsMajor)
    void update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                            int obs_num, int frame_stack, ObsStackOrder stack_order,
                            const std::vector<int>& field_sizes, bool is_first_frame);

    // =========================================================================
    // 观测采集函数 — 各观测分量的 getter
    //
    // 每个函数从对应的数据源读取当前值，写入 segment 中。
    // 数据来源:
    //   传感器数据:   joint_pos_buffer_, ang_vel_buffer_, quat_buffer_
    //   指令数据:     cmd_vel_ (手柄或 /cmd_vel)
    //   历史数据:     last_act_ (上一帧动作输出)
    //   运动轨迹数据:  motion_loader (NPZ 文件中的参考关节位置/速度)
    //   感知数据:     perception_obs_buffer_ (外部发布的高程数据)
    //   模式数据:     is_interrupt_ (中断模式标志)
    // =========================================================================
    void get_cmd_vel_obs(std::vector<float>& segment);      // 速度指令 "cmd_vel":3
    void get_ang_vel_obs(std::vector<float>& segment);      // 角速度 "ang_vel":3
    void get_gravity_b_obs(std::vector<float>& segment);    // 重力方向 "gravity_b":3
    void get_dof_pos_obs(std::vector<float>& segment);      // 关节位置 "dof_pos":23
    void get_dof_vel_obs(std::vector<float>& segment);      // 关节速度 "dof_vel":23
    void get_last_action_obs(std::vector<float>& segment);  // 上一帧动作 "last_action":23
    void get_interrupt_obs(std::vector<float>& segment);    // 中断标志 "interrupt":1

    // =========================================================================
    // ROS2 Service 回调
    // =========================================================================
    void init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stand_up_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // =========================================================================
    // 状态发布
    // =========================================================================
    void publish_joint_states();  // 发布 /joint_states (位置+速度+力矩)
    void publish_action();        // 发布 /action (目标关节位置)
    void publish_imu();           // 发布 /imu (四元数姿态+角速度)

    // 调试辅助: 打印 vector 内容到日志
    template <typename T>
    void print_vector(const std::string& name, const std::vector<T>& vec) {
        std::stringstream ss;
        ss << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
};
