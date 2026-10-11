// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

// ---------------------------------------------------------------------------
// 解析帧栈顺序配置字符串 → 枚举值
//
// YAML 中 obs_stack_orders 字段控制观测历史如何在 ONNX 输入 buffer 中排列:
//   - "frame_major": 按帧优先，连续的整帧观测排在一起
//   - "obs_major":   按观测分量优先，每个分量的历史帧排在一起
// ---------------------------------------------------------------------------
ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

// ---------------------------------------------------------------------------
// 维护滑动窗口式的观测历史缓冲区（帧栈）
//
// 每次推理前调用，将最新的单帧观测 obs[] 写入 input_buffer 的历史窗口中。
// 支持两种内存布局:
//
// ┌─ FrameMajor (默认, 用于10帧历史) ─────────────────────────────┐
// │  input_buffer = [frm₀(78) │ frm₁(78) │ ... │ frm₉(78)]       │
// │                                                               │
// │  首帧: 用同一帧填充全部10个位置（历史缓冲区初始化）              │
// │  后续: 整体左移1帧 → 最新帧填入第10个槽位                       │
// │                                                               │
// │  例如 obs_num=78, frame_stack=10:                              │
// │    [frm₀...frm₉] → 左移78 → [frm₁...frm₉, ] ← new_obs[78]   │
// └───────────────────────────────────────────────────────────────┘
//
// ┌─ ObsMajor (AMP 使用, 用于3帧历史) ────────────────────────────┐
// │  input_buffer = [ang_vel₀ ang_vel₁ ang_vel₂ │ grav₀ grav₁ grav₂ │ ...]│
// │                                                               │
// │  每个观测分量独立维护自己的历史窗口:                             │
// │    field_sizes = [3, 3, 3, 12, 16, 16]                        │
// │    (ang_vel, gravity, cmd, joint_pos, joint_vel, action)       │
// │                                                               │
// │  首帧: 每个分量的历史槽位全部填入当前值                          │
// │  后续: 每个分量独立左移1格 → 最新值填入最后槽位                   │
// │                                                               │
// │  例如 ang_vel(3维), frame_stack=3:                             │
// │    [v₀ v₁ v₂] → 左移3 → [v₁ v₂, ] ← new_ang_vel[3]           │
// └───────────────────────────────────────────────────────────────┘
//
// 参数:
//   input_buffer:  ONNX 模型的输入缓冲区（会被原地修改）
//   obs:           最新一帧的原始观测向量 [obs_num]
//   obs_num:       单帧观测维度（wheel_quad: 53）
//   frame_stack:   保留多少帧历史（如 10）
//   stack_order:   FrameMajor 或 ObsMajor
//   field_sizes:   仅在 ObsMajor 模式下使用，指定每个分量的维度
//   is_first_frame: true 表示首帧（全部填充），false 表示后续帧（滑动窗口）
// ---------------------------------------------------------------------------
void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    // ============================================================
    // FrameMajor 模式：整帧为单位滑动
    // ============================================================
    if (stack_order == ObsStackOrder::FrameMajor) {
        if (is_first_frame) {
            // 首帧: 将当前观测复制 frame_stack 次，填满整个历史窗口
            // 这样网络看到的是一段"静止的"初始状态
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin(), obs.end(), input_buffer.begin() + frame * obs_num);
            }
        } else {
            // 后续帧: 整帧左移，丢弃最老的一帧
            //   [frm₀ frm₁ ... frm₉]                 (移动前)
            //   ↓ std::move 左移 obs_num 个元素
            //   [frm₁ frm₂ ... frm₉ frm₉]            (移动后，最后一帧是脏数据)
            //   ↓ std::copy 写入最新观测
            //   [frm₁ frm₂ ... frm₉ new_frm]          (完成)
            std::move(input_buffer.begin() + obs_num,                     // 源起始: 第1帧
                      input_buffer.begin() + frame_stack * obs_num,       // 源结束: 最后一帧之后
                      input_buffer.begin());                              // 目标: 第0帧位置
            std::copy(obs.begin(), obs.end(),
                      input_buffer.begin() + (frame_stack - 1) * obs_num); // 最新帧写入最后槽位
        }
        return;
    }

    // ============================================================
    // ObsMajor 模式：每个观测分量独立滑动
    //
    // 布局示例 (obs_num=78, frame_stack=3, field_sizes=[3,3,3,23,23,23]):
    //
    //   [ang_vel₀ ang_vel₁ ang_vel₂ │ grav₀ grav₁ grav₂ │ cmd₀ cmd₁ cmd₂ │
    //    jpos₀ jpos₁ jpos₂ │ jvel₀ jvel₁ jvel₂ │ act₀ act₁ act₂]
    //
    //   input_offset  = 当前分量在 input_buffer 中的起始位置
    //   obs_offset    = 当前分量在 obs[] 中的起始位置
    // ============================================================
    int input_offset = 0;  // input_buffer 中的偏移，按分量逐个推进
    int obs_offset = 0;    // obs 向量中的偏移，按分量逐个推进

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            // 首帧: 将每个分量的当前值复制 frame_stack 次
            // 例如 ang_vel = [1.0, 2.0, 3.0], frame_stack=3:
            //   → [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset,
                          obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            // 后续帧: 该分量的历史窗口左移一个 slot，丢弃最老的值
            // 例如 ang_vel field_size=3, frame_stack=3:
            //   [v₀ₓ v₀ᵧ v₀𝓏 │ v₁ₓ v₁ᵧ v₁𝓏 │ v₂ₓ v₂ᵧ v₂𝓏]
            //   ↓ std::move 左移 field_size=3 个元素
            //   [v₁ₓ v₁ᵧ v₁𝓏 │ v₂ₓ v₂ᵧ v₂𝓏 │ v₂ₓ v₂ᵧ v₂𝓏]  (最后一格是脏数据)
            //   ↓ std::copy 从新观测中取对应分量填入最后槽位
            //   [v₁ₓ v₁ᵧ v₁𝓏 │ v₂ₓ v₂ᵧ v₂𝓏 │ newₓ newᵧ new𝓏] (完成)
            std::move(input_buffer.begin() + input_offset + field_size,               // 源起始: 该分量第1帧
                      input_buffer.begin() + input_offset + frame_stack * field_size, // 源结束: 最后帧之后
                      input_buffer.begin() + input_offset);                           // 目标: 该分量第0帧
            std::copy(obs.begin() + obs_offset,
                      obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size); // 最新值写入最后槽位
        }
        // 推进到下一个分量在两种 buffer 中的位置
        input_offset += field_size * frame_stack;  // input: 跳过该分量全部 frame_stack 帧
        obs_offset += field_size;                   // obs:   跳过该分量的一帧
    }
}

// ============================================================================
// setup_model — 创建 ONNX Runtime 推理会话并绑定输入/输出张量
//
// 流程:
//   1. 配置 SessionOptions (图优化、CPU 内存池、禁用多线程)
//   2. 从 .onnx 文件创建 Ort::Session
//   3. 读取模型的 input/output shape 和节点名
//   4. 校验模型 input_size 与 YAML 配置是否一致
//   5. 创建 input_tensor / output_tensor (绑定到 input_buffer / output_buffer)
//
// ONNX Runtime API 要求节点名以 const char* 数组传入，所以维护两套:
//   input_names       → std::vector<std::string>  (拥有字符串内存)
//   input_names_raw   → std::vector<const char*>  (指向 input_names, 给 ORT API 用)
//
// 为什么禁用 ONNX 内部线程:
//   推理循环已绑定到实时线程 (SCHED_FIFO 70)，ONNX 内部线程池会干扰调度
//   且每个 Session 只被一个线程使用，不需要内部并行
// ============================================================================
void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size) {
    // 如果 ctx 尚未创建则分配
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    // ── 配置 ONNX Runtime 会话选项 ───────────────────────────────────────
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();                      // 禁用 ONNX 内部线程池
    session_options.EnableCpuMemArena();                             // 启用 CPU 内存池 (减少 malloc)
    session_options.EnableMemPattern();                              // 启用内存复用模式
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // 全量图优化

    // ── 创建推理会话 ─────────────────────────────────────────────────────
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // ── 解析模型输入 ─────────────────────────────────────────────────────
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();                      // 保存节点名字符串
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();  // 如 [-1, 780]
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;     // 动态 batch → 固定为 1
    }

    // ── 校验输入维度 ─────────────────────────────────────────────────────
    // 将 shape 各维乘起来，与 YAML 配置的 input_size 对比
    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " +
            std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    // ── 解析模型输出 ─────────────────────────────────────────────────────
    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);                            // 输出 = 关节目标位置 [16]

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();  // 如 [1, 16]
    }

    // ── 构建 const char* 指针数组 (供 ORT Run API 使用) ──────────────────
    ctx->input_names_raw = std::vector<const char*>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char*>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();       // 指向 input_names 内部 buffer
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    // ── 创建 CPU 内存分配器 ──────────────────────────────────────────────
    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    // ── 创建输入/输出张量 (直接绑定到 buffer, 零拷贝) ─────────────────────
    // CreateTensor 不复制数据，而是用 data() 指针创建 tensor view
    // 推理前写 input_buffer → 推理后读 output_buffer
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(),
        ctx->input_shape.data(), ctx->input_shape.size()));

    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(),
        ctx->output_shape.data(), ctx->output_shape.size()));
}

// ============================================================================
// reset_runtime_state — 重置所有运行时状态到初始值
//
// 在以下场景调用:
//   - 节点启动时
//   - 手柄按 B 键暂停推理
//   - 手柄按 A 键复位关节
//
// 重置内容:
//   - 停止推理、退出中断/运动模式
//   - 速度指令清零
//   - 动作输出复位到默认关节角度
//   - 所有策略的观测历史和运动帧归零
// ============================================================================
void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    active_policy_idx_ = 0;

    // 速度指令清零 (加锁保护 joy/cmd_vel 回调)
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    }
    // 感知观测清零
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    // 动作输出复位到默认关节角度
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (int i = 0; i < joint_num_; i++) {
            act_[i] = static_cast<float>(joint_default_angle_[i]);
            last_act_[i] = static_cast<float>(joint_default_angle_[i]);
            act_vel_[i] = 0.0f;
            last_act_vel_[i] = 0.0f;
        }
    }
    // 中断模式: 将中断关节目标复位到默认角度
    // 注意 interrupt_action_ 可能只覆盖部分关节 (如最后 10 个)
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    // 重置所有策略的观测历史和运动帧
    for (PolicyRuntime& policy : policies_) {
        reset_policy_runtime(policy);
    }
}

// ============================================================================
// active_policy — 获取当前活跃策略的引用
//
// active_policy_idx_ 变化场景:
//   - 初始化: 0 (默认策略)
//   - 运动模式切换 (LB 键): 在默认策略和运动策略间切换
//   - 运动策略轮转 (RB 键): 在多个运动策略间切换
//
// 提供 const 和非 const 两个重载:
//   non-const: 推理线程使用 (修改 is_first_frame, input_buffer 等)
//   const:     查询时使用
// ============================================================================
InferenceNode::PolicyRuntime& InferenceNode::active_policy() {
    return policies_[active_policy_idx_];
}

const InferenceNode::PolicyRuntime& InferenceNode::active_policy() const {
    return policies_[active_policy_idx_];
}

// ============================================================================
// initialize_runtime_state — 首次初始化所有 buffer 和消息结构
//
// 仅在构造函数中调用一次。分配各缓冲区的初始大小并填零。
// 与 reset_runtime_state() 的区别:
//   - initialize:  首次分配内存 + 设置消息元数据 (joint name 等)
//   - reset:       已有 buffer, 仅将值归零
// ============================================================================
void InferenceNode::initialize_runtime_state() {
    active_policy_idx_ = 0;

    // 预分配 JointState 消息的字段名
    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i] = "action_" + std::to_string(i + 1);
    }

    // 分配数据缓冲区
    cmd_vel_.assign(3, 0.0f);                                     // [vx, vy, ωz]
    act_.assign(joint_num_, 0.0f);                                 // 当前动作输出 [16]
    last_act_.assign(joint_num_, 0.0f);                            // 上一帧动作 (用于 EMA 平滑)
    act_vel_.assign(joint_num_, 0.0f);                             // 当前速度输出 (轮子) [16]
    last_act_vel_.assign(joint_num_, 0.0f);                        // 上一帧速度输出 (用于 EMA 平滑)
    joint_pos_buffer_.assign(joint_num_, 0.0f);                    // 关节位置 [16]
    joint_vel_buffer_.assign(joint_num_, 0.0f);                    // 关节速度 [16]
    joint_torques_buffer_.assign(joint_num_, 0.0f);               // 关节力矩 [16]
    quat_buffer_.assign(4, 0.0f);                                  // 姿态四元数 [w,x,y,z]
    ang_vel_buffer_.assign(3, 0.0f);                               // 角速度 [ωx,ωy,ωz]

    // 感知观测: 仅当 obs_layout 中有 "perception" 分量时才分配
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }

    // 中断模式: 仅当 obs_layout 中有 "interrupt" 分量时才分配
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);                       // 中断关节数固定 10 个
    } else {
        interrupt_action_.clear();
    }
}


// ============================================================================
// supports_interrupt — 是否支持中断模式
//
// 判断依据: interrupt_action_ 是否已分配
// (在 initialize_runtime_state 中根据 obs_layout 中是否有 "interrupt" 分量决定)
// ============================================================================
bool InferenceNode::supports_interrupt() const {
    return !interrupt_action_.empty();
}

// ============================================================================
// reset_policy_runtime — 重置指定策略的运行时状态
//
// 清空该策略的:
//   - 单帧观测向量 obs[]
//   - 各观测分量缓存 obs_segments[][]
//   - ONNX 输入/输出 buffer
//   - 运动帧序号 → 0
//   - 首帧标志 → true (下次推理时会初始化帧栈历史)
// ============================================================================
void InferenceNode::reset_policy_runtime(PolicyRuntime& policy) {
    std::fill(policy.obs.begin(), policy.obs.end(), 0.0f);
    for (auto& segment : policy.obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    for (auto& segment : policy.extra_obs_segments) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    if (policy.ctx) {
        std::fill(policy.ctx->input_buffer.begin(), policy.ctx->input_buffer.end(), 0.0f);
        std::fill(policy.ctx->output_buffer.begin(), policy.ctx->output_buffer.end(), 0.0f);
    }
    policy.is_first_frame = true;   // 下次推理时触发帧栈初始化
}

// ============================================================================
// apply_action — 将动作输出写入机器人硬件
//
// 由 control 线程以 250Hz 频率调用。
// 对策略输出的 act_ 做 EMA (指数移动平均) 平滑后写入机器人:
//   last_act = α × act + (1-α) × last_act
// 其中 α = act_alpha_ (如 0.9)，值越小平滑越多，值越大响应越快
// ============================================================================
void InferenceNode::apply_action() {
    // 推理未运行或电机未初始化 → 不写硬件
    if (!is_running_.load() || !robot_->is_init_.load()) {
        return;
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        // EMA 平滑: 使动作输出连续变化，避免突变导致机器人抖动
        // 位置 (腿部) 与速度 (轮子) 分别平滑
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
            last_act_vel_[i] = act_alpha_ * act_vel_[i] + (1 - act_alpha_) * last_act_vel_[i];
        }
    }
    // 位置目标 (腿部) + 速度目标 (轮子) 一并写入硬件
    robot_->apply_action(last_act_, last_act_vel_);
}

// ============================================================================
// control — 控制线程主循环 (250Hz, SCHED_FIFO priority=70)
//
// 职责: 以固定频率将最新动作输出写入机器人硬件
// 频率 = 1 / dt_ (如 dt=0.004 → 250Hz)
//
// 与推理线程的关系:
//   inference (50Hz)           control (250Hz)
//   ──────────────             ──────────────
//   ONNX Run → 写入 act_  ──→  读取 last_act_ → EMA → 写入硬件
//                               ↑ 每 5 次 inference 触发 1 次 control
//                               但 control 每次都用最新 act_ 做 EMA 平滑
// ============================================================================
void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    // 设置实时调度策略 SCHED_FIFO, 优先级 70 (最高)
    struct sched_param sp {};
    sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    while (rclcpp::ok()) {
        auto loop_start = std::chrono::steady_clock::now();
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;
        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

// ============================================================================
// inference — 推理线程主循环 (50Hz, SCHED_FIFO priority=70)
//
// 这是整个系统的核心循环，每步执行:
//
//   1. 采集传感器观测 → update_obs_segments()
//      ├─ 关节位置/速度/力矩 → 从硬件读取
//      ├─ IMU 角速度/四元数   → 从硬件读取
//      ├─ 速度指令            → cmd_vel_ (手柄或 /cmd_vel)
//      ├─ 上一帧动作          → last_act_
//      └─ 感知观测            → perception_obs_buffer_ (外部发布)
//
//   2. 拼接观测 → flatten_obs_segments() → obs[53]
//      各分量首尾相接: [ang_vel:3 | gravity_b:3 | cmd:3 | dof_pos:12 | dof_vel:16 | last_action:16]
//
//   3. 观测截断 → clamp(obs, -clip_observations_, +clip_observations_)
//
//   4. 帧栈更新 → update_stacked_obs()
//      将单帧 obs 写入 input_buffer 的滑动窗口 → 构建多帧历史输入
//
//   5. (可选) 额外观测拼接 (RPO 感知用, wheel_quad 无)
//
//   6. (可选) 运动帧推进 → step_motion_frame()
//      运动策略模式下，从 NPZ 文件读取下一帧参考姿态
//
//   7. ONNX 推理 → session->Run()
//      input_buffer → 模型 → output_buffer
//
//   8. 后处理 → 写入 act_
//      output → clamp → usd2urdf 索引重映射 → scale + offset → act_
//      (中断模式下，最后 N 个关节用外部指令覆盖)
//
//   9. 发布动作 → publish_action()
//
// 频率 = 1 / (dt × decimation) (如 0.004×5=0.02 → 50Hz)
// ============================================================================
void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    // 设置实时调度策略 SCHED_FIFO, 优先级 70 (与控制线程同级)
    struct sched_param sp {};
    sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    // 推理周期 = dt × decimation (控制周期 × 降采样因子)
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while (rclcpp::ok()) {
        auto loop_start = std::chrono::steady_clock::now();

        // 推理暂停 → 跳过本轮，睡眠等待
        if (!is_running_.load()) {
            std::this_thread::sleep_for(period);
            continue;
        }

        try {
            // 获取模式锁，防止推理过程中策略被切换
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            auto& policy = active_policy();

            // ── 步骤 1: 采集观测 ──────────────────────────────────────
            update_obs_segments(policy.obs_segments, policy.obs_layout);
            publish_imu();
            publish_joint_states();

            // ── 步骤 2: 拼接单帧观测 ──────────────────────────────────
            // obs_segments → obs[78]
            flatten_obs_segments(policy.obs_segments, policy.obs.begin());

            // ── 步骤 3: 观测截断 ──────────────────────────────────────
            std::transform(policy.obs.begin(), policy.obs.end(), policy.obs.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            // ── 步骤 4: 帧栈更新 ──────────────────────────────────────
            // 将 obs[78] 写入 input_buffer 的历史窗口
            update_stacked_obs(policy.ctx->input_buffer, policy.obs, policy.obs_num, policy.frame_stack,
                               policy.stack_order, policy.obs_layout_sizes, policy.is_first_frame);

            // ── 步骤 5: (可选) 额外观测 ───────────────────────────────
            // 拼接到 input_buffer 末尾 (frame_stack × obs_num 之后)
            if (policy.extra_obs_num > 0) {
                update_obs_segments(policy.extra_obs_segments, policy.extra_obs_layout);
                flatten_obs_segments(policy.extra_obs_segments,
                                     policy.ctx->input_buffer.begin() + policy.frame_stack * policy.obs_num);
            }

            // ── 首帧标志清除 (下次推理不再初始化帧栈)
            policy.is_first_frame = false;

            // ── 步骤 7: ONNX 推理 ─────────────────────────────────────
            // 入参: input_names_raw → input_tensor → 模型 → output_tensor → output_names_raw
            policy.ctx->session->Run(Ort::RunOptions{nullptr},
                                     policy.ctx->input_names_raw.data(), policy.ctx->input_tensor.get(),
                                     policy.ctx->num_inputs,
                                     policy.ctx->output_names_raw.data(), policy.ctx->output_tensor.get(),
                                     policy.ctx->num_outputs);

            // ── 步骤 8: 后处理 ────────────────────────────────────────
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                const int n = static_cast<int>(policy.ctx->output_buffer.size());
                for (int i = 0; i < n; i++) {
                    // 8a: 动作截断
                    policy.ctx->output_buffer[i] = std::clamp(policy.ctx->output_buffer[i],
                                                               -clip_actions_, clip_actions_);
                    // 8b: USD→URDF 关节索引重映射
                    //     模型输出按 USD 关节顺序，但硬件使用 URDF 顺序
                    const size_t urdf = static_cast<size_t>(usd2urdf_[i]);
                    const float scaled = policy.ctx->output_buffer[i] *
                                         static_cast<float>(action_scales_[i]);
                    // 8c: 速度/位置分流
                    //     轮子 (wheel_joint_indices) → 速度控制, 不叠加默认角度
                    //     腿部                     → 位置控制, scale + 默认角度
                    const bool is_wheel =
                        std::find(wheel_joint_indices_.begin(), wheel_joint_indices_.end(),
                                  static_cast<long int>(i)) != wheel_joint_indices_.end();
                    if (is_wheel) {
                        act_vel_[urdf] = scaled;
                    } else {
                        act_[urdf] = scaled + static_cast<float>(joint_default_angle_[urdf]);
                    }
                }

                // ── 中断模式: 用外部指令覆盖尾部关节 ──────────────────
                // interrupt_action_ 覆盖 act_ 的最后 N 个关节
                if (supports_interrupt() && is_interrupt_.load()) {
                    std::unique_lock<std::mutex> lock(interrupt_mutex_);
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }

                // ── 步骤 9: 发布动作 ──────────────────────────────────
                publish_action();
            }
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        // 精确睡眠控制: 用周期减去本轮耗时 = 需睡眠时间
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;

        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            // 循环超时告警 (推理耗时超过周期)
            RCLCPP_WARN(this->get_logger(),
                        "Inference loop overran! Took %lld us, but period is %lld us.",
                        static_cast<long long>(elapsed_time.count()),
                        static_cast<long long>(period.count()));
        }
    }
}

// ============================================================================
// main — 程序入口
//
// 启动流程:
//   1. rclcpp::init()                 — 初始化 ROS2
//   2. mlockall(MCL_CURRENT|MCL_FUTURE) — 锁定所有内存页，防止 swap 导致实时性抖动
//   3. 设置 main 线程为 SCHED_FIFO 50 — 保证 ROS2 回调响应及时
//   4. 创建 InferenceNode              — 加载配置、初始化硬件、启动工作线程
//   5. MultiThreadedExecutor(2)        — 2 个 executor 线程处理 ROS2 回调
//   6. executor.spin()                — 进入事件循环，阻塞直到 SIGINT/Ctrl-C
//
// 线程拓扑 (4 线程):
//   main       - SCHED_FIFO 50  - ROS2 初始化 + executor 管理
//   executor₀  - 默认调度       - 话题订阅回调 (joy, cmd_vel, joint_state, elevation)
//   executor₁  - 默认调度       - Service 回调 (reset_joints, init_motors, ...)
//   control    - SCHED_FIFO 70  - 动作发布 (250Hz)
//   inference  - SCHED_FIFO 70  - ONNX 推理 (50Hz)
// ============================================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    // 锁定所有已分配和将来分配的内存页，防止被 swap 到磁盘
    // 实时系统必须: swap 导致的缺页中断延迟不可预测
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }

    // 设置 main 线程实时调度 (优先级 50, 低于工作线程的 70)
    pthread_setname_np(pthread_self(), "main");
    struct sched_param sp {};
    sp.sched_priority = 50;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Failed to set realtime priority for main thread");
        rclcpp::shutdown();
        return 1;
    }

    std::shared_ptr<InferenceNode> node;
    try {
        // 构造 InferenceNode → load_config + 初始化硬件 + 启动推理/控制线程
        node = std::make_shared<InferenceNode>();

        // 使用 2 线程的 executor 并行处理 ROS2 回调和 Service
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);

        // 打印手柄操作提示
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
        RCLCPP_INFO(node->get_logger(),
                    "Press 'Y' to switch between Gamepad Control / cmd_vel Control");
        if (node->supports_interrupt()) {
            RCLCPP_INFO(node->get_logger(),
                        "Press 'LB' to switch interrupt mode");
        }
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "LT/RT: Control turning (left / right rotation)");

        // 阻塞直到 SIGINT / rclcpp::shutdown()
        executor.spin();
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();  // 显式析构 InferenceNode → join 工作线程 → 释放硬件
    return 0;
}
