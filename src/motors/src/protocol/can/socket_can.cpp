/**
 * @file
 * SocketCAN 标准帧（can_frame）收发实现。
 *
 * 架构：双线程模型
 *   - RX 线程：select 轮询 → 读取帧 → 查回调表 → 分发给电机驱动
 *   - TX 线程：从无锁队列取帧 → write 发送
 *
 * 注意：
 *   - RX/TX 绑到同一核（见 cpu_id 计算），SCHED_FIFO 同优先级可能互相抢占
 *   - send_sleep_us_ 默认为 0，连续发帧无间隔，CAN 控制器硬件 TX 队列浅时易丢帧
 *   - write 失败仅重试 3 次 + 打日志，不检查 errno，bus-off 后无自动恢复逻辑
 */

#include "socket_can.hpp"

// ── MotorsCAN 工厂：根据后端名称创建实例 ──
std::shared_ptr<MotorsCAN> MotorsCAN::get(const std::string& interface, const std::string& backend) {
    ensure_logger();
    if (backend == "socketcan") {
        return MotorsSocketCAN::get(interface);
    }
    throw std::runtime_error("Unknown CAN backend: " + backend);
}

// ── MotorsSocketCAN 单例管理：每个 CAN 口一个共享实例 ──
std::unordered_map<std::string, std::shared_ptr<MotorsSocketCAN>> MotorsSocketCAN::instances_;

std::shared_ptr<MotorsSocketCAN> MotorsSocketCAN::get(const std::string& interface) {
    if (instances_.find(interface) == instances_.end()) instances_[interface] = createInstance(interface);
    return instances_[interface];
}

MotorsSocketCAN::MotorsSocketCAN(const std::string& interface)
    : interface_(interface), sockfd_(INIT_FD), receiving_(false), tx_queue_(TX_QUEUE_SIZE) {
    open(interface);
}

MotorsSocketCAN::~MotorsSocketCAN() { this->close(); }

void MotorsSocketCAN::open(const std::string& interface) {
    // ① 创建原始 CAN 套接字
    sockfd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sockfd_ == INIT_FD) {
        logger_->error("Failed to create CAN socket");
        throw std::runtime_error("Failed to create CAN socket");
    }

    // ② 设置发送缓冲区 1MB（减少用户态→内核态拷贝失败）
    int bufsize = 1024 * 1024;
    setsockopt(sockfd_, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    // ③ 查找网络接口索引
    strncpy(if_request_.ifr_name, interface.c_str(), IFNAMSIZ);
    if (ioctl(sockfd_, SIOCGIFINDEX, &if_request_) == -1) {
        logger_->error("Unable to detect CAN interface {}", interface);
        this->close();
        throw std::runtime_error("Unable to detect CAN interface " + interface);
    }

    // ④ 绑定套接字到指定 CAN 接口
    addr_.can_family = AF_CAN;
    addr_.can_ifindex = if_request_.ifr_ifindex;
    int rc = ::bind(sockfd_, reinterpret_cast<struct sockaddr *>(&addr_), sizeof(addr_));
    if (rc == -1) {
        logger_->error("Failed to bind socket to network interface {}", interface);
        this->close();
        throw std::runtime_error("Failed to bind socket to network interface " + interface);
    }

    // ⑤ 设为非阻塞模式（select + 非阻塞读，避免忙等）
    int flags = fcntl(sockfd_, F_GETFL, 0);
    if (flags == -1) {
        logger_->error("Failed to get socket flags");
        this->close();
        throw std::runtime_error("Failed to get socket flags");
    }
    if (fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        logger_->error("Failed to set socket to non-blocking");
        this->close();
        throw std::runtime_error("Failed to set socket to non-blocking");
    }

    receiving_ = true;

    // ═══════════════════════════════════════════
    // RX 线程：select 轮询 + 回调分发
    // ═══════════════════════════════════════════
    receiver_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "can_rx");

        // 设为 SCHED_FIFO 实时调度，优先级 80（确保 CAN 帧不被丢）
        struct sched_param sp{}; sp.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            logger_->error("Failed to set realtime priority for CAN RX thread");
        }

        // CPU 绑定：can0→最后核, can1→倒数第二核, ...
        // ⚠️ 同一接口的 RX/TX 线程绑到同一核，同优先级可能互相抢占
        int total_cores = std::thread::hardware_concurrency();
        if (total_cores == 0) total_cores = 4;
        int cpu_id = total_cores - 1;

        char last_char = interface_.back();
        if (isdigit(last_char)) {
            int port_num = last_char - '0';
            cpu_id = total_cores - 1 - port_num;
            if (cpu_id < 0) cpu_id = 0;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            logger_->error("Failed to bind CAN RX thread to Core {}", cpu_id);
        }

        fd_set descriptors;
        int maxfd = sockfd_;
        struct timeval timeout;
        can_frame rx_frame;

        while (receiving_) {
            FD_ZERO(&descriptors);
            FD_SET(sockfd_, &descriptors);

            timeout.tv_sec = TIMEOUT_SEC;      // 0 秒
            timeout.tv_usec = TIMEOUT_USEC;    // 1000 微秒（1ms 超时，兼顾响应与 CPU）

            int sel_ret = ::select(maxfd + 1, &descriptors, NULL, NULL, &timeout);
            if (sel_ret < 0) {
                if (errno == EINTR) continue;  // 被信号打断，继续
                logger_->error("CAN select error: {}", strerror(errno));
                break;
            }

            if (sel_ret == 1) {
                // 内核缓冲区可能有多帧，读完为止
                while (true) {
                    int len = ::read(sockfd_, &rx_frame, CAN_MTU);
                    if (len < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;  // 内核缓冲区已空
                        }
                        logger_->warn("CAN read error: {}", strerror(errno));
                        break;
                    }
                    if (len == 0) break;  // 对端关闭（CAN 口 down）

                    // 查回调表：根据帧 ID 找到对应电机的回调
                    CanCbkFunc callback_to_run;
                    {
                        std::lock_guard<std::mutex> lock(can_callback_mutex_);
                        CanCbkId key = key_extractor_(rx_frame);  // 默认用 can_id 做键
                        auto it = can_callback_list_.find(key);
                        if (it != can_callback_list_.end()) {
                            callback_to_run = it->second;
                        }
                    }
                    // 回调在 RX 线程内执行，耗时操作会阻塞接收
                    if (callback_to_run) {
                        callback_to_run(rx_frame);
                    }
                }
            }
        }
    });

    // ═══════════════════════════════════════════
    // TX 线程：从无锁队列取帧 → write 发送
    // ═══════════════════════════════════════════
    sender_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "can_tx");

        struct sched_param sp{}; sp.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            logger_->error("Failed to set realtime priority for CAN TX thread");
        }

        // CPU 绑定逻辑与 RX 线程相同 → ⚠️ 同一接口的收发线程共用一个核
        int total_cores = std::thread::hardware_concurrency();
        if (total_cores == 0) total_cores = 4;
        int cpu_id = total_cores - 1;

        char last_char = interface_.back();
        if (isdigit(last_char)) {
            int port_num = last_char - '0';
            cpu_id = total_cores - 1 - port_num;
            if (cpu_id < 0) cpu_id = 0;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            logger_->error("Failed to bind CAN TX thread to Core {}", cpu_id);
        }

        can_frame tx_frame;
        int count = 0;

        while (receiving_) {
            {
                std::unique_lock<std::mutex> lock(tx_mutex_);
                // 等待队列非空 或 停止信号
                tx_cv_.wait(lock, [this]() { return !tx_queue_.empty() || !receiving_; });
                if (!receiving_) break;
                if (!tx_queue_.pop(tx_frame)) continue;  // 假唤醒
            }

            // 带重试的发送（最多 3 次，间隔 1ms）
            // ⚠️ 不检查 errno：bus-off 时 write 返回 -1，仅打日志丢弃帧
            while (::write(sockfd_, &tx_frame, sizeof(can_frame)) < 0 && count < MAX_RETRY_COUNT) {
                count += 1;
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
            }

            if (count >= MAX_RETRY_COUNT) {
                logger_->error("Failed to transmit CAN frame");
            } else if (send_sleep_us_ > 0) {
                // 帧间间隔（默认 0，多电机同时发时连续怼帧 → 硬件 TX 队列溢出）
                std::this_thread::sleep_for(std::chrono::microseconds(send_sleep_us_));
            }
            count = 0;
        }
    });
}

void MotorsSocketCAN::close() {
    // 通知 TX 线程退出
    receiving_ = false;
    tx_cv_.notify_one();

    if (receiver_thread_.joinable()) receiver_thread_.join();
    if (sender_thread_.joinable()) sender_thread_.join();

    if (sockfd_ != INIT_FD) {
        if (::close(sockfd_) < 0) {
            logger_->warn("Failed to close socket {}: {}", interface_, strerror(errno));
        } else {
            logger_->info("CAN interface {} closed successfully.", interface_);
        }
    }
    sockfd_ = INIT_FD;
}

// 发送入口（多线程安全）：将帧推入无锁队列，通知 TX 线程
void MotorsSocketCAN::transmit(const can_frame &frame) {
    if (sockfd_ == INIT_FD) {
        logger_->error("Unable to transmit: Socket not open");
        return;
    }
    tx_queue_.bounded_push(frame);  // 队满时丢弃最旧帧
    tx_cv_.notify_one();
}

// 注册 CAN 帧回调：电机驱动通过 can_id 匹配自己的回包
void MotorsSocketCAN::add_can_callback(const CanCbkFunc& callback, const CanCbkId id) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_[id] = callback;
}

// 注销 CAN 帧回调
void MotorsSocketCAN::remove_can_callback(CanCbkId id) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.erase(id);
}

// 清空全部回调
void MotorsSocketCAN::clear_can_callbacks() {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.clear();
}

// 自定义回调键提取器（默认用 can_id，可改为 can_id + dlc 等）
void MotorsSocketCAN::set_can_key_extractor(CanCbkKeyExtractor extractor) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    key_extractor_ = std::move(extractor);
}
