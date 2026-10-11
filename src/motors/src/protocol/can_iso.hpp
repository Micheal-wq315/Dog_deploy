#pragma once

#include <linux/can.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "logger.hpp"

// Callback types shared by all CAN backends
using CanCbkFunc = std::function<void(const can_frame&)>;
using CanCbkId = uint16_t;
using CanCbkKeyExtractor = std::function<CanCbkId(const can_frame&)>;
using CanCbkMap = std::unordered_map<CanCbkId, CanCbkFunc>;

/**
 * @brief Abstract interface for classic CAN communication backends.
 *
 * Concrete implementations:
 *   - MotorsSocketCAN: Linux SocketCAN (direct CAN 2.0)
 *   - MotorsEthercatCAN: EtherCAT-to-CAN bridge
 *
 * All backends share the same transmit/callback API, allowing motor drivers
 * to switch transport transparently.
 */
class MotorsSocketCAN;

class MotorsCAN {
public:
    virtual ~MotorsCAN() = default;

    /// Send a CAN frame.
    virtual void transmit(const can_frame& frame) = 0;

    /// Register a receive callback keyed by CAN ID.
    virtual void add_can_callback(const CanCbkFunc& callback, CanCbkId id) = 0;

    /// Remove a previously registered callback.
    virtual void remove_can_callback(CanCbkId id) = 0;

    /// Remove all registered callbacks.
    virtual void clear_can_callbacks() = 0;

    /// Customize the key extractor (default: frame.can_id).
    virtual void set_can_key_extractor(CanCbkKeyExtractor extractor) = 0;

    /**
     * @brief Factory: create or retrieve a CAN backend instance.
     * @param interface  Interface name (e.g. "can0", or EtherCAT master name)
     * @param backend    Backend type: "socketcan" (default) | "ethercat"
     */
    static std::shared_ptr<MotorsCAN> get(
        const std::string& interface,
        const std::string& backend = "socketcan");

    static void init_logger(std::shared_ptr<Logger> logger);

protected:
    MotorsCAN() = default;
    inline static std::shared_ptr<Logger> logger_ = nullptr;

    inline static void ensure_logger() {
        if (!logger_) {
            logger_ = Logger::get_or_create("motors");
        }
    }
};

inline void MotorsCAN::init_logger(std::shared_ptr<Logger> logger) { logger_ = std::move(logger); }
