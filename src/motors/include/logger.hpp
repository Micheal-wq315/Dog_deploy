#pragma once

#include <fmt/color.h>
#include <fmt/core.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// ============================================================================
// Logger — drop-in replacement for spdlog::logger backed by fmt::print
//
// Supports the subset of spdlog used by the motors package:
//   logger->info/error/warn/debug(fmt_str, args...)
//   Logger::get(name) / Logger::register_logger(logger)
//   setup_logger(name) → std::shared_ptr<Logger>
// ============================================================================
class Logger {
   public:
    explicit Logger(std::string name) : name_(std::move(name)) {}

    // ── Log methods ──────────────────────────────────────────────────
    template <typename... Args>
    void info(const std::string& fmt_str, Args&&... args) {
        log("INFO", fmt::color::white, fmt_str, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(const std::string& fmt_str, Args&&... args) {
        log("ERROR", fmt::color::red, fmt_str, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(const std::string& fmt_str, Args&&... args) {
        log("WARN", fmt::color::yellow, fmt_str, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(const std::string& fmt_str, Args&&... args) {
        log("DEBUG", fmt::color::gray, fmt_str, std::forward<Args>(args)...);
    }

    // ── Registry (mimics spdlog::get / spdlog::register_logger) ─────
    static std::shared_ptr<Logger> get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry().find(name);
        return (it != registry().end()) ? it->second : nullptr;
    }

    static void register_logger(std::shared_ptr<Logger> logger) {
        if (!logger) return;
        std::lock_guard<std::mutex> lock(mutex_);
        registry()[logger->name_] = std::move(logger);
    }

    // Convenience: always returns a valid logger (creates one if missing)
    static std::shared_ptr<Logger> get_or_create(const std::string& name) {
        if (auto existing = get(name)) return existing;
        auto logger = std::make_shared<Logger>(name);
        register_logger(logger);
        return logger;
    }

   private:
    template <typename... Args>
    void log(const char* level, fmt::color color, const std::string& fmt_str,
             Args&&... args) {
        auto msg = fmt::format(fmt_str, std::forward<Args>(args)...);
        fmt::print(stderr, fg(color), "[{}] ", level);
        fmt::print(stderr, "{}\n", msg);
    }

    std::string name_;
    static inline std::mutex mutex_;

    static std::unordered_map<std::string, std::shared_ptr<Logger>>& registry() {
        static std::unordered_map<std::string, std::shared_ptr<Logger>> reg;
        return reg;
    }
};

// ── Drop-in for spdlog::error("msg") ────────────────────────────────
namespace spdlog {
inline void error(const std::string& msg) {
    fmt::print(stderr, fg(fmt::color::red), "[ERROR] ");
    fmt::print(stderr, "{}\n", msg);
}
}  // namespace spdlog

// ── Replacement for setup_logger() in utils.hpp ──────────────────────
inline std::shared_ptr<Logger> setup_logger(const std::string& logger_name = "motors") {
    return Logger::get_or_create(logger_name);
}
