#pragma once

#include "lt-runtime.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace lt {

class tcp_channel {
public:
    tcp_channel();
    ~tcp_channel();

    tcp_channel(const tcp_channel &) = delete;
    tcp_channel & operator=(const tcp_channel &) = delete;

    bool connect_to(
        const std::string & endpoint,
        std::string & error);

    bool listen_and_accept(
        const std::string & endpoint,
        std::string & error);

    bool send_plan(const plan & value);
    bool recv_plan(plan & value);

    bool send_message(const message & value);
    bool recv_message(message & value);

    void close();
    bool valid() const;

private:
    bool send_all(const void * data, size_t size);
    bool recv_all(void * data, size_t size);

    bool send_frame(
        uint32_t kind,
        const std::vector<uint8_t> & payload);

    bool recv_frame(
        uint32_t & kind,
        std::vector<uint8_t> & payload);

private:
    mutable std::mutex send_mutex_;
    intptr_t socket_ = -1;
};

class socket_transport final : public transport {
public:
    explicit socket_transport(
        std::shared_ptr<tcp_channel> channel);

    bool send(const message & value) override;

private:
    std::shared_ptr<tcp_channel> channel_;
};

} // namespace lt
