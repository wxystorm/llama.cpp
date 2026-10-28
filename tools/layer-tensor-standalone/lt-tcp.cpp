#include "lt-tcp.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace lt {
namespace {

constexpr uint32_t k_magic = 0x3156544cU; // LTV1
constexpr uint32_t k_version = 7;
constexpr uint32_t k_frame_plan = 1;
constexpr uint32_t k_frame_message = 2;
constexpr uint32_t k_max_frame = 256U * 1024U * 1024U;

struct frame_header {
    uint32_t magic;
    uint32_t version;
    uint32_t kind;
    uint32_t size;
};

struct plan_header {
    uint32_t execution_mode;
    int32_t layers;
    uint32_t chunks;
    uint32_t first_tensor_owner;
    int32_t owner_burst_limit;
    int32_t max_owner_ahead;
    uint32_t debug_phone_partial;
    uint32_t layer_owners;
    float c2p_pc_ratio;
    float p2c_pc_ratio;
};

struct message_header {
    uint32_t kind;
    int32_t layer;
    int32_t chunk;
    uint64_t seq;

    uint32_t residual_count;
    uint32_t hidden_count;
    uint32_t topk_count;
    uint32_t weights_count;
    uint32_t partial_count;
    uint32_t layer_hidden_count;

    double checksum;
};

bool parse_endpoint(
        const std::string & endpoint,
        std::string & host,
        std::string & port) {
    const size_t pos = endpoint.rfind(':');
    if (pos == std::string::npos ||
        pos == 0 ||
        pos + 1 >= endpoint.size()) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = endpoint.substr(pos + 1);
    return true;
}

template <typename T>
void append_pod(
        std::vector<uint8_t> & out,
        const T & value) {
    const size_t old = out.size();
    out.resize(old + sizeof(T));
    std::memcpy(out.data() + old, &value, sizeof(T));
}

template <typename T>
void append_vector(
        std::vector<uint8_t> & out,
        const std::vector<T> & values) {
    if (values.empty()) {
        return;
    }
    const size_t bytes = values.size() * sizeof(T);
    const size_t old = out.size();
    out.resize(old + bytes);
    std::memcpy(out.data() + old, values.data(), bytes);
}

template <typename T>
bool read_pod(
        const std::vector<uint8_t> & input,
        size_t & pos,
        T & value) {
    if (pos > input.size() ||
        sizeof(T) > input.size() - pos) {
        return false;
    }
    std::memcpy(&value, input.data() + pos, sizeof(T));
    pos += sizeof(T);
    return true;
}

template <typename T>
bool read_vector(
        const std::vector<uint8_t> & input,
        size_t & pos,
        uint32_t count,
        std::vector<T> & values) {
    const size_t bytes =
        static_cast<size_t>(count) * sizeof(T);
    if (pos > input.size() ||
        bytes > input.size() - pos) {
        return false;
    }
    values.resize(count);
    if (bytes > 0) {
        std::memcpy(values.data(), input.data() + pos, bytes);
    }
    pos += bytes;
    return true;
}

#ifdef _WIN32
using native_socket = SOCKET;
constexpr native_socket invalid_socket = INVALID_SOCKET;

bool ensure_socket_runtime() {
    static const bool initialized = []() {
        WSADATA data {};
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }();
    return initialized;
}

void close_socket(native_socket value) {
    if (value != invalid_socket) {
        closesocket(value);
    }
}
#else
using native_socket = int;
constexpr native_socket invalid_socket = -1;

bool ensure_socket_runtime() {
    return true;
}

void close_socket(native_socket value) {
    if (value != invalid_socket) {
        ::close(value);
    }
}
#endif

native_socket as_native(intptr_t value) {
    return static_cast<native_socket>(value);
}

intptr_t as_stored(native_socket value) {
    return static_cast<intptr_t>(value);
}

bool set_tcp_nodelay(
        native_socket value,
        std::string & error) {
    int yes = 1;
    if (setsockopt(
            value,
            IPPROTO_TCP,
            TCP_NODELAY,
            reinterpret_cast<const char *>(&yes),
            sizeof(yes)) != 0) {
        error = "setsockopt(TCP_NODELAY) failed";
        return false;
    }
    return true;
}

} // namespace

tcp_channel::tcp_channel() = default;

tcp_channel::~tcp_channel() {
    close();
}

bool tcp_channel::connect_to(
        const std::string & endpoint,
        std::string & error) {
    close();
    if (!ensure_socket_runtime()) {
        error = "socket runtime initialization failed";
        return false;
    }

    std::string host;
    std::string port;
    if (!parse_endpoint(endpoint, host, port)) {
        error = "invalid endpoint, expected HOST:PORT";
        return false;
    }

    addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * result = nullptr;
    if (getaddrinfo(
            host.c_str(),
            port.c_str(),
            &hints,
            &result) != 0) {
        error = "getaddrinfo failed";
        return false;
    }

    native_socket connected = invalid_socket;
    for (addrinfo * it = result;
         it != nullptr;
         it = it->ai_next) {
        native_socket socket_value =
            ::socket(
                it->ai_family,
                it->ai_socktype,
                it->ai_protocol);
        if (socket_value == invalid_socket) {
            continue;
        }

        if (::connect(
                socket_value,
                it->ai_addr,
                static_cast<int>(it->ai_addrlen)) == 0) {
            connected = socket_value;
            break;
        }

        close_socket(socket_value);
    }

    freeaddrinfo(result);

    if (connected == invalid_socket) {
        error = "connect failed";
        return false;
    }

    if (!set_tcp_nodelay(
            connected,
            error)) {
        close_socket(connected);
        return false;
    }

    socket_ = as_stored(connected);
    return true;
}

bool tcp_channel::listen_and_accept(
        const std::string & endpoint,
        std::string & error) {
    close();
    if (!ensure_socket_runtime()) {
        error = "socket runtime initialization failed";
        return false;
    }

    std::string host;
    std::string port;
    if (!parse_endpoint(endpoint, host, port)) {
        error = "invalid endpoint, expected HOST:PORT";
        return false;
    }

    addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    const char * host_arg =
        (host == "0.0.0.0" ||
         host == "*" ||
         host.empty()) ?
        nullptr :
        host.c_str();

    addrinfo * result = nullptr;
    if (getaddrinfo(
            host_arg,
            port.c_str(),
            &hints,
            &result) != 0) {
        error = "getaddrinfo failed";
        return false;
    }

    native_socket listener = invalid_socket;
    for (addrinfo * it = result;
         it != nullptr;
         it = it->ai_next) {
        native_socket socket_value =
            ::socket(
                it->ai_family,
                it->ai_socktype,
                it->ai_protocol);
        if (socket_value == invalid_socket) {
            continue;
        }

        int yes = 1;
        setsockopt(
            socket_value,
            SOL_SOCKET,
            SO_REUSEADDR,
            reinterpret_cast<const char *>(&yes),
            sizeof(yes));

        if (::bind(
                socket_value,
                it->ai_addr,
                static_cast<int>(it->ai_addrlen)) == 0 &&
            ::listen(socket_value, 1) == 0) {
            listener = socket_value;
            break;
        }

        close_socket(socket_value);
    }

    freeaddrinfo(result);

    if (listener == invalid_socket) {
        error = "bind/listen failed";
        return false;
    }

    native_socket accepted =
        ::accept(listener, nullptr, nullptr);
    close_socket(listener);

    if (accepted == invalid_socket) {
        error = "accept failed";
        return false;
    }

    if (!set_tcp_nodelay(
            accepted,
            error)) {
        close_socket(accepted);
        return false;
    }

    socket_ = as_stored(accepted);
    return true;
}

bool tcp_channel::send_plan(const plan & value) {
    plan_header header {};
    header.execution_mode =
        static_cast<uint32_t>(value.execution_mode);
    header.layers = value.layers;
    header.chunks =
        static_cast<uint32_t>(value.chunk_tokens.size());
    header.first_tensor_owner =
        static_cast<uint32_t>(value.first_tensor_owner);
    header.owner_burst_limit =
        value.owner_burst_limit;
    header.max_owner_ahead =
        value.max_owner_ahead;
    header.debug_phone_partial =
        value.debug_phone_partial ? 1U : 0U;
    header.layer_owners =
        static_cast<uint32_t>(value.layer_owners.size());
    header.c2p_pc_ratio = value.c2p_pc_ratio;
    header.p2c_pc_ratio = value.p2c_pc_ratio;

    std::vector<uint8_t> payload;
    append_pod(payload, header);
    append_vector(payload, value.chunk_tokens);

    for (const side owner : value.layer_owners) {
        const uint8_t raw =
            static_cast<uint8_t>(owner);
        append_pod(payload, raw);
    }

    return send_frame(k_frame_plan, payload);
}

bool tcp_channel::recv_plan(plan & value) {
    uint32_t kind = 0;
    std::vector<uint8_t> payload;
    if (!recv_frame(kind, payload) ||
        kind != k_frame_plan) {
        return false;
    }

    size_t pos = 0;
    plan_header header {};
    if (!read_pod(payload, pos, header)) {
        return false;
    }

    if (header.execution_mode >
            static_cast<uint32_t>(mode::layer) ||
        header.first_tensor_owner >
            static_cast<uint32_t>(side::phone) ||
        header.layers <= 0 ||
        header.chunks == 0 ||
        header.owner_burst_limit <= 0 ||
        header.max_owner_ahead < 0 ||
        header.debug_phone_partial > 1U) {
        return false;
    }

    value = {};
    value.execution_mode =
        static_cast<mode>(header.execution_mode);
    value.layers = header.layers;
    value.first_tensor_owner =
        static_cast<side>(header.first_tensor_owner);
    value.owner_burst_limit =
        header.owner_burst_limit;
    value.max_owner_ahead =
        header.max_owner_ahead;
    value.debug_phone_partial =
        header.debug_phone_partial != 0;
    value.c2p_pc_ratio = header.c2p_pc_ratio;
    value.p2c_pc_ratio = header.p2c_pc_ratio;

    if (!read_vector(
            payload,
            pos,
            header.chunks,
            value.chunk_tokens)) {
        return false;
    }

    value.layer_owners.resize(header.layer_owners);
    for (uint32_t i = 0;
         i < header.layer_owners;
         ++i) {
        uint8_t raw = 0;
        if (!read_pod(payload, pos, raw) ||
            raw > static_cast<uint8_t>(side::phone)) {
            return false;
        }
        value.layer_owners[i] =
            static_cast<side>(raw);
    }

    return pos == payload.size() && value.valid();
}

bool tcp_channel::send_message(const message & value) {
    message_header header {};
    header.kind =
        static_cast<uint32_t>(value.kind);
    header.layer = value.task.layer;
    header.chunk = value.task.chunk;
    header.seq = value.seq;

    header.residual_count =
        static_cast<uint32_t>(
            value.ready.residual.size());
    header.hidden_count =
        static_cast<uint32_t>(
            value.ready.hidden.size());
    header.topk_count =
        static_cast<uint32_t>(
            value.ready.topk.size());
    header.weights_count =
        static_cast<uint32_t>(
            value.ready.weights.size());
    header.partial_count =
        static_cast<uint32_t>(
            value.partial.size());
    header.layer_hidden_count =
        static_cast<uint32_t>(
            value.hidden.size());
    header.checksum = value.checksum;

    std::vector<uint8_t> payload;
    append_pod(payload, header);
    append_vector(payload, value.ready.residual);
    append_vector(payload, value.ready.hidden);
    append_vector(payload, value.ready.topk);
    append_vector(payload, value.ready.weights);
    append_vector(payload, value.partial);
    append_vector(payload, value.hidden);

    return send_frame(k_frame_message, payload);
}

bool tcp_channel::recv_message(message & value) {
    uint32_t frame_kind = 0;
    std::vector<uint8_t> payload;
    if (!recv_frame(frame_kind, payload) ||
        frame_kind != k_frame_message) {
        return false;
    }

    size_t pos = 0;
    message_header header {};
    if (!read_pod(payload, pos, header)) {
        return false;
    }

    if (header.kind <
            static_cast<uint32_t>(
                message_kind::ready) ||
        header.kind >
            static_cast<uint32_t>(
                message_kind::bench_pull_batch)) {
        return false;
    }

    value = {};
    value.kind =
        static_cast<message_kind>(header.kind);
    value.task.layer = header.layer;
    value.task.chunk = header.chunk;
    value.seq = header.seq;
    value.checksum = header.checksum;

    if (!read_vector(
            payload, pos,
            header.residual_count,
            value.ready.residual) ||
        !read_vector(
            payload, pos,
            header.hidden_count,
            value.ready.hidden) ||
        !read_vector(
            payload, pos,
            header.topk_count,
            value.ready.topk) ||
        !read_vector(
            payload, pos,
            header.weights_count,
            value.ready.weights) ||
        !read_vector(
            payload, pos,
            header.partial_count,
            value.partial) ||
        !read_vector(
            payload, pos,
            header.layer_hidden_count,
            value.hidden)) {
        return false;
    }

    return pos == payload.size();
}

void tcp_channel::close() {
    if (socket_ == -1) {
        return;
    }

    close_socket(as_native(socket_));
    socket_ = -1;
}

bool tcp_channel::valid() const {
    return socket_ != -1;
}

bool tcp_channel::send_all(
        const void * data,
        size_t size) {
    if (!valid()) {
        return false;
    }

    const uint8_t * ptr =
        static_cast<const uint8_t *>(data);
    size_t sent = 0;

    while (sent < size) {
#ifdef _WIN32
        const int count = ::send(
            as_native(socket_),
            reinterpret_cast<const char *>(
                ptr + sent),
            static_cast<int>(
                std::min<size_t>(
                    size - sent,
                    1U << 30)),
            0);
#else
        const ssize_t count = ::send(
            as_native(socket_),
            ptr + sent,
            size - sent,
            0);
#endif
        if (count <= 0) {
            return false;
        }
        sent += static_cast<size_t>(count);
    }

    return true;
}

bool tcp_channel::recv_all(
        void * data,
        size_t size) {
    if (!valid()) {
        return false;
    }

    uint8_t * ptr =
        static_cast<uint8_t *>(data);
    size_t received = 0;

    while (received < size) {
#ifdef _WIN32
        const int count = ::recv(
            as_native(socket_),
            reinterpret_cast<char *>(
                ptr + received),
            static_cast<int>(
                std::min<size_t>(
                    size - received,
                    1U << 30)),
            0);
#else
        const ssize_t count = ::recv(
            as_native(socket_),
            ptr + received,
            size - received,
            0);
#endif
        if (count <= 0) {
            return false;
        }
        received += static_cast<size_t>(count);
    }

    return true;
}

bool tcp_channel::send_frame(
        uint32_t kind,
        const std::vector<uint8_t> & payload) {
    if (payload.size() > k_max_frame) {
        return false;
    }

    frame_header header {};
    header.magic = k_magic;
    header.version = k_version;
    header.kind = kind;
    header.size =
        static_cast<uint32_t>(payload.size());

    std::lock_guard<std::mutex> lock(send_mutex_);
    return send_all(&header, sizeof(header)) &&
        (payload.empty() ||
         send_all(payload.data(), payload.size()));
}

bool tcp_channel::recv_frame(
        uint32_t & kind,
        std::vector<uint8_t> & payload) {
    frame_header header {};
    if (!recv_all(&header, sizeof(header))) {
        return false;
    }

    if (header.magic != k_magic ||
        header.version != k_version ||
        header.size > k_max_frame) {
        return false;
    }

    kind = header.kind;
    payload.resize(header.size);

    return payload.empty() ||
        recv_all(payload.data(), payload.size());
}

socket_transport::socket_transport(
        std::shared_ptr<tcp_channel> channel)
    : channel_(std::move(channel)) {
}

bool socket_transport::send(
        const message & value) {
    return channel_ != nullptr &&
        channel_->send_message(value);
}

} // namespace lt
