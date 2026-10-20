message(STATUS "Using RPC backend")

ggml_add_backend_library(ggml-rpc
                         ggml-rpc.cpp
                         transport.cpp
                         inference-transport.cpp
                        )

target_compile_features(ggml-rpc PRIVATE cxx_std_17)

option(GGML_RPC_BUILD_TRANSPORT_BENCH "ggml: build inference transport microbenchmark" ON)

if (GGML_RPC_BUILD_TRANSPORT_BENCH)
    add_executable(test-inference-transport
                   test-inference-transport.cpp
                   inference-transport.cpp
                   transport.cpp)

    target_link_libraries(test-inference-transport PRIVATE ggml-base Threads::Threads)
    target_compile_features(test-inference-transport PRIVATE cxx_std_17)
    target_include_directories(test-inference-transport PRIVATE ..)

    if (WIN32)
        target_link_libraries(test-inference-transport PRIVATE ws2_32)
    endif()

    # Exercise the actual socket-backed READY/PARTIAL/ACK transport in both
    # directions, with a host-side P2C single-owner Reduce.
    add_executable(test-inference-cycle-wire
                   test-inference-cycle-wire.cpp
                   inference-transport.cpp
                   transport.cpp)
    target_link_libraries(test-inference-cycle-wire PRIVATE ggml-base Threads::Threads)
    target_compile_features(test-inference-cycle-wire PRIVATE cxx_std_17)
    target_include_directories(test-inference-cycle-wire PRIVATE ..)

    if (WIN32)
        target_link_libraries(test-inference-cycle-wire PRIVATE ws2_32)
    endif()
    add_test(NAME test-inference-cycle-wire COMMAND $<TARGET_FILE:test-inference-cycle-wire>)
endif()

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()

# RDMA auto-detection (Linux only, requires libibverbs)
if (NOT WIN32 AND NOT APPLE)
    find_library(IBVERBS_LIB ibverbs)
    if (IBVERBS_LIB)
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" ON)
    else()
        option(GGML_RPC_RDMA "ggml: enable RDMA transport for RPC" OFF)
    endif()
else()
    set(GGML_RPC_RDMA OFF CACHE BOOL "RDMA not available on this platform" FORCE)
endif()

if (GGML_RPC_RDMA)
    if (NOT IBVERBS_LIB)
        find_library(IBVERBS_LIB ibverbs REQUIRED)
    endif()
    target_compile_definitions(ggml-rpc PRIVATE GGML_RPC_RDMA)
    target_link_libraries(ggml-rpc PRIVATE ${IBVERBS_LIB})
    message(STATUS "  RDMA transport enabled (auto-detected)")
else()
    message(STATUS "  RDMA transport disabled")
endif()
