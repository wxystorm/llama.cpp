add_library(layer-tensor-runtime STATIC
    lt-runtime.cpp
    lt-tcp.cpp
    lt-ggml.cpp
)

target_include_directories(layer-tensor-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(layer-tensor-runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(layer-tensor-runtime PRIVATE cxx_std_17)
target_link_libraries(layer-tensor-runtime PUBLIC llama)
target_link_libraries(layer-tensor-runtime PRIVATE Threads::Threads)

if (WIN32)
    target_link_libraries(layer-tensor-runtime PRIVATE ws2_32)
endif()

add_executable(llama-layer-tensor-bench
    bench.cpp
)
target_compile_features(llama-layer-tensor-bench PRIVATE cxx_std_17)
target_link_libraries(llama-layer-tensor-bench PRIVATE
    layer-tensor-runtime
    Threads::Threads
)

add_executable(llama-layer-tensor-worker
    worker.cpp
)
target_compile_features(llama-layer-tensor-worker PRIVATE cxx_std_17)
target_link_libraries(llama-layer-tensor-worker PRIVATE
    layer-tensor-runtime
    Threads::Threads
)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS
        llama-layer-tensor-bench
        llama-layer-tensor-worker
        RUNTIME
    )
endif()

if (BUILD_TESTING)
    add_test(
        NAME layer-tensor-runtime-self-test
        COMMAND llama-layer-tensor-bench --self-test
    )
endif()
