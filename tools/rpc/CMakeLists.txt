set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_include_directories(${TARGET} PRIVATE ../../src)
target_link_libraries(${TARGET} PRIVATE llama ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Build the real-layer phone backend benchmark in the same CMake tree as the
# RPC server. This is intentional: phone NPU/HTP is configured by the
# hexagon-rpc build, while OpenCL is loaded from the same runtime bin directory.
set(PHONE_LAYER_BENCH_TARGET ggml-phone-layer-speed)
add_executable(${PHONE_LAYER_BENCH_TARGET}
    ${PROJECT_SOURCE_DIR}/tests/test-phone-layer-speed.cpp)
target_include_directories(${PHONE_LAYER_BENCH_TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PHONE_LAYER_BENCH_TARGET} PRIVATE llama ggml)
target_compile_features(${PHONE_LAYER_BENCH_TARGET} PRIVATE cxx_std_17)

if (GGML_OPENCL)
    target_compile_definitions(${PHONE_LAYER_BENCH_TARGET} PRIVATE
        PHONE_BENCH_HAS_OPENCL=1)
endif()
if (GGML_HEXAGON)
    target_compile_definitions(${PHONE_LAYER_BENCH_TARGET} PRIVATE
        PHONE_BENCH_HAS_HEXAGON=1)
endif()

if (TARGET ggml-opencl)
    add_dependencies(${PHONE_LAYER_BENCH_TARGET} ggml-opencl)
endif()
if (TARGET ggml-hexagon)
    add_dependencies(${PHONE_LAYER_BENCH_TARGET} ggml-hexagon)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${PHONE_LAYER_BENCH_TARGET} RUNTIME)
endif()
