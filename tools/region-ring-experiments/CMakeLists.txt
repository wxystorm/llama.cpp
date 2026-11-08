# Native isolated G/C/N/A Stage profiler.
# This executable does not link the llama_context inference scheduler or RPC.
# On Android, the N backend is explicitly HTP0 with quantized weights in
# the repack buffer and a mandatory precompile + CPU parity check.
add_executable(region-ring-v2-stage-profile v2_stage_profile.cpp)
target_include_directories(region-ring-v2-stage-profile PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/src
    ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc)
target_link_libraries(region-ring-v2-stage-profile PRIVATE ggml ggml-cpu)
target_compile_features(region-ring-v2-stage-profile PRIVATE cxx_std_17)
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS region-ring-v2-stage-profile RUNTIME)
endif()
