set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_include_directories(${TARGET} PRIVATE ../../src)
target_link_libraries(${TARGET} PRIVATE llama ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# On Phone, compile the original Qwen3 G/C/N/A native profiler into the
# inference RPC executable itself. This does not launch a subprocess and
# has no Python or standalone-Profiler runtime dependency.
if (ANDROID)
    target_sources(${TARGET} PRIVATE
        v2-profile-native.cpp
        ../region-ring-experiments/v2_stage_profile.cpp)
    target_compile_definitions(${TARGET} PRIVATE V2_STAGE_PROFILE_EMBEDDED)
    target_include_directories(${TARGET} PRIVATE
        ${PROJECT_SOURCE_DIR}/vendor
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc
        ${PROJECT_SOURCE_DIR}/tools/region-ring-experiments)
    target_link_libraries(${TARGET} PRIVATE ggml-cpu)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
