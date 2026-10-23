# Native Cycle prefill and one next-token output, separate from test programs.
# ggml-rpc supplies the low-level Cycle ACK/transport symbols used by the
# real stage dispatcher; keep that dependency off ordinary libllama.
if (TARGET ggml-rpc)
    add_library(llama-cycle-runtime STATIC
        ${PROJECT_SOURCE_DIR}/src/llama-hybrid-cycle-forward.cpp
        ${PROJECT_SOURCE_DIR}/src/llama-hybrid-cycle-model-io.cpp)
    target_include_directories(llama-cycle-runtime PUBLIC
        ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(llama-cycle-runtime PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc)
    target_link_libraries(llama-cycle-runtime PUBLIC llama ggml-rpc)
    target_compile_features(llama-cycle-runtime PRIVATE cxx_std_17)

    add_executable(llama-cycle-prefill llama-cycle-prefill.cpp)
    target_link_libraries(llama-cycle-prefill PRIVATE
        llama-cycle-runtime ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-cycle-prefill PRIVATE cxx_std_17)
    install(TARGETS llama-cycle-prefill RUNTIME)

    add_executable(llama-cycle-cli llama-cycle-cli.cpp)
    target_link_libraries(llama-cycle-cli PRIVATE
        llama-cycle-runtime ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-cycle-cli PRIVATE cxx_std_17)
    install(TARGETS llama-cycle-cli RUNTIME)
endif()
