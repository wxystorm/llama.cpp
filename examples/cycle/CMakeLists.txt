# Native Cycle prefill and one next-token output, separate from test programs.
# ggml-rpc supplies the low-level Cycle ACK/transport symbols used by the
# real stage dispatcher; keep that dependency off ordinary libllama.
if (TARGET ggml-rpc)
    add_library(llama-cycle-runtime INTERFACE)
    target_include_directories(llama-cycle-runtime INTERFACE
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(llama-cycle-runtime INTERFACE llama ggml-rpc)

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

    # Verification using only standard llama_model/context/decode/logits APIs.
    add_executable(llama-cycle-standard llama-cycle-standard.cpp)
    target_link_libraries(llama-cycle-standard PRIVATE
        llama-cycle-runtime ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-cycle-standard PRIVATE cxx_std_17)
    install(TARGETS llama-cycle-standard RUNTIME)
endif()
