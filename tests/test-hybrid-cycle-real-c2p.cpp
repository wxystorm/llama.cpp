// Compile the same physical Qwen3-MoE capture and Cycle helpers as the P2C
// diagnostic with the reverse Tensor boundary as the default. Separate test
// targets keep their public command lines and environment variables simple.
#define LLAMA_CYCLE_C2P_TEST 1
#include "test-hybrid-cycle-real-p2c.cpp"
