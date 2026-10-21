// Opt-in physical first Cycle period (one complete repeating topology).
// With X=5,Y=1 this executes real layers 0..7 using ONE bounded FIFO:
//   PC_FULL[0..4] -> C2P[5] -> PHONE_FULL[6] -> P2C[7].
// Initial token embeddings come from independent CPU reference capture;
// every Transformer layer executes its real physical CPU/Phone weights.
// Distinct main() configuration preserves the single-boundary diagnostics.
#define LLAMA_CYCLE_LOOP_TEST 1
#include "test-hybrid-cycle-real-p2c.cpp"
