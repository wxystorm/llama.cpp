#pragma once

struct common_params;

// Special-purpose direct path for the official llama-cli executable when
// LLAMA_HYBRID_CYCLE_STANDARD_DECODE=1. No side effects unless opted in.
// This is separate from the regular llama-cli -> llama-server HTTP UI.
bool cli_cycle_requested();
int cli_cycle_run(common_params & params);
