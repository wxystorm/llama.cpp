# Motivation Fig. 2(b): phone NPU layer switching versus execution

**Base:** `feat/phone-only-htp-q4k-q6k-stream-20261115`. **Experiment branch:** `exp/motivation-phone-npu-layer-20261117`.

Aim: measure the cost of staging one layer's **HTP-offloaded weights** from file-backed CPU mmap into HTP REPACK storage and compare it with the time spent executing the same layer's HTP graph splits. The latter is **not the compute time of an entire Transformer layer** when unsupported operators fall back to CPU.

This code is experimental and **has not been built or run on the Snapdragon phone**. Rebuild the Android Host binary and `libggml-hexagon.so`. Keep the matching DSP skeleton `libggml-htp-v79.so` deployed (rebuild it if your existing DSP image predates the original Q4_K/Q6_K support).

## On-device command

After deploying the updated binary, libraries, and the two `tools/phone-only-streaming` scripts into `$HOME/phone-stream-htp/`:

```bash
export LD_LIBRARY_PATH="$HOME/phone-stream-htp/lib:$PREFIX/lib"
export ADSP_LIBRARY_PATH="$HOME/phone-stream-htp/lib"

LLAMA_BIN="$HOME/phone-stream-htp/bin/llama-cli" \
  LAYER=20 PRETOUCH=0 \
  bash "$HOME/phone-stream-htp/run_motivation_2b.sh" \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
  "$HOME/sharegpt40/prompt_512_01.txt" \
  "$HOME/phone-moti/14b-l20-natural"

# Deliberate page-touch before CPU repack: separates cold/warm memory
# page-in activity from host repack. This is NOT natural runtime behavior.
LLAMA_BIN="$HOME/phone-stream-htp/bin/llama-cli" \
  LAYER=20 PRETOUCH=1 \
  bash "$HOME/phone-stream-htp/run_motivation_2b.sh" \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
  "$HOME/sharegpt40/prompt_512_01.txt" \
  "$HOME/phone-moti/14b-l20-pretouch"
```

Use `GGML_HEXAGON_NDEV=1` for the isolated HTP session, `-npu -n 1`, and identical prompt/context/ubatch parameters in both runs. Repeat 3-5 times if practical, keeping output directories separate to preserve logs.

## Output

`run.log` and `metrics.log` contain:
- `[PHONE_MOTI_MAP]`: cost of **fastrpc_mmap for a shared buffer allocation**. These mappings may be reused; do not attribute every one to an individual layer.
- `[PHONE_MOTI_ALLOC]`: cost of `rpcmem_alloc2` + FD retrieval (separate from mapping).
- `[PHONE_MOTI_WEIGHT]`: one quantized layer weight copied to HTP; `page_touch_ms`, `repack_wall_ms`, `page_touch_read_mib`, `repack_read_mib` and major faults.
- `[PHONE_MOTI_SPLIT]`: one HTP split attributed to a single `blk.N`; `stage_ms`, `npu_wall_ms`, weight bytes, stage storage I/O and major faults.
- `[PHONE_MOTI_LAYER]`: per-layer aggregation across its exclusively attributed HTP splits for **one scheduler graph evaluation**. A longer prompt produces several of these lines (one per ubatch).
- `[PHONE_MOTI_DIAG]`: count of splits skipped because the layer could not be uniquely attributed.

The parser creates `layer_samples.csv`, `map_events.csv` and `summary.txt` with median values. For Fig. 2(b), use `median_stage_ms` and `median_htp_wall_ms` from `PRETOUCH=0`, and show the breakdown from `PRETOUCH=1` only as auxiliary analysis. Do not add mapping time to the stage time unless evidence proves the mapping occurred on that specific layer's critical path.

**Measurement cautions:** `/proc/self/io read_bytes` is process-attributed storage I/O, not exact per-weight flash read time. `getrusage(ru_majflt)` counts major page faults. `PRETOUCH=0` repack timing can include page-in stalls. `PRETOUCH=1` performs an extra explicit page walk; it changes cache residency and should **not** be used for the headline layer-switch latency. NPU execution walls include graph submission and synchronization and are not HTP kernel-only. The measured ratio may be tens or hundreds, not necessarily orders of magnitude; report the observed ratio.

## Resident-weight OOM ceiling (separate experiment)

Do **not** interpret `GGML_PHONE_STREAM_DEVICE_BUDGET_MIB` as the pinned-resident OOM threshold. This `-npu` path uses temporary on-demand REPACK buffers and cannot directly establish the number of complete model layers that remain resident in HTP memory.

For the main-text resident ceiling, use a controlled pinned-resident allocation / actual resident-layer configuration and record layer count, resident tensor bytes, process RSS and Android `lmkd`/OOM or FastRPC allocation failures. Avoid inferring OOM from `GGML_HEXAGON_VMEM` address-space limits alone.
