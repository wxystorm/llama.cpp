# Jupiter Native offline DP

The default full suite now uses **jupiter-offline-dp**. It is not the old
per-prompt exact-q refinement experiment. Keep their output directories
and results separate.

* Two physical pipeline stages: PC prefix (CPU + CUDA local service) and
  Phone suffix (all serial HTP sessions). The native bridge wait/transfer
  and splits after that bridge are reported separately. The local residual
  in the tagged CPU split is retained; this is split-level attribution.
* Fixed Context/Batch 2048, default maximum Chunk 256, paper DP k <= 8.
  Auxiliary calibration plans may use up to the 16-slot wire capacity.
* Verify a complete 2048-token max-width execution first. This checks
  actual allocation/Prepare/Prefill feasibility, not peak Phone RAM usage.
* Measure CUDA and HTP throughput saturation using exact synthetic chat
  prompts; the 90% throughput proxy is not hardware occupancy. One b is
  selected for all lengths, with explicit feasibility/plateau fallback.
* Measure a shared (x, KV-prefix y) grid with real native KV writes.
  Median physical-stage wall times -> q(x,y)=max(PC,Phone).
  The q costs exclude graph preparation, network and final output path.
* Independent off-grid samples report interpolation error. The 20% maximum
  error threshold is a diagnostic warning, not a proof of exact q or a
  condition to skip measurements. No per-Prompt refinement occurs.
* Use one shared Eq. (2)-(4) DP table for lengths 2..2048. Short N<b is
  an explicitly labeled single-Chunk exception. Formal prefill timings
  still measure the original DECODE_TO_LOGITS metric, including actual
  communication and graph preparation when incurred.

## PowerShell

Existing RPC patch 8 PC/Phone binaries suffice; no C++ changes in this
offline migration. Restart the Phone RPC service with the correct model
before profiling. The first capacity check is required: 256-token Native
feasibility on the real devices has not been assumed.

    .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -CpuLayers 36 -Layouts 52 -Sessions 4 -UseServer -SequencePolicy jupiter-offline-dp -MaxChunk 256 -OfflineProfileDir ".\jupiter-offline-profile\32b-l52" -OutDir ".\jupiter-32b-l52-offline-full50" -ProfileOnly -Resume

Remove -ProfileOnly to run the full 50; all profile options must match.
Or omit -ProfileOnly from the first run to build the missing profile and
then automatically execute the formal benchmark. A ready profile is reused
without starting a calibration worker.

Changing the Phone driver/binary or relevant runtime environment requires
a new profile directory/tag, even if the RPC endpoint is unchanged. The
protocol does not identify the exact remote software image. Do not run
concurrent profile/experiment workers against the same Phone service.

## Artifacts

hardware-config.json, capacity.json, min-chunk.json, stage2-samples.csv,
q-profile.csv, q-samples.csv, validation.csv, validation-samples.csv,
validation.json, sequence.plan, plan-summary.csv, profile-ready.json.
Each completed measurement keeps its validated request trace and sidecar.
Incomplete/failed requests are never accepted into the shared profile.
The formal output records offline-profile-binding.json and each case
plan_source. New and historical results cannot be resumed into one directory.

The old explicit -SequencePolicy jupiter-dp retains the historical
per-Prompt route. -RefineRepeats and -MaxRounds belong to that route,
not the new offline sampler. -LegacyCalibration is historical only.

## Model-free regression tests

    python -m unittest discover -s tools/jupiter -p "test_offline_profile.py"
    python -m unittest discover -s tools/jupiter -p "test_run_phone_owned_full_suite.py"

