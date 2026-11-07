# Reuse a plan.json produced by region-ring-offline-plan. No planner search.
# The model, token-size hint and Phone NPU Region mapping are checked before
# handing off to the existing fixed-layout run_prompt_sweep.ps1 path.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptDir,
    [string]$Out = ".\offline-plan-runs",
    [string]$Exe = "",
    [int]$Threads = 8,
    [int]$NPredict = 32,
    [int]$Batch = 512,
    [int]$UBatch = 512,
    [switch]$NoMmap,
    [switch]$ConfirmBankedPhoneReady,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$planFile = (Resolve-Path $Plan).Path
$p = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
if ($p.schema -eq 2 -and $p.planner -eq 'phone-four-backend-dual-bank-exact-dag-v1') {
    # The schema-2 Runner uses only the supported experimental Runtime
    # subset. Never reinterpret an arbitrary DAG or fall back to OpenCL.
    if (!$p.runtime_compatible -or !$p.runtime_experimental) {
        throw "The selected G/C/N/A plan is still offline-only. Re-run run_plan.ps1 -PhoneMixed -RuntimeCompatible."
    }
    if (!$ConfirmBankedPhoneReady) {
        throw "Patch32 Android Phone with Plan-controlled two HTP Banks, MIXED_OPENCL=1 and NPU_GPU_PIPELINE=1 must be built and checked. Pass -ConfirmBankedPhoneReady only after checking those conditions."
    }
    if ([IO.Path]::GetFileName($Model) -ine [IO.Path]::GetFileName([string]$p.model_id)) {
        throw "Saved plan/model mismatch. Re-plan for this GGUF."
    }
    $stages = @($p.execution_stages)
    $edges = @($p.execution_edges)
    $cycles = @($p.cycles)
    if ($cycles.Count -lt 2 -or $cycles.Count -gt 4 -or
            $stages.Count -lt 4 -or $edges.Count -ne $stages.Count - 1 -or
            $stages.Count -gt 16) {
        throw "Invalid schema-2 Cycle or Region count."
    }
    for ($cyIndex=0; $cyIndex -lt $cycles.Count; $cyIndex++) {
        $cy=$cycles[$cyIndex]
        $phoneDepth=[int]$cy.phone_npu + [int]$cy.phone_gpu
        $pcTail=($cyIndex -eq $cycles.Count - 1 -and $phoneDepth -eq 0)
        if ([int]$cy.pc_gpu -lt 1 -or [int]$cy.pc_cpu -lt 0 -or
                [int]$cy.phone_npu -lt 0 -or [int]$cy.phone_npu -gt 1 -or
                [int]$cy.phone_gpu -lt 0 -or
                ($phoneDepth -lt 1 -and !$pcTail)) {
            throw "Unsupported schema-2 Runtime Cycle: G-first, optional CPU, HTP P1, optional last PC tail."
        }
    }
    $currentLayer = 0
    $runtimeParts = @()
    $htp = @()
    $adreno = @()
    for ($i = 0; $i -lt $stages.Count; $i++) {
        $st = $stages[$i]
        $depth = [int]$st.last_layer_exclusive - [int]$st.first_layer
        if ([int]$st.region -ne $i -or
                [int]$st.first_layer -ne $currentLayer -or $depth -le 0) {
            throw "Schema-2 Regions must cover all layers in strictly ordered contiguous ranges."
        }
        $code = switch ([string]$st.backend) {
            'G' { 'G' }
            'C' { 'C' }
            'N' { 'P' }
            'A' { 'P' }
            default { throw "Unknown physical backend in schema-2 Region." }
        }
        $runtimeParts += "$code$depth"
        if ($st.backend -eq 'N') {
            if ($depth -ne 1) { throw "HTP local executor only supports P1." }
            $htp += $st
        } elseif ($st.backend -eq 'A') {
            $adreno += $st
        }
        $currentLayer = [int]$st.last_layer_exclusive
    }
    if ($currentLayer -ne [int]$p.layers -or
            $htp.Count -lt 2 -or $adreno.Count -lt 1 -or
            $stages[0].backend -ne 'G' -or
            @('G','C','N','A') -notcontains [string]$stages[-1].backend) {
        throw "Schema-2 model coverage, two-Bank HTP or final backend invalid."
    }
    $localEdges = @($edges | Where-Object { $_.handoff -eq 'HTP_TO_ADRENO' })
    if ($localEdges.Count -lt 1) {
        throw "Missing adjacent Phone HTP->Adreno edge; refusing ordinary P2C/C2P fallback."
    }
    for ($i = 0; $i -lt $edges.Count; $i++) {
        $e = $edges[$i]
        if ([int]$e.from_region -ne $i -or [int]$e.to_region -ne $i+1) {
            throw "Schema-2 execution_edges must match adjacent physical Region IDs."
        }
        if ($e.handoff -eq 'HTP_TO_ADRENO' -and
                ($stages[$i].backend -ne 'N' -or
                 $stages[$i+1].backend -ne 'A')) {
            throw "Invalid local HTP->Adreno edge; only directly adjacent Phone stages are legal."
        }
    }
    $expectNpu = @($p.phone_npu_mapping)
    if ($expectNpu.Count -ne $htp.Count) { throw "Schema-2 NPU mapping count mismatch." }
    for ($i = 0; $i -lt $htp.Count; $i++) {
        if ([int]$expectNpu[$i].region -ne [int]$htp[$i].region -or
                [int]$expectNpu[$i].layer -ne [int]$htp[$i].first_layer) {
            throw "Schema-2 NPU mapping does not match selected physical Regions."
        }
    }
    $inputPrompts = @(Get-ChildItem -LiteralPath $PromptDir -File -Filter 'prompt_*.txt')
    if ($inputPrompts.Count -eq 0) { throw "No prompt_N.txt found." }
    foreach ($sample in $inputPrompts) {
        if ($sample.BaseName -notmatch '^prompt_([0-9]+)$' -or
                [int]$Matches[1] -ne [int]$p.tokens) {
            throw "This Plan was calibrated for $($p.tokens) tokens; provide the matching Prompt group."
        }
    }
    $layout = $runtimeParts -join ','
    $npuRegions = (@($htp | ForEach-Object { [string]$_.region })) -join ','
    $x = [int]$p.chunk_tokens
    if ($x -notin @(16,32,64,128)) { throw "Unsupported shared Chunk width." }
    Write-Host "[PHONE_LOCAL_HANDOFF] phase=APPLY layout=$layout terminal=$($stages[-1].backend) npu_regions=$npuRegions local_links=$($localEdges.Count) chunk=$x status=EXPERIMENTAL"
    # These PC settings are model/runtime capabilities. The actual Phone NPU
    # layer mapping travels in the Patch32 Plan RPC, not in Phone exports.
    $capabilities = @{
        "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP" = "1"
        "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS" = $npuRegions
        "LLAMA_REGION_RING_PHONE_GRAPHLESS" = "0"
        "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW" = "1"
        "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES" = "1"
        "GGML_META_PHONE_FULL_REGION_GRAPH" = "1"
        "LLAMA_REGION_RING_PHONE_AUTONOMOUS" = "1"
        "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL" = "1"
        "LLAMA_REGION_RING_PHONE_INPUT_PUSH" = "1"
        "LLAMA_REGION_RING_PHONE_INPUT_STREAM" = "1"
        "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM" = "1"
        "LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP" = "0"
        "LLAMA_REGION_RING_REAL_STAGE_PROFILE" = "1"
        "LLAMA_REGION_RING_TASK_TRACE" = "1"
    }
    $oldVars = @{}
    foreach ($key in $capabilities.Keys) {
        $oldVars[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
    }
    try {
        foreach ($key in $capabilities.Keys) {
            [Environment]::SetEnvironmentVariable($key, $capabilities[$key], "Process")
        }
        $params = @{
            Model = $Model
            Rpc = $Rpc
            PromptDir = $PromptDir
            Out = $Out
            Threads = $Threads
            NPredict = $NPredict
            Batch = $Batch
            UBatch = $UBatch
            Planner = "apply-wide"
            SkipProfile = $true
            ForceLayout = $layout
            ForceXC = $x
            ForceXG = $x
            ForceXP = $x
            LogVerbosity = 4
        }
        if ($Exe -ne "") { $params.Exe = $Exe }
        if ($NoMmap) { $params.NoMmap = $true }
        if ($DryRun) { $params.DryRun = $true }
        & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @params
        if ($LASTEXITCODE -ne 0) {
            throw "Four-backend execution failed; check final-owner/PC_HEAD and Phone LOCAL_HANDOFF logs."
        }
    }
    finally {
        foreach ($key in $capabilities.Keys) {
            [Environment]::SetEnvironmentVariable($key, $oldVars[$key], "Process")
        }
    }
    return
}
if ($p.schema -ne 1 -or $p.planner -ne 'offline-exact-dag') {
    throw "Invalid plan file schema or planner type: $planFile"
}
if ($p.npu_streaming_required -and -not $p.npu_streaming_runtime_ready) {
    throw "Plan requires two HTP weight Banks. Re-run offline planner with --npu-banked-runtime-confirmed 1 ONLY after the patched Phone HTP server has been built and tested."
}
if ($p.npu_streaming_required -and -not $ConfirmBankedPhoneReady) {
    throw "Banked Plan requires -ConfirmBankedPhoneReady after verifying the Phone server has GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2 and matching Region/Layer lists."
}
if ($p.max_npu_region_layers -gt 1) {
    throw "HTP local-only executor currently supports one Transformer layer per Phone Region."
}
if ([IO.Path]::GetFileName($Model) -ine [IO.Path]::GetFileName([string]$p.model_id)) {
    throw "Model file name does not match saved plan: expected $($p.model_id), got $Model. Re-plan for this model."
}
$prompts = @(Get-ChildItem -LiteralPath $PromptDir -File -Filter 'prompt_*.txt')
if ($prompts.Count -eq 0) { throw "No prompt_N.txt in $PromptDir" }
foreach ($f in $prompts) {
    if ($f.BaseName -notmatch '^prompt_([0-9]+)$' -or
            [int]$Matches[1] -ne [int]$p.tokens) {
        throw "Saved plan expects $($p.tokens)-token hints; this directory contains $($f.Name). Use a plan matching each prompt length."
    }
}
if ([int]$p.xc -le 0 -or [int]$p.xg -le 0 -or
        [int]$p.xp -le 0 -or [string]::IsNullOrWhiteSpace($p.layout)) {
    throw "Plan missing fixed layout or Chunk sizes."
}
Write-Host "[OFFLINE_PLAN_APPLY] model=$($p.model_id) tokens=$($p.tokens)"
Write-Host "[OFFLINE_PLAN_APPLY] layout=$($p.layout) XC=$($p.xc) XG=$($p.xg) XP=$($p.xp)"
Write-Host "[OFFLINE_PLAN_APPLY] Phone RPC MUST be started with GGML_RPC_PHONE_NPU_SERIAL_REGIONS=$($p.phone_npu_regions)"
if ($p.npu_streaming_required) {
    Write-Host "[OFFLINE_PLAN_APPLY] Phone MUST use GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2, the matching ordered GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS, Qwen3 LOCAL_ONLY+GRAPHLESS+DIRECT_INPUT+DIRECT_P2C+HTP_PRECOMPILE, and tested Android Hexagon binaries."
}
Write-Host "[OFFLINE_PLAN_APPLY] This runs APPLY_WIDE -SkipProfile. It does NOT run the Planner or collect new calibration."
$call = @{
    Model = $Model
    Rpc = $Rpc
    PromptDir = $PromptDir
    Out = $Out
    Threads = $Threads
    NPredict = $NPredict
    Batch = $Batch
    UBatch = $UBatch
    Planner = "apply-wide"
    SkipProfile = $true
    ForceLayout = [string]$p.layout
    ForceXC = [int]$p.xc
    ForceXG = [int]$p.xg
    ForceXP = [int]$p.xp
    LogVerbosity = 4
}
if ($Exe -ne "") { $call.Exe = $Exe }
if ($NoMmap) { $call.NoMmap = $true }
if ($DryRun) { $call.DryRun = $true }
& (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @call
exit $LASTEXITCODE
