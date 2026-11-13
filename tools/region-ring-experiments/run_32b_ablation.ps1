# Qwen3-32B (64 layers): A0/A1 fixed G8-P6, A2 profiled Cycles,
# A3 one memory-greedy G->CPU->HTP pass. Default: two prompts only.
[CmdletBinding()]
param(
    [string]$Model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
    [string]$Rpc = "10.39.52.114:50052",
    [string]$PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$Out = ".\32b-bank-cycle-ablation",
    [string]$ProfileCsv = "",
    [ValidateSet("A0","A1","A2","A3")][string[]]$Mode = @(),
    [string[]]$Prompt = @(),
    [ValidateRange(1,20)][int]$Repeat = 1,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
$entry = Join-Path $PSScriptRoot "run_32b_ablation.py"
$arguments = @($entry, "--model", $Model, "--rpc", $Rpc,
               "--prompt-dir", $PromptDir, "--out", $Out,
               "--repeat", "$Repeat")
if ($ProfileCsv) { $arguments += @("--profile-csv", $ProfileCsv) }
foreach ($name in $Mode) { $arguments += @("--mode", $name) }
foreach ($name in $Prompt) { $arguments += @("--prompt", $name) }
if ($DryRun) { $arguments += "--dry-run" }
& $python @arguments
exit $LASTEXITCODE
