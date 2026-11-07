# Export Hybrid Profile real transfer-only tensor-copy curves.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$ProfileRoot="",
    [switch]$Update
)
$ErrorActionPreference="Stop"
$repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$source=(Resolve-Path -LiteralPath $Model).Path
$stem=[IO.Path]::GetFileNameWithoutExtension($source)
if (!$ProfileRoot) { $ProfileRoot=Join-Path $repo "model-profiles\$stem" }
$folder=[IO.Path]::GetFullPath($ProfileRoot)
$logs=Join-Path $folder "runs"
if (!(Test-Path -LiteralPath $logs)) {
    throw "No base Profile transfer-only runs: $logs. Run run_model_profile.ps1 first."
}
$output=Join-Path $folder "communication-profile.csv"
if ((Test-Path -LiteralPath $output) -and !$Update) {
    throw "Existing $output. Pass -Update to explicitly refresh."
}
$python=(Get-Command python -ErrorAction Stop).Source
& $python (Join-Path $PSScriptRoot "build_communication_profile.py") --run $logs --output $output
if ($LASTEXITCODE -ne 0) { throw "Missing real transfer probes; nothing fabricated." }
Write-Host "[COMMUNICATION_PROFILE] status=COMPLETE csv=$output"
