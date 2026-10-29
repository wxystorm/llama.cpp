param(
    [Parameter(Mandatory = $true)]
    [string] $Exe,

    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Rpc,

    [Parameter(Mandatory = $true)]
    [string] $PromptFile,

    [int] $TargetTokens = 358,
    [string] $XValues = "32,48,64,80,96,112,128",
    [int] $MinChunk = 32,
    [int] $MaxChunk = 128,
    [int] $PcLayers = 25,
    [int] $Repeats = 1,

    [string] $OutDir = ".\\jupiter-q-sweep",
    [string] $PythonExe = "python",

    [int] $Batch = 512,
    [int] $Context = 512,
    [int] $Threads = 8,

    [switch] $RunDp,
    [string] $PlanOutput = ".\\jupiter.plan",
    [switch] $DryRun,

    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

function Convert-ToIntList {
    param([string] $Text)

    $values = @()
    foreach ($item in $Text.Split(",")) {
        $trimmed = $item.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }

        $value = 0
        if (-not [int]::TryParse($trimmed, [ref] $value)) {
            throw "invalid integer in list: '$trimmed'"
        }
        $values += $value
    }
    return $values
}

function Split-FeasibleTail {
    param(
        [int] $Total,
        [int] $Min,
        [int] $Max
    )

    if ($Total -lt $Min) {
        throw "tail $Total is smaller than min chunk $Min"
    }

    $count = [int] [Math]::Ceiling($Total / [double] $Max)
    while ($count -gt 1 -and [Math]::Floor($Total / [double] $count) -lt $Min) {
        $count--
    }

    if ($count -le 0) {
        throw "cannot split tail $Total into [$Min,$Max]"
    }

    $base = [int] [Math]::Floor($Total / [double] $count)
    $extra = $Total - $base * $count
    $peak = $base
    if ($extra -gt 0) {
        $peak++
    }

    if ($base -lt $Min -or $peak -gt $Max) {
        throw "cannot split tail $Total into [$Min,$Max]"
    }

    $chunks = @()
    for ($i = 0; $i -lt $count; $i++) {
        $value = $base
        if ($i -lt $extra) {
            $value++
        }
        $chunks += $value
    }
    return $chunks
}

function New-SweepPlan {
    param(
        [int] $Total,
        [int] $X,
        [int] $Min,
        [int] $Max
    )

    if ($X -lt $Min -or $X -gt $Max) {
        throw "x=$X is outside [$Min,$Max]"
    }
    if ($Total -lt $Min) {
        throw "target token count $Total is too small"
    }

    $n = [int] [Math]::Floor($Total / [double] $X)
    $rem = $Total - $n * $X

    if ($n -eq 0) {
        return Split-FeasibleTail -Total $Total -Min $Min -Max $Max
    }

    $chunks = @()

    if ($rem -eq 0) {
        for ($i = 0; $i -lt $n; $i++) {
            $chunks += $X
        }
        return $chunks
    }

    if ($rem -ge $Min) {
        for ($i = 0; $i -lt $n; $i++) {
            $chunks += $X
        }
        $chunks += $rem
        return $chunks
    }

    # Fold a tiny remainder into the last X-sized chunk and rebalance only
    # that tail. This preserves as many exact X samples as possible.
    for ($i = 0; $i -lt ($n - 1); $i++) {
        $chunks += $X
    }

    $tail = $X + $rem
    $chunks += Split-FeasibleTail -Total $tail -Min $Min -Max $Max
    return $chunks
}

function Save-ProcessEnv {
    param([string[]] $Names)

    $saved = @{}
    foreach ($name in $Names) {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    return $saved
}

function Restore-ProcessEnv {
    param(
        [hashtable] $Saved,
        [string[]] $Names
    )

    foreach ($name in $Names) {
        $value = $Saved[$name]
        if ($null -eq $value) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $value
        }
    }
}

if ($MinChunk -le 0 -or $MaxChunk -lt $MinChunk) {
    throw "invalid chunk range [$MinChunk,$MaxChunk]"
}
if ($Repeats -le 0) {
    throw "Repeats must be positive"
}

$exePath = (Resolve-Path $Exe).Path
$modelPath = (Resolve-Path $Model).Path
$promptPath = (Resolve-Path $PromptFile).Path
$promptText = Get-Content -Raw $promptPath

$plannerPath = Join-Path $PSScriptRoot "jupiter_planner.py"
if (-not (Test-Path $plannerPath)) {
    throw "planner not found: $plannerPath"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outDirPath = (Resolve-Path $OutDir).Path

$xList = Convert-ToIntList $XValues
if ($xList.Count -eq 0) {
    throw "XValues is empty"
}

$plans = @()
foreach ($x in $xList) {
    $chunks = @(New-SweepPlan -Total $TargetTokens -X $x -Min $MinChunk -Max $MaxChunk)
    $sum = ($chunks | Measure-Object -Sum).Sum

    if ($sum -ne $TargetTokens) {
        throw "internal plan error for x=${x}: sum=$sum target=$TargetTokens"
    }

    foreach ($chunk in $chunks) {
        if ($chunk -lt $MinChunk -or $chunk -gt $MaxChunk) {
            throw "internal plan error for x=${x}: chunk=$chunk outside [$MinChunk,$MaxChunk]"
        }
    }

    $planText = ($chunks -join ",")
    $plans += [pscustomobject]@{
        X = $x
        Chunks = $chunks
        Text = $planText
    }

    Write-Host "[JUPITER_SWEEP_PLAN] x=$x total=$TargetTokens chunks=$planText"
}

if ($DryRun) {
    return
}

$envNames = @(
    "LLAMA_JUPITER_PREFILL",
    "LLAMA_JUPITER_PC_LAYERS",
    "LLAMA_JUPITER_WINDOW",
    "LLAMA_JUPITER_TRACE",
    "LLAMA_JUPITER_PROFILE_STAGE",
    "LLAMA_JUPITER_SEQUENCE_PLAN",
    "LLAMA_JUPITER_PLAN_FILE"
)
$savedEnv = Save-ProcessEnv $envNames

$csvFiles = @()

try {
    $env:LLAMA_JUPITER_PREFILL = "1"
    $env:LLAMA_JUPITER_PC_LAYERS = "$PcLayers"
    $env:LLAMA_JUPITER_WINDOW = "1"
    $env:LLAMA_JUPITER_TRACE = "1"
    $env:LLAMA_JUPITER_PROFILE_STAGE = "1"

    Remove-Item Env:LLAMA_JUPITER_PLAN_FILE -ErrorAction SilentlyContinue

    foreach ($plan in $plans) {
        for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
            $env:LLAMA_JUPITER_SEQUENCE_PLAN = $plan.Text

            $stem = "q-x{0}-r{1}" -f $plan.X, $repeat
            $logPath = Join-Path $outDirPath "$stem.log"
            $csvPath = Join-Path $outDirPath "$stem.csv"

            Write-Host ""
            Write-Host "[JUPITER_SWEEP_RUN] x=$($plan.X) repeat=$repeat/$Repeats chunks=$($plan.Text)"

            $cliArgs = @(
                "-m", $modelPath,
                "--rpc", $Rpc,
                "--split-mode", "layer",
                "-ngl", "all",
                "-ub", "$MaxChunk",
                "-b", "$Batch",
                "-c", "$Context",
                "-n", "1",
                "-t", "$Threads",
                "--flash-attn", "on",
                "-lv", "3",
                "--no-conversation",
                "--single-turn",
                "--simple-io",
                "-p", $promptText
            )
            if ($ExtraArgs.Count -gt 0) {
                $cliArgs += $ExtraArgs
            }

            # Windows PowerShell 5.x wraps native stderr as ErrorRecord objects.
            # llama.cpp intentionally writes normal INFO logs to stderr, so the
            # script must not treat those records as terminating PowerShell errors.
            # The real failure signal for the native process is $LASTEXITCODE.
            $savedErrorActionPreference = $ErrorActionPreference
            $hasNativePreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference
            if ($hasNativePreference) {
                $savedNativePreference = $PSNativeCommandUseErrorActionPreference
            }

            try {
                $ErrorActionPreference = "Continue"
                if ($hasNativePreference) {
                    $PSNativeCommandUseErrorActionPreference = $false
                }

                & $exePath @cliArgs 2>&1 |
                    ForEach-Object {
                        if ($_ -is [System.Management.Automation.ErrorRecord]) {
                            $_.Exception.Message
                        } else {
                            [string] $_
                        }
                    } |
                    Tee-Object -FilePath $logPath

                $exitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $savedErrorActionPreference
                if ($hasNativePreference) {
                    $PSNativeCommandUseErrorActionPreference = $savedNativePreference
                }
            }

            if ($exitCode -ne 0) {
                throw "llama-cli failed for x=$($plan.X), repeat=$repeat, exit=$exitCode"
            }

            $planLine = Select-String -Path $logPath -Pattern "\[JUPITER_SEQUENCE_PLAN\].*total=(\d+).*chunks=([^ ]+)" | Select-Object -Last 1
            $actualTotal = 0
            $actualPlan = $null
            $planSource = "SEQUENCE_PLAN"

            if ($null -ne $planLine) {
                $match = [regex]::Match($planLine.Line, "total=(\d+).*chunks=([^ ]+)")
                if (-not $match.Success) {
                    throw "could not parse runtime plan line: $($planLine.Line)"
                }

                $actualTotal = [int] $match.Groups[1].Value
                $actualPlan = $match.Groups[2].Value.Trim()
            } else {
                # Some CLI runs can omit the summary line even though the
                # runtime still executes the requested sequence plan. Rebuild
                # the formal prompt partition from JUPITER_UBATCH BEGIN lines.
                # Warmup ubatches are excluded by requiring tokens >= MinChunk.
                $ubatchChunks = @()
                $ubatchLines = Select-String -Path $logPath -Pattern "\[JUPITER_UBATCH\].*event=BEGIN.*prefix_tokens=(\d+).*tokens=(\d+)"

                foreach ($line in $ubatchLines) {
                    $m = [regex]::Match($line.Line, "prefix_tokens=(\d+).*tokens=(\d+)")
                    if (-not $m.Success) {
                        continue
                    }

                    $prefix = [int] $m.Groups[1].Value
                    $tokens = [int] $m.Groups[2].Value

                    if ($tokens -lt $MinChunk) {
                        continue
                    }

                    $ubatchChunks += [pscustomobject]@{
                        Prefix = $prefix
                        Tokens = $tokens
                    }
                }

                if ($ubatchChunks.Count -gt 0) {
                    $ordered = @($ubatchChunks | Sort-Object Prefix)
                    $expectedPrefix = 0
                    $chunks = @()
                    $valid = $true

                    foreach ($item in $ordered) {
                        if ($item.Prefix -ne $expectedPrefix) {
                            $valid = $false
                            break
                        }
                        $chunks += $item.Tokens
                        $expectedPrefix += $item.Tokens
                    }

                    if ($valid -and $expectedPrefix -eq $TargetTokens) {
                        $actualTotal = $expectedPrefix
                        $actualPlan = ($chunks -join ",")
                        $planSource = "UBATCH_RECONSTRUCT"
                        Write-Host "[JUPITER_SWEEP_PLAN_FALLBACK] x=$($plan.X) repeat=$repeat source=$planSource chunks=$actualPlan"
                    }
                }
            }

            if ($null -eq $actualPlan) {
                $tail = Get-Content -Path $logPath -Tail 40 -ErrorAction SilentlyContinue
                Write-Host "[JUPITER_SWEEP_DIAG] missing plan summary and no complete ubatch reconstruction; log=$logPath"
                if ($null -ne $tail) {
                    $tail | ForEach-Object { Write-Host $_ }
                }
                throw "runtime did not expose a complete Jupiter sequence plan for x=$($plan.X)"
            }

            if ($actualTotal -ne $TargetTokens -or $actualPlan -ne $plan.Text) {
                throw "runtime plan mismatch: expected total=$TargetTokens chunks=$($plan.Text), got total=$actualTotal chunks=$actualPlan source=$planSource"
            }

            $extractArgs = @(
                $plannerPath,
                "extract-q",
                "--log", $logPath,
                "--output", $csvPath,
                "--metric", "compute_ms",
                "--min-tokens", "$MinChunk"
            )
            & $PythonExe @extractArgs

            if ($LASTEXITCODE -ne 0) {
                throw "extract-q failed for x=$($plan.X), repeat=$repeat"
            }

            $csvFiles += $csvPath
        }
    }

    $profilePath = Join-Path $outDirPath "q-profile-all.csv"

    $allRows = @()
    foreach ($csvFile in $csvFiles) {
        $allRows += Import-Csv $csvFile
    }

    if ($allRows.Count -eq 0) {
        throw "sweep produced no q samples"
    }

    # Keep repeated observations. The Python planner collapses repeated
    # (x,y,device) samples with a median before computing the slowest stage.
    $allRows | Export-Csv -Path $profilePath -NoTypeInformation -Encoding UTF8

    $uniquePairs = ($allRows | Group-Object -Property x,y | Measure-Object).Count

    Write-Host ""
    Write-Host "[JUPITER_SWEEP_DONE] rows=$($allRows.Count) unique_xy=$uniquePairs profile=$profilePath"

    if ($RunDp) {
        $planOutPath = [System.IO.Path]::GetFullPath($PlanOutput)

        $dpArgs = @(
            $plannerPath,
            "sequence",
            "--profile", $profilePath,
            "--output", $planOutPath,
            "--devices", "2",
            "--min-chunk", "$MinChunk",
            "--max-chunk", "$MaxChunk",
            "--target-length", "$TargetTokens"
        )
        & $PythonExe @dpArgs

        if ($LASTEXITCODE -ne 0) {
            throw "sequence DP failed"
        }

        Write-Host "[JUPITER_SWEEP_DP] plan=$planOutPath"
        Get-Content $planOutPath
    }
}
finally {
    Restore-ProcessEnv -Saved $savedEnv -Names $envNames
}
