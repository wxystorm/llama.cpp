$ProcessName = "llama-cli"
$OutputFile = "E:\llama\llama.cpp\pc_memory.csv"
$IntervalSeconds = 1

# CSV 表头
"Time,Elapsed_s,WorkingSet_MB,PeakWorkingSet_MB,PrivateMemory_MB,AvailableRAM_MB,UsedRAM_MB,TotalRAM_MB" |
    Out-File -FilePath $OutputFile -Encoding utf8

Write-Host "Waiting for $ProcessName ..."

# 等待新的 llama-cli 启动
do {
    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
            Sort-Object StartTime -Descending |
            Select-Object -First 1

    if (-not $proc) {
        Start-Sleep -Milliseconds 500
    }
} while (-not $proc)

$pid_target = $proc.Id
$startTime = Get-Date

# 总物理内存基本不变，只读取一次
$os = Get-CimInstance Win32_OperatingSystem
$totalMB = $os.TotalVisibleMemorySize / 1024

Write-Host "Found $ProcessName PID=$pid_target"
Write-Host "Total RAM: $([math]::Round($totalMB, 0)) MB"
Write-Host "Logging to: $OutputFile"
Write-Host ""

while ($true) {

    $proc = Get-Process -Id $pid_target -ErrorAction SilentlyContinue

    if (-not $proc) {
        Write-Host ""
        Write-Host "llama-cli exited."
        break
    }

    $proc.Refresh()

    # -------------------------
    # 进程内存
    # -------------------------

    # 当前驻留在物理 RAM 中的进程内存
    $workingSetMB =
        $proc.WorkingSet64 / 1MB

    # 运行以来最大 Working Set
    $peakWorkingSetMB =
        $proc.PeakWorkingSet64 / 1MB

    # 进程私有提交内存
    $privateMemoryMB =
        $proc.PrivateMemorySize64 / 1MB

    # -------------------------
    # 系统内存
    # -------------------------

    $os = Get-CimInstance Win32_OperatingSystem

    $availableMB =
        $os.FreePhysicalMemory / 1024

    $usedMB =
        $totalMB - $availableMB

    # -------------------------
    # 时间
    # -------------------------

    $elapsed =
        ((Get-Date) - $startTime).TotalSeconds

    $timestamp =
        Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # -------------------------
    # CSV
    # -------------------------

    $line =
        "{0},{1:F1},{2:F2},{3:F2},{4:F2},{5:F2},{6:F2},{7:F2}" -f `
        $timestamp, `
        $elapsed, `
        $workingSetMB, `
        $peakWorkingSetMB, `
        $privateMemoryMB, `
        $availableMB, `
        $usedMB, `
        $totalMB

    Add-Content -Path $OutputFile -Value $line

    # -------------------------
    # 控制台显示
    # -------------------------

    Write-Host (
        "[{0,5:F1}s] WS={1,8:F0} MB  Peak={2,8:F0} MB  Private={3,8:F0} MB  Available={4,8:F0} MB" -f `
        $elapsed, `
        $workingSetMB, `
        $peakWorkingSetMB, `
        $privateMemoryMB, `
        $availableMB
    )

    Start-Sleep -Seconds $IntervalSeconds
}

Write-Host "CSV saved to: $OutputFile"