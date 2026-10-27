# Restart the sweeps between two runs: wait until a sweep log shows its next job starting, stop every python
# started from this folder (twice, seconds apart), then launch.ps1. Loses only the new job's few seconds of loading.
# Stops ALL of this folder's sweeps, so stop launch_after / launch_when_free waiters first. Detached (src/detach.py).
#   powershell -ExecutionPolicy Bypass -File relaunch_at_boundary.ps1 -Log runs\scout_gpu1.log -Config configs/queue.txt -Gpu 1
param(
  [Parameter(Mandatory = $true)][string]$Log,
  [Parameter(Mandatory = $true)][string]$Config,
  [int]$Gpu = 0,
  [string]$Shard = "queue"
)
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$launch = Join-Path $PSScriptRoot "launch.ps1"
$jobLine = "'^\[\d+/\d+\] '"
$wait = "`$n0 = @(Select-String -Path '$Log' -Pattern $jobLine).Count; " +
        "while (@(Select-String -Path '$Log' -Pattern $jobLine).Count -eq `$n0) { Start-Sleep 5 }"
$stop = "Get-Process python | Where-Object { `$_.Path -like '$PSScriptRoot*' } | Stop-Process -Force"
$cmd = "$wait; $stop; Start-Sleep 5; $stop; Start-Sleep 5; & '$launch' -Config '$Config' -Gpu $Gpu -Shard '$Shard'"
$newPid = & $python "src/detach.py" "runs\relaunch_gpu$Gpu.log" "runs\relaunch_gpu$Gpu.err" -- powershell -NoProfile -ExecutionPolicy Bypass -Command $cmd
"relauncher pid ${newPid}: restarts on $Config at the next job start in $Log (output -> runs\relaunch_gpu$Gpu.log)"
