# Queue a sweep shard behind one that is still running: wait for that shard's process to exit, then run launch.ps1.
# Detached like launch.ps1, so it survives the SSH session ending. Use the pid that launch.ps1 printed for that GPU.
#   powershell -ExecutionPolicy Bypass -File launch_after.ps1 -WaitPid 20224 -Config "configs/scout.json" -Gpu 0 -Shard 0/2
# Shards are assigned from the full job list (sweep.py), so the two GPUs may start their next batch at different times.
param(
  [Parameter(Mandatory = $true)][int]$WaitPid,
  [Parameter(Mandatory = $true)][string]$Config,
  [int]$Gpu = 0,
  [string]$Shard = "0/1"
)
Set-Location $PSScriptRoot
$launch = Join-Path $PSScriptRoot "launch.ps1"
$cmd = "Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue; & '$launch' -Config '$Config' -Gpu $Gpu -Shard '$Shard' *> 'runs\after_gpu$Gpu.log'"
$p = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) `
      -WindowStyle Hidden -PassThru
"waiter pid $($p.Id): after pid $WaitPid exits, starts $Config shard $Shard on GPU $Gpu (launch output -> runs\after_gpu$Gpu.log)"
