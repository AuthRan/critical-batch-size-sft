# Start launch.ps1 on a GPU once nobody else is using it: memory in use under 1.5 GB at two checks 5 min apart.
# For a GPU another lab user's job is holding. Detached (src/detach.py), so it survives the SSH session ending.
#   powershell -ExecutionPolicy Bypass -File launch_when_free.ps1 -Config "configs/a.json,configs/b.json" -Gpu 0 -Shard queue
param(
  [Parameter(Mandatory = $true)][string]$Config,
  [int]$Gpu = 0,
  [string]$Shard = "queue"
)
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$launch = Join-Path $PSScriptRoot "launch.ps1"
# a failed nvidia-smi call prints no number and counts as busy
$wait = "`$n = 0; while (`$n -lt 2) { `$u = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $Gpu; " +
        "if (`$u -match '^\d+$' -and [int]`$u -lt 1500) { `$n++ } else { `$n = 0 }; if (`$n -lt 2) { Start-Sleep 300 } }"
$cmd = "$wait; & '$launch' -Config '$Config' -Gpu $Gpu -Shard '$Shard'"
$newPid = & $python "src/detach.py" "runs\free_gpu$Gpu.log" "runs\free_gpu$Gpu.err" -- powershell -NoProfile -ExecutionPolicy Bypass -Command $cmd
"waiter pid ${newPid}: starts $Config shard $Shard on GPU $Gpu once it is free (launch output -> runs\free_gpu$Gpu.log)"
