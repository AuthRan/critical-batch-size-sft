# Keep one GPU's sweep going for days with nobody logged in: each time its sweep exits (its lists ran out, or it
# crashed), wait -Every seconds and launch it again. Finished runs are skipped, so a relaunch with nothing to do only
# re-reads the lists, and runs that src/autoplan.py queued meanwhile get picked up. With -FreeConfig, a relaunch uses
# that list instead whenever the GPU has under 1.5 GB in use (nobody else on it). Detached (src/detach.py).
# -After <pid>: the sweep already running on this GPU; the loop starts when it exits.
#   powershell -ExecutionPolicy Bypass -File keepalive.ps1 -Config configs/queue.txt -Gpu 1 -After 8616
param(
  [Parameter(Mandatory = $true)][string]$Config,
  [string]$FreeConfig = "",
  [int]$Gpu = 0,
  [int]$After = 0,
  [int]$Every = 1200
)
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$launch = Join-Path $PSScriptRoot "launch.ps1"
# no double quotes inside: the command line passes through subprocess quoting into a new powershell
$pick = "`$c = '$Config'"
if ($FreeConfig) {
  $pick = "`$u = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $Gpu; " +
          "`$c = if (`$u -match '^\d+$' -and [int]`$u -lt 1500) { '$FreeConfig' } else { '$Config' }"
}
$cmd = "if ($After) { Wait-Process -Id $After -ErrorAction SilentlyContinue }; while (`$true) { Start-Sleep $Every; $pick; " +
       "`$out = & '$launch' -Config `$c -Gpu $Gpu -Shard queue; (Get-Date -Format 'MM-dd HH:mm') + ' ' + `$out; " +
       "if (`$out -match 'started pid (\d+)') { Wait-Process -Id ([int]`$Matches[1]) -ErrorAction SilentlyContinue } }"
$newPid = & $python "src/detach.py" "runs\keepalive_gpu$Gpu.log" "runs\keepalive_gpu$Gpu.err" -- powershell -NoProfile -ExecutionPolicy Bypass -Command $cmd
"keepalive pid ${newPid}: relaunches $Config on GPU $Gpu whenever its sweep exits (log runs\keepalive_gpu$Gpu.log)"
