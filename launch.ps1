# Launch a sweep shard as a detached process (src/detach.py) that survives the SSH session ending.
#   powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json" -Gpu 0 -Shard 0/2
#   powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json" -Gpu 1 -Shard 1/2
# Several configs separated by commas run back to back. The log is named after the first one.
# Follow progress:   Get-Content runs\pilot_gpu0.log -Tail 20 -Wait
# Runs offline from the HF cache by default (the college proxy fails HF SSL checks, ~1 min lost per run);
# pass -Online the first time a model or corpus is used on this machine.
param(
  [string]$Config = "configs/pilot.json",
  [int]$Gpu = 0,
  [string]$Shard = "0/1",
  [switch]$Online
)
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$cfgs = @($Config -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$name = [IO.Path]::GetFileNameWithoutExtension($cfgs[0]) + "_gpu$Gpu"
New-Item -ItemType Directory -Force -Path runs | Out-Null
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
if (-not $Online) { $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1" }
# detach.py, not Start-Process: OpenSSH ends every process a session started when it closes, unless it breaks away
$argList = @("src/sweep.py") + $cfgs + @("--shard", $Shard)
$newPid = & $python "src/detach.py" "runs\$name.log" "runs\$name.err" -- $python @argList
"started pid ${newPid}: $($cfgs -join ' + ') shard $Shard on GPU $Gpu -> runs\$name.log"
