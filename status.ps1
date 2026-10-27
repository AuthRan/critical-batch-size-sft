# One-shot status on the GPU machine, from cmd or PowerShell:
#   powershell -ExecutionPolicy Bypass -File status.ps1
# A log tail can be hours old: a queue is alive only if its venv python is listed and its log was written minutes ago.
Set-Location $PSScriptRoot
"now: " + (Get-Date).ToString("MM-dd HH:mm")
# event logs and net statistics are denied to this account; query user shows who is at the console, and since when
query user 2>$null | ForEach-Object { "session: $_" }
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$PSScriptRoot*" } |
  ForEach-Object { "venv python pid $($_.Id), started $(if ($_.StartTime) { $_.StartTime.ToString('MM-dd HH:mm') } else { 'just now' })" }
foreach ($f in Get-ChildItem runs\*.jsonl) {
  "$($f.Name): $((Get-Content $f.FullName | Measure-Object -Line).Lines) rows, last write $($f.LastWriteTime.ToString('MM-dd HH:mm'))"
}
foreach ($f in Get-ChildItem runs\*_gpu*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 4) {
  $age = [int]((Get-Date) - $f.LastWriteTime).TotalMinutes
  "== $($f.Name), last write $($f.LastWriteTime.ToString('MM-dd HH:mm')) ($age min ago)"
  Select-String -Path $f.FullName -Pattern '^\[\d+/\d+\]|jobs to run|FAILED' | Select-Object -Last 3 | ForEach-Object { "   " + $_.Line }
  "   " + (Get-Content $f.FullName -Tail 1)
}
Select-String -Path runs\*_gpu*.log -Pattern 'FAILED' | Group-Object Filename | ForEach-Object { "FAILED x$($_.Count) in $($_.Name)" }
# Get-PSDrive reports 0 free for this account on the GPU machine; dir does not
if ((cmd /c dir /-c runs) -join " " -match "(\d+) bytes free") { "free on C: " + [math]::Round([double]$Matches[1] / 1GB, 1) + " GB" }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
