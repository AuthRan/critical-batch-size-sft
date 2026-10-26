# One-shot status on the GPU machine, from cmd or PowerShell:
#   powershell -ExecutionPolicy Bypass -File status.ps1
# A log tail can be hours old: a queue is alive only if its venv python is listed and its log was written minutes ago.
Set-Location $PSScriptRoot
"now: " + (Get-Date).ToString("MM-dd HH:mm")
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*paper2_sft_cbs*" } |
  ForEach-Object { "venv python pid $($_.Id), started $($_.StartTime.ToString('MM-dd HH:mm'))" }
foreach ($f in Get-ChildItem runs\*.jsonl) {
  "$($f.Name): $((Get-Content $f.FullName | Measure-Object -Line).Lines) rows, last write $($f.LastWriteTime.ToString('MM-dd HH:mm'))"
}
foreach ($f in Get-ChildItem runs\*_gpu*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 4) {
  $age = [int]((Get-Date) - $f.LastWriteTime).TotalMinutes
  "== $($f.Name), last write $($f.LastWriteTime.ToString('MM-dd HH:mm')) ($age min ago)"
  Select-String -Path $f.FullName -Pattern '^\[\d+/\d+\]|jobs to run|FAILED' | Select-Object -Last 3 | ForEach-Object { "   " + $_.Line }
  "   " + (Get-Content $f.FullName -Tail 1)
}
"FAILED lines in all logs: " + @(Select-String -Path runs\*_gpu*.log -Pattern 'FAILED').Count
"free on C: " + [math]::Round((Get-PSDrive C).Free / 1GB, 1) + " GB"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
