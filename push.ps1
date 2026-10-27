# Push the project (code, configs, docs; not runs/ or .venv) to the GPU machine.
# Usage, from any PowerShell window on the laptop:
#   powershell -ExecutionPolicy Bypass -File "D:\Research paper\Paper 2\push.ps1"
# It asks for the account password once. Existing files on the remote are overwritten;
# remote-only files (runs\*.jsonl, .venv) are left alone.
Set-Location $PSScriptRoot
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force
scp -r configs src tests launch.ps1 launch_after.ps1 status.ps1 setup_remote.cmd PLAN.md README.md requirements.txt <user>@<gpu-host>:G:/paper2_sft_cbs/
if ($LASTEXITCODE -eq 0) { "pushed to <user>@<gpu-host>:G:/paper2_sft_cbs/" } else { "push failed (exit $LASTEXITCODE)" }
