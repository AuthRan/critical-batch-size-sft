@echo off
REM One-time setup on the GPU machine (Windows). Run from inside the project folder:
REM   cd /d G:\paper2_sft_cbs && setup_remote.cmd
REM Uses the system torch (already CUDA-enabled) via --system-site-packages; installs the rest into a venv.
python -m venv --system-site-packages .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q "transformers>=4.46" "datasets>=2.20" "accelerate>=0.30" "numpy>=1.26" "matplotlib>=3.8" "pytest>=8" || exit /b 1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"
echo === tests ===
python -m pytest tests -q || exit /b 1
echo === smoke run (downloads SmolLM2-135M + dolly-15k, 30 steps) ===
set CUDA_VISIBLE_DEVICES=0
python src\sweep.py configs\smoke.json
echo === done. Next: powershell -ExecutionPolicy Bypass -File launch.ps1 -Config configs/pilot.json -Gpu 0 -Shard 0/2  (and -Gpu 1 -Shard 1/2) ===
