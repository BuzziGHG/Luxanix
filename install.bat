@echo off
setlocal enabledelayedexpansion
title SimRTX Studio - 1-Klick Installer (NVIDIA RTX 3080 Ti)

echo ================================================================
echo         SimRTX Studio - Installation & Umgebungseinrichtung
echo ================================================================
echo.
echo [1/4] Ueberpruefe uv Package Manager...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    ) else (
        echo [INFO] uv wird heruntergeladen und installiert...
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    )
)

echo [2/4] Initialisiere Python 3.11 Umgebung...
uv python install 3.11
if not exist ".venv" (
    uv venv .venv --python 3.11
)

echo [3/4] Installiere PyTorch mit CUDA 12.4 & RTX Beschleunigung...
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo [4/4] Installiere SimRTX Studio Bibliotheken (Gradio, OpenCV, Transformers)...
uv pip install --python .venv\Scripts\python.exe numpy opencv-python pillow gradio transformers accelerate timm imageio imageio-ffmpeg tqdm

echo.
echo ================================================================
echo      Installation erfolgreich abgeschlossen!
echo      Starte das Tool jederzeit per Doppelklick auf 'run.bat'
echo ================================================================
echo.
pause
