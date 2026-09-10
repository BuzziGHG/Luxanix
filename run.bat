@echo off
setlocal
title Luxanix Studio — RTX Raytracing & Video Remaster
cd /d "%~dp0"

echo ================================================================
echo           LUXANIX STUDIO — RAYTRACING VIDEO REMASTER
echo              AI Photorealism & Multi-GPU Acceleration
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] Die Umgebung wurde noch nicht installiert!
    echo Bitte starte zuerst die Datei 'install.bat'.
    echo.
    pause
    exit /b 1
)

echo Starte native Luxanix Studio Desktop-App...
echo.

.venv\Scripts\python.exe ui\desktop_app.py

if %errorlevel% neq 0 (
    echo.
    echo [HINWEIS] Das Programm wurde beendet.
    pause
)
