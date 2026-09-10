@echo off
setlocal
title SimRTX Studio - Launcher
cd /d "%~dp0"

echo ================================================================
echo           SimRTX Studio — Raytracing Video Remaster
echo                  NVIDIA RTX 3080 Ti Edition
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] Die Umgebung wurde noch nicht installiert!
    echo Bitte starte zuerst die Datei 'install.bat'.
    echo.
    pause
    exit /b 1
)

echo Starte SimRTX Studio Web-Interface...
echo Die Benutzeroberflaeche oeffnet sich automatisch in deinem Standard-Browser.
echo URL: http://127.0.0.1:7860
echo Druecke Strg+C in diesem Fenster, um das Programm zu beenden.
echo.

.venv\Scripts\python.exe ui\app.py

if %errorlevel% neq 0 (
    echo.
    echo [HINWEIS] Das Programm wurde beendet.
    pause
)
