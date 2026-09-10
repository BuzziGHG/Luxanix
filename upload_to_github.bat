@echo off
setlocal enabledelayedexpansion
title Luxanix — 1-Klick GitHub Upload

echo ================================================================
echo               LUXANIX STUDIO — GITHUB UPLOAD TOOL
echo            Repository: https://github.com/BuzziGHG/Luxanix.git
echo ================================================================
echo.

cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd;%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"

echo [1/3] Bereite Git Repository vor...
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin https://github.com/BuzziGHG/Luxanix.git

git add .
git commit -m "Luxanix Studio: AI Raytracing, Video Cutter, Detail Clarity & Multi-GPU Remaster" >nul 2>&1

echo.
echo [2/3] Ueberpruefe GitHub Autorisierung...
echo Du hast zwei Moeglichkeiten:
echo.
echo   [1] Per Browser anmelden (Empfohlen - oeffnet GitHub Login)
echo   [2] Personal Access Token (PAT) eingeben
echo.
set /p choice="Waehle Option (1 oder 2, Standard: 1): "

if "%choice%"=="2" (
    echo.
    set /p token="Fuege dein GitHub Personal Access Token (mit 'repo'-Rechten) ein: "
    if "!token!"=="" (
        echo [FEHLER] Kein Token eingegeben!
        pause
        exit /b 1
    )
    echo Lade Code hoch zu https://github.com/BuzziGHG/Luxanix.git...
    git push https://!token!@github.com/BuzziGHG/Luxanix.git main --force
) else (
    echo.
    echo Starte GitHub Web-Login... Ein Browserfenster oeffnet sich gleich.
    gh auth login -h github.com -p https -w
    echo.
    echo Lade Code hoch zu https://github.com/BuzziGHG/Luxanix.git...
    git push -u origin main --force
)

if %errorlevel% equ 0 (
    echo.
    echo ================================================================
    echo      ERFOLG! Alle Dateien wurden erfolgreich hochgeladen!
    echo      Repository: https://github.com/BuzziGHG/Luxanix
    echo ================================================================
) else (
    echo.
    echo [HINWEIS] Falls der Upload fehlgeschlagen ist, pruefe bitte deine
    echo Berechtigungen fuer das Repository 'BuzziGHG/Luxanix'.
)

echo.
pause
