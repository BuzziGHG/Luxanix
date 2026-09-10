# ⚡ LUXANIX STUDIO PRO — v1.1.0

[![Aktuelle Version](https://img.shields.io/badge/Version-v1.1.0%20(CapCut%20%2B%20Blackwell%20Setup)-76b900.svg?style=for-the-badge&logo=github)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![Download Luxanix-Setup.exe](https://img.shields.io/badge/📥%20Download-Luxanix--Setup.exe%20(v1.1.0)-00c4cc.svg?style=for-the-badge&logo=windows)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![NVIDIA RTX 50 Ready](https://img.shields.io/badge/NVIDIA%20RTX-50%20%7C%2040%20%7C%2030%20%7C%2020%20Series-76b900.svg?logo=nvidia)](https://www.nvidia.com)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Professionelles CapCut-inspiriertes Videoschnitt- & Remastering-Studio mit echtem NVIDIA Tensor-Core Raytracing (RTGI, SSR, RTAO), Auto-Adaptivem AI-Preset, nativer Desktop-App, 8K Ultra HD Upscaling und dauerhafter Windows-Installation via Setup-Assistent.**
> 
> 📥 **[Hier klicken: Neueste Version `Luxanix-Setup.exe (v1.1.0)` herunterladen](https://github.com/BuzziGHG/Luxanix/releases/latest)**

---

## 🎨 CapCut Pro Desktop NLE Benutzeroberfläche

Luxanix Studio besitzt eine moderne 4-Zonen-Desktop-Oberfläche im vertrauten Stil professioneller Schnittsoftware (wie CapCut Desktop Pro):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ LUXANIX PRO [Menü] [Datei]   🟢 Auto-Save: Aktiv   [Projekt-Titel]          [ 🚀 Export ] │ Top Header
├──────────────────────────┬──────────────────────────────────────────┬───────────────────────┤
│ 📁 MEDIEN & PRESETS      │ 🎬 PLAYER & VIEWPORT                     │ ⚙️ INSPEKTOR          │
│ • [📁 Medien] [✨ Shader] │ • [16:9 ▾] [◫ Split 50/50] [⚡ Remaster]  │ • Tab 1: Raytracing   │ 3-Spalten
│ • [🤖 Auto-AI] [🎨 LUTs] │ • Responsiver Monitor mit Live-Vorschau │ • Tab 2: Hardware RTX │ Arbeitsbereich
│ • Filter: RTX 50, Regen, │ • Digitaler Timecode (00:00:02:15)       │ • Tab 3: Belichtung   │
│   Sunset, Nürburgring    │ • Transport: ⏮ ◀ [ ▶ Play ] ▶ ⏭          │ • Tab 4: 8K Export    │
│ • Visuelles Card-Grid    │ • [ ⚡ Vorschau rendern ]                │                       │
├──────────────────────────┴──────────────────────────────────────────┴───────────────────────┤
│ 🎬 MULTI-TRACK TIMELINE                                                                     │
│ • Werkzeuge: [ ✂️ Teilen (Strg+B) ]  [ ⏮ Start trimmen ]  [ ⏭ Ende trimmen ]  [ 🗑️ Löschen ]  │ Schnitt-
│ • Timecode-Lineal mit vertikaler Playhead-Nadel zum interaktiven Frame-Scrubbing            │ Timeline
│ • Spur V1: Videoclip mit Thumbnail-Strip-Vorschau & Schnittmarken                           │
│ • Spur FX: RTX 50 Hyper-Path Tracing & Auto-Scene Dynamic Shader                            │
│ • Spur A1: Audio-Wellenform-Visualisierung                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ ══════════════════════════════════ 100% ═══════════════════════════════════════════════════ │ Statusleiste
│ Bereit. NVIDIA RTX 50 Ready | FP16 Tensor Cores                                             │ & Render-ETA
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Vollwertiger Windows Setup-Assistent (`Luxanix-Setup.exe`)

Nie wieder jedes Mal eine temporäre `.exe` im Download-Ordner suchen!

1. Lade dir die Datei **[`Luxanix-Setup.exe`](https://github.com/BuzziGHG/Luxanix/releases/latest)** herunter.
2. Führe `Luxanix-Setup.exe` einmalig aus:
   - **4-Schritte Windows Setup-Wizard:**
     - **Schritt 1:** Willkommen & Feature-Übersicht
     - **Schritt 2:** Zielordner (`%LOCALAPPDATA%\Programs\Luxanix Studio`) und Verknüpfungsoptionen (Desktop, Startmenü, Windows Apps)
     - **Schritt 3:** Installation mit animiertem Fortschrittsbalken & detaillierter Schrittanzeige
     - **Schritt 4:** "Installation abgeschlossen!" mit Option `[X] Luxanix Studio jetzt starten`
3. **Fertig installiert!** Du kannst `Luxanix-Setup.exe` jetzt löschen. Luxanix Studio ist dauerhaft auf deinem PC eingerichtet und startet ab sofort bequem:
   - Über das **Windows-Startmenü** (einfach "Luxanix" eintippen)
   - Über die **Desktop-Verknüpfung** `Luxanix Studio`
   - In unter 1 Sekunde direkt im schnellen App-Modus ohne Wizard!

---

## 🖥️ Unterstützte Grafikkarten & Generationen

Luxanix Studio erkennt deine Grafikkarte automatisch und wählt das passende Speicher- und Performanceprofil:

| Generation | Architektur | Unterstützte Modelle | Optimierungs-Features |
| :--- | :--- | :--- | :--- |
| **RTX 50-Serie (Neu!)** | **Blackwell** | **RTX 5090 (32GB GDDR7)**, **5080 (16GB)**, **5070 / 5070 Ti**, **5060** | **Hyper-Path Tracing Profil:** 18 RTGI Bounces, 24 SSR Steps, 12 RTAO Samples, native 8K Tensor-Berechnung (4320p), 5. Gen Tensor Cores (BF16/FP8), AV1 Dual-NVENC Hardware-Encoding. |
| **RTX 40-Serie** | **Ada Lovelace** | RTX 4060, 4070, 4080, 4090 | **Ultra Quality Profil:** 4. Gen Tensor Cores, AV1/Dual-NVENC Support, 12 RTGI / 16 SSR / 8 RTAO. |
| **RTX 30-Serie** | **Ampere** | RTX 3060, 3070, 3080, **RTX 3080 Ti**, 3090 | **Balanced / Ultra Profil:** Volle Ausnutzung von 12 GB+ VRAM, FP16/BF16 Tensor Cores, bis zu 4K/8K-Rendering, HEVC NVENC (`preset p6/p7`). |
| **RTX 20-Serie** | **Turing** | RTX 2060 (6GB/12GB), 2070, 2080, 2080 Ti | **Low-VRAM Profil:** Automatische adaptive Skalierung gegen VRAM-Überlauf, periodischer CUDA Cache-Flush, Turing NVENC (`preset p4`). |

---

## 🎮 Enthaltene Presets für Simracing & Gaming

Preset | Beschreibung
:--- | :---
🔥 **RTX 50 Blackwell: Hyper-Path Tracing (8K Ultra)** | 18 RTGI Bounces, 24 SSR Steps, 12 RTAO Samples, AV1 Dual-NVENC Export.
⚡ **RTX 50 Blackwell: Nürburgring 24h Photorealism** | Diffuses Nordschleifen-Licht, nasse Eifel-Kurven, mikrofeine Asphaltschärfe.
🌧️ **Simracing: Wet Track & Reflections** | Extreme Screen-Space-Reflexionen auf der Fahrbahn, kühler Ton, verstärkte Kontaktschatten für Regenschlachten.
🌅 **Simracing: Golden Hour Sunset** | Warme Farbtemperatur, weiches RTGI-Bounce-Licht, Sonnenuntergangs-Glow und cinematische Kontraste.
☁️ **Simracing: Nürburgring Overcast** | Realistische, neutrale Farbwiedergabe, diffuse Schatten und natürliche Strecken-Details.
🌃 **Simracing: Night Race & Headlights** | Knackige Tiefschwarzwerte, starker Bloom auf Scheinwerfern und Rückleuchten, Reflexionen auf Asphalt.
🏆 **Assetto Corsa / ACC Hyper-Realism** | Balanciertes RTGI + SSR, subtile Texturschärfung für Cockpit- und Verfolger-Perspektiven.

---

## 📺 8K Ultra-Upscaling & AI Super-Resolution

Luxanix Studio bietet integriertes High-End Upscaling bis zu **8K Ultra HD (7680x4320 / 4320p)**:
- **Lanczos4 & Detail-Clarity Super-Resolution:** Schärft feine Streckentexturen, Curbs, Sponsoren-Aufkleber und Cockpit-Displays kristallklar nach.
- **Hardware-beschleunigter AV1 & HEVC NVENC Export:** Volle Ausnutzung der NVIDIA-Hardware-Encoder bis zu 8192x8192 bei variabler Bitrate bis 160 Mbps.
- **Volle Ultrawide- & Triple-Screen Unterstützung:**
  - Standard 16:9: Bis zu 7680 x 4320 (8K UHD)
  - Ultrawide 21:9: Bis zu 8192 x 3480
  - Super-Ultrawide 32:9: Bis zu 8192 x 2304

---

## 📜 Lizenz

Dieses Projekt steht unter der [MIT Lizenz](LICENSE). Frei verwendbar für private und kommerzielle Videoproduktionen.
