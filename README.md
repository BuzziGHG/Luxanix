# ⚡ LUXANIX STUDIO — v1.1.0

[![Aktuelle Version](https://img.shields.io/badge/Version-v1.1.0%20(Blackwell%20Update)-76b900.svg?style=for-the-badge&logo=github)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![Download Luxanix.exe](https://img.shields.io/badge/📥%20Download-Luxanix.exe%20(v1.1.0)-10b981.svg?style=for-the-badge&logo=windows)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![NVIDIA RTX Acceleration](https://img.shields.io/badge/NVIDIA%20RTX-50%20%7C%2040%20%7C%2030%20%7C%2020%20Series-76b900.svg?logo=nvidia)](https://www.nvidia.com)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Verwandle normale Gaming- und Simracing-Videoaufnahmen in photorealistische Meisterwerke mit echtem Screen-Space Raytracing (RTGI, SSR, RTAO), KI-Tiefenrekonstruktion, Auto-Adaptivem AI-Preset, nativer Desktop-App und 8K Ultra HD Upscaling.**
> 
> 📥 **[Hier klicken: Neueste Version `Luxanix.exe (v1.1.0)` herunterladen](https://github.com/BuzziGHG/Luxanix/releases/latest)**

---

## 🏎️ Was ist Luxanix Studio?

Gameplay-Aufnahmen aus Rennsimulationen (*Assetto Corsa*, *ACC*, *iRacing*, *F1 24*, *Forza*, *Le Mans Ultimate*) oder Action-Games wirken auf Videoaufnahmen oft flach, verwaschen durch TAA-Bewegungsunschärfe und missen die physikalische Lichtstimmung echter Rennübertragungen.

Normale Schnittprogramme wie CapCut bieten lediglich einfache Farbfilter, besitzen aber **keinerlei Verständnis für 3D-Räume, Oberflächenkrümmungen oder Lichteinfall**.

**Luxanix Studio bietet eine vollwertige Suite zur Videoveredelung:**
1. **Video-Metadaten & Maße:** Sofortige Erkennung von Videoauflösung (1080p, 1440p, 4K, 8K), Seitenverhältnis (16:9, 21:9 Ultrawide, 32:9 Triple-Screen), Bildrate (FPS) und Gesamtlaufzeit.
2. **Integrierter Video-Cutter (Timeline-Trimming):** Bestimme Start- und Endzeitpunkt auf die Zehntelsekunde genau, um gezielt Renn-Highlights, Drifts oder Überholmanöver zu rendern, ohne stundenlanges Rohmaterial bearbeiten zu müssen.
3. **AI-Tiefenrekonstruktion (Neural Monocular Depth):** Verwandelt 2D-Gameplay-Frames mithilfe von *Depth Anything v2* in dichte metrische 3D-Tiefenkarten – direkt auf den Tensor Cores berechnet.
4. **Ray Traced Global Illumination (RTGI):** Berechnet physikalisches Abprall-Licht. Farbige Curbs, Leitplanken oder der Himmel strahlen realistisch auf den Asphalt und die Karosserie ab.
5. **Screen-Space Reflections (SSR):** Echte Spiegelungen mit Fresnel-Gleichungen und Rauheits-Parametern für nasse Strecken, Regenpfützen und Fahrzeuglacke.
6. **Ray Traced Ambient Occlusion (RTAO):** Physikalische Kontaktschatten unter dem Fahrzeugchassis und in den Radkästen lassen Autos fest mit der Strecke verschmelzen.
7. **Detail-Clarity & Texturschärfung (Anti-TAA):** Beseitigt die typische Bewegungsunschärfe moderner Spiele und holt feine Strecken- und Karbondetails hervor.
8. **Live Render-Monitor mit ETA & Fortschrittsbalken:** Präzise Anzeige von berechneten Frames, aktuellem Prozentwert, Render-FPS und minutengenauer Restzeit-Berechnung.
9. **NVIDIA NVENC Export & 8K Ultra-Upscaling:** Blitzschneller Hardware-Export von 1080p bis **8K Ultra HD (7680x4320)** mit NVIDIA AV1 & HEVC NVENC Hardware-Beschleunigung bei 100 % Audio-Erhalt.

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

## 🚀 Schnellstart (Windows)

1. Lade dir die neueste **[`Luxanix.exe`](https://github.com/BuzziGHG/Luxanix/releases/latest)** herunter.
2. Mache einen **Doppelklick auf `Luxanix.exe`**:
   - **Vollautomatische Festinstallation:** Das Tool installiert sich beim ersten Klick automatisch dauerhaft auf deinem PC (`AppData\Local\Programs\Luxanix`).
   - **Desktop- & Startmenü-Verknüpfung:** Es wird automatisch ein Desktop-Icon sowie ein Startmenü-Eintrag erstellt.
   - Du musst nie wieder nach der heruntergeladenen Datei im Download-Ordner suchen – drücke einfach die **Windows-Taste**, tippe **`Luxanix`** und drücke Enter!
3. Das Programm startet direkt als eigenständige, native Windows Desktop-Applikation mit NVIDIA RTX Beschleunigung.
4. Video auswählen, Auto-Preset aktivieren und die Remaster-Vorschau direkt im integrierten Player ansehen!

---

## 📺 8K Ultra-Upscaling & AI Super-Resolution

Luxanix Studio bietet integriertes High-End Upscaling bis zu **8K Ultra HD (7680x4320 / 4320p)**:
- **Lanczos4 & Detail-Clarity Super-Resolution:** Schärft feine Streckentexturen, Curbs, Sponsoren-Aufkleber und Cockpit-Displays kristallklar nach.
- **Hardware-beschleunigter HEVC / H.265 NVENC Export:** Volle Ausnutzung der NVIDIA-Hardware-Encoder bis zu 8192x8192 bei variabler Bitrate bis 160 Mbps.
- **Volle Ultrawide- & Triple-Screen Unterstützung:**
  - Standard 16:9: Bis zu 7680 x 4320 (8K UHD)
  - Ultrawide 21:9: Bis zu 8192 x 3480
  - Super-Ultrawide 32:9: Bis zu 8192 x 2304

---

## 📜 Lizenz

Dieses Projekt steht unter der [MIT Lizenz](LICENSE). Frei verwendbar für private und kommerzielle Videoproduktionen.
