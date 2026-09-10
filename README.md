# ⚡ LUXANIX STUDIO — AI Raytracing & Photorealistic Video Remaster

[![Download Luxanix.exe](https://img.shields.io/badge/📥%20Download-Luxanix.exe%20(Windows%20Release)-76b900.svg?style=for-the-badge&logo=windows)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![GitHub Repository](https://img.shields.io/badge/GitHub-BuzziGHG%2FLuxanix-181717.svg?logo=github)](https://github.com/BuzziGHG/Luxanix)
[![NVIDIA RTX Acceleration](https://img.shields.io/badge/NVIDIA%20RTX-20%20%7C%2030%20%7C%2040%20%7C%2050%20Series-76b900.svg?logo=nvidia)](https://www.nvidia.com)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 👉 **[Klicke hier für das neueste Release: `Luxanix.exe` (GitHub Releases)](https://github.com/BuzziGHG/Luxanix/releases/latest)**
> 💾 **[Direkt-Download: `Luxanix.exe` (Sofort-Download)](https://github.com/BuzziGHG/Luxanix/raw/main/Luxanix.exe)**

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
9. **NVIDIA NVENC Export & 8K Ultra-Upscaling:** Blitzschneller Hardware-Export von 1080p bis **8K Ultra HD (7680x4320)** mit NVIDIA HEVC NVENC Hardware-Beschleunigung bei 100 % Audio-Erhalt.

---

## 🖥️ Unterstützte Grafikkarten & Generationen

Luxanix Studio erkennt deine Grafikkarte automatisch und wählt das passende Speicher- und Performanceprofil:

| Generation | Architektur | Unterstützte Modelle | Optimierungs-Features |
| :--- | :--- | :--- | :--- |
| **Ältere RTX-Karten** | **Turing (RTX 20)** | RTX 2060 (6GB/12GB), 2070, 2080, 2080 Ti | **Low-VRAM Profil:** Automatische adaptive Skalierung gegen VRAM-Überlauf, periodischer CUDA Cache-Flush, Turing NVENC (`preset p4`). |
| **Aktuelle Generation** | **Ampere (RTX 30)** | RTX 3060, 3070, 3080, **RTX 3080 Ti**, 3090 | **Ultra / Balanced Profil:** Volle Ausnutzung von 12 GB+ VRAM, FP16 Tensor Cores, bis zu 4K-Rendering, NVENC (`preset p6/p7`). |
| **Neuere Generationen** | **Ada Lovelace (RTX 40)** | RTX 4060, 4070, 4080, 4090 | **Maximum Quality Profil:** 4. Gen Tensor Cores, AV1/Dual-NVENC, ultra-dichte Raymarching-Samples. |
| **Zukünftige Generationen** | **Blackwell (RTX 50)** | RTX 50-Serie (via Update) | Vorbereitet für Compute 9.0+, SM 10.0 und nächste Tensor-Core Generationen. |

---

## 🚀 Schnellstart (Windows)

1. **Repository klonen:**
   ```bash
   git clone https://github.com/BuzziGHG/Luxanix.git
   cd Luxanix
   ```
2. **Installation starten:**
   - Doppelklicke auf **`install.bat`** (richtet Python 3.11, PyTorch mit CUDA 12.4 und alle Bibliotheken vollautomatisch ein).
3. **Studio starten:**
   - Doppelklicke auf **`run.bat`** (öffnet das Studio automatisch im Standard-Browser unter `http://127.0.0.1:7860`).

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
