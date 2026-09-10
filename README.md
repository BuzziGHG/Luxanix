# ⚡ SimRTX Studio — AI Raytracing & Video Remaster Studio

[![NVIDIA RTX](https://img.shields.io/badge/Optimized%20for-NVIDIA%20RTX%203080%20Ti-76b900.svg?logo=nvidia)](https://www.nvidia.com)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://python.org)

> **Verwandle normale Simracing- und Gaming-Videoaufnahmen in photorealistische Meisterwerke mit echtem Screen-Space Raytracing (RTGI, SSR, RTAO) und kinoreifem Color-Grading.**

---

## 🏎️ Was ist SimRTX Studio?

Viele Simracing- und Gaming-Videos (z. B. aus *Assetto Corsa*, *Assetto Corsa Competizione*, *iRacing*, *F1 24*, *Forza*, *Le Mans Ultimate*) sehen in Standard-Aufnahmen oft flach aus, da ihnen erweiterte Raytracing-Shader und kinoreife Lichtstimmungen fehlen.

Normale Videobearbeitungsprogramme wie CapCut können zwar Sättigung und Kontrast anpassen, haben jedoch **keinerlei Verständnis für 3D-Geometrie, Lichteinfall oder Reflexionen**.

**SimRTX Studio löst dieses Problem:**
1. **AI-Tiefenrekonstruktion (Neural Monocular Depth):** Verwandelt 2D-Gameplay-Frames mithilfe von *Depth Anything v2* in dichte, metrische 3D-Tiefenkarten – direkt berechnet auf den Tensor Cores deiner RTX 3080 Ti.
2. **Ray Traced Global Illumination (RTGI):** Berechnet physikalisches Abprall-Licht (Indirect Bounce Light). Helle Curbs, Grasflächen, Werbebanden oder der Himmel strahlen realistisch auf den Asphalt und das Fahrzeugchassis ab.
3. **Screen-Space Reflections (SSR):** Simuliert physikalisch basierte Reflexionen mit Fresnel-Gleichungen und Rauheits-Parametern. Perfekt für nasse Rennstrecken, Pfützen, Windschutzscheiben und glänzende Autolacke.
4. **Ray Traced Ambient Occlusion (RTAO):** Präzise Kontaktschatten unter Fahrzeugen, im Radkasten und an Streckenrändern lassen Rennwagen fest mit der Fahrbahn verschmelzen.
5. **Kanten-erhaltendes Bilateral Denoising:** Beseitigt Ray-Rauschen bei maximaler Silhouetten-Schärfe.
6. **Feintuning & Color-Grading:** Präzise Regler für Sättigung, dynamische Vibrance, Kontrast, ACES Filmic Tone Mapping und Scheinwerfer-Bloom.
7. **NVIDIA NVENC Hardware-Export:** Render-Export mit maximaler Geschwindigkeit unter voller Beibehaltung aller Audiospuren (Motorsound, Reifenquietschen, Funk).

---

## 🖥️ Unterstützte Grafikkarten & Generationen

SimRTX Studio erkennt deine NVIDIA Grafikkarte automatisch und wählt das ideale Performance- und Speicherprofil:

| Generation | Architektur | Unterstützte Modelle | Optimierungs-Features |
| :--- | :--- | :--- | :--- |
| **Ältere RTX-Karten** | **Turing (RTX 20)** | RTX 2060 (6GB/12GB), 2070, 2080, 2080 Ti | **Low-VRAM Profil:** Automatische adaptive Skalierung gegen VRAM-Überlauf, periodischer CUDA Cache-Flush, Turing NVENC (`preset p4`). |
| **Aktuelle Generation** | **Ampere (RTX 30)** | RTX 3060, 3070, 3080, **RTX 3080 Ti**, 3090 | **Ultra / Balanced Profil:** Volle Ausnutzung von 12 GB+ VRAM, FP16 Tensor Cores, bis zu 4K-Rendering, NVENC (`preset p6/p7`). |
| **Neuere Generationen** | **Ada Lovelace (RTX 40)** | RTX 4060, 4070, 4080, 4090 | **Maximum Quality Profil:** 4. Gen Tensor Cores, AV1/Dual-NVENC, ultra-dichte Raymarching-Samples. |
| **Zukünftige Generationen** | **Blackwell (RTX 50)** | RTX 50-Serie (via Update) | Bereits vorbereitet für Compute 9.0+, SM 10.0 und nächste Tensor-Core Generationen. |

- **Betriebssystem:** Windows 10 / 11 (64-bit)
- **Treiber:** NVIDIA Game Ready / Studio Driver mit CUDA-Unterstützung (Driver 530+)
- **System-RAM:** Mind. 16 GB empfohlen

---

## 🚀 1-Klick Schnellstart (Windows)

1. **Repository herunterladen oder klonen:**
   ```bash
   git clone https://github.com/your-username/SimRTX-Studio.git
   cd SimRTX-Studio
   ```
2. **Installation starten:**
   - Doppelklicke auf **`install.bat`**.
   - Das Skript richtet automatisch Python 3.11, PyTorch mit CUDA 12.4 und alle benötigten Bibliotheken ein.
3. **Studio starten:**
   - Doppelklicke auf **`run.bat`**.
   - Die Weboberfläche öffnet sich automatisch in deinem Browser unter `http://127.0.0.1:7860`.

---

## 🎮 Enthaltene Presets für Simracing & Gaming

SimRTX Studio enthält maßgeschneiderte Presets für unterschiedliche Strecken- und Wetterszenarien:

| Preset | Beschreibung |
| :--- | :--- |
| 🌧️ **Simracing: Wet Track & Reflections** | Extrem hohe Screen-Space-Reflexionen auf der Fahrbahn, kühler Ton, verstärkte Kontaktschatten für Regenschlachten. |
| 🌅 **Simracing: Golden Hour Sunset** | Warme Farbtemperatur, weiches RTGI-Bounce-Licht, Sonnenuntergangs-Glow und cinematische Kontraste. |
| ☁️ **Simracing: Nürburgring Overcast** | Realistische, neutrale Farbwiedergabe, diffuse Schatten und natürliche Strecken-Details. |
| 🌃 **Simracing: Night Race & Headlights** | Knackige Tiefschwarzwerte, starker Bloom auf Scheinwerfern und Rückleuchten, Reflexionen auf Asphalt. |
| 🏆 **Assetto Corsa / ACC Hyper-Realism** | Balanciertes RTGI + RTAO Profil mit ACES-Tonemapping für maximale Authentizität. |
| ⚡ **Subtle Clean RTX Boost** | Dezente Lichtaufwertung ohne Überzeichnung – ideal für saubere Broadcast- und E-Sports-Streams. |

---

## 🛠️ Manuelle Installation (für Entwickler)

Falls du das Tool manuell in einer bestehenden Umgebung installieren möchtest:

```bash
# Virtuelle Umgebung erstellen
python -m venv .venv
.venv\Scripts\activate

# PyTorch mit CUDA 12.4 installieren
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Abhängigkeiten installieren
pip install -r requirements.txt

# Starten
python ui/app.py
```

---

## 📂 Projektstruktur

```
SimRTX-Studio/
├── engine/
│   ├── depth_estimator.py    # AI-Tiefenmodell (Depth Anything v2 / MiDaS) auf CUDA FP16
│   ├── geometry.py           # 3D-Kameraraum & Oberflächennormalen-Rekonstruktion
│   ├── raytracer.py          # Screen-Space Raymarching (RTGI, SSR, RTAO)
│   ├── denoiser.py           # Cross-Bilateral Denoising Filter
│   ├── postprocess.py        # Sättigung, Vibrance, Kontrast, Bloom, ACES Tonemap
│   └── video_pipeline.py     # Frame-Processing & NVENC-Encoder mit Audio
├── ui/
│   └── app.py                # Gradio Studio Interface mit Before/After Slider
├── presets.json              # Shader- und Farbprofile
├── install.bat               # 1-Klick Windows Installer
├── run.bat                   # 1-Klick Starter
├── requirements.txt          # Paketabhängigkeiten
├── pyproject.toml            # Pip / GitHub Projekt-Konfiguration
└── README.md                 # Dokumentation
```

---

## 📜 Lizenz

Dieses Projekt steht unter der [MIT Lizenz](LICENSE). Frei verwendbar für private und kommerzielle Videoproduktionen.
