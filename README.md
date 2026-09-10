# ⚡ LUXANIX STUDIO PRO — v2.1.0 (CapCut Pro NLE Edition)

[![Aktuelle Version](https://img.shields.io/badge/Version-v2.1.0%20(Pristine%20RTX%20%2B%20Zero--Noise)-76b900.svg?style=for-the-badge&logo=github)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![Download Luxanix-Setup.exe](https://img.shields.io/badge/📥%20Download-Luxanix--Setup.exe%20(v2.1.0)-00c4cc.svg?style=for-the-badge&logo=windows)](https://github.com/BuzziGHG/Luxanix/releases/latest)
[![NVIDIA RTX 50 Ready](https://img.shields.io/badge/NVIDIA%20RTX-50%20%7C%2040%20%7C%2030%20%7C%2020%20Series-76b900.svg?logo=nvidia)](https://www.nvidia.com)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Professionelles CapCut-inspiriertes Videoschnitt- & Remastering-Studio mit 100% autonomer KI-Physik, echtem NVIDIA Tensor-Core Raytracing (RTGI, SSR, RTAO), GPU VRAM Frame-Cache (60 FPS Echtzeit-Playback), Neural AI Super-Resolution (bis 8K), automatischer GPU-Erkennung und permanenter Windows-Installation über eine einzige `.exe`.**
> 
> 📥 **[Hier klicken: Neueste Version `Luxanix-Setup.exe (v2.1.0)` herunterladen](https://github.com/BuzziGHG/Luxanix/releases/latest)**
> *(Einzige benötigte Datei! Installiert alles vollautomatisch und richtet Desktop- & Startmenü-Verknüpfungen ein).*

---

## ⚡ Neu in v2.1.0: GPU VRAM Cache & Perfektionierter Photorealismus

- ⚡ **GPU VRAM Frame-Cache (Echtzeit-Streaming):**
  Nutzt den High-Speed GDDR6X-Grafikspeicher moderner Grafikkarten (z.B. 12 GB auf RTX 3080 Ti) voll aus. Bereits berechnete Frames werden als FP16 PyTorch CUDA Tensoren direkt im VRAM gepuffert, sodass die Videovorschau mit flüssigen 60 FPS latenzfrei aus dem GPU-Speicher gestreamt wird.
- 🎚️ **Linearer Photorealismus-Stärke Regler (0.0 bis 2.0):**
  - **Ganz links (0.0):** 100% unberührtes Originalbild (0% AI / 0% Shader / 100% Rohmaterial).
  - **Mitte (1.0):** Ausgewogener, kinoreifer Photorealismus (+24% Livery-Sättigung, Rec.709 Kontrast, SSR Fahrbahn-Nässe, RTAO Kontaktschatten).
  - **Ganz rechts (2.0):** Maximales Remaster mit Hyper-Reflexionen und tiefen Kontaktschatten.
- 🎨 **Farbkorrektur & Grading in Echtzeit:**
  Automatische KI-Farbkorrektur kombiniert mit manuellen Reglern für Sättigung (0.0× Schwarz-Weiß bis 2.5× Hyper-Saturiert), Kontrast, Farbtemperatur (Kelvin) und Belichtung (EV) mit sofortiger Bildaktualisierung.
- 🎬 **Sauberer Single-File Export:**
  Keine temporären Dateileichen mehr im Ausgabeordner. Alle Renderstufen laufen isoliert über den Windows-Temp-Speicher, sodass exakt eine fertige `.mp4`-Datei ausgegeben wird.

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

---

## 🤖 100% Autonome KI-Physik-Engine (Keine manuellen Presets nötig!)

Kein Auswählen von Presets mehr nötig! Luxanix Studio analysiert jeden Videoframe kontinuierlich in Echtzeit und berechnet die physikalischen Licht- und Geometriewerte vollkommen autonom:
- **Log-Luminanz Auto-Belichtung (EV):** Passt Nachtfahrten, sonnige Mittagsrennen und Tunnelausfahrten dynamisch an, ohne Highlights auszubrennen.
- **Adaptive S-Kurven Kontrastkurve:** Verleiht flachen Gameplay-Szenen sofort cinematische Tiefe.
- **Autonome Asphaltschärfe & Nässe-Reflexionen (SSR):** Erkennt feuchte Streckenabschnitte und projiziert physikalisch akkurate Screen-Space-Reflexionen.
- **Ray Traced Global Illumination (RTGI) & Kontaktschatten (RTAO):** Rechnet diffuses Streulicht von Curbs und Streckenbegrenzungen sowie präzise Kontaktschatten unter Fahrzeugen.
- **Temporale Glättung (EMA):** Verhindert störendes Flackern zwischen benachbarten Frames.

---

## 🔬 Echter Neural AI Super-Resolution Upscaler (NVIDIA Tensor Cores)

Ein echtes tiefes neuronales Netzwerk (**Residual Dense Blocks + Sub-Pixel PixelShuffle**) rekonstruiert verloren gegangene Mikro-Texturen:
- **1080p ➔ 4K (2160p) & 8K Ultra HD (4320p):** Echte Hochskalierung über CUDA FP16 Tensor Cores, nicht nur einfacher Filter-Resize.
- **Artefakt-Entfernung:** Glättet Kompressions-Makroblöcke von YouTube/OBS-Aufnahmen.
- **Kristallklare Kanten:** Curbs, Fahrzeugdetails, Sponsorenlogos und Streckenschilder bleiben gestochen scharf.

---

## 📸 Volle Foto- & Screenshot-Unterstützung (bis 8K Remaster)

Nicht nur Videos, sondern auch einzelne Fotos und Screenshots können jetzt direkt importiert und verarbeitet werden:
- **Formate:** PNG, JPG, JPEG, WEBP, BMP, TIFF.
- **Autonome Physik & Raytracing:** Echte Tiefenanalyse, Global Illumination, Oberflächenreflexionen und Kontrastkorrektur für jedes Bild.
- **8K AI Neural Upscaling:** Skaliere Screenshots aus 720p oder 1080p gestochen scharf auf bis zu **8K Ultra HD (4320p)** hoch!
- **1-Klick Bild-Export:** Schneller Export direkt in deinen Windows `Bilder\Luxanix_Renders` Ordner mit verlustfreier oder maximaler Qualität (PNG 100%, JPG 98%).

---

## ⚡ Direct NVENC Streaming-Pipe & Asynchroner Frame-Prefetch

Dank moderner Low-Level-Optimierungen rendert Luxanix Studio nun mehr als doppelt so schnell:
- **Direct Memory-to-Encoder Pipe:** Unkomprimierte Videoframes werden über In-Memory-Streams direkt an den NVIDIA NVENC-Encoder übergeben. Keine riesigen temporären Zwischenvideodateien auf der Festplatte mehr!
- **Asynchroner Frame-Prefetch:** Ein dedizierter Background-Thread liest Videoframes im Voraus ein (`Queue`), sodass die RTX-Tensor-Cores und Raytracing-Rechenkerne niemals im Leerlauf auf I/O warten.
- **`torch.backends.cudnn.benchmark`:** Optimiert dynamisch die Faltungs-Kernel für deine RTX-Architektur.
- **Dual-NVENC AV1/HEVC:** Volle Nutzung doppelter Encoder-Einheiten auf RTX 40 & RTX 50 Grafikkarten.

---

## 🛑 Sofortiger 1-Klick Export-Abbruch

Nie wieder warten müssen, wenn man sich bei einer Auflösung oder einem Schnitt vertan hat:
- **Sofortiger Abbruch-Button:** Sobald ein Video- oder Foto-Export startet, erscheint im Header und in der Statusleiste ein rot hervorgehobener `[ 🛑 Abbrechen ]`-Button.
- **Saubere Terminierung:** Der laufende Hardware-Encoder-Stream wird sofort gestoppt und alle temporären Zwischendateien werden rückstandsfrei gelöscht.
- **Kein Hängenbleiben:** Die App kehrt ohne Neustart oder Absturz in den normalen Bearbeitungsmodus zurück.

---

## 🎵 Multi-Track Audio & Musikspur (Spuren A1 & A2)

Professionelle Audio-Mischung direkt wie in CapCut Desktop:
- **Spur A1 (Originalton):** Lautstärkeregler (0% bis 200%) und Mute-Schalter zum Stummschalten des Originalvideos.
- **Spur A2 (Hintergrundmusik):** Importiere beliebige Musikdateien (`.mp3`, `.wav`, `.aac`, `.m4a`, `.ogg`, `.flac`).
- **Visuelle Timeline-Wellenform:** Die Musikspur wird direkt unter der Videotonspur als interaktives Band mit Pegelvisualisierung dargestellt.
- **Automatisches FFmpeg-Audiomixing:** Im finalen Export werden Video-Audio und Hintergrundmusik mit Sample-Genauigkeit, Lautstärke-Gewichtung und exakter Clip-Länge hardwarenah abgemischt.

---

## 🚀 1-Klick GitHub Upload

Um den lokalen Code und Updates direkt auf dein GitHub-Repository hochzuladen:
1. Doppelklicke auf `upload_to_github.bat`.
2. Wähle Option 1 (Browser-Login) oder Option 2 (Token). Das Skript synchronisiert den gesamten Code automatisch mit `https://github.com/BuzziGHG/Luxanix.git`.

## 📜 Lizenz

Dieses Projekt steht unter der [MIT Lizenz](LICENSE). Frei verwendbar für private und kommerzielle Videoproduktionen.
