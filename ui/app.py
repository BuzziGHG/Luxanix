"""
Luxanix Studio — AI Raytracing & Photorealistic Video Remaster
Sleek, dark, GPU-accelerated video enhancement tool for Gaming & Simracing.
Features:
- Video metadata inspection (resolution, aspect ratio, fps, duration)
- Video timeline trimming & cutting (render only the best racing moments)
- Screen-Space Ray Tracing (RTGI, SSR, RTAO)
- AI Photorealism & Detail Clarity (Anti-TAA blur, ACES Tonemapping, Bloom, Film Grain)
- Realtime Render Monitor with Frame Count, Progress Bar, FPS, and ETA countdown
- Multi-Generation GPU support (RTX 20, 30, 40, and 50-Series)
"""

import sys
import os
import json
import math
import time

# Ensure project root is in Python module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional

from engine.depth_estimator import DepthEstimator
from engine.video_pipeline import VideoPipeline, get_video_info
from engine.hardware import detect_gpu_hardware, get_profile_settings, HARDWARE_PROFILES

# Load Presets
PRESETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets.json")
if os.path.exists(PRESETS_PATH):
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        PRESETS = json.load(f)
else:
    PRESETS = {}

# Global Hardware Profile & Engine Singleton
GPU_INFO = detect_gpu_hardware()
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = VideoPipeline()
    return PIPELINE


def get_gpu_badge_html():
    if GPU_INFO.vram_gb > 0:
        return (
            f"🟢 <b>GPU:</b> {GPU_INFO.device_name} | "
            f"<b>Architektur:</b> {GPU_INFO.generation} | "
            f"<b>VRAM:</b> {GPU_INFO.vram_gb:.1f} GB | "
            f"<b>Profil:</b> {GPU_INFO.recommended_profile.upper()}"
        )
    return "⚠️ Warnung: Keine CUDA GPU gefunden. CPU-Software-Modus aktiv."


def on_video_upload(video_path: Optional[str]):
    """Reads video file metadata and configures the trimming sliders and stats badge."""
    if not video_path or not os.path.exists(video_path):
        return (
            "<div class='stats-card'>Kein Video geladen. Lade ein Gaming- oder Simracing-Video hoch.</div>",
            gr.update(maximum=60.0, value=0.0),
            gr.update(maximum=60.0, value=60.0),
            gr.update(maximum=60.0, value=2.0),
            "00:00 bis 00:00 (0.0s)"
        )

    info = get_video_info(video_path)
    dur = info.get("duration_sec", 60.0)

    html_stats = f"""
    <div class='stats-card'>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;'>
            <div><span class='stat-label'>Auflösung</span><br><b>{info.get('resolution_label', 'Unbekannt')}</b></div>
            <div><span class='stat-label'>Format</span><br><b>{info.get('aspect_ratio', '16:9')}</b></div>
            <div><span class='stat-label'>Framerate</span><br><b>{info.get('fps', 30.0)} FPS</b></div>
            <div><span class='stat-label'>Dauer</span><br><b>{info.get('duration_str', '00:00')}</b></div>
            <div><span class='stat-label'>Frames</span><br><b>{info.get('total_frames', 0):,} Frames</b></div>
        </div>
    </div>
    """

    preview_default = min(2.0, dur)
    trim_summary = f"Volle Videolänge: 00:00 bis {info.get('duration_str', '00:00')}"

    return (
        html_stats,
        gr.update(maximum=dur, value=0.0),
        gr.update(maximum=dur, value=dur),
        gr.update(maximum=dur, value=preview_default),
        trim_summary
    )


def update_trim_label(start_sec: float, end_sec: float, video_path: Optional[str]):
    """Computes and updates the cut duration text."""
    if end_sec <= start_sec:
        return "⚠️ Fehler: Endzeitpunkt muss nach dem Startzeitpunkt liegen!"

    diff_sec = end_sec - start_sec
    m_start, s_start = int(start_sec // 60), int(start_sec % 60)
    m_end, s_end = int(end_sec // 60), int(end_sec % 60)
    m_diff, s_diff = int(diff_sec // 60), int(diff_sec % 60)

    fps = 30.0
    if video_path and os.path.exists(video_path):
        info = get_video_info(video_path)
        fps = info.get("fps", 30.0)

    frames = int(diff_sec * fps)
    return f"✂️ Schnittbereich: {m_start:02d}:{s_start:02d} bis {m_end:02d}:{s_end:02d} | Rendern: {m_diff:02d}:{s_diff:02d} Min ({diff_sec:.1f}s / ~{frames:,} Frames)"


def extract_frame_at_time(video_path: str, timestamp_sec: float) -> np.ndarray:
    """Extracts a single frame from video at given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Kann Video nicht öffnen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        cap = cv2.VideoCapture(video_path)
        ret, frame_bgr = cap.read()
        cap.release()

    if not ret or frame_bgr is None:
        raise ValueError("Konnte keinen Frame aus dem Video extrahieren.")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def apply_preset_to_sliders(preset_name: str):
    """Returns updated slider values based on selected preset."""
    if preset_name not in PRESETS:
        return [gr.skip()] * 20

    p = PRESETS[preset_name]
    return [
        p.get("rtgi_intensity", 0.65),
        p.get("rtgi_range", 4.0),
        p.get("rtgi_steps", 10),
        p.get("ssr_intensity", 0.5),
        p.get("roughness", 0.25),
        p.get("wet_track_mode", False),
        p.get("rtao_intensity", 0.75),
        p.get("rtao_radius", 1.2),
        p.get("exposure", 0.0),
        p.get("contrast", 1.1),
        p.get("saturation", 1.15),
        p.get("vibrance", 0.18),
        p.get("temperature", 0.0),
        p.get("bloom_intensity", 0.18),
        p.get("bloom_threshold", 0.78),
        p.get("use_aces", True),
        p.get("vignette", 0.1),
        p.get("clarity", 0.35),
        p.get("film_grain", 0.02),
        p.get("denoise", True),
    ]


def parse_hardware_profile(selection_str: str) -> str:
    """Extracts profile key ('low_vram', 'balanced', 'ultra') from UI dropdown."""
    if "20-Serie" in selection_str or "6-8 GB" in selection_str:
        return "low_vram"
    elif "30-Serie" in selection_str or "Ausgewogen" in selection_str:
        return "balanced"
    elif "40" in selection_str or "50" in selection_str or "Maximum" in selection_str:
        return "ultra"
    return GPU_INFO.recommended_profile


def on_hardware_profile_change(selection_str: str):
    """Adjusts raymarching sample defaults when user switches GPU profile."""
    prof_key = parse_hardware_profile(selection_str)
    settings = get_profile_settings(prof_key)
    return [
        settings.get("rtgi_steps", 10),
        settings.get("rtao_radius", 1.2),
    ]


def process_preview_frame(
    input_image,
    video_file,
    video_timestamp,
    hardware_profile_choice,
    rtgi_intensity,
    rtgi_range,
    rtgi_steps,
    ssr_intensity,
    roughness,
    wet_track_mode,
    rtao_intensity,
    rtao_radius,
    exposure,
    contrast,
    saturation,
    vibrance,
    temperature,
    bloom_intensity,
    bloom_threshold,
    use_aces,
    vignette,
    clarity,
    film_grain,
    denoise,
):
    """Processes a single preview frame and returns comparison slider."""
    if input_image is not None:
        frame_rgb = np.array(input_image)
    elif video_file is not None:
        frame_rgb = extract_frame_at_time(video_file, video_timestamp)
    else:
        frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_rgb[360:, :] = [40, 42, 45]
        frame_rgb[:360, :] = [80, 140, 210]
        cv2.rectangle(frame_rgb, (440, 280), (840, 520), (220, 20, 30), -1)
        cv2.rectangle(frame_rgb, (400, 500), (880, 530), (10, 10, 10), -1)

    prof_key = parse_hardware_profile(hardware_profile_choice)
    prof_settings = get_profile_settings(prof_key)

    params = {
        "rtgi_intensity": float(rtgi_intensity),
        "rtgi_range": float(rtgi_range),
        "rtgi_steps": int(rtgi_steps),
        "ssr_intensity": float(ssr_intensity),
        "roughness": float(roughness),
        "wet_track_mode": bool(wet_track_mode),
        "rtao_intensity": float(rtao_intensity),
        "rtao_radius": float(rtao_radius),
        "exposure": float(exposure),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "vibrance": float(vibrance),
        "temperature": float(temperature),
        "bloom_intensity": float(bloom_intensity),
        "bloom_threshold": float(bloom_threshold),
        "use_aces": bool(use_aces),
        "vignette": float(vignette),
        "clarity": float(clarity),
        "film_grain": float(film_grain),
        "denoise": bool(denoise),
    }
    for k, v in prof_settings.items():
        if k not in params:
            params[k] = v

    pipeline = get_pipeline()
    out_rgb, depth_viz, normals_viz = pipeline.process_single_frame(frame_rgb, params)

    slider_tuple = (Image.fromarray(frame_rgb), Image.fromarray(out_rgb))
    return slider_tuple, Image.fromarray(depth_viz), Image.fromarray(normals_viz)


def render_full_video(
    video_file,
    hardware_profile_choice,
    enable_trim,
    trim_start,
    trim_end,
    output_resolution,
    encoder_codec,
    bitrate_mbps,
    rtgi_intensity,
    rtgi_range,
    rtgi_steps,
    ssr_intensity,
    roughness,
    wet_track_mode,
    rtao_intensity,
    rtao_radius,
    exposure,
    contrast,
    saturation,
    vibrance,
    temperature,
    bloom_intensity,
    bloom_threshold,
    use_aces,
    vignette,
    clarity,
    film_grain,
    denoise,
    progress=gr.Progress(track_tqdm=False),
):
    """Renders the full or cut video with detailed live ETA, speed, and status."""
    if video_file is None:
        raise gr.Error("Bitte lade zuerst ein Video hoch!")

    prof_key = parse_hardware_profile(hardware_profile_choice)
    prof_settings = get_profile_settings(prof_key)

    params = {
        "enable_trim": bool(enable_trim),
        "trim_start": float(trim_start),
        "trim_end": float(trim_end),
        "output_resolution": output_resolution,
        "encoder_codec": "hevc_nvenc" if "HEVC" in encoder_codec else "h264_nvenc",
        "bitrate_mbps": int(bitrate_mbps),
        "rtgi_intensity": float(rtgi_intensity),
        "rtgi_range": float(rtgi_range),
        "rtgi_steps": int(rtgi_steps),
        "ssr_intensity": float(ssr_intensity),
        "roughness": float(roughness),
        "wet_track_mode": bool(wet_track_mode),
        "rtao_intensity": float(rtao_intensity),
        "rtao_radius": float(rtao_radius),
        "exposure": float(exposure),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "vibrance": float(vibrance),
        "temperature": float(temperature),
        "bloom_intensity": float(bloom_intensity),
        "bloom_threshold": float(bloom_threshold),
        "use_aces": bool(use_aces),
        "vignette": float(vignette),
        "clarity": float(clarity),
        "film_grain": float(film_grain),
        "denoise": bool(denoise),
    }
    for k, v in prof_settings.items():
        if k not in params:
            params[k] = v

    output_dir = os.path.join(os.path.expanduser("~"), "Videos", "Luxanix_Renders")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_file))[0]
    out_file = os.path.join(output_dir, f"{base_name}_Luxanix_RTX.mp4")

    pipeline = get_pipeline()

    def update_progress(curr, total, fps, eta, elapsed):
        prog = curr / total if total > 0 else 0.0
        pct = prog * 100.0
        m_eta, s_eta = int(eta // 60), int(eta % 60)
        m_el, s_el = int(elapsed // 60), int(elapsed % 60)

        desc_str = (
            f"[{pct:.1f}%] Frame {curr}/{total} | "
            f"Speed: {fps:.1f} FPS | "
            f"Verstrichen: {m_el:02d}:{s_el:02d} | "
            f"Restzeit: {m_eta:02d}:{s_eta:02d} Min"
        )
        progress(prog, desc=desc_str)

    rendered_path = pipeline.process_video(
        input_path=video_file,
        output_path=out_file,
        params=params,
        progress_callback=update_progress,
    )

    return rendered_path, f"✅ Video erfolgreich gerendert & gespeichert in:\n{rendered_path}"


custom_css = """
body, .gradio-container {
    background-color: #0b0d10 !important;
    color: #e4e9f0 !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}
.lux-title {
    background: linear-gradient(90deg, #00f3ff 0%, #76b900 60%, #00ff88 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    margin-bottom: 2px !important;
    letter-spacing: -0.5px;
}
.gpu-badge {
    background-color: #121815;
    border: 1px solid #76b900;
    color: #8ce600;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 12px;
}
.stats-card {
    background: #14171d;
    border: 1px solid #28303d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
.stat-label {
    color: #8c9ba5;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.gr-button-primary {
    background: linear-gradient(135deg, #76b900 0%, #5b9200 100%) !important;
    border: none !important;
    color: #000 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    padding: 12px 20px !important;
}
.gr-button-primary:hover {
    filter: brightness(1.2) !important;
    box-shadow: 0 0 15px rgba(118, 185, 0, 0.4) !important;
}
"""


def build_app():
    with gr.Blocks(title="Luxanix Studio | RTX Video Remaster") as demo:
        gr.HTML(
            f"""
            <div style="padding: 12px 0 6px 0;">
                <h1 class="lux-title">⚡ LUXANIX STUDIO — RTX RAYTRACING & VIDEO REMASTER</h1>
                <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 0;">
                    Professionelles Screen-Space Raytracing, AI-Detailverbesserung, Video-Trimming & Hardware NVENC-Export.
                </p>
                <div class="gpu-badge">{get_gpu_badge_html()}</div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Inputs, Cutter & Tuning Controls
            with gr.Column(scale=4):
                with gr.Tab("📁 1. Video-Import & Maße"):
                    video_input = gr.Video(label="Simracing / Gaming Video hochladen", sources=["upload"])
                    video_stats_html = gr.HTML(
                        "<div class='stats-card'>Lade ein Video hoch, um Auflösung, Format, FPS und Dauer anzuzeigen.</div>"
                    )

                with gr.Tab("✂️ 2. Video Schneiden & Trimmen"):
                    enable_trim = gr.Checkbox(value=False, label="✂️ Zuschneiden aktivieren (Nur Highlight rendern)")
                    with gr.Row():
                        trim_start = gr.Slider(0.0, 300.0, value=0.0, step=0.5, label="Startpunkt (Sekunden)")
                        trim_end = gr.Slider(0.0, 300.0, value=60.0, step=0.5, label="Endpunkt (Sekunden)")
                    trim_info_label = gr.Markdown("Volle Videolänge wird gerendert.")

                with gr.Tab("🎨 3. Presets & GPU"):
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Simracing: Wet Track & Reflections" if "Simracing: Wet Track & Reflections" in PRESETS else None,
                        label="Shader- & Look-Preset"
                    )
                    hw_choices = [
                        f"Automatisch ({GPU_INFO.generation} - {GPU_INFO.recommended_profile.upper()})",
                        "RTX 20-Serie / 6-8 GB VRAM (Performance & Memory-Saver)",
                        "RTX 30-Serie / 8-12 GB VRAM (Ausgewogen)",
                        "RTX 40/50-Serie / 12GB+ (Maximum Quality)"
                    ]
                    hardware_profile_choice = gr.Dropdown(
                        choices=hw_choices,
                        value=hw_choices[0],
                        label="Grafikkarten-Profil (VRAM & Performance Optimierung)",
                        info="Wähle deine GPU-Klasse, um Speicherverbrauch und Render-Geschwindigkeit optimal abzustimmen."
                    )

                with gr.Accordion("✨ Raytracing-Shader Einstellungen (RTX Core)", open=True):
                    rtgi_intensity = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="RTGI: Indirektes Licht (Bounce Light)")
                    rtgi_range = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="RTGI: Licht-Reichweite (Radius)")
                    rtgi_steps = gr.Slider(4, 24, value=10, step=1, label="RTGI: Raymarching Samples / Qualität")

                    gr.HTML("<hr style='border-color: #242b35; margin: 10px 0;'>")
                    ssr_intensity = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="SSR: Reflexionen (Asphalt / Lack / Wasser)")
                    roughness = gr.Slider(0.01, 1.0, value=0.22, step=0.02, label="SSR: Oberflächen-Rauheit (Glossiness)")
                    wet_track_mode = gr.Checkbox(value=False, label="🌧️ Nasse Strecke Modus (Boostet Bodenreflexionen)")

                    gr.HTML("<hr style='border-color: #242b35; margin: 10px 0;'>")
                    rtao_intensity = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="RTAO: Kontaktschatten (Chassis / Radkästen)")
                    rtao_radius = gr.Slider(0.5, 4.0, value=1.3, step=0.1, label="RTAO: Schatten-Radius")
                    denoise = gr.Checkbox(value=True, label="Kanten-erhaltendes Denoising (Kein Flimmern)")

                with gr.Accordion("💎 Photorealismus & Detail-Clarity (Anti-Blur)", open=True):
                    clarity = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Detail-Klarheit (Entfernt TAA-Bewegungsunschärfe)")
                    film_grain = gr.Slider(0.0, 0.1, value=0.02, step=0.005, label="Subtiles Filmkorn (Beseitigt Farb-Banding)")
                    bloom_intensity = gr.Slider(0.0, 1.0, value=0.2, step=0.02, label="Bloom: Scheinwerfer & Glanz-Glow")
                    bloom_threshold = gr.Slider(0.5, 0.95, value=0.78, step=0.02, label="Bloom-Schwellenwert")
                    use_aces = gr.Checkbox(value=True, label="ACES Filmic Tone Mapping (Kino-Kontrast)")

                with gr.Accordion("🌈 Farbstimmung & Feintuning (Color-Grading)", open=False):
                    saturation = gr.Slider(0.0, 2.5, value=1.18, step=0.02, label="Sättigung (Saturation)")
                    vibrance = gr.Slider(-0.5, 1.0, value=0.2, step=0.05, label="Dynamik (Smart Vibrance)")
                    contrast = gr.Slider(0.5, 2.0, value=1.1, step=0.02, label="Kontrast (S-Curve)")
                    exposure = gr.Slider(-2.0, 2.0, value=0.05, step=0.05, label="Belichtung (Exposure EV)")
                    temperature = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Farbtemperatur (Kalt 🔵 / Warm 🟠)")
                    vignette = gr.Slider(0.0, 0.6, value=0.12, step=0.02, label="Vignette (Randabdunklung)")

                with gr.Row():
                    video_timestamp = gr.Slider(0.0, 300.0, value=2.0, step=0.5, label="Vorschau-Sekunde")
                    btn_preview = gr.Button("⚡ Vorschau-Frame Aktualisieren", variant="primary")

            # Right Column: Live Comparison & Export
            with gr.Column(scale=6):
                with gr.Tab("👁️ Interaktiver Vorher / Nachher Vergleich"):
                    preview_slider = gr.ImageSlider(
                        label="Ziehe den Regler, um den Unterschied vor und nach dem Raytracing zu sehen!",
                        show_label=True,
                        type="pil",
                    )
                    with gr.Accordion("🔍 AI-Tiefenkarte & Normalen anzeigen (Diagnose)", open=False):
                        with gr.Row():
                            depth_view = gr.Image(label="AI-Tiefenkarte (Depth Anything v2)")
                            normals_view = gr.Image(label="3D-Normalenvektoren (Oberflächen-Ausrichtung)")

                with gr.Tab("🎬 Video Rendern & Export"):
                    gr.Markdown("### 🚀 Export-Einstellungen (NVIDIA NVENC Beschleunigung)")
                    with gr.Row():
                        output_resolution = gr.Dropdown(
                            choices=[
                                "Original",
                                "1080p Full HD",
                                "1440p 2K QHD",
                                "4K Ultra HD (2160p)",
                                "8K Ultra HD (4320p / 7680x4320)",
                            ],
                            value="Original",
                            label="Ausgabe-Auflösung (inkl. 8K Super-Resolution Upscaling)"
                        )
                        encoder_codec = gr.Dropdown(
                            choices=[
                                "HEVC / H.265 (NVIDIA NVENC - Empfohlen für 4K/8K)",
                                "H.264 (NVIDIA NVENC - Bis 4K, maximale Kompatibilität)",
                            ],
                            value="HEVC / H.265 (NVIDIA NVENC - Empfohlen für 4K/8K)",
                            label="Video-Codec"
                        )
                        bitrate_mbps = gr.Slider(10, 160, value=60, step=5, label="Bitrate (Mbps) — für 4K/8K 60-120 Mbps empfohlen")

                    btn_render_video = gr.Button("🚀 Starte Video-Raytracing Export", variant="primary", size="lg")
                    render_status = gr.Textbox(label="Live Render-Status & Speicherort", interactive=False, lines=2)
                    rendered_video_output = gr.Video(label="Fertig gerendertes Video (mit Original-Audio)")

                with gr.Tab("ℹ️ Grafikkarte & Generationen"):
                    gr.Markdown(
                        """
                        ### 🎮 Unterstützte Grafikkarten-Generationen in Luxanix
                        
                        - **Turing (RTX 20-Serie):** RTX 2060 (6GB/12GB), RTX 2070, RTX 2080 (Low-VRAM Profil mit automatischer Speicherschonung).
                        - **Ampere (RTX 30-Serie):** RTX 3060, 3070, 3080, **RTX 3080 Ti (deine Karte)**, RTX 3090.
                        - **Ada Lovelace (RTX 40-Serie):** RTX 4060, 4070, 4080, 4090.
                        - **Blackwell (RTX 50-Serie):** Zukunftssicher vorbereitet für Compute 9.0+ & nächste Tensor Cores.
                        """
                    )

        # Wire Up Events
        # Video Upload -> Update Metadata, Trimming Sliders
        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[video_stats_html, trim_start, trim_end, video_timestamp, trim_info_label]
        )

        # Trimming Sliders -> Update Trim Summary
        for t_ctrl in [trim_start, trim_end]:
            t_ctrl.change(
                fn=update_trim_label,
                inputs=[trim_start, trim_end, video_input],
                outputs=[trim_info_label]
            )

        all_sliders = [
            rtgi_intensity, rtgi_range, rtgi_steps,
            ssr_intensity, roughness, wet_track_mode,
            rtao_intensity, rtao_radius,
            exposure, contrast, saturation, vibrance, temperature,
            bloom_intensity, bloom_threshold, use_aces, vignette,
            clarity, film_grain, denoise
        ]

        preset_dropdown.change(
            fn=apply_preset_to_sliders,
            inputs=[preset_dropdown],
            outputs=all_sliders
        )

        hardware_profile_choice.change(
            fn=on_hardware_profile_change,
            inputs=[hardware_profile_choice],
            outputs=[rtgi_steps, rtao_radius]
        )

        preview_inputs = [gr.State(None), video_input, video_timestamp, hardware_profile_choice] + all_sliders
        btn_preview.click(
            fn=process_preview_frame,
            inputs=preview_inputs,
            outputs=[preview_slider, depth_view, normals_view]
        )

        render_inputs = [
            video_input,
            hardware_profile_choice,
            enable_trim,
            trim_start,
            trim_end,
            output_resolution,
            encoder_codec,
            bitrate_mbps
        ] + all_sliders

        btn_render_video.click(
            fn=render_full_video,
            inputs=render_inputs,
            outputs=[rendered_video_output, render_status]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(inbrowser=True, server_port=7860, css=custom_css)
