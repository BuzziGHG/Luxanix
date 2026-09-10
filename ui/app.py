"""
SimRTX Studio - Modern User Interface
Dark RTX / Simracing Themed Studio for Ray Tracing and Color Grading Video Reworks.
"""

import os
import json
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from typing import Dict, Any, Tuple

from engine.depth_estimator import DepthEstimator
from engine.video_pipeline import VideoPipeline

# Load Presets
PRESETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets.json")
if os.path.exists(PRESETS_PATH):
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        PRESETS = json.load(f)
else:
    PRESETS = {}

# Global Engine Pipeline Singleton
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = VideoPipeline()
    return PIPELINE


def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"🟢 GPU Aktiv: {gpu_name} ({vram_gb:.1f} GB VRAM) | CUDA FP16 Beschleunigung Bereit"
    return "⚠️ Warnung: Keine CUDA GPU gefunden. CPU-Modus aktiv."


def extract_frame_from_video(video_path: str, timestamp_sec: float) -> np.ndarray:
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
        # Fallback to frame 0
        cap = cv2.VideoCapture(video_path)
        ret, frame_bgr = cap.read()
        cap.release()

    if not ret or frame_bgr is None:
        raise ValueError("Konnte keinen Frame aus dem Video extrahieren.")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def apply_preset_to_sliders(preset_name: str):
    """Returns updated slider values based on selected preset."""
    if preset_name not in PRESETS:
        return [gr.skip()] * 18

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
        p.get("denoise", True),
    ]


def process_preview_frame(
    input_image,
    video_file,
    video_timestamp,
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
    denoise,
):
    """Processes a single frame and returns comparison slider plus diagnostics."""
    # Determine input frame
    if input_image is not None:
        frame_rgb = np.array(input_image)
    elif video_file is not None:
        frame_rgb = extract_frame_from_video(video_file, video_timestamp)
    else:
        # Create a test gaming synthetic frame if nothing provided
        frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_rgb[360:, :] = [40, 42, 45]  # Road
        frame_rgb[:360, :] = [80, 140, 210]  # Sky
        cv2.rectangle(frame_rgb, (440, 280), (840, 520), (220, 20, 30), -1)  # Car
        cv2.rectangle(frame_rgb, (400, 500), (880, 530), (10, 10, 10), -1)  # Shadow

    # Build params dict
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
        "denoise": bool(denoise),
    }

    pipeline = get_pipeline()
    out_rgb, depth_viz, normals_viz = pipeline.process_single_frame(frame_rgb, params)

    # Return slider tuple: (original_frame, raytraced_frame)
    slider_tuple = (Image.fromarray(frame_rgb), Image.fromarray(out_rgb))
    return slider_tuple, Image.fromarray(depth_viz), Image.fromarray(normals_viz)


def render_full_video(
    video_file,
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
    denoise,
    progress=gr.Progress(),
):
    """Renders the full video with ray tracing and hardware NVENC encoding."""
    if video_file is None:
        raise gr.Error("Bitte lade zuerst ein Video hoch!")

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
        "denoise": bool(denoise),
    }

    output_dir = os.path.join(os.path.expanduser("~"), "Videos", "SimRTX_Renders")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_file))[0]
    out_file = os.path.join(output_dir, f"{base_name}_RTX_Overhauled.mp4")

    pipeline = get_pipeline()

    def update_progress(curr, total, fps, eta):
        prog = curr / total if total > 0 else 0.0
        progress(prog, desc=f"Frame {curr}/{total} | {fps:.1f} FPS | Restzeit: {eta:.0f}s")

    rendered_path = pipeline.process_video(
        input_path=video_file,
        output_path=out_file,
        params=params,
        progress_callback=update_progress,
    )

    return rendered_path, f"✅ Video erfolgreich fertig gerendert: {rendered_path}"


# Custom Theme & Styling
custom_css = """
body, .gradio-container {
    background-color: #0d0f12 !important;
    color: #e0e6ed !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}
.rtx-title {
    background: linear-gradient(90deg, #76b900, #00f3ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    margin-bottom: 2px !important;
}
.gpu-badge {
    background-color: #1a231d;
    border: 1px solid #76b900;
    color: #8ce600;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 12px;
}
.gr-button-primary {
    background: linear-gradient(135deg, #76b900 0%, #5b9200 100%) !important;
    border: none !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}
.gr-button-primary:hover {
    filter: brightness(1.15) !important;
}
"""


def build_app():
    with gr.Blocks(title="SimRTX Studio | Raytracing Video Remaster") as demo:
        gr.HTML(
            f"""
            <div style="padding: 10px 0;">
                <h1 class="rtx-title">⚡ SimRTX Studio — Raytracing Video Remaster</h1>
                <p style="color: #8c9ba5; font-size: 1.05rem; margin-top: 0;">
                    Photorealistisches Screen-Space Raytracing (RTGI, SSR, RTAO) & Color-Grading für Simracing- und Gaming-Videos.
                </p>
                <div class="gpu-badge">{get_gpu_info()}</div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=4):
                with gr.Tab("📁 Video- & Frame-Import"):
                    video_input = gr.Video(label="Simracing / Gaming Video hochladen", sources=["upload"])
                    video_timestamp = gr.Slider(
                        minimum=0.0, maximum=300.0, value=2.0, step=0.5,
                        label="Vorschau-Zeitpunkt (Sekunden im Video)"
                    )
                    image_input = gr.Image(
                        label="Oder Screenshot / Einzelbild einfügen",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )

                with gr.Tab("🎨 Presets & Profile"):
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Simracing: Wet Track & Reflections" if "Simracing: Wet Track & Reflections" in PRESETS else None,
                        label="Preset auswählen"
                    )

                with gr.Accordion("✨ Raytracing-Shader Einstellungen (RTX Core)", open=True):
                    rtgi_intensity = gr.Slider(0.0, 2.0, value=0.65, step=0.05, label="RTGI: Indirektes Licht (Bounce Light)")
                    rtgi_range = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="RTGI: Licht-Reichweite (Radius)")
                    rtgi_steps = gr.Slider(4, 20, value=10, step=1, label="RTGI: Raymarching Samples / Qualität")

                    gr.HTML("<hr style='border-color: #2a2e35; margin: 10px 0;'>")
                    ssr_intensity = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="SSR: Reflexionen (Asphalt / Lack / Wasser)")
                    roughness = gr.Slider(0.01, 1.0, value=0.22, step=0.02, label="SSR: Oberflächen-Rauheit (Glossiness)")
                    wet_track_mode = gr.Checkbox(value=False, label="🌧️ Nasse Strecke Modus (Boostet Bodenreflexionen)")

                    gr.HTML("<hr style='border-color: #2a2e35; margin: 10px 0;'>")
                    rtao_intensity = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="RTAO: Kontaktschatten (Chassis / Radkästen)")
                    rtao_radius = gr.Slider(0.5, 4.0, value=1.3, step=0.1, label="RTAO: Schatten-Radius")
                    denoise = gr.Checkbox(value=True, label="Kanten-erhaltendes Denoising (Kein Flimmern)")

                with gr.Accordion("🌈 Feintuning & Color-Grading (CapCut-Killer)", open=True):
                    saturation = gr.Slider(0.0, 2.5, value=1.18, step=0.02, label="Sättigung (Saturation)")
                    vibrance = gr.Slider(-0.5, 1.0, value=0.2, step=0.05, label="Dynamik (Smart Vibrance)")
                    contrast = gr.Slider(0.5, 2.0, value=1.1, step=0.02, label="Kontrast (S-Curve)")
                    exposure = gr.Slider(-2.0, 2.0, value=0.05, step=0.05, label="Belichtung (Exposure EV)")
                    temperature = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Farbtemperatur (Kalt 🔵 / Warm 🟠)")
                    bloom_intensity = gr.Slider(0.0, 1.0, value=0.2, step=0.02, label="Bloom: Scheinwerfer & Glanz-Glow")
                    bloom_threshold = gr.Slider(0.5, 0.95, value=0.78, step=0.02, label="Bloom-Schwellenwert")
                    use_aces = gr.Checkbox(value=True, label="ACES Filmic Tone Mapping (Kino-Kontrast)")
                    vignette = gr.Slider(0.0, 0.6, value=0.12, step=0.02, label="Vignette (Randabdunklung)")

                btn_preview = gr.Button("⚡ Vorschau-Frame Aktualisieren", variant="primary")

            # Right Column: Live Comparison & Video Export
            with gr.Column(scale=6):
                with gr.Tab("👁️ Interaktiver Vorher / Nachher Vergleich"):
                    preview_slider = gr.ImageSlider(
                        label="Ziehe den Regler, um den Raytracing-Unterschied zu sehen!",
                        show_label=True,
                        type="pil",
                    )

                    with gr.Accordion("🔍 AI-Tiefenkarte & Normalen anzeigen (Diagnose)", open=False):
                        with gr.Row():
                            depth_view = gr.Image(label="AI-Rekonstruierte Tiefenkarte (Depth Anything v2)")
                            normals_view = gr.Image(label="Oberflächen-Normalenvektoren (3D-Geometrie)")

                with gr.Tab("🎬 Komplettes Video Rendern"):
                    gr.Markdown("### Berechne den finalen Clip mit RTX NVENC Hardware-Beschleunigung")
                    gr.Markdown("Alle Audio-Spuren (Motor-Sound, Reifenquietschen, Spotter) bleiben erhalten!")

                    btn_render_video = gr.Button("🚀 Starte Video-Raytracing Export", variant="primary", size="lg")
                    render_status = gr.Textbox(label="Status & Speicherort", interactive=False)
                    rendered_video_output = gr.Video(label="Fertiges RTX-Video")

        all_sliders = [
            rtgi_intensity, rtgi_range, rtgi_steps,
            ssr_intensity, roughness, wet_track_mode,
            rtao_intensity, rtao_radius,
            exposure, contrast, saturation, vibrance, temperature,
            bloom_intensity, bloom_threshold, use_aces, vignette, denoise
        ]

        preset_dropdown.change(
            fn=apply_preset_to_sliders,
            inputs=[preset_dropdown],
            outputs=all_sliders
        )

        preview_inputs = [image_input, video_input, video_timestamp] + all_sliders
        btn_preview.click(
            fn=process_preview_frame,
            inputs=preview_inputs,
            outputs=[preview_slider, depth_view, normals_view]
        )

        render_inputs = [video_input] + all_sliders
        btn_render_video.click(
            fn=render_full_video,
            inputs=render_inputs,
            outputs=[rendered_video_output, render_status]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(inbrowser=True, server_port=7860, css=custom_css)
