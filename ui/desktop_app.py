"""
Luxanix Studio — Native Desktop Edition
GPU-accelerated desktop application for Raytracing, 8K Upscaling & Photorealistic Remastering.
Runs directly on the desktop (no web browser).
Features:
- Native GUI window via CustomTkinter
- Embedded Video Player (Play/Pause, Timeline Scrub, Split-Screen Before/After)
- Auto-Adaptive AI Preset (Per-Millisecond Dynamic Scene Analyzer & Optimizer)
- Hardware-accelerated NVIDIA RTX Pipeline (RTGI, SSR, RTAO, Anti-TAA Clarity, 8K Ultra HD)
"""

import os
import sys
import time
import json
import threading
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.video_pipeline import VideoPipeline, get_video_info
from engine.hardware import detect_gpu_hardware, get_profile_settings
from engine.auto_preset import AutoSceneOptimizer

# Set CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class LuxanixDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚡ Luxanix Studio — AI Raytracing & 8K Video Remaster (NVIDIA RTX)")
        self.geometry("1420x920")
        self.minsize(1180, 780)

        # State Variables
        self.video_path = None
        self.video_info = {}
        self.pipeline = None
        self.auto_optimizer = AutoSceneOptimizer(smoothing_alpha=0.25)
        self.gpu_info = detect_gpu_hardware()

        # Playback / Preview State
        self.preview_cap = None
        self.preview_frames_cache = []
        self.is_playing = False
        self.playback_idx = 0
        self.split_view_mode = "Remaster"  # "Remaster", "Split", "Original"
        self.is_rendering = False
        self.stop_render_flag = False

        self._build_ui()
        self._init_pipeline_async()

    def _init_pipeline_async(self):
        def init():
            self.lbl_status.configure(text="Initialisiere RTX Pipeline & Depth Anything v2...")
            try:
                self.pipeline = VideoPipeline()
                self.lbl_status.configure(text="✅ RTX 3080 Ti Tensor-Core Engine bereit.")
            except Exception as e:
                self.lbl_status.configure(text=f"Warnung bei Initialisierung: {e}")
        t = threading.Thread(target=init, daemon=True)
        t.start()

    def _build_ui(self):
        # Configure Grid Layout (2 columns: Sidebar 380px, Main Display rest)
        self.grid_columnconfigure(0, weight=0, minsize=390)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # Bottom status bar

        # =======================
        # LEFT SIDEBAR (Scrollable)
        # =======================
        self.sidebar = ctk.CTkScrollableFrame(self, width=390, corner_radius=0, fg_color="#0e1115")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Title / Branding
        lbl_brand = ctk.CTkLabel(
            self.sidebar,
            text="⚡ LUXANIX STUDIO",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#76b900"
        )
        lbl_brand.pack(anchor="w", padx=16, pady=(16, 2))

        lbl_sub = ctk.CTkLabel(
            self.sidebar,
            text="AI Raytracing & 8K Ultra Remaster",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        lbl_sub.pack(anchor="w", padx=16, pady=(0, 10))

        # GPU Badge
        gpu_name = self.gpu_info.get("device_name", "NVIDIA GeForce RTX")
        vram_gb = self.gpu_info.get("vram_gb", 12.0)
        self.lbl_gpu = ctk.CTkLabel(
            self.sidebar,
            text=f"🟢 {gpu_name} ({vram_gb:.1f} GB VRAM)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#76b900",
            fg_color="#18231c",
            corner_radius=6,
            padx=8,
            pady=4
        )
        self.lbl_gpu.pack(fill="x", padx=16, pady=(0, 14))

        # 1. Video Import Section
        sec_video = ctk.CTkFrame(self.sidebar, fg_color="#14181d", corner_radius=8)
        sec_video.pack(fill="x", padx=14, pady=6)

        btn_select_video = ctk.CTkButton(
            sec_video,
            text="📁 Video auswählen...",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1e2630",
            hover_color="#2b3644",
            command=self._choose_video
        )
        btn_select_video.pack(fill="x", padx=12, pady=10)

        self.lbl_video_info = ctk.CTkLabel(
            sec_video,
            text="Kein Video geladen.\nBitte wähle ein Gameplay- oder Simracing-Video aus.",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            justify="left"
        )
        self.lbl_video_info.pack(anchor="w", padx=12, pady=(0, 10))

        # 2. Timeline Cutter / Trimmer
        sec_cutter = ctk.CTkFrame(self.sidebar, fg_color="#14181d", corner_radius=8)
        sec_cutter.pack(fill="x", padx=14, pady=6)

        self.sw_trim = ctk.CTkSwitch(
            sec_cutter,
            text="✂️ Video-Schnitt (Highlight Trimmer)",
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#76b900",
            command=self._update_trim_labels
        )
        self.sw_trim.pack(anchor="w", padx=12, pady=(10, 6))

        grid_trim = ctk.CTkFrame(sec_cutter, fg_color="transparent")
        grid_trim.pack(fill="x", padx=10, pady=4)
        grid_trim.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(grid_trim, text="Start (Sek):", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w", padx=4)
        ctk.CTkLabel(grid_trim, text="Ende (Sek):", font=ctk.CTkFont(size=11)).grid(row=0, column=1, sticky="w", padx=4)

        self.entry_trim_start = ctk.CTkEntry(grid_trim, placeholder_text="0.0", width=80)
        self.entry_trim_start.insert(0, "0.0")
        self.entry_trim_start.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        self.entry_trim_end = ctk.CTkEntry(grid_trim, placeholder_text="10.0", width=80)
        self.entry_trim_end.insert(0, "10.0")
        self.entry_trim_end.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        self.lbl_trim_summary = ctk.CTkLabel(
            sec_cutter,
            text="Schnittbereich: Deaktiviert",
            font=ctk.CTkFont(size=11),
            text_color="#64748b"
        )
        self.lbl_trim_summary.pack(anchor="w", padx=12, pady=(2, 8))

        # 3. AUTO-ADAPTIVE PRESET (Per-Millisecond Scene Optimizer)
        sec_auto = ctk.CTkFrame(self.sidebar, fg_color="#16201a", border_width=1, border_color="#76b900", corner_radius=8)
        sec_auto.pack(fill="x", padx=14, pady=8)

        self.sw_auto_preset = ctk.CTkSwitch(
            sec_auto,
            text="🤖 Auto-Preset (Echtzeit-Optimierung)",
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#76b900",
            command=self._on_auto_preset_toggle
        )
        self.sw_auto_preset.select()  # Default enabled!
        self.sw_auto_preset.pack(anchor="w", padx=12, pady=(10, 4))

        self.lbl_auto_desc = ctk.CTkLabel(
            sec_auto,
            text="Analysiert Szenen dynamisch & berechnet optimales RTGI, Nässe-Reflexionen & Belichtung.",
            font=ctk.CTkFont(size=10.5),
            text_color="#86efac",
            justify="left",
            wraplength=340
        )
        self.lbl_auto_desc.pack(anchor="w", padx=12, pady=(0, 4))

        self.lbl_auto_telemetry = ctk.CTkLabel(
            sec_auto,
            text="Dynamik: Aktiviert (passt sich pro Frame flüssig an)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#4ade80",
            justify="left"
        )
        self.lbl_auto_telemetry.pack(anchor="w", padx=12, pady=(0, 8))

        # 4. Manual Shader & Lighting Sliders (Accordion/Frame)
        self.sec_sliders = ctk.CTkFrame(self.sidebar, fg_color="#14181d", corner_radius=8)
        self.sec_sliders.pack(fill="x", padx=14, pady=6)

        lbl_tune = ctk.CTkLabel(self.sec_sliders, text="🎛️ Manuelles Feintuning", font=ctk.CTkFont(size=12, weight="bold"))
        lbl_tune.pack(anchor="w", padx=12, pady=(8, 4))

        self.slider_rtgi = self._create_slider(self.sec_sliders, "RTGI Intensität (Streulicht)", 0.0, 1.5, 0.65)
        self.slider_ssr = self._create_slider(self.sec_sliders, "SSR Reflexionen (Spiegelung)", 0.0, 1.5, 0.55)
        self.slider_rtao = self._create_slider(self.sec_sliders, "RTAO Kontaktschatten", 0.0, 1.5, 0.60)
        self.slider_clarity = self._create_slider(self.sec_sliders, "Detail-Clarity (Anti-TAA)", 0.0, 1.0, 0.35)
        self.slider_exposure = self._create_slider(self.sec_sliders, "Belichtung (Exposure EV)", -1.5, 1.5, 0.05)
        self.slider_contrast = self._create_slider(self.sec_sliders, "Kontrast (S-Kurve)", 0.6, 1.8, 1.12)
        self.slider_bloom = self._create_slider(self.sec_sliders, "Scheinwerfer-Bloom", 0.0, 1.0, 0.30)
        self.slider_grain = self._create_slider(self.sec_sliders, "Filmkorn (Anti-Banding)", 0.0, 0.3, 0.08)

        # 5. Export Settings (8K Ultra HD)
        sec_export = ctk.CTkFrame(self.sidebar, fg_color="#14181d", corner_radius=8)
        sec_export.pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(sec_export, text="🚀 Export & Auflösung", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=12, pady=(8, 2))

        self.combo_res = ctk.CTkComboBox(
            sec_export,
            values=[
                "Original",
                "1080p Full HD",
                "1440p 2K QHD",
                "4K Ultra HD (2160p)",
                "8K Ultra HD (4320p / 7680x4320)"
            ],
            width=340
        )
        self.combo_res.set("Original")
        self.combo_res.pack(fill="x", padx=12, pady=4)

        self.combo_codec = ctk.CTkComboBox(
            sec_export,
            values=[
                "HEVC / H.265 (NVIDIA NVENC - Empfohlen für 4K/8K)",
                "H.264 (NVIDIA NVENC - Bis 4K)"
            ],
            width=340
        )
        self.combo_codec.set("HEVC / H.265 (NVIDIA NVENC - Empfohlen für 4K/8K)")
        self.combo_codec.pack(fill="x", padx=12, pady=4)

        self.slider_bitrate = self._create_slider(sec_export, "Bitrate (Mbps) — für 8K 60-120 Mbps", 10, 160, 60)

        # Render Full Video Button
        self.btn_render_full = ctk.CTkButton(
            self.sidebar,
            text="🚀 Vollständiges Video Rendern & Speichern",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#76b900",
            hover_color="#5a8e00",
            text_color="#000000",
            height=42,
            command=self._start_full_render
        )
        self.btn_render_full.pack(fill="x", padx=14, pady=(12, 16))

        # =======================
        # RIGHT MAIN DISPLAY AREA
        # =======================
        self.main_area = ctk.CTkFrame(self, fg_color="#0b0d10", corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Top Bar: Tabs & View Mode
        top_bar = ctk.CTkFrame(self.main_area, fg_color="#12161b", height=48, corner_radius=8)
        top_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 8))

        self.btn_tab_video = ctk.CTkButton(
            top_bar,
            text="🎬 Video-Vorschau (Player)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#76b900",
            text_color="#000000",
            width=170,
            command=lambda: self._switch_tab("video")
        )
        self.btn_tab_video.pack(side="left", padx=8, pady=8)

        self.btn_tab_frame = ctk.CTkButton(
            top_bar,
            text="👁️ Einzelbild A/B Vergleich",
            font=ctk.CTkFont(size=12),
            fg_color="#1e2630",
            text_color="#cbd5e1",
            width=170,
            command=lambda: self._switch_tab("frame")
        )
        self.btn_tab_frame.pack(side="left", padx=8, pady=8)

        self.btn_tab_diag = ctk.CTkButton(
            top_bar,
            text="🔍 AI-Tiefenkarte & Normalen",
            font=ctk.CTkFont(size=12),
            fg_color="#1e2630",
            text_color="#cbd5e1",
            width=180,
            command=lambda: self._switch_tab("diag")
        )
        self.btn_tab_diag.pack(side="left", padx=8, pady=8)

        # Split Mode Segmented Button
        self.seg_view_mode = ctk.CTkSegmentedButton(
            top_bar,
            values=["Remaster", "Split (50/50)", "Original"],
            command=self._on_view_mode_change
        )
        self.seg_view_mode.set("Remaster")
        self.seg_view_mode.pack(side="right", padx=12, pady=8)

        # Video Canvas / Screen
        self.canvas_container = ctk.CTkFrame(self.main_area, fg_color="#050608", corner_radius=8)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.lbl_screen = ctk.CTkLabel(
            self.canvas_container,
            text="Lade ein Video hoch und klicke auf '▶️ Video-Vorschau rendern', um die überarbeitete Szene zu sehen.",
            font=ctk.CTkFont(size=13),
            text_color="#64748b"
        )
        self.lbl_screen.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Video Player Controls Bar
        self.player_bar = ctk.CTkFrame(self.main_area, fg_color="#12161b", height=54, corner_radius=8)
        self.player_bar.grid(row=2, column=0, sticky="ew", padx=6, pady=(8, 0))

        self.btn_play_pause = ctk.CTkButton(
            self.player_bar,
            text="▶️ Play",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=80,
            command=self._toggle_playback
        )
        self.btn_play_pause.pack(side="left", padx=8, pady=8)

        self.btn_gen_preview = ctk.CTkButton(
            self.player_bar,
            text="⚡ Video-Vorschau generieren & abspielen",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f538d",
            hover_color="#14375e",
            command=self._generate_video_preview
        )
        self.btn_gen_preview.pack(side="left", padx=8, pady=8)

        self.slider_timeline = ctk.CTkSlider(
            self.player_bar,
            from_=0,
            to=100,
            command=self._on_timeline_scrub
        )
        self.slider_timeline.set(0)
        self.slider_timeline.pack(side="left", fill="x", expand=True, padx=12, pady=8)

        self.lbl_player_time = ctk.CTkLabel(self.player_bar, text="00:00 / 00:00", font=ctk.CTkFont(size=11), text_color="#94a3b8")
        self.lbl_player_time.pack(side="right", padx=12, pady=8)

        # =======================
        # BOTTOM STATUS & RENDER BAR
        # =======================
        self.bottom_bar = ctk.CTkFrame(self, fg_color="#080a0c", height=50, corner_radius=0)
        self.bottom_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        self.progress_bar = ctk.CTkProgressBar(self.bottom_bar, progress_color="#76b900", height=10)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=16, pady=(6, 2))

        status_row = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 6))

        self.lbl_status = ctk.CTkLabel(
            status_row,
            text="Bereit. Lade ein Video, um das RTX-Remastering zu starten.",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        self.lbl_status.pack(side="left")

        self.lbl_eta = ctk.CTkLabel(
            status_row,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#76b900"
        )
        self.lbl_eta.pack(side="right")

    def _create_slider(self, parent, label_text, min_val, max_val, default_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=12, pady=2)

        lbl = ctk.CTkLabel(frame, text=f"{label_text}: {default_val:.2f}", font=ctk.CTkFont(size=10.5), text_color="#cbd5e1")
        lbl.pack(anchor="w")

        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, number_of_steps=100)
        slider.set(default_val)
        slider.pack(fill="x", pady=(1, 4))

        def on_change(val):
            lbl.configure(text=f"{label_text}: {float(val):.2f}")
        slider.configure(command=on_change)
        return slider

    def _choose_video(self):
        file_path = filedialog.askopenfilename(
            title="Wähle ein Gameplay- oder Simracing-Video",
            filetypes=[("Video-Dateien", "*.mp4 *.mkv *.avi *.mov *.webm"), ("Alle Dateien", "*.*")]
        )
        if not file_path:
            return

        self.video_path = file_path
        self.video_info = get_video_info(file_path)

        w = self.video_info.get("width", 1920)
        h = self.video_info.get("height", 1080)
        fps = self.video_info.get("fps", 60.0)
        dur = self.video_info.get("duration", 0.0)
        frames = self.video_info.get("total_frames", 0)
        aspect = self.video_info.get("aspect_ratio", "16:9")

        self.lbl_video_info.configure(
            text=f"📄 {os.path.basename(file_path)}\n"
                 f"Auflösung: {w}x{h} ({aspect}) | FPS: {fps:.1f}\n"
                 f"Dauer: {dur:.1f}s ({frames} Frames)",
            text_color="#e2e8f0"
        )

        self.entry_trim_start.delete(0, "end")
        self.entry_trim_start.insert(0, "0.0")

        preview_end = min(dur, 6.0) if dur > 0 else 6.0
        self.entry_trim_end.delete(0, "end")
        self.entry_trim_end.insert(0, f"{preview_end:.1f}")

        self._update_trim_labels()
        self._generate_first_frame_preview()

    def _update_trim_labels(self):
        if not self.sw_trim.get():
            self.lbl_trim_summary.configure(text="Schnittbereich: Deaktiviert (Ganzes Video wird gerendert)")
            return
        try:
            st = float(self.entry_trim_start.get())
            en = float(self.entry_trim_end.get())
            length = max(0.0, en - st)
            fps = self.video_info.get("fps", 60.0)
            f_count = int(length * fps)
            self.lbl_trim_summary.configure(
                text=f"Schnitt: {st:.1f}s bis {en:.1f}s (Dauer: {length:.1f}s, ~{f_count} Frames)",
                text_color="#76b900"
            )
        except Exception:
            pass

    def _on_auto_preset_toggle(self):
        is_auto = bool(self.sw_auto_preset.get())
        if is_auto:
            self.lbl_auto_telemetry.configure(text="Dynamik: Aktiviert (passt sich pro Frame flüssig an)", text_color="#4ade80")
        else:
            self.lbl_auto_telemetry.configure(text="Dynamik: Deaktiviert (Manuelle Slider-Werte aktiv)", text_color="#f87171")

    def _collect_params(self):
        is_auto = bool(self.sw_auto_preset.get())
        base_params = {
            "auto_preset": is_auto,
            "enable_trim": bool(self.sw_trim.get()),
            "trim_start": float(self.entry_trim_start.get() or 0.0),
            "trim_end": float(self.entry_trim_end.get() or 10.0),
            "output_resolution": self.combo_res.get(),
            "encoder_codec": "hevc_nvenc" if "HEVC" in self.combo_codec.get() else "h264_nvenc",
            "bitrate_mbps": int(self.slider_bitrate.get()),
            "rtgi_intensity": float(self.slider_rtgi.get()),
            "ssr_intensity": float(self.slider_ssr.get()),
            "rtao_intensity": float(self.slider_rtao.get()),
            "clarity": float(self.slider_clarity.get()),
            "exposure": float(self.slider_exposure.get()),
            "contrast": float(self.slider_contrast.get()),
            "bloom_intensity": float(self.slider_bloom.get()),
            "film_grain": float(self.slider_grain.get()),
            "denoise": True,
            "use_aces": True,
        }
        return base_params

    def _generate_first_frame_preview(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return

        def task():
            cap = cv2.VideoCapture(self.video_path)
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            params = self._collect_params()

            if self.pipeline is None:
                self.pipeline = VideoPipeline()

            out_rgb, depth, normals = self.pipeline.process_single_frame(frame_rgb, params)
            self._display_image_on_screen(out_rgb)

        threading.Thread(target=task, daemon=True).start()

    def _generate_video_preview(self):
        """Renders a 3-5 second playable preview clip and stores frames in memory for instant playback."""
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Video", "Bitte wähle zuerst eine Videodatei aus!")
            return

        if self.is_rendering:
            return

        self.btn_gen_preview.configure(text="⏳ Rendere Video-Vorschau...", state="disabled")
        self.lbl_status.configure(text="Rendere interaktive Video-Vorschau auf RTX 3080 Ti...")
        self.progress_bar.set(0)

        def task():
            try:
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                start_sec = float(self.entry_trim_start.get() or 0.0) if self.sw_trim.get() else 0.0
                preview_duration = 3.5  # 3.5 seconds loop
                max_preview_frames = int(preview_duration * min(fps, 30.0))

                start_frame = int(start_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                params = self._collect_params()
                if self.pipeline is None:
                    self.pipeline = VideoPipeline()

                self.auto_optimizer.reset()
                frames_cache = []

                for i in range(max_preview_frames):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Downscale preview slightly for ultra-smooth 60fps playback
                    ph = 540
                    pw = int(frame_rgb.shape[1] * (ph / frame_rgb.shape[0]))
                    frame_small = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_AREA)

                    out_rgb, _, _ = self.pipeline.process_single_frame(frame_small, params)

                    frames_cache.append((frame_small, out_rgb))
                    prog = (i + 1) / max_preview_frames
                    self.after(0, lambda p=prog, idx=i+1: self._update_preview_progress(p, idx, max_preview_frames))

                cap.release()
                self.preview_frames_cache = frames_cache
                self.playback_idx = 0

                self.after(0, self._on_preview_ready)
            except Exception as e:
                self.after(0, lambda err=str(e): self.lbl_status.configure(text=f"Fehler bei Vorschau: {err}"))
            finally:
                self.after(0, lambda: self.btn_gen_preview.configure(text="⚡ Video-Vorschau generieren & abspielen", state="normal"))

        threading.Thread(target=task, daemon=True).start()

    def _update_preview_progress(self, prog, curr, total):
        self.progress_bar.set(prog)
        self.lbl_status.configure(text=f"Rendere Video-Vorschau: Frame {curr}/{total}...")

    def _on_preview_ready(self):
        self.lbl_status.configure(text="✅ Video-Vorschau fertig! Spielt in Schleife ab.")
        self.progress_bar.set(1.0)
        self.is_playing = True
        self.btn_play_pause.configure(text="⏸️ Pause")
        self._run_playback_loop()

    def _toggle_playback(self):
        if not self.preview_frames_cache:
            self._generate_video_preview()
            return
        self.is_playing = not self.is_playing
        self.btn_play_pause.configure(text="⏸️ Pause" if self.is_playing else "▶️ Play")
        if self.is_playing:
            self._run_playback_loop()

    def _run_playback_loop(self):
        if not self.is_playing or not self.preview_frames_cache:
            return

        orig_frame, remaster_frame = self.preview_frames_cache[self.playback_idx]

        # Apply View Mode
        if self.split_view_mode == "Remaster":
            display_img = remaster_frame
        elif self.split_view_mode == "Original":
            display_img = orig_frame
        else:
            # 50/50 Split Screen
            h, w = remaster_frame.shape[:2]
            mid = w // 2
            display_img = np.copy(remaster_frame)
            display_img[:, :mid] = orig_frame[:, :mid]
            # White divider line
            display_img[:, mid-2:mid+2] = [255, 255, 255]

        self._display_image_on_screen(display_img)

        # Update Timeline Slider & Time Label
        n = len(self.preview_frames_cache)
        prog_pct = (self.playback_idx / n) * 100.0 if n > 0 else 0
        self.slider_timeline.set(prog_pct)
        self.lbl_player_time.configure(text=f"Frame {self.playback_idx + 1} / {n}")

        self.playback_idx = (self.playback_idx + 1) % n
        self.after(33, self._run_playback_loop)  # ~30 FPS playback loop

    def _on_timeline_scrub(self, val):
        if not self.preview_frames_cache:
            return
        idx = int((float(val) / 100.0) * len(self.preview_frames_cache))
        self.playback_idx = max(0, min(len(self.preview_frames_cache) - 1, idx))
        orig_frame, remaster_frame = self.preview_frames_cache[self.playback_idx]
        self._display_image_on_screen(remaster_frame if self.split_view_mode != "Original" else orig_frame)

    def _on_view_mode_change(self, mode):
        self.split_view_mode = mode
        if not self.is_playing and self.preview_frames_cache:
            self._on_timeline_scrub(self.slider_timeline.get())

    def _switch_tab(self, tab):
        # Update tab button colors
        for b in [self.btn_tab_video, self.btn_tab_frame, self.btn_tab_diag]:
            b.configure(fg_color="#1e2630", text_color="#cbd5e1")

        if tab == "video":
            self.btn_tab_video.configure(fg_color="#76b900", text_color="#000000")
            if self.preview_frames_cache:
                self.is_playing = True
                self._run_playback_loop()
        elif tab == "frame":
            self.btn_tab_frame.configure(fg_color="#76b900", text_color="#000000")
            self.is_playing = False
            self.btn_play_pause.configure(text="▶️ Play")
            self._generate_first_frame_preview()
        elif tab == "diag":
            self.btn_tab_diag.configure(fg_color="#76b900", text_color="#000000")
            self.is_playing = False
            self._show_diagnostics_preview()

    def _show_diagnostics_preview(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return

        def task():
            cap = cv2.VideoCapture(self.video_path)
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.pipeline is None:
                self.pipeline = VideoPipeline()

            params = self._collect_params()
            out_rgb, depth, normals = self.pipeline.process_single_frame(frame_rgb, params)

            # Combine depth and normals side by side
            h, w = depth.shape[:2]
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = depth
            combined[:, w:] = normals
            self._display_image_on_screen(combined)

        threading.Thread(target=task, daemon=True).start()

    def _display_image_on_screen(self, img_rgb):
        cw = self.canvas_container.winfo_width() or 900
        ch = self.canvas_container.winfo_height() or 600

        # Fit image while keeping aspect ratio
        ih, iw = img_rgb.shape[:2]
        scale = min((cw - 20) / iw, (ch - 20) / ih, 1.0)
        target_w = max(1, int(iw * scale))
        target_h = max(1, int(ih * scale))

        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

        self.lbl_screen.configure(image=ctk_img, text="")
        self.lbl_screen.image = ctk_img

    def _start_full_render(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Video", "Bitte wähle zuerst ein Video aus!")
            return

        if self.is_rendering:
            return

        self.is_rendering = True
        self.is_playing = False
        self.btn_render_full.configure(state="disabled", text="⏳ Video wird gerendert...")

        output_dir = os.path.join(os.path.expanduser("~"), "Videos", "Luxanix_Renders")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_file = os.path.join(output_dir, f"{base_name}_Luxanix_RTX.mp4")

        params = self._collect_params()

        def progress_cb(curr, total, fps, eta, elapsed):
            prog = curr / total if total > 0 else 0.0
            pct = prog * 100.0
            m_eta, s_eta = int(eta // 60), int(eta % 60)
            m_el, s_el = int(elapsed // 60), int(elapsed % 60)

            self.after(0, lambda: self._update_render_ui(prog, pct, curr, total, fps, m_eta, s_eta, m_el, s_el))

        def render_thread():
            try:
                if self.pipeline is None:
                    self.pipeline = VideoPipeline()

                rendered_path = self.pipeline.process_video(
                    input_path=self.video_path,
                    output_path=out_file,
                    params=params,
                    progress_callback=progress_cb
                )

                self.after(0, lambda: self._on_render_complete(rendered_path))
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_render_error(err))

        threading.Thread(target=render_thread, daemon=True).start()

    def _update_render_ui(self, prog, pct, curr, total, fps, m_eta, s_eta, m_el, s_el):
        self.progress_bar.set(prog)
        self.lbl_status.configure(
            text=f"Rendere: Frame {curr}/{total} ({pct:.1f}%) | Geschwindigkeit: {fps:.1f} FPS | Verstrichen: {m_el:02d}:{s_el:02d}"
        )
        self.lbl_eta.configure(text=f"Restzeit: {m_eta:02d}:{s_eta:02d} Min")

    def _on_render_complete(self, output_path):
        self.is_rendering = False
        self.progress_bar.set(1.0)
        self.btn_render_full.configure(state="normal", text="🚀 Vollständiges Video Rendern & Speichern")
        self.lbl_status.configure(text=f"✅ Fertig! Gespeichert in: {output_path}")
        self.lbl_eta.configure(text="FERTIG")

        resp = messagebox.askyesno(
            "Render Erfolgreich!",
            f"Das Video wurde erfolgreich in 8K/RTX gerendert:\n\n{output_path}\n\nMöchtest du den Ordner im Explorer öffnen?"
        )
        if resp:
            os.system(f'explorer /select,"{output_path}"')

    def _on_render_error(self, err_msg):
        self.is_rendering = False
        self.btn_render_full.configure(state="normal", text="🚀 Vollständiges Video Rendern & Speichern")
        self.lbl_status.configure(text=f"Fehler: {err_msg}")
        messagebox.showerror("Fehler beim Rendern", f"Ein Fehler ist aufgetreten:\n{err_msg}")


if __name__ == "__main__":
    app = LuxanixDesktopApp()
    app.mainloop()
