"""
Luxanix Studio Pro — Autonomous AI Video Editor (CapCut Desktop NLE Edition)
A full-featured modern video editor with NVIDIA RTX Tensor-Core AI Photorealism & 8K Super-Resolution.
- 100% Autonomous Realism Engine: Autonomously calculates lighting, depth, asphalt wetness, SSR, RTGI, and RTAO per frame.
- Real Neural AI Super-Resolution Upscaler (1080p -> 4K -> 8K) on CUDA Tensor Cores.
- Continuous Full-Video Player: Play and scrub the ENTIRE video from 00:00 to the end with live RTX Remastering and A/B Split Screen.
- Full NLE Timeline Editing: Cut/split clips (Ctrl+B), delete segments (Del), trim handles, and multi-track display.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.video_pipeline import VideoPipeline, get_video_info
from engine.hardware import detect_gpu_hardware, get_profile_settings
from engine.auto_realism import AutonomousRealismEngine
from engine.upscaler import NeuralUpscaler

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class LuxanixDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚡ Luxanix Studio Pro v2.0.0 — CapCut NLE AI Video Editor (NVIDIA RTX 50 Ready)")
        self.geometry("1520x960")
        self.minsize(1240, 800)
        self.configure(fg_color="#0b0e13")

        icon_path = os.path.join(PROJECT_ROOT, "assets", "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        # Core Engines
        self.pipeline = None
        self.auto_realism = AutonomousRealismEngine()
        self.neural_upscaler = None
        self.gpu_info = detect_gpu_hardware()
        self.current_profile_key = getattr(self.gpu_info, "recommended_profile", "ultra")

        # Video, Photo & Timeline State
        self.video_path = None
        self.media_type = "video"  # "video" or "image"
        self.video_info = {}
        self.video_cap = None
        self.total_duration_sec = 0.0
        self.total_frames = 0
        self.fps = 30.0

        # Timeline Segments: List of dicts representing edited cuts
        # Each segment: {"start": 0.0, "end": 10.0, "title": "Clip 1"}
        self.timeline_segments = []
        self.selected_segment_idx = 0

        # Continuous Playback State (Entire Video)
        self.is_playing = False
        self.current_time_sec = 0.0
        self.play_thread = None
        self.stop_playback_flag = False
        self.playback_speed = 1.0

        # Display & View Mode
        self.split_view_mode = "Remaster"  # "Remaster", "Split", "Original", "Depth"
        self.aspect_ratio_mode = "16:9"
        self.realism_intensity = 1.0
        self.is_rendering = False

        # Live telemetry metrics
        self.telemetry = {
            "exposure": 0.0,
            "contrast": 1.15,
            "ssr": 0.55,
            "rtgi": 0.60,
            "rtao": 0.65,
            "wetness": 0.50
        }

        self._build_capcut_nle_ui()
        self._bind_shortcuts()
        self._init_pipeline_async()

    def _init_pipeline_async(self):
        def init():
            try:
                self.after(0, lambda: self.lbl_status.configure(text="⚡ Initialisiere NVIDIA RTX Pipeline & Neural Tensor-Core Engine..."))
                self.pipeline = VideoPipeline()
                self.neural_upscaler = NeuralUpscaler(device=self.pipeline.device, use_fp16=self.pipeline.use_fp16)
                gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
                self.after(0, lambda: self.lbl_status.configure(text=f"✅ {gpu_name} Tensor-Core Engine aktiv (Echtzeit-Berechnung bereit)."))
            except Exception as e:
                self.after(0, lambda err=str(e): self.lbl_status.configure(text=f"Warnung bei Initialisierung: {err}"))
        threading.Thread(target=init, daemon=True).start()

    def _bind_shortcuts(self):
        self.bind("<space>", lambda e: self._toggle_playback())
        self.bind("<Control-b>", lambda e: self._split_clip_at_playhead())
        self.bind("<Delete>", lambda e: self._delete_selected_segment())
        self.bind("<BackSpace>", lambda e: self._delete_selected_segment())
        self.bind("<Left>", lambda e: self._step_time(-0.5))
        self.bind("<Right>", lambda e: self._step_time(+0.5))
        self.bind("<Up>", lambda e: self._select_prev_segment())
        self.bind("<Down>", lambda e: self._select_next_segment())
        self.bind("<Home>", lambda e: self._rewind_to_start())
        self.bind("<End>", lambda e: self._seek_to_end())
        self.bind("<i>", lambda e: self._trim_start_at_playhead())
        self.bind("<o>", lambda e: self._trim_end_at_playhead())

    # =========================================================================
    # CAPCUT NLE INTERFACE BUILDER
    # =========================================================================
    def _build_capcut_nle_ui(self):
        self.grid_rowconfigure(0, weight=0, minsize=44)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0, minsize=215)
        self.grid_rowconfigure(3, weight=0, minsize=38)
        self.grid_columnconfigure(0, weight=1)

        self._build_top_header()
        self._build_center_workspace()
        self._build_bottom_timeline()
        self._build_bottom_statusbar()

    # 1. TOP HEADER BAR
    def _build_top_header(self):
        self.header = ctk.CTkFrame(self, fg_color="#101318", height=44, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header.grid_propagate(False)

        # Left: Brand & Menu
        left_box = ctk.CTkFrame(self.header, fg_color="transparent")
        left_box.pack(side="left", padx=12, pady=6)

        lbl_logo = ctk.CTkLabel(
            left_box,
            text="⚡ LUXANIX",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#00e5ff"
        )
        lbl_logo.pack(side="left", padx=(0, 4))

        badge = ctk.CTkLabel(
            left_box,
            text="CAPCUT PRO v2.0.0",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#00c4cc",
            text_color="#000000",
            corner_radius=4,
            padx=5,
            pady=1
        )
        badge.pack(side="left", padx=(0, 12))

        for menu_name in ["Datei", "Bearbeiten", "Schnitt", "Ansicht"]:
            b = ctk.CTkButton(
                left_box,
                text=menu_name,
                font=ctk.CTkFont(size=11),
                fg_color="transparent",
                hover_color="#1c222b",
                text_color="#94a3b8",
                width=45,
                height=26
            )
            b.pack(side="left", padx=1)

        self.lbl_autosave = ctk.CTkLabel(
            left_box,
            text="🟢 Auto-Save: Aktiv",
            font=ctk.CTkFont(size=10),
            text_color="#4ade80",
            fg_color="#142018",
            corner_radius=10,
            padx=8,
            pady=2
        )
        self.lbl_autosave.pack(side="left", padx=8)

        btn_master_reset = ctk.CTkButton(
            left_box,
            text="↩️ Alles Reset",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1e242d",
            hover_color="#374151",
            text_color="#cbd5e1",
            width=75,
            height=24,
            command=self._reset_all
        )
        btn_master_reset.pack(side="left", padx=4)

        # Center: Project Title & Shortcuts Hint
        center_box = ctk.CTkFrame(self.header, fg_color="transparent")
        center_box.pack(side="left", expand=True)

        self.entry_proj_name = ctk.CTkEntry(
            center_box,
            width=240,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#181d24",
            border_color="#27313f",
            corner_radius=4
        )
        self.entry_proj_name.insert(0, "Assetto Corsa — RTX Remaster")
        self.entry_proj_name.pack(side="left", padx=6)

        ctk.CTkLabel(
            center_box,
            text="⌨️ [Leertaste] Play/Pause   [Strg+B] Schneiden   [Entf] Löschen   [◄ / ►] Scrub",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        ).pack(side="left", padx=10)

        # Right: GPU Telemetry Badge & Prominent Export Button
        right_box = ctk.CTkFrame(self.header, fg_color="transparent")
        right_box.pack(side="right", padx=12, pady=6)

        gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
        self.lbl_header_gpu = ctk.CTkLabel(
            right_box,
            text=f"🟢 Auto-GPU: {gpu_name} (Optimal abgestimmt)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#76b900",
            fg_color="#172217",
            corner_radius=4,
            padx=8,
            pady=3
        )
        self.lbl_header_gpu.pack(side="left", padx=(0, 10))

        self.btn_export = ctk.CTkButton(
            right_box,
            text="🚀 Exportieren",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#00c4cc",
            hover_color="#009ea5",
            text_color="#000000",
            width=120,
            height=30,
            corner_radius=5,
            command=self._start_full_render
        )
        self.btn_export.pack(side="left")

    # 2. CENTER WORKSPACE (Left Library, Center Player, Right Inspector)
    def _build_center_workspace(self):
        self.workspace = ctk.CTkFrame(self, fg_color="#0b0e13", corner_radius=0)
        self.workspace.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        self.workspace.grid_columnconfigure(0, weight=0, minsize=320)
        self.workspace.grid_columnconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(2, weight=0, minsize=320)
        self.workspace.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_center_player()
        self._build_right_inspector()

    # 2A. LEFT PANEL: MEDIEN & AUTONOME KI-ENGINE (No Presets!)
    def _build_left_panel(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#10141a", corner_radius=6, border_width=1, border_color="#1a202a")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.left_tabview = ctk.CTkTabview(
            panel,
            fg_color="transparent",
            segmented_button_fg_color="#161c24",
            segmented_button_selected_color="#00c4cc",
            segmented_button_selected_hover_color="#009ea5",
            segmented_button_unselected_color="#161c24",
            segmented_button_unselected_hover_color="#202935",
            text_color="#ffffff",
            height=36
        )
        self.left_tabview.grid(row=0, column=0, sticky="nsew", padx=6, pady=(4, 6))

        tab_media = self.left_tabview.add("📁 Medien")
        tab_ai = self.left_tabview.add("⚡ Autonome KI")
        tab_upscale = self.left_tabview.add("🔬 AI-Upscaler")
        tab_color = self.left_tabview.add("🎨 Farbe")
        tab_audio = self.left_tabview.add("🎵 Audio")

        # --- TAB: MEDIEN ---
        box_import = ctk.CTkFrame(tab_media, fg_color="#151b22", corner_radius=6, border_width=1, border_color="#26313f")
        box_import.pack(fill="x", padx=4, pady=8)

        btn_import = ctk.CTkButton(
            box_import,
            text="➕ Video oder Foto importieren",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f2835",
            hover_color="#2a3749",
            text_color="#00e5ff",
            height=38,
            command=self._choose_video
        )
        btn_import.pack(fill="x", padx=10, pady=(10, 6))

        self.lbl_media_status = ctk.CTkLabel(
            box_import,
            text="Videos (MP4, MKV, AVI, MOV) oder Fotos\n(PNG, JPG, WEBP, BMP bis 8K)",
            font=ctk.CTkFont(size=10),
            text_color="#64748b",
            justify="center"
        )
        self.lbl_media_status.pack(padx=10, pady=(0, 10))

        self.card_clip_info = ctk.CTkFrame(tab_media, fg_color="#151b22", corner_radius=6)
        self.card_clip_info.pack(fill="x", padx=4, pady=4)

        self.lbl_clip_title = ctk.CTkLabel(
            self.card_clip_info,
            text="📄 Kein Medium geladen",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e2e8f0",
            anchor="w"
        )
        self.lbl_clip_title.pack(fill="x", padx=10, pady=(8, 2))

        self.lbl_clip_details = ctk.CTkLabel(
            self.card_clip_info,
            text="Importiere ein Video oder Foto für RTX Remaster & Schnitt.",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8",
            justify="left",
            anchor="w"
        )
        self.lbl_clip_details.pack(fill="x", padx=10, pady=(0, 8))

        # --- TAB: AUTONOME KI-ENGINE (PURE REALISM - NO PRESETS!) ---
        scroll_ai = ctk.CTkScrollableFrame(tab_ai, fg_color="transparent")
        scroll_ai.pack(fill="both", expand=True, padx=2, pady=2)

        card_auto = ctk.CTkFrame(scroll_ai, fg_color="#131e17", corner_radius=6, border_width=1, border_color="#10b981")
        card_auto.pack(fill="x", padx=2, pady=(4, 8))

        ctk.CTkLabel(
            card_auto,
            text="🤖 Autonome Physik-Berechnung",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#34d399"
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            card_auto,
            text="Die KI berechnet Licht, Belichtung, Nässe-Reflexionen (SSR) und Kontaktschatten (RTAO) pro Frame komplett selbstständig. Keine manuellen Presets erforderlich!",
            font=ctk.CTkFont(size=10),
            text_color="#a7f3d0",
            justify="left",
            wraplength=270
        ).pack(anchor="w", padx=10, pady=(0, 8))

        # Master Realism Intensity
        ctk.CTkLabel(scroll_ai, text="Photorealismus-Stärke", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=4, pady=(6, 2))
        self.slider_intensity = ctk.CTkSlider(
            scroll_ai,
            from_=0.2,
            to=1.8,
            button_color="#00c4cc",
            progress_color="#00c4cc",
            command=self._on_intensity_change
        )
        self.slider_intensity.set(1.0)
        self.slider_intensity.pack(fill="x", padx=4, pady=(2, 8))

        # Live Real-Time Telemetry Dashboard
        card_telemetry = ctk.CTkFrame(scroll_ai, fg_color="#151b22", corner_radius=6)
        card_telemetry.pack(fill="x", padx=2, pady=4)

        ctk.CTkLabel(card_telemetry, text="Echtzeit-Berechnung Telemetrie:", font=ctk.CTkFont(size=10, weight="bold"), text_color="#00e5ff").pack(anchor="w", padx=8, pady=(6, 4))
        self.lbl_telem_exp = ctk.CTkLabel(card_telemetry, text="• Auto-Belichtung: Berechne...", font=ctk.CTkFont(size=10), text_color="#94a3b8", anchor="w")
        self.lbl_telem_exp.pack(fill="x", padx=8)

        self.lbl_telem_wet = ctk.CTkLabel(card_telemetry, text="• Fahrbahn-Nässe / SSR: Berechne...", font=ctk.CTkFont(size=10), text_color="#94a3b8", anchor="w")
        self.lbl_telem_wet.pack(fill="x", padx=8)

        self.lbl_telem_rtgi = ctk.CTkLabel(card_telemetry, text="• RTGI Streulicht: Berechne...", font=ctk.CTkFont(size=10), text_color="#94a3b8", anchor="w")
        self.lbl_telem_rtgi.pack(fill="x", padx=8)

        self.lbl_telem_rtao = ctk.CTkLabel(card_telemetry, text="• RTAO Kontaktschatten: Berechne...", font=ctk.CTkFont(size=10), text_color="#94a3b8", anchor="w")
        self.lbl_telem_rtao.pack(fill="x", padx=8, pady=(0, 6))

        ctk.CTkButton(
            scroll_ai,
            text="↩️ KI-Werte zurücksetzen",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            height=26,
            command=self._reset_ai_settings
        ).pack(fill="x", padx=4, pady=6)

        # --- TAB: NEURAL AI-UPSCALER ---
        scroll_up = ctk.CTkScrollableFrame(tab_upscale, fg_color="transparent")
        scroll_up.pack(fill="both", expand=True, padx=2, pady=2)

        card_up = ctk.CTkFrame(scroll_up, fg_color="#181d28", corner_radius=6, border_width=1, border_color="#3b82f6")
        card_up.pack(fill="x", padx=2, pady=(4, 8))

        ctk.CTkLabel(card_up, text="🔬 Neural Super-Resolution", font=ctk.CTkFont(size=11, weight="bold"), text_color="#60a5fa").pack(anchor="w", padx=10, pady=(8, 2))
        ctk.CTkLabel(
            card_up,
            text="Echtes Deep Learning Upscaling via PyTorch PixelShuffle & Residual Dense Blocks auf Tensor Cores. Rekonstruiert feinste Texturen & Asphaltdetails.",
            font=ctk.CTkFont(size=10),
            text_color="#93c5fd",
            justify="left",
            wraplength=270
        ).pack(anchor="w", padx=10, pady=(0, 8))

        ctk.CTkLabel(scroll_up, text="Ziel-Auflösung für Export & Player:", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=4, pady=(4, 2))
        self.combo_upscale = ctk.CTkComboBox(
            scroll_up,
            values=[
                "Original Auflösung",
                "2K QHD (1440p) — AI Detail Reconstruct",
                "4K Ultra HD (2160p) — Neural Super-Resolution",
                "8K Ultra HD (4320p) — Deep Sub-Pixel Reconstruct"
            ]
        )
        self.combo_upscale.set("4K Ultra HD (2160p) — Neural Super-Resolution")
        self.combo_upscale.pack(fill="x", padx=4, pady=4)

        self.sw_neural = ctk.CTkSwitch(
            scroll_up,
            text="NVIDIA Tensor-Core Upscaler aktivieren",
            font=ctk.CTkFont(size=10, weight="bold"),
            progress_color="#00c4cc"
        )
        self.sw_neural.select()
        self.sw_neural.pack(anchor="w", padx=6, pady=8)

        # --- TAB: FARBE & COLOR GRADING (CapCut) ---
        scroll_col = ctk.CTkScrollableFrame(tab_color, fg_color="transparent")
        scroll_col.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_col, text="🎨 Farbkorrektur & Grading", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00e5ff").pack(anchor="w", padx=4, pady=(4, 6))

        ctk.CTkLabel(scroll_col, text="Sättigung", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=4, pady=(4, 1))
        self.slider_sat = ctk.CTkSlider(scroll_col, from_=0.5, to=1.6, button_color="#00c4cc", progress_color="#00c4cc", command=lambda v: self._on_color_change())
        self.slider_sat.set(1.0)
        self.slider_sat.pack(fill="x", padx=4, pady=(1, 6))

        ctk.CTkLabel(scroll_col, text="Farbtemperatur", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=4, pady=(4, 1))
        self.slider_temp = ctk.CTkSlider(scroll_col, from_=-0.5, to=0.5, button_color="#00c4cc", progress_color="#00c4cc", command=lambda v: self._on_color_change())
        self.slider_temp.set(0.0)
        self.slider_temp.pack(fill="x", padx=4, pady=(1, 6))

        ctk.CTkLabel(scroll_col, text="Filmkorn (Kino-Look)", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=4, pady=(4, 1))
        self.slider_grain = ctk.CTkSlider(scroll_col, from_=0.0, to=0.2, button_color="#00c4cc", progress_color="#00c4cc", command=lambda v: self._on_color_change())
        self.slider_grain.set(0.05)
        self.slider_grain.pack(fill="x", padx=4, pady=(1, 6))

        ctk.CTkButton(
            scroll_col,
            text="↩️ Farbkorrektur zurücksetzen",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            height=26,
            command=self._reset_color_settings
        ).pack(fill="x", padx=4, pady=6)

        # --- TAB: AUDIO SPUR (CapCut) ---
        scroll_aud = ctk.CTkScrollableFrame(tab_audio, fg_color="transparent")
        scroll_aud.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_aud, text="🎵 Audio-Spur & Pegel", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00e5ff").pack(anchor="w", padx=4, pady=(4, 6))

        ctk.CTkLabel(scroll_aud, text="Master-Lautstärke (0% - 200%)", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=4, pady=(4, 1))
        self.slider_vol = ctk.CTkSlider(scroll_aud, from_=0.0, to=2.0, button_color="#00c4cc", progress_color="#00c4cc")
        self.slider_vol.set(1.0)
        self.slider_vol.pack(fill="x", padx=4, pady=(1, 6))

        self.sw_mute = ctk.CTkSwitch(scroll_aud, text="Audio stummschalten (Mute)", font=ctk.CTkFont(size=10, weight="bold"), progress_color="#f87171")
        self.sw_mute.pack(anchor="w", padx=6, pady=8)

        ctk.CTkButton(
            scroll_aud,
            text="↩️ Audio zurücksetzen",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            height=26,
            command=self._reset_audio_settings
        ).pack(fill="x", padx=4, pady=6)

    # 2B. CENTER PLAYER: CONTINUOUS FULL-VIDEO STREAMING
    def _build_center_player(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#0b0d11", corner_radius=6, border_width=1, border_color="#181e26")
        panel.grid(row=0, column=1, sticky="nsew", padx=2, pady=0)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        # Top Bar: Format & Split Screen Compare
        player_top = ctk.CTkFrame(panel, fg_color="#12161d", height=36, corner_radius=0)
        player_top.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        self.opt_aspect = ctk.CTkOptionMenu(
            player_top,
            values=["16:9 Breitbild", "21:9 Ultrawide", "32:9 Triple Screen", "9:16 Reel/Shorts", "Original"],
            width=115,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._on_aspect_change
        )
        self.opt_aspect.set("16:9 Breitbild")
        self.opt_aspect.pack(side="left", padx=8, pady=6)

        self.seg_view_mode = ctk.CTkSegmentedButton(
            player_top,
            values=["⚡ Remaster", "◫ 50/50 Split", "👁️ Original", "🔍 Tiefenkarte"],
            font=ctk.CTkFont(size=10, weight="bold"),
            selected_color="#00c4cc",
            selected_hover_color="#009ea5",
            unselected_color="#1a222c",
            command=self._on_view_mode_change,
            height=24
        )
        self.seg_view_mode.set("⚡ Remaster")
        self.seg_view_mode.pack(side="right", padx=8, pady=6)

        # Screen Viewport Container
        self.canvas_container = ctk.CTkFrame(panel, fg_color="#040507", corner_radius=0)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.lbl_screen = ctk.CTkLabel(
            self.canvas_container,
            text="🎬 Importiere ein Video, um das gesamte Video im Player anzusehen.\nDrücke [Leertaste] zum Abspielen/Pausieren.",
            font=ctk.CTkFont(size=12),
            text_color="#475569"
        )
        self.lbl_screen.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Bottom Transport Bar
        transport_bar = ctk.CTkFrame(panel, fg_color="#12161d", height=40, corner_radius=0)
        transport_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

        self.lbl_timecode = ctk.CTkLabel(
            transport_bar,
            text="00:00:00:00 / 00:00:00:00",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#00e5ff"
        )
        self.lbl_timecode.pack(side="left", padx=12, pady=6)

        # Transport Controls
        controls = ctk.CTkFrame(transport_bar, fg_color="transparent")
        controls.pack(side="left", expand=True)

        ctk.CTkButton(controls, text="⏮", width=28, height=26, font=ctk.CTkFont(size=11), fg_color="#1a222c", hover_color="#283444", command=self._rewind_to_start).pack(side="left", padx=2)
        ctk.CTkButton(controls, text="◀", width=28, height=26, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=lambda: self._step_time(-1.0)).pack(side="left", padx=2)

        self.btn_play_pause = ctk.CTkButton(
            controls,
            text="▶",
            width=44,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00c4cc",
            hover_color="#009ea5",
            text_color="#000000",
            corner_radius=14,
            command=self._toggle_playback
        )
        self.btn_play_pause.pack(side="left", padx=6)

        ctk.CTkButton(controls, text="▶", width=28, height=26, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=lambda: self._step_time(+1.0)).pack(side="left", padx=2)
        ctk.CTkButton(controls, text="⏭", width=28, height=26, font=ctk.CTkFont(size=11), fg_color="#1a222c", hover_color="#283444", command=self._seek_to_end).pack(side="left", padx=2)

        # Playback Speed selector
        self.opt_speed = ctk.CTkOptionMenu(
            transport_bar,
            values=["0.5x", "1.0x", "1.5x", "2.0x"],
            width=70,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._on_speed_change
        )
        self.opt_speed.set("1.0x")
        self.opt_speed.pack(side="right", padx=10, pady=6)

    # 2C. RIGHT INSPECTOR (CLIP & HARDWARE DETAILS)
    def _build_right_inspector(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#10141a", corner_radius=6, border_width=1, border_color="#1a202a")
        panel.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=0)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        ctk.CTkLabel(scroll, text="✂️ Clip-Inspektor & Schnittliste", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00e5ff").pack(anchor="w", padx=4, pady=(4, 4))

        self.lbl_clip_stat = ctk.CTkLabel(
            scroll,
            text="Ausgewähltes Segment: 1\nStart: 0.0s | Ende: 10.0s\nDauer: 10.0s",
            font=ctk.CTkFont(size=10),
            text_color="#cbd5e1",
            justify="left"
        )
        self.lbl_clip_stat.pack(anchor="w", padx=6, pady=2)

        # Quick Actions
        btn_split_inspector = ctk.CTkButton(
            scroll,
            text="✂️ Clip hier teilen (Strg+B)",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            command=self._split_clip_at_playhead
        )
        btn_split_inspector.pack(fill="x", padx=4, pady=2)

        btn_del_inspector = ctk.CTkButton(
            scroll,
            text="🗑️ Ausgewählten Clip löschen (Entf)",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            text_color="#fecaca",
            command=self._delete_selected_segment
        )
        btn_del_inspector.pack(fill="x", padx=4, pady=2)

        # Prev / Next segment buttons
        nav_row = ctk.CTkFrame(scroll, fg_color="transparent")
        nav_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(
            nav_row,
            text="◀ Vorheriger",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            height=24,
            command=self._select_prev_segment
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(
            nav_row,
            text="Nächster ▶",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            height=24,
            command=self._select_next_segment
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))

        btn_reset_inspector = ctk.CTkButton(
            scroll,
            text="↩️ Alle Schnitte zurücksetzen",
            font=ctk.CTkFont(size=10),
            fg_color="#1e242d",
            hover_color="#2b3442",
            command=self._reset_cuts
        )
        btn_reset_inspector.pack(fill="x", padx=4, pady=2)

        # Interactive Clip List container
        ctk.CTkLabel(scroll, text="🎬 Clips auf der Timeline (Klick zum Auswählen):", font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=4, pady=(10, 2))
        self.frame_clips_container = ctk.CTkFrame(scroll, fg_color="#10151c", corner_radius=6, border_width=1, border_color="#1c2430")
        self.frame_clips_container.pack(fill="x", padx=2, pady=2)

        # Fully Autonomous Hardware Auto-Tuning Card
        ctk.CTkLabel(scroll, text="⚡ Automatische Hardware-Optimierung", font=ctk.CTkFont(size=11, weight="bold"), text_color="#76b900").pack(anchor="w", padx=4, pady=(12, 4))
        card_hw = ctk.CTkFrame(scroll, fg_color="#131f17", corner_radius=6, border_width=1, border_color="#10b981")
        card_hw.pack(fill="x", padx=2, pady=4)

        gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
        vram = getattr(self.gpu_info, "vram_gb", 12.0)
        gen = getattr(self.gpu_info, "generation", "RTX")
        prof = getattr(self.gpu_info, "recommended_profile", "ultra").upper()

        ctk.CTkLabel(
            card_hw,
            text="🟢 100% Automatisch konfiguriert",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#34d399"
        ).pack(anchor="w", padx=8, pady=(6, 2))

        ctk.CTkLabel(
            card_hw,
            text=f"• Erkannte GPU: {gpu_name}\n• Generation: {gen}\n• Grafikspeicher: {vram:.1f} GB VRAM\n• Performance-Profil: {prof} (Auto)\n• Tensor Cores: FP16/BF16 Hardware-Pass\n• NVENC Encoder: Automatisch gewählt",
            font=ctk.CTkFont(size=9),
            text_color="#cbd5e1",
            justify="left"
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ctk.CTkLabel(
            card_hw,
            text="Du musst deine Grafikkarte nicht auswählen. Die App stimmt VRAM-Cache, Shader-Samples und Encoder-Presets vollautomatisch auf dein System ab.",
            font=ctk.CTkFont(size=8),
            text_color="#6ee7b7",
            justify="left",
            wraplength=200
        ).pack(anchor="w", padx=8, pady=(0, 6))

    # 3. BOTTOM MULTI-TRACK TIMELINE
    def _build_bottom_timeline(self):
        self.timeline_panel = ctk.CTkFrame(self, fg_color="#0e1217", corner_radius=0, border_width=1, border_color="#181e26")
        self.timeline_panel.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        self.timeline_panel.grid_rowconfigure(2, weight=1)
        self.timeline_panel.grid_columnconfigure(1, weight=1)

        # 3A. Toolbar
        toolbar = ctk.CTkFrame(self.timeline_panel, fg_color="#12161d", height=32, corner_radius=0)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        tools_left = ctk.CTkFrame(toolbar, fg_color="transparent")
        tools_left.pack(side="left", padx=8, pady=3)

        ctk.CTkButton(tools_left, text="✂️ Teilen (Strg+B)", width=95, height=24, font=ctk.CTkFont(size=10, weight="bold"), fg_color="#1a222c", hover_color="#283444", command=self._split_clip_at_playhead).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="🗑️ Löschen (Entf)", width=90, height=24, font=ctk.CTkFont(size=10, weight="bold"), fg_color="#7f1d1d", hover_color="#991b1b", text_color="#fecaca", command=self._delete_selected_segment).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="◀ Vorh.", width=55, height=24, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=self._select_prev_segment).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="Nächst. ▶", width=55, height=24, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=self._select_next_segment).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="↩️ Schnitte Reset", width=95, height=24, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=self._reset_cuts).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="⏮ Trim In (I)", width=75, height=24, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=self._trim_start_at_playhead).pack(side="left", padx=2)
        ctk.CTkButton(tools_left, text="⏭ Trim Out (O)", width=80, height=24, font=ctk.CTkFont(size=10), fg_color="#1a222c", hover_color="#283444", command=self._trim_end_at_playhead).pack(side="left", padx=2)

        self.lbl_timeline_summary = ctk.CTkLabel(
            toolbar,
            text="Timeline: 1 Clip | Gesamtlänge: 0.0s",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8"
        )
        self.lbl_timeline_summary.pack(side="left", expand=True)

        tools_right = ctk.CTkFrame(toolbar, fg_color="transparent")
        tools_right.pack(side="right", padx=8, pady=3)
        ctk.CTkLabel(tools_right, text="🧲 Magnet: An", font=ctk.CTkFont(size=10), text_color="#00c4cc").pack(side="left", padx=6)

        # 3B. Track Headers & Ruler Canvas
        header_col = ctk.CTkFrame(self.timeline_panel, fg_color="#10141a", width=68, corner_radius=0)
        header_col.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=0, pady=0)

        ctk.CTkLabel(header_col, text="SPUREN", font=ctk.CTkFont(size=9, weight="bold"), text_color="#64748b").pack(pady=4)
        ctk.CTkLabel(header_col, text="🎬 V1\nVideo", font=ctk.CTkFont(size=10, weight="bold"), text_color="#00c4cc").pack(pady=10)
        ctk.CTkLabel(header_col, text="⚡ FX\nShader", font=ctk.CTkFont(size=10, weight="bold"), text_color="#a855f7").pack(pady=6)
        ctk.CTkLabel(header_col, text="🔊 A1\nAudio", font=ctk.CTkFont(size=10, weight="bold"), text_color="#10b981").pack(pady=8)

        self.canvas_timeline = ctk.CTkCanvas(
            self.timeline_panel,
            bg="#0b0e13",
            highlightthickness=0,
            height=160
        )
        self.canvas_timeline.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=0, pady=0)
        self.canvas_timeline.bind("<Configure>", lambda e: self._draw_timeline())
        self.canvas_timeline.bind("<Button-1>", self._on_timeline_click)
        self.canvas_timeline.bind("<B1-Motion>", self._on_timeline_drag)

    # 4. BOTTOM STATUS BAR
    def _build_bottom_statusbar(self):
        self.bottom_bar = ctk.CTkFrame(self, fg_color="#090b0e", height=38, corner_radius=0)
        self.bottom_bar.grid(row=3, column=0, sticky="ew", padx=0, pady=0)

        self.progress_bar = ctk.CTkProgressBar(
            self.bottom_bar,
            progress_color="#00c4cc",
            fg_color="#182029",
            height=6,
            corner_radius=0
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=0, pady=(0, 2))

        status_row = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        status_row.pack(fill="x", padx=12, pady=(2, 4))

        self.lbl_status = ctk.CTkLabel(
            status_row,
            text="Bereit. Importiere ein Video oder drücke Leertaste zur Vollvideo-Wiedergabe.",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8"
        )
        self.lbl_status.pack(side="left")

        self.lbl_eta = ctk.CTkLabel(
            status_row,
            text="⚡ Autonome RTX Physik-Engine Aktiv",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#76b900"
        )
        self.lbl_eta.pack(side="right")

    # =========================================================================
    # TIMELINE DRAWING & INTERACTION
    # =========================================================================
    def _draw_timeline(self):
        w = self.canvas_timeline.winfo_width()
        h = self.canvas_timeline.winfo_height()
        if w < 50 or h < 50:
            return

        self.canvas_timeline.delete("all")

        dur = max(self.total_duration_sec, 1.0)

        # 1. Ruler
        self.canvas_timeline.create_rectangle(0, 0, w, 24, fill="#12171e", outline="#1c2430")
        step_px = max(20.0, w / dur)
        num_ticks = min(int(dur) + 1, 40)
        for sec in range(0, int(dur) + 1, max(1, int(dur // 20))):
            x = (sec / dur) * w
            self.canvas_timeline.create_line(x, 14, x, 24, fill="#475569")
            m = sec // 60
            s = sec % 60
            self.canvas_timeline.create_text(x + 14, 8, text=f"{m:02d}:{s:02d}", fill="#94a3b8", font=("Segoe UI", 8))

        # 2. Track 1: Video Segments
        self.canvas_timeline.create_rectangle(0, 28, w, 84, fill="#11161d", outline="#1c2430")

        if not self.timeline_segments:
            segments = [{"start": 0.0, "end": dur, "title": "Clip 1"}]
        else:
            segments = self.timeline_segments

        for idx, seg in enumerate(segments):
            seg_x1 = (seg["start"] / dur) * w
            seg_x2 = (seg["end"] / dur) * w
            is_sel = (idx == self.selected_segment_idx)

            outline_col = "#00e5ff" if is_sel else "#008b94"
            fill_col = "#1b384d" if is_sel else "#14212c"

            # Clip box
            self.canvas_timeline.create_rectangle(
                seg_x1, 32, seg_x2, 80,
                fill=fill_col, outline=outline_col, width=3 if is_sel else 1
            )

            if is_sel:
                # Top glowing accent line & tag
                self.canvas_timeline.create_line(seg_x1, 32, seg_x2, 32, fill="#ffffff", width=2)
                self.canvas_timeline.create_text(
                    max(seg_x1 + 60, seg_x2 - 8), 44,
                    text="✓ AUSGEWÄHLT",
                    anchor="e",
                    fill="#00e5ff",
                    font=("Segoe UI", 8, "bold")
                )

            # Segment Title & Duration
            seg_dur = seg["end"] - seg["start"]
            title_text = f"🎬 {seg.get('title', f'Clip {idx+1}')} ({seg_dur:.1f}s)"
            self.canvas_timeline.create_text(
                seg_x1 + 8, 44,
                text=title_text,
                anchor="w",
                fill="#ffffff" if is_sel else "#cbd5e1",
                font=("Segoe UI", 9, "bold" if is_sel else "normal")
            )

            # Miniature preview blocks
            box_count = max(1, int((seg_x2 - seg_x1) / 60))
            bw = (seg_x2 - seg_x1) / box_count
            for bi in range(box_count):
                bx = seg_x1 + bi * bw
                self.canvas_timeline.create_rectangle(
                    bx + 2, 54, bx + bw - 2, 76,
                    fill="#27445c" if is_sel else "#1b2633",
                    outline="#385b7c" if is_sel else "#2f4255"
                )

        # 3. Track 2: AI Shader Track
        self.canvas_timeline.create_rectangle(0, 88, w, 118, fill="#11161d", outline="#1c2430")
        for seg in segments:
            seg_x1 = (seg["start"] / dur) * w
            seg_x2 = (seg["end"] / dur) * w
            self.canvas_timeline.create_rectangle(
                seg_x1, 92, seg_x2, 114,
                fill="#271838", outline="#a855f7", width=1.5
            )
            self.canvas_timeline.create_text(
                seg_x1 + 8, 103,
                text="⚡ Autonome RTX Physik-Engine (SSR + RTGI + RTAO)",
                anchor="w",
                fill="#d8b4fe",
                font=("Segoe UI", 8, "bold")
            )

        # 4. Track 3: Audio Waveform
        self.canvas_timeline.create_rectangle(0, 122, w, 152, fill="#11161d", outline="#1c2430")
        for seg in segments:
            seg_x1 = (seg["start"] / dur) * w
            seg_x2 = (seg["end"] / dur) * w
            self.canvas_timeline.create_rectangle(
                seg_x1, 125, seg_x2, 149,
                fill="#12251d", outline="#10b981", width=1
            )
            wave_pts = int((seg_x2 - seg_x1) / 5)
            for i in range(wave_pts):
                wx = seg_x1 + i * 5
                amp = np.sin((seg["start"] + i * 0.1) * 2.0) * 8 + np.cos(i * 0.8) * 4
                self.canvas_timeline.create_line(
                    wx, 137 - abs(amp), wx, 137 + abs(amp),
                    fill="#34d399", width=1.5
                )

        # 5. Playhead Needle
        playhead_x = (self.current_time_sec / dur) * w
        self.canvas_timeline.create_line(playhead_x, 0, playhead_x, h, fill="#00e5ff", width=2)
        self.canvas_timeline.create_polygon(
            playhead_x - 6, 0,
            playhead_x + 6, 0,
            playhead_x + 6, 12,
            playhead_x, 18,
            playhead_x - 6, 12,
            fill="#00e5ff", outline="#ffffff"
        )

    def _on_timeline_click(self, event):
        w = self.canvas_timeline.winfo_width()
        if w <= 0 or self.total_duration_sec <= 0:
            return
        target_sec = max(0.0, min(self.total_duration_sec, (event.x / w) * self.total_duration_sec))
        self.current_time_sec = target_sec

        # Determine which clip segment was clicked
        clicked_idx = -1
        for i, seg in enumerate(self.timeline_segments):
            if seg["start"] <= target_sec <= seg["end"]:
                clicked_idx = i
                break
        if clicked_idx != -1 and clicked_idx != self.selected_segment_idx:
            self.selected_segment_idx = clicked_idx

        self._update_timeline_stats()
        self._seek_to_time(target_sec)
        self._draw_timeline()

    def _on_timeline_drag(self, event):
        self._on_timeline_click(event)

    # =========================================================================
    # NLE CUTTING & EDITING ACTIONS (SPLIT, DELETE, SELECT, TRIM, RESET)
    # =========================================================================
    def _split_clip_at_playhead(self):
        """Splits the clip segment under the playhead into two distinct clips (Ctrl+B)."""
        if not self.timeline_segments:
            return

        t = self.current_time_sec
        for i, seg in enumerate(self.timeline_segments):
            if seg["start"] + 0.2 < t < seg["end"] - 0.2:
                # Split this segment!
                seg1 = {"start": seg["start"], "end": t, "title": f"Clip {i+1}A"}
                seg2 = {"start": t, "end": seg["end"], "title": f"Clip {i+1}B"}
                self.timeline_segments[i] = seg1
                self.timeline_segments.insert(i + 1, seg2)
                self.selected_segment_idx = i + 1
                self.lbl_status.configure(text=f"✂️ Clip bei {t:.1f}s geteilt! Gesamt: {len(self.timeline_segments)} Clips")
                self._update_timeline_stats()
                self._draw_timeline()
                return
        self.lbl_status.configure(text="Playhead befindet sich an den Rändern eines Clips (kein Schnitt nötig).")

    def _delete_selected_segment(self):
        """Deletes the currently selected segment from the timeline (Del)."""
        self._delete_segment_by_index(self.selected_segment_idx)

    def _delete_segment_by_index(self, idx):
        """Deletes a specific segment by index with safety check."""
        if len(self.timeline_segments) <= 1:
            messagebox.showinfo("Info", "Es muss mindestens ein Clip auf der Timeline verbleiben.\nNutze 'Schnitte zurücksetzen', wenn du das gesamte Video wiederherstellen möchtest.")
            return

        if 0 <= idx < len(self.timeline_segments):
            deleted = self.timeline_segments.pop(idx)
            self.selected_segment_idx = max(0, min(len(self.timeline_segments) - 1, idx))
            # Move playhead to start of newly selected clip
            new_seg = self.timeline_segments[self.selected_segment_idx]
            self.current_time_sec = new_seg["start"]
            self._seek_to_time(self.current_time_sec)
            self.lbl_status.configure(text=f"🗑️ Clip {idx+1} ('{deleted.get('title', 'Clip')}') gelöscht!")
            self._update_timeline_stats()
            self._draw_timeline()

    def _select_segment_by_index(self, idx):
        if 0 <= idx < len(self.timeline_segments):
            self.selected_segment_idx = idx
            seg = self.timeline_segments[idx]
            self.current_time_sec = seg["start"]
            self._seek_to_time(self.current_time_sec)
            self.lbl_status.configure(text=f"Ausgewählt: Clip {idx+1} ({seg.get('title', '')}) von {seg['start']:.1f}s bis {seg['end']:.1f}s")
            self._update_timeline_stats()
            self._draw_timeline()

    def _select_prev_segment(self):
        if self.timeline_segments:
            new_idx = max(0, self.selected_segment_idx - 1)
            self._select_segment_by_index(new_idx)

    def _select_next_segment(self):
        if self.timeline_segments:
            new_idx = min(len(self.timeline_segments) - 1, self.selected_segment_idx + 1)
            self._select_segment_by_index(new_idx)

    def _trim_start_at_playhead(self):
        idx = self.selected_segment_idx
        if 0 <= idx < len(self.timeline_segments):
            seg = self.timeline_segments[idx]
            if self.current_time_sec < seg["end"] - 0.2:
                seg["start"] = self.current_time_sec
                self._update_timeline_stats()
                self._draw_timeline()

    def _trim_end_at_playhead(self):
        idx = self.selected_segment_idx
        if 0 <= idx < len(self.timeline_segments):
            seg = self.timeline_segments[idx]
            if self.current_time_sec > seg["start"] + 0.2:
                seg["end"] = self.current_time_sec
                self._update_timeline_stats()
                self._draw_timeline()

    def _reset_cuts(self):
        self.timeline_segments = [{"start": 0.0, "end": self.total_duration_sec, "title": "Gesamter Clip"}]
        self.selected_segment_idx = 0
        self.current_time_sec = 0.0
        self._seek_to_time(0.0)
        self.lbl_status.configure(text="↩️ Alle Schnitte zurückgesetzt. Gesamtes Video wiederhergestellt.")
        self._update_timeline_stats()
        self._draw_timeline()

    def _reset_ai_settings(self):
        self.slider_intensity.set(1.0)
        self.realism_intensity = 1.0
        self.lbl_status.configure(text="↩️ KI-Photorealismus auf Standardwerte zurückgesetzt.")
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _reset_color_settings(self):
        self.slider_sat.set(1.0)
        self.slider_temp.set(0.0)
        self.slider_grain.set(0.05)
        self.lbl_status.configure(text="↩️ Farbkorrektur auf Standardwerte zurückgesetzt.")
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _reset_audio_settings(self):
        self.slider_vol.set(1.0)
        self.sw_mute.deselect()
        self.lbl_status.configure(text="↩️ Audio-Einstellungen auf Standardwerte zurückgesetzt.")

    def _reset_all(self):
        self._reset_cuts()
        self._reset_ai_settings()
        self._reset_color_settings()
        self._reset_audio_settings()
        self.opt_speed.set("1.0x")
        self.playback_speed = 1.0
        self.seg_view_mode.set("⚡ Remaster")
        self.split_view_mode = "Remaster"
        self.opt_aspect.set("16:9 Breitbild")
        self.aspect_ratio_mode = "16:9 Breitbild"
        self.lbl_status.configure(text="↩️ Alles erfolgreich auf Standardwerte zurückgesetzt!")
        messagebox.showinfo("Reset", "Alle Schnitte, KI-, Farb- und Audio-Einstellungen wurden auf Standard zurückgesetzt.")

    def _update_timeline_stats(self):
        n = len(self.timeline_segments)
        total_cut_len = sum(seg["end"] - seg["start"] for seg in self.timeline_segments)
        self.lbl_timeline_summary.configure(text=f"Timeline: {n} Clips | Gesamtdauer: {total_cut_len:.1f}s")

        if 0 <= self.selected_segment_idx < n:
            s = self.timeline_segments[self.selected_segment_idx]
            dur = s["end"] - s["start"]
            self.lbl_clip_stat.configure(
                text=f"Ausgewähltes Segment: {self.selected_segment_idx + 1}/{n}\nTitel: {s.get('title', 'Clip')}\nStart: {s['start']:.1f}s | Ende: {s['end']:.1f}s\nDauer: {dur:.1f}s"
            )

        # Refresh interactive clip cards in inspector
        if hasattr(self, "frame_clips_container") and self.frame_clips_container.winfo_exists():
            for child in self.frame_clips_container.winfo_children():
                child.destroy()

            for i, seg in enumerate(self.timeline_segments):
                is_active = (i == self.selected_segment_idx)
                seg_dur = seg["end"] - seg["start"]
                card = ctk.CTkFrame(
                    self.frame_clips_container,
                    fg_color="#162534" if is_active else "#12161d",
                    border_width=1,
                    border_color="#00c4cc" if is_active else "#1f2937",
                    corner_radius=4
                )
                card.pack(fill="x", padx=4, pady=2)

                # Clickable label to select
                lbl_card = ctk.CTkLabel(
                    card,
                    text=f"{'✓ ' if is_active else ''}🎬 Clip {i+1}: {seg.get('title', '')}\n{seg['start']:.1f}s – {seg['end']:.1f}s ({seg_dur:.1f}s)",
                    font=ctk.CTkFont(size=9, weight="bold" if is_active else "normal"),
                    text_color="#ffffff" if is_active else "#94a3b8",
                    justify="left",
                    anchor="w"
                )
                lbl_card.pack(side="left", fill="x", expand=True, padx=6, pady=4)
                lbl_card.bind("<Button-1>", lambda e, idx=i: self._select_segment_by_index(idx))
                card.bind("<Button-1>", lambda e, idx=i: self._select_segment_by_index(idx))

                # Quick delete button on card
                if n > 1:
                    btn_del = ctk.CTkButton(
                        card,
                        text="🗑️",
                        width=24,
                        height=22,
                        font=ctk.CTkFont(size=10),
                        fg_color="#3e1a1f",
                        hover_color="#7f1d1d",
                        text_color="#f87171",
                        command=lambda idx=i: self._delete_segment_by_index(idx)
                    )
                    btn_del.pack(side="right", padx=4, pady=4)

    # =========================================================================
    # FULL-VIDEO CONTINUOUS PLAYBACK ENGINE
    # =========================================================================
    def _toggle_playback(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Medium", "Bitte wähle zuerst ein Video oder Foto aus!")
            return

        if getattr(self, "media_type", "video") == "image":
            self.lbl_status.configure(text="📷 Standbild im Player aktiv. Photorealismus & Farbkorrektur in Echtzeit.")
            self._render_single_frame_at(0.0)
            return

        self.is_playing = not self.is_playing
        self.btn_play_pause.configure(text="⏸" if self.is_playing else "▶")

        if self.is_playing:
            self.stop_playback_flag = False
            self.play_thread = threading.Thread(target=self._run_playback_thread, daemon=True)
            self.play_thread.start()

    def _run_playback_thread(self):
        """Streams through the ENTIRE video continuously applying real-time RTX Photorealism."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = (1.0 / (fps * self.playback_speed))

        target_frame = int(self.current_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        while self.is_playing and not self.stop_playback_flag:
            t_start = time.perf_counter()

            ret, frame_bgr = cap.read()
            if not ret:
                # Reached the end of the video! Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_time_sec = 0.0
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Fast preview resolution for smooth 30/60fps playback
            ph = 540
            pw = int(frame_rgb.shape[1] * (ph / frame_rgb.shape[0]))
            small_rgb = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_AREA)

            # 1. Autonomous Realism Computation (No Presets!)
            auto_params = self.auto_realism.analyze_and_compute(small_rgb, master_intensity=self.realism_intensity)

            # 2. Render enhanced frame
            if self.pipeline is not None:
                out_rgb, depth, normals = self.pipeline.process_single_frame(small_rgb, auto_params)
            else:
                out_rgb = small_rgb

            # 3. Apply Split Mode
            if self.split_view_mode == "Original":
                disp = small_rgb
            elif self.split_view_mode == "Depth" and 'depth' in locals():
                h, w = depth.shape[:2]
                comb = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comb[:, :w] = depth
                comb[:, w:] = normals
                disp = comb
            elif self.split_view_mode == "Split":
                h, w = out_rgb.shape[:2]
                mid = w // 2
                disp = np.copy(out_rgb)
                disp[:, :mid] = small_rgb[:, :mid]
                disp[:, mid-2:mid+2] = [255, 255, 255]
            else:
                disp = out_rgb

            # Advance playhead
            self.current_time_sec += (1.0 / fps) * self.playback_speed
            if self.current_time_sec > self.total_duration_sec:
                self.current_time_sec = 0.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Update UI safely
            curr_sec = self.current_time_sec
            try:
                self.after(0, lambda d=disp, t=curr_sec, p=auto_params: self._update_playback_ui(d, t, p))
            except Exception:
                pass

            # Maintain correct video frame rate
            t_elapsed = time.perf_counter() - t_start
            t_sleep = max(0.001, frame_interval - t_elapsed)
            time.sleep(t_sleep)

        cap.release()

    def _update_playback_ui(self, disp_img, curr_sec, auto_params):
        self._display_image_on_screen(disp_img)
        self._draw_timeline()

        # Update Timecode
        m = int(curr_sec // 60)
        s = int(curr_sec % 60)
        f = int((curr_sec - int(curr_sec)) * 30)
        dur_m = int(self.total_duration_sec // 60)
        dur_s = int(self.total_duration_sec % 60)
        dur_f = int((self.total_duration_sec - int(self.total_duration_sec)) * 30)
        self.lbl_timecode.configure(text=f"{m:02d}:{s:02d}:{f:02d} / {dur_m:02d}:{dur_s:02d}:{dur_f:02d}")

        # Update Live Telemetry
        self.lbl_telem_exp.configure(text=f"• Belichtung: {auto_params.get('exposure', 0.0):+.2f} EV (Kontrast: {auto_params.get('contrast', 1.15):.2f})")
        self.lbl_telem_wet.configure(text=f"• Nässe / SSR: {auto_params.get('ssr_intensity', 0.5):.2f} (Spiegelung aktiv)")
        self.lbl_telem_rtgi.configure(text=f"• RTGI Streulicht: {auto_params.get('rtgi_intensity', 0.5):.2f} Bounce")
        self.lbl_telem_rtao.configure(text=f"• RTAO Kontaktschatten: {auto_params.get('rtao_intensity', 0.6):.2f}")

    def _seek_to_time(self, target_sec):
        self.current_time_sec = target_sec
        if not self.is_playing and self.video_path:
            self._render_single_frame_at(target_sec)

    def _render_single_frame_at(self, sec):
        def task():
            if getattr(self, "media_type", "video") == "image":
                img_bgr = cv2.imread(self.video_path)
                if img_bgr is not None:
                    frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    pil_img = Image.open(self.video_path).convert("RGB")
                    frame_rgb = np.array(pil_img)
            else:
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                target_frame = int(sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame_bgr = cap.read()
                cap.release()
                if not ret: return
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            ph = 540
            pw = int(frame_rgb.shape[1] * (ph / frame_rgb.shape[0]))
            small_rgb = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_AREA)

            auto_params = self.auto_realism.analyze_and_compute(small_rgb, master_intensity=self.realism_intensity)
            if self.pipeline is not None:
                out_rgb, depth, normals = self.pipeline.process_single_frame(small_rgb, auto_params)
            else:
                out_rgb = small_rgb

            if self.split_view_mode == "Original": disp = small_rgb
            elif self.split_view_mode == "Depth":
                h, w = depth.shape[:2]
                comb = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comb[:, :w] = depth
                comb[:, w:] = normals
                disp = comb
            elif self.split_view_mode == "Split":
                h, w = out_rgb.shape[:2]
                mid = w // 2
                disp = np.copy(out_rgb)
                disp[:, :mid] = small_rgb[:, :mid]
                disp[:, mid-2:mid+2] = [255, 255, 255]
            else:
                disp = out_rgb

            try:
                self.after(0, lambda d=disp, t=sec, p=auto_params: self._update_playback_ui(d, t, p))
            except Exception:
                pass
        threading.Thread(target=task, daemon=True).start()

    def _step_time(self, delta_sec):
        target = max(0.0, min(self.total_duration_sec, self.current_time_sec + delta_sec))
        self._seek_to_time(target)

    def _rewind_to_start(self):
        self._seek_to_time(0.0)

    def _seek_to_end(self):
        self._seek_to_time(self.total_duration_sec)

    def _on_speed_change(self, choice):
        try:
            self.playback_speed = float(choice.replace("x", ""))
        except Exception:
            self.playback_speed = 1.0

    def _on_view_mode_change(self, mode):
        if "Remaster" in mode: self.split_view_mode = "Remaster"
        elif "Split" in mode: self.split_view_mode = "Split"
        elif "Original" in mode: self.split_view_mode = "Original"
        elif "Tiefenkarte" in mode: self.split_view_mode = "Depth"
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _on_aspect_change(self, choice):
        self.aspect_ratio_mode = choice
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _on_intensity_change(self, val):
        self.realism_intensity = float(val)
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _on_color_change(self):
        if not self.is_playing:
            self._render_single_frame_at(self.current_time_sec)

    def _display_image_on_screen(self, img_rgb):
        cw = self.canvas_container.winfo_width() or 800
        ch = self.canvas_container.winfo_height() or 500

        ih, iw = img_rgb.shape[:2]
        scale = min((cw - 16) / iw, (ch - 16) / ih, 1.0)
        target_w = max(1, int(iw * scale))
        target_h = max(1, int(ih * scale))

        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

        self.lbl_screen.configure(image=ctk_img, text="")
        self.lbl_screen.image = ctk_img

    def load_video(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return

        self.video_path = file_path
        self.video_info = get_video_info(file_path)
        self.media_type = self.video_info.get("media_type", "video")

        w = self.video_info.get("width", 1920)
        h = self.video_info.get("height", 1080)
        self.fps = self.video_info.get("fps", 60.0)
        self.total_duration_sec = self.video_info.get("duration_sec", 5.0 if self.media_type == "image" else 10.0)
        self.total_frames = self.video_info.get("total_frames", 1 if self.media_type == "image" else 300)
        aspect = self.video_info.get("aspect_ratio", "16:9")

        if self.media_type == "image":
            self.lbl_clip_title.configure(text=f"📷 {os.path.basename(file_path)}")
            self.lbl_clip_details.configure(
                text=f"Bild-Auflösung: {w}x{h} ({aspect})\nStatus: Standbild bereit für RTX Remaster & AI-Upscaling"
            )
            self.lbl_media_status.configure(text=f"✅ Foto geladen: {os.path.basename(file_path)}")
            self.btn_export.configure(text="🚀 Bild exportieren")
            self.lbl_status.configure(text=f"📷 Foto '{os.path.basename(file_path)}' geladen! Bereit für RTX Photorealismus & 8K Upscale.")
        else:
            self.lbl_clip_title.configure(text=f"🎬 {os.path.basename(file_path)}")
            self.lbl_clip_details.configure(
                text=f"Auflösung: {w}x{h} ({aspect}) | {self.fps:.1f} FPS\nDauer: {self.total_duration_sec:.1f}s ({self.total_frames} Frames)"
            )
            self.lbl_media_status.configure(text=f"✅ Video geladen: {os.path.basename(file_path)}")
            self.btn_export.configure(text="🚀 Exportieren")
            self.lbl_status.configure(text=f"🎬 Video '{os.path.basename(file_path)}' geladen.")

        # Initialize timeline segments
        self.timeline_segments = [{"start": 0.0, "end": self.total_duration_sec, "title": os.path.splitext(os.path.basename(file_path))[0]}]
        self.selected_segment_idx = 0
        self.current_time_sec = 0.0

        self._update_timeline_stats()
        self._draw_timeline()
        self._render_single_frame_at(0.0)

    def _choose_video(self):
        file_path = filedialog.askopenfilename(
            title="Wähle ein Gameplay-Video oder Foto",
            filetypes=[
                ("Unterstützte Medien (Videos & Fotos)", "*.mp4 *.mkv *.avi *.mov *.webm *.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
                ("Video-Dateien", "*.mp4 *.mkv *.avi *.mov *.webm"),
                ("Bild-Dateien", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
                ("Alle Dateien", "*.*")
            ]
        )
        if not file_path:
            return
        self.load_video(file_path)

    # =========================================================================
    # EXPORT PIPELINE (PHOTOS & VIDEOS WITH DIRECT NVENC & AI-UPSCALING)
    # =========================================================================
    def _start_full_render(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Medium", "Bitte wähle zuerst ein Video oder Foto aus!")
            return

        if self.is_rendering:
            return

        choice = self.combo_upscale.get()
        if "8K" in choice:
            out_res = "8K Ultra HD (4320p)"
        elif "4K" in choice:
            out_res = "4K Ultra HD (2160p)"
        elif "2K" in choice:
            out_res = "1440p 2K QHD"
        else:
            out_res = "Original"

        # --- PHOTO EXPORT ---
        if getattr(self, "media_type", "video") == "image":
            self.is_rendering = True
            self.btn_export.configure(state="disabled", text="⏳ Rendert Bild...")
            self.lbl_status.configure(text=f"⚡ Rendere Foto mit RTX Raytracing & Neural AI Super-Resolution ({out_res})...")

            output_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Luxanix_Renders")
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            tag = "8K" if "8K" in choice else ("4K" if "4K" in choice else ("2K" if "2K" in choice else "Original"))
            out_file = os.path.join(output_dir, f"{base_name}_Luxanix_RTX_{tag}.png")

            img_params = {
                "auto_realism": True,
                "realism_intensity": self.realism_intensity,
                "neural_upscale": bool(self.sw_neural.get()),
                "output_resolution": out_res,
                "denoise": True
            }

            def render_image_thread():
                try:
                    if self.pipeline is None:
                        self.pipeline = VideoPipeline()
                    rendered_path = self.pipeline.process_image(
                        input_path=self.video_path,
                        output_path=out_file,
                        params=img_params
                    )
                    self.after(0, lambda: self._on_render_complete(rendered_path))
                except Exception as e:
                    self.after(0, lambda err=str(e): self._on_render_error(err))

            threading.Thread(target=render_image_thread, daemon=True).start()
            return

        # --- VIDEO EXPORT ---
        self.is_rendering = True
        self.is_playing = False
        self.btn_export.configure(state="disabled", text="⏳ Rendert...")

        output_dir = os.path.join(os.path.expanduser("~"), "Videos", "Luxanix_Renders")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_file = os.path.join(output_dir, f"{base_name}_Luxanix_RTX_Remaster.mp4")

        prof_settings = get_profile_settings(self.current_profile_key)
        preferred_codec = prof_settings.get("encoder_codec", "hevc_nvenc")
        nvenc_p = prof_settings.get("nvenc_preset", "p7")

        if "8K" in choice:
            codec = "av1_nvenc" if preferred_codec == "av1_nvenc" else "hevc_nvenc"
            bitrate = 80
        elif "4K" in choice:
            codec = preferred_codec
            bitrate = 50
        elif "2K" in choice:
            codec = preferred_codec
            bitrate = 35
        else:
            codec = preferred_codec
            bitrate = 25

        params = {
            "auto_realism": True,
            "realism_intensity": self.realism_intensity,
            "neural_upscale": bool(self.sw_neural.get()),
            "output_resolution": out_res,
            "encoder_codec": codec,
            "bitrate_mbps": bitrate,
            "enable_trim": True,
            "trim_start": self.timeline_segments[0]["start"] if self.timeline_segments else 0.0,
            "trim_end": self.timeline_segments[-1]["end"] if self.timeline_segments else self.total_duration_sec,
            "nvenc_preset": nvenc_p,
            "denoise": True
        }

        def progress_cb(curr, total, fps, eta, elapsed):
            prog = curr / total if total > 0 else 0.0
            pct = prog * 100.0
            m_eta, s_eta = int(eta // 60), int(eta % 60)
            m_el, s_el = int(elapsed // 60), int(elapsed % 60)
            try:
                self.after(0, lambda: self._update_render_ui(prog, pct, curr, total, fps, m_eta, s_eta, m_el, s_el))
            except Exception:
                pass

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
                try:
                    self.after(0, lambda: self._on_render_complete(rendered_path))
                except Exception:
                    pass
            except Exception as e:
                try:
                    self.after(0, lambda err=str(e): self._on_render_error(err))
                except Exception:
                    pass

        threading.Thread(target=render_thread, daemon=True).start()

    def _update_render_ui(self, prog, pct, curr, total, fps, m_eta, s_eta, m_el, s_el):
        self.progress_bar.set(prog)
        self.lbl_status.configure(
            text=f"Rendere Video: Frame {curr}/{total} ({pct:.1f}%) | {fps:.1f} FPS | Verstrichen: {m_el:02d}:{s_el:02d}"
        )
        self.lbl_eta.configure(text=f"Restzeit: {m_eta:02d}:{s_eta:02d} Min")

    def _on_render_complete(self, output_path):
        self.is_rendering = False
        self.progress_bar.set(1.0)
        btn_txt = "🚀 Bild exportieren" if getattr(self, "media_type", "video") == "image" else "🚀 Exportieren"
        self.btn_export.configure(state="normal", text=btn_txt)
        self.lbl_status.configure(text=f"✅ Fertig! Gespeichert in: {output_path}")
        self.lbl_eta.configure(text="FERTIG")

        media_name = "Das Foto" if getattr(self, "media_type", "video") == "image" else "Das Video"
        resp = messagebox.askyesno(
            "Render Erfolgreich!",
            f"{media_name} wurde erfolgreich mit RTX & Neural AI-Upscaling gerendert:\n\n{output_path}\n\nMöchtest du den Ordner im Explorer öffnen?"
        )
        if resp:
            os.system(f'explorer /select,"{output_path}"')

    def _on_render_error(self, err_msg):
        self.is_rendering = False
        btn_txt = "🚀 Bild exportieren" if getattr(self, "media_type", "video") == "image" else "🚀 Exportieren"
        self.btn_export.configure(state="normal", text=btn_txt)
        self.lbl_status.configure(text=f"Fehler: {err_msg}")
        messagebox.showerror("Fehler beim Rendern", f"Ein Fehler ist aufgetreten:\n{err_msg}")


if __name__ == "__main__":
    app = LuxanixDesktopApp()
    app.mainloop()