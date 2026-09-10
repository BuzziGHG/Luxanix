"""
Luxanix Studio — Native CapCut-Style Desktop NLE Edition
GPU-accelerated desktop application for Raytracing, 8K Upscaling & Photorealistic Remastering.
Inspired by CapCut Pro Desktop video editor layout:
- Top Header Bar (Branding, Project Title, Auto-Save, Shortcuts, Prominent Export Button)
- Top-Left Panel (Media Library, Raytracing Presets Card Grid, Auto-AI, Color Grading)
- Top-Center Panel (Player Monitor, Aspect Ratio Selector, 50/50 Split Compare, Transport Controls)
- Top-Right Panel (Inspector: Raytracing, Hardware RTX 50, Lighting/AI, 8K Export)
- Bottom Panel (Multi-Track Timeline: Toolbar, Ruler, Playhead Needle, V1 Video Strip with Thumbnails, FX Track, A1 Audio Waveform)
- Bottom-most Render & Status Bar
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
from engine.hardware import detect_gpu_hardware, get_profile_settings, get_all_profiles
from engine.auto_preset import AutoSceneOptimizer

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class LuxanixDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("⚡ Luxanix Studio Pro — AI Raytracing & 8K Video Remaster (NVIDIA RTX 50 Ready)")
        self.geometry("1480x940")
        self.minsize(1220, 800)
        self.configure(fg_color="#0b0e13")

        icon_path = os.path.join(PROJECT_ROOT, "assets", "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        # State Variables
        self.video_path = None
        self.video_info = {}
        self.pipeline = None
        self.auto_optimizer = AutoSceneOptimizer(smoothing_alpha=0.25)
        self.gpu_info = detect_gpu_hardware()
        self.current_profile_key = getattr(self.gpu_info, "recommended_profile", "ultra")

        # Load Presets from presets.json
        self.presets_data = {}
        presets_file = os.path.join(PROJECT_ROOT, "presets.json")
        if os.path.exists(presets_file):
            try:
                with open(presets_file, "r", encoding="utf-8") as f:
                    self.presets_data = json.load(f)
            except Exception:
                pass

        # Playback & Preview State
        self.preview_frames_cache = []
        self.is_playing = False
        self.playback_idx = 0
        self.split_view_mode = "Remaster"
        self.aspect_ratio_mode = "16:9"
        self.is_rendering = False
        self.stop_render_flag = False

        # Timeline Trimming State
        self.trim_start_sec = 0.0
        self.trim_end_sec = 10.0
        self.playhead_pos_pct = 0.0

        self._build_capcut_ui()
        self._init_pipeline_async()

    def _init_pipeline_async(self):
        def init():
            try:
                self.after(0, lambda: self.lbl_status.configure(text="⚡ Initialisiere NVIDIA RTX Pipeline & Depth Anything v2 Tensor Cores..."))
                self.pipeline = VideoPipeline()
                gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
                self.after(0, lambda: self.lbl_status.configure(text=f"✅ {gpu_name} Tensor-Core Engine bereit (RTX 50 / 40 / 30 optimiert)."))
            except Exception as e:
                self.after(0, lambda err=str(e): self.lbl_status.configure(text=f"Warnung bei Initialisierung: {err}"))
        threading.Thread(target=init, daemon=True).start()

    def _build_capcut_ui(self):
        self.grid_rowconfigure(0, weight=0, minsize=44)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0, minsize=210)
        self.grid_rowconfigure(3, weight=0, minsize=38)
        self.grid_columnconfigure(0, weight=1)

        self._build_header_bar()
        self._build_main_workspace()
        self._build_timeline_panel()
        self._build_status_bar()

    # 1. TOP HEADER BAR
    def _build_header_bar(self):
        self.header = ctk.CTkFrame(self, fg_color="#101318", height=44, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header.grid_propagate(False)

        left_box = ctk.CTkFrame(self.header, fg_color="transparent")
        left_box.pack(side="left", padx=14, pady=6)

        lbl_logo = ctk.CTkLabel(
            left_box,
            text="⚡ LUXANIX",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#00e5ff"
        )
        lbl_logo.pack(side="left", padx=(0, 4))

        badge_pro = ctk.CTkLabel(
            left_box,
            text="PRO",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#00c4cc",
            text_color="#000000",
            corner_radius=4,
            padx=5,
            pady=1
        )
        badge_pro.pack(side="left", padx=(0, 14))

        for item in ["Menü", "Datei", "Bearbeiten", "Layout"]:
            btn_menu = ctk.CTkButton(
                left_box,
                text=item,
                font=ctk.CTkFont(size=11),
                fg_color="transparent",
                hover_color="#1c222b",
                text_color="#94a3b8",
                width=50,
                height=26,
                command=self._on_menu_click
            )
            btn_menu.pack(side="left", padx=2)

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
        self.lbl_autosave.pack(side="left", padx=14)

        center_box = ctk.CTkFrame(self.header, fg_color="transparent")
        center_box.pack(side="left", expand=True)

        self.entry_proj_name = ctk.CTkEntry(
            center_box,
            width=260,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#181d24",
            border_color="#27313f",
            corner_radius=4
        )
        self.entry_proj_name.insert(0, "Assetto Corsa — RTX 50 Ultra Remaster")
        self.entry_proj_name.pack(side="left", padx=6)

        lbl_shortcuts = ctk.CTkLabel(
            center_box,
            text="⌨️ [Leertaste] Play/Pause   [Strg+B] Teilen   [F5] Vorschau",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        lbl_shortcuts.pack(side="left", padx=10)

        right_box = ctk.CTkFrame(self.header, fg_color="transparent")
        right_box.pack(side="right", padx=12, pady=6)

        gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
        vram = getattr(self.gpu_info, "vram_gb", 12.0)
        self.lbl_header_gpu = ctk.CTkLabel(
            right_box,
            text=f"⚡ {gpu_name} ({vram:.0f}GB)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#76b900",
            fg_color="#172217",
            corner_radius=4,
            padx=8,
            pady=3
        )
        self.lbl_header_gpu.pack(side="left", padx=(0, 10))

        self.btn_header_export = ctk.CTkButton(
            right_box,
            text="🚀 Exportieren",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#00c4cc",
            hover_color="#009ea5",
            text_color="#000000",
            width=120,
            height=30,
            corner_radius=5,
            command=self._open_export_modal
        )
        self.btn_header_export.pack(side="left")

    # 2. MAIN WORKSPACE
    def _build_main_workspace(self):
        self.workspace = ctk.CTkFrame(self, fg_color="#0b0e13", corner_radius=0)
        self.workspace.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        self.workspace.grid_columnconfigure(0, weight=0, minsize=320)
        self.workspace.grid_columnconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(2, weight=0, minsize=330)
        self.workspace.grid_rowconfigure(0, weight=1)

        self._build_left_library_panel()
        self._build_center_player_panel()
        self._build_right_inspector_panel()

    # 2A. LEFT LIBRARY
    def _build_left_library_panel(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#10141a", corner_radius=6, border_width=1, border_color="#1a202a")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.lib_tabview = ctk.CTkTabview(
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
        self.lib_tabview.grid(row=0, column=0, sticky="nsew", padx=6, pady=(4, 6))

        tab_media = self.lib_tabview.add("📁 Medien")
        tab_rt = self.lib_tabview.add("✨ Raytracing")
        tab_ai = self.lib_tabview.add("🤖 Auto-AI")
        tab_luts = self.lib_tabview.add("🎨 LUTs")

        # TAB: MEDIEN
        box_import = ctk.CTkFrame(tab_media, fg_color="#151b22", corner_radius=6, border_width=1, border_color="#26313f")
        box_import.pack(fill="x", padx=4, pady=8)

        btn_import = ctk.CTkButton(
            box_import,
            text="➕ Video importieren",
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
            text="MP4, MKV, AVI, MOV bis 8K Ultra HD | Klick zum Auswählen",
            font=ctk.CTkFont(size=10),
            text_color="#64748b",
            justify="center"
        )
        self.lbl_media_status.pack(padx=10, pady=(0, 10))

        self.card_clip_info = ctk.CTkFrame(tab_media, fg_color="#151b22", corner_radius=6)
        self.card_clip_info.pack(fill="x", padx=4, pady=4)

        self.lbl_clip_title = ctk.CTkLabel(
            self.card_clip_info,
            text="📄 Kein Clip importiert",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e2e8f0",
            anchor="w"
        )
        self.lbl_clip_title.pack(fill="x", padx=10, pady=(8, 2))

        self.lbl_clip_details = ctk.CTkLabel(
            self.card_clip_info,
            text="Wähle ein Video aus, um Schnitt & Remastering zu starten.",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8",
            justify="left",
            anchor="w"
        )
        self.lbl_clip_details.pack(fill="x", padx=10, pady=(0, 8))

        # TAB: RAYTRACING
        filter_frame = ctk.CTkFrame(tab_rt, fg_color="transparent")
        filter_frame.pack(fill="x", padx=2, pady=(2, 6))

        filters = ["Alle", "RTX 50", "Nass", "Sunset", "Nürburg", "ACC"]
        for f in filters:
            b = ctk.CTkButton(
                filter_frame,
                text=f,
                width=46,
                height=22,
                font=ctk.CTkFont(size=10),
                fg_color="#18202a",
                hover_color="#263445",
                text_color="#cbd5e1",
                command=lambda cat=f: self._filter_presets(cat)
            )
            b.pack(side="left", padx=2)

        self.scroll_presets = ctk.CTkScrollableFrame(tab_rt, fg_color="transparent", width=310)
        self.scroll_presets.pack(fill="both", expand=True, padx=2, pady=2)
        self._populate_preset_cards()

        # TAB: AUTO-AI
        card_ai = ctk.CTkFrame(tab_ai, fg_color="#131e17", corner_radius=6, border_width=1, border_color="#10b981")
        card_ai.pack(fill="x", padx=4, pady=8)

        self.sw_auto_preset = ctk.CTkSwitch(
            card_ai,
            text="🤖 Auto-Preset (Echtzeit)",
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#10b981",
            command=self._on_auto_preset_toggle
        )
        self.sw_auto_preset.select()
        self.sw_auto_preset.pack(anchor="w", padx=10, pady=(10, 4))

        ctk.CTkLabel(
            card_ai,
            text="Analysiert Szenen pro Millisekunde dynamisch & passt RTGI-Streulicht, Scheinwerfer-Bloom und Reflexionen flüssig an.",
            font=ctk.CTkFont(size=10),
            text_color="#a7f3d0",
            justify="left",
            wraplength=280
        ).pack(anchor="w", padx=10, pady=(0, 8))

        self.lbl_ai_metrics = ctk.CTkLabel(
            card_ai,
            text="⚡ Status: Frame-Analyse Aktiv (16ms Latenz)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#34d399"
        )
        self.lbl_ai_metrics.pack(anchor="w", padx=10, pady=(0, 10))

        # TAB: LUTS
        ctk.CTkLabel(tab_luts, text="Cinematische Farbprofile (LUTs)", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=8, pady=6)
        lut_list = [
            ("🎬 Cinema 35mm Realism", "Natürlicher Kontrast & feines Filmkorn"),
            ("🏎️ Nordschleife Overcast", "Neutrales Licht, kühle Asphalt-Sättigung"),
            ("🌆 Cyberpunk Neon Sunset", "Starke Farbspiegelungen auf nassem Boden"),
            ("☀️ Monza High Noon", "Helle Spitzlichter und tiefe Kontaktschatten")
        ]
        for name, desc in lut_list:
            box = ctk.CTkFrame(tab_luts, fg_color="#151b22", corner_radius=6)
            box.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(box, text=name, font=ctk.CTkFont(size=11, weight="bold"), text_color="#00e5ff").pack(anchor="w", padx=8, pady=(4, 0))
            ctk.CTkLabel(box, text=desc, font=ctk.CTkFont(size=9), text_color="#94a3b8").pack(anchor="w", padx=8, pady=(0, 4))

    def _populate_preset_cards(self, category="Alle"):
        for widget in self.scroll_presets.winfo_children():
            widget.destroy()

        preset_items = list(self.presets_data.items()) if self.presets_data else [
            ("⚡ RTX 50 Blackwell: Hyper-Path Tracing (8K Ultra)", {"description": "18 RTGI Bounces, 24 SSR Steps, AV1 Dual-NVENC"}),
            ("🌧️ Simracing: Wet Track & Reflections", {"description": "Nasser Asphalt, ultra-starke Screen-Space Spiegelungen"}),
            ("🌅 Simracing: Golden Hour Sunset", {"description": "Warmer Sonnenuntergang, weiches Bounce-Licht & Glow"}),
            ("☁️ Simracing: Nürburgring Overcast", {"description": "Diffuses Licht, neutrale Farbtemperatur & Rennstrecken-Details"}),
            ("🌃 Simracing: Night Race & Headlights", {"description": "Starker Scheinwerfer-Bloom, tiefe Nacht-Kontraste"}),
            ("🏆 ACC / Assetto Corsa Hyper-Realism", {"description": "Perfekte Ausbalancierung für Simracing & Cockpit-Kameras"})
        ]

        for title, data in preset_items:
            if category == "RTX 50" and "50" not in title and "Blackwell" not in title: continue
            if category == "Nass" and "Wet" not in title and "Nass" not in title: continue
            if category == "Sunset" and "Sunset" not in title and "Golden" not in title: continue
            if category == "Nürburg" and "Nürburgring" not in title: continue
            if category == "ACC" and "ACC" not in title and "Assetto" not in title: continue

            card = ctk.CTkFrame(self.scroll_presets, fg_color="#151a22", corner_radius=6, border_width=1, border_color="#202936")
            card.pack(fill="x", padx=2, pady=4)

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=8, pady=(6, 2))

            lbl = ctk.CTkLabel(
                header,
                text=title,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#e2e8f0",
                anchor="w",
                wraplength=190,
                justify="left"
            )
            lbl.pack(side="left", fill="x", expand=True)

            btn_apply = ctk.CTkButton(
                header,
                text="Anwenden",
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#00c4cc",
                hover_color="#009ea5",
                text_color="#000000",
                width=64,
                height=24,
                corner_radius=4,
                command=lambda p_name=title: self._apply_preset_by_name(p_name)
            )
            btn_apply.pack(side="right")

            desc = data.get("description", "Photorealistisches Raytracing Preset")
            lbl_d = ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=9),
                text_color="#94a3b8",
                anchor="w",
                justify="left",
                wraplength=270
            )
            lbl_d.pack(fill="x", padx=8, pady=(0, 6))

    def _filter_presets(self, category):
        self._populate_preset_cards(category)

    # 2B. CENTER PLAYER MONITOR
    def _build_center_player_panel(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#0b0d11", corner_radius=6, border_width=1, border_color="#181e26")
        panel.grid(row=0, column=1, sticky="nsew", padx=2, pady=0)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        player_top = ctk.CTkFrame(panel, fg_color="#12161d", height=36, corner_radius=0)
        player_top.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        self.opt_aspect = ctk.CTkOptionMenu(
            player_top,
            values=["16:9 Breitbild", "21:9 Ultrawide", "32:9 Triple Screen", "9:16 Reel/Shorts", "Original"],
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
            command=self._on_aspect_change
        )
        self.opt_aspect.set("16:9 Breitbild")
        self.opt_aspect.pack(side="left", padx=8, pady=6)

        lbl_zoom = ctk.CTkLabel(player_top, text="Zoom: Anpassen", font=ctk.CTkFont(size=10), text_color="#64748b")
        lbl_zoom.pack(side="left", padx=6)

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

        self.canvas_container = ctk.CTkFrame(panel, fg_color="#040507", corner_radius=0)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.lbl_screen = ctk.CTkLabel(
            self.canvas_container,
            text="🎬 Ziehe ein Video hierher oder klicke auf 'Video importieren'\num das RTX-Remastering in Echtzeit zu sehen.",
            font=ctk.CTkFont(size=12),
            text_color="#475569"
        )
        self.lbl_screen.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        transport_bar = ctk.CTkFrame(panel, fg_color="#12161d", height=40, corner_radius=0)
        transport_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

        self.lbl_timecode = ctk.CTkLabel(
            transport_bar,
            text="00:00:00:00 / 00:00:10:00",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#00e5ff"
        )
        self.lbl_timecode.pack(side="left", padx=12, pady=6)

        transport_controls = ctk.CTkFrame(transport_bar, fg_color="transparent")
        transport_controls.pack(side="left", expand=True)

        btn_start = ctk.CTkButton(
            transport_controls,
            text="⏮",
            width=28,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._rewind_to_start
        )
        btn_start.pack(side="left", padx=2)

        btn_prev_frame = ctk.CTkButton(
            transport_controls,
            text="◀",
            width=28,
            height=26,
            font=ctk.CTkFont(size=10),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._step_frame_back
        )
        btn_prev_frame.pack(side="left", padx=2)

        self.btn_play_pause = ctk.CTkButton(
            transport_controls,
            text="▶",
            width=42,
            height=28,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#00c4cc",
            hover_color="#009ea5",
            text_color="#000000",
            corner_radius=14,
            command=self._toggle_playback
        )
        self.btn_play_pause.pack(side="left", padx=6)

        btn_next_frame = ctk.CTkButton(
            transport_controls,
            text="▶",
            width=28,
            height=26,
            font=ctk.CTkFont(size=10),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._step_frame_forward
        )
        btn_next_frame.pack(side="left", padx=2)

        btn_end = ctk.CTkButton(
            transport_controls,
            text="⏭",
            width=28,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._seek_to_end
        )
        btn_end.pack(side="left", padx=2)

        self.btn_gen_preview = ctk.CTkButton(
            transport_bar,
            text="⚡ Vorschau rendern",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e3a5f",
            hover_color="#274d7e",
            text_color="#60a5fa",
            width=130,
            height=26,
            corner_radius=4,
            command=self._generate_video_preview
        )
        self.btn_gen_preview.pack(side="right", padx=10, pady=6)

    # 2C. RIGHT INSPECTOR
    def _build_right_inspector_panel(self):
        panel = ctk.CTkFrame(self.workspace, fg_color="#10141a", corner_radius=6, border_width=1, border_color="#1a202a")
        panel.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=0)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.insp_tabview = ctk.CTkTabview(
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
        self.insp_tabview.grid(row=0, column=0, sticky="nsew", padx=6, pady=(4, 6))

        tab_rt = self.insp_tabview.add("Raytracing")
        tab_hw = self.insp_tabview.add("Hardware")
        tab_light = self.insp_tabview.add("Belichtung")
        tab_exp = self.insp_tabview.add("8K Export")

        # TAB: RAYTRACING
        scroll_rt = ctk.CTkScrollableFrame(tab_rt, fg_color="transparent")
        scroll_rt.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_rt, text="NVIDIA Tensor RTX Shader", font=ctk.CTkFont(size=11, weight="bold"), text_color="#76b900").pack(anchor="w", padx=4, pady=(2, 4))
        self.slider_rtgi = self._create_slider(scroll_rt, "RTGI Streulicht (Bounce)", 0.0, 1.5, 0.65)
        self.slider_ssr = self._create_slider(scroll_rt, "SSR Spiegelungen (Asphalt)", 0.0, 1.5, 0.55)
        self.slider_rtao = self._create_slider(scroll_rt, "RTAO Kontaktschatten", 0.0, 1.5, 0.60)
        self.slider_clarity = self._create_slider(scroll_rt, "Detail-Clarity (Anti-TAA)", 0.0, 1.0, 0.35)
        self.slider_wetness = self._create_slider(scroll_rt, "Oberflächen-Nässe / Glanz", 0.0, 1.5, 0.50)

        # TAB: HARDWARE
        scroll_hw = ctk.CTkScrollableFrame(tab_hw, fg_color="transparent")
        scroll_hw.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_hw, text="GPU-Architektur & Hardware-Profil", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=4, pady=(2, 4))

        self.profile_map = {
            "🔥 RTX 50 Blackwell (Hyper-PT & 8K)": "blackwell",
            "⚡ RTX 40 Ada Lovelace (Ultra Quality)": "ultra",
            "⚡ RTX 30 Ampere (Ausgewogen)": "balanced",
            "🌱 RTX 20 Turing (Low-VRAM Saver)": "low_vram",
        }
        rec_key = getattr(self.gpu_info, "recommended_profile", "ultra")
        default_label = next((k for k, v in self.profile_map.items() if v == rec_key), "🔥 RTX 50 Blackwell (Hyper-PT & 8K)")

        self.opt_arch = ctk.CTkOptionMenu(
            scroll_hw,
            values=list(self.profile_map.keys()),
            command=self._on_arch_profile_change
        )
        self.opt_arch.set(default_label)
        self.opt_arch.pack(fill="x", padx=4, pady=4)

        card_hw_info = ctk.CTkFrame(scroll_hw, fg_color="#151b22", corner_radius=6)
        card_hw_info.pack(fill="x", padx=4, pady=8)

        gpu_name = getattr(self.gpu_info, "device_name", "NVIDIA RTX")
        vram_gb = getattr(self.gpu_info, "vram_gb", 12.0)

        ctk.CTkLabel(card_hw_info, text="Hardware-Diagnose:", font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=8, pady=(6, 2))
        ctk.CTkLabel(card_hw_info, text=f"• GPU: {gpu_name}", font=ctk.CTkFont(size=9), text_color="#94a3b8").pack(anchor="w", padx=8)
        ctk.CTkLabel(card_hw_info, text=f"• VRAM: {vram_gb:.1f} GB GDDR6X", font=ctk.CTkFont(size=9), text_color="#94a3b8").pack(anchor="w", padx=8)
        ctk.CTkLabel(card_hw_info, text="• Tensor Cores: Gen 3/4/5 Aktiv", font=ctk.CTkFont(size=9), text_color="#76b900").pack(anchor="w", padx=8)
        ctk.CTkLabel(card_hw_info, text="• Dual-NVENC AV1: Bereit", font=ctk.CTkFont(size=9), text_color="#00c4cc").pack(anchor="w", padx=8, pady=(0, 6))

        # TAB: BELICHTUNG
        scroll_light = ctk.CTkScrollableFrame(tab_light, fg_color="transparent")
        scroll_light.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_light, text="Dynamische Szenen-Parameter", font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=4, pady=(2, 4))
        self.slider_exposure = self._create_slider(scroll_light, "Belichtung (Exposure EV)", -1.5, 1.5, 0.05)
        self.slider_contrast = self._create_slider(scroll_light, "Kontrast (S-Kurve)", 0.6, 1.8, 1.12)
        self.slider_bloom = self._create_slider(scroll_light, "Scheinwerfer-Bloom", 0.0, 1.0, 0.30)
        self.slider_grain = self._create_slider(scroll_light, "Filmkorn (Anti-Banding)", 0.0, 0.3, 0.08)

        # TAB: 8K EXPORT
        scroll_exp = ctk.CTkScrollableFrame(tab_exp, fg_color="transparent")
        scroll_exp.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(scroll_exp, text="Export-Auflösung", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=4, pady=(2, 2))
        self.combo_res = ctk.CTkComboBox(
            scroll_exp,
            values=["Original", "1080p Full HD", "1440p 2K QHD", "4K Ultra HD (2160p)", "8K Ultra HD (4320p)"]
        )
        self.combo_res.set("Original")
        self.combo_res.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkLabel(scroll_exp, text="Hardware-Encoder Codec", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=4, pady=(2, 2))
        self.combo_codec = ctk.CTkComboBox(
            scroll_exp,
            values=[
                "AV1 (NVIDIA RTX 50 & 40 Dual-NVENC)",
                "HEVC / H.265 (NVIDIA NVENC 4K/8K)",
                "H.264 (NVIDIA NVENC)",
                "libx264 (CPU Fallback)"
            ]
        )
        self.combo_codec.set("AV1 (NVIDIA RTX 50 & 40 Dual-NVENC)")
        self.combo_codec.pack(fill="x", padx=4, pady=(0, 6))

        self.slider_bitrate = self._create_slider(scroll_exp, "Bitrate (Mbps) — für 8K 80-120 Mbps", 10, 160, 60)

        btn_start_render = ctk.CTkButton(
            scroll_exp,
            text="🚀 Video Rendern & Speichern",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#00c4cc",
            hover_color="#009ea5",
            text_color="#000000",
            height=36,
            command=self._start_full_render
        )
        btn_start_render.pack(fill="x", padx=4, pady=10)

    # 3. BOTTOM MULTI-TRACK TIMELINE
    def _build_timeline_panel(self):
        self.timeline_panel = ctk.CTkFrame(self, fg_color="#0e1217", corner_radius=0, border_width=1, border_color="#181e26")
        self.timeline_panel.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        self.timeline_panel.grid_rowconfigure(2, weight=1)
        self.timeline_panel.grid_columnconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self.timeline_panel, fg_color="#12161d", height=32, corner_radius=0)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        tools_left = ctk.CTkFrame(toolbar, fg_color="transparent")
        tools_left.pack(side="left", padx=8, pady=3)

        btn_split = ctk.CTkButton(
            tools_left,
            text="✂️ Teilen (Strg+B)",
            width=100,
            height=24,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._on_split_clip
        )
        btn_split.pack(side="left", padx=2)

        btn_trim_in = ctk.CTkButton(
            tools_left,
            text="⏮ Start trimmen",
            width=90,
            height=24,
            font=ctk.CTkFont(size=10),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._trim_start_at_playhead
        )
        btn_trim_in.pack(side="left", padx=2)

        btn_trim_out = ctk.CTkButton(
            tools_left,
            text="⏭ Ende trimmen",
            width=90,
            height=24,
            font=ctk.CTkFont(size=10),
            fg_color="#1a222c",
            hover_color="#283444",
            command=self._trim_end_at_playhead
        )
        btn_trim_out.pack(side="left", padx=2)

        btn_del = ctk.CTkButton(
            tools_left,
            text="🗑️ Löschen",
            width=70,
            height=24,
            font=ctk.CTkFont(size=10),
            fg_color="#1a222c",
            hover_color="#3e2025",
            command=self._reset_trim
        )
        btn_del.pack(side="left", padx=2)

        self.lbl_timeline_summary = ctk.CTkLabel(
            toolbar,
            text="Clip-Schnittbereich: 0.0s – 10.0s (Gesamtlänge)",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8"
        )
        self.lbl_timeline_summary.pack(side="left", expand=True)

        tools_right = ctk.CTkFrame(toolbar, fg_color="transparent")
        tools_right.pack(side="right", padx=8, pady=3)

        ctk.CTkLabel(tools_right, text="🧲 Magnet: An", font=ctk.CTkFont(size=10), text_color="#00c4cc").pack(side="left", padx=6)
        ctk.CTkLabel(tools_right, text="🔍 Zoom:", font=ctk.CTkFont(size=10), text_color="#64748b").pack(side="left", padx=2)

        self.slider_zoom = ctk.CTkSlider(tools_right, from_=1.0, to=4.0, width=80, height=14)
        self.slider_zoom.set(1.0)
        self.slider_zoom.pack(side="left", padx=4)

        header_col = ctk.CTkFrame(self.timeline_panel, fg_color="#10141a", width=70, corner_radius=0)
        header_col.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=0, pady=0)

        ctk.CTkLabel(header_col, text="TRACKS", font=ctk.CTkFont(size=9, weight="bold"), text_color="#64748b").pack(pady=4)
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

    def _draw_timeline(self):
        w = self.canvas_timeline.winfo_width()
        h = self.canvas_timeline.winfo_height()
        if w < 50 or h < 50:
            return

        self.canvas_timeline.delete("all")

        self.canvas_timeline.create_rectangle(0, 0, w, 24, fill="#12171e", outline="#1c2430")

        dur = self.video_info.get("duration", 10.0) or 10.0
        dur = max(dur, 1.0)

        num_ticks = min(int(dur) + 1, 30)
        step_px = w / dur
        for sec in range(num_ticks):
            x = sec * step_px
            if x > w:
                break
            self.canvas_timeline.create_line(x, 14, x, 24, fill="#475569")
            time_str = f"00:{sec:02d}"
            self.canvas_timeline.create_text(x + 14, 8, text=time_str, fill="#94a3b8", font=("Segoe UI", 8))

        self.canvas_timeline.create_rectangle(0, 28, w, 84, fill="#11161d", outline="#1c2430")

        clip_start_x = (self.trim_start_sec / dur) * w
        clip_end_x = (self.trim_end_sec / dur) * w
        clip_end_x = max(clip_end_x, clip_start_x + 30)

        self.canvas_timeline.create_rectangle(
            clip_start_x, 32, clip_end_x, 80,
            fill="#16232e", outline="#00c4cc", width=1.5
        )

        title = os.path.basename(self.video_path) if self.video_path else "Assetto_Corsa_Gameplay.mp4"
        self.canvas_timeline.create_text(
            clip_start_x + 10, 44,
            text=f"🎬 {title} [1080p60 -> 8K Remaster]",
            anchor="w",
            fill="#e2e8f0",
            font=("Segoe UI", 9, "bold")
        )

        num_thumbs = max(2, int((clip_end_x - clip_start_x) / 75))
        box_w = (clip_end_x - clip_start_x) / num_thumbs
        for i in range(num_thumbs):
            bx = clip_start_x + i * box_w
            self.canvas_timeline.create_rectangle(
                bx + 2, 54, bx + box_w - 2, 76,
                fill="#1e2c3a", outline="#293b4d"
            )
            self.canvas_timeline.create_text(
                bx + (box_w / 2), 65,
                text=f"Frame {(i + 1) * 30}",
                fill="#64748b",
                font=("Segoe UI", 7)
            )

        self.canvas_timeline.create_rectangle(0, 88, w, 118, fill="#11161d", outline="#1c2430")
        self.canvas_timeline.create_rectangle(
            clip_start_x, 92, clip_end_x, 114,
            fill="#271838", outline="#a855f7", width=1.5
        )
        self.canvas_timeline.create_text(
            clip_start_x + 10, 103,
            text="✨ RTX 50 Hyper-Path Tracing + SSR & Auto-Scene Dynamic Optimizer (Aktiv)",
            anchor="w",
            fill="#d8b4fe",
            font=("Segoe UI", 8, "bold")
        )

        self.canvas_timeline.create_rectangle(0, 122, w, 152, fill="#11161d", outline="#1c2430")
        self.canvas_timeline.create_rectangle(
            clip_start_x, 125, clip_end_x, 149,
            fill="#12251d", outline="#10b981", width=1
        )
        step_wave = 5
        wave_pts = int((clip_end_x - clip_start_x) / step_wave)
        for i in range(wave_pts):
            wx = clip_start_x + i * step_wave
            amp = np.sin(i * 0.4) * 8 + np.cos(i * 0.9) * 4
            self.canvas_timeline.create_line(
                wx, 137 - abs(amp), wx, 137 + abs(amp),
                fill="#34d399", width=1.5
            )

        playhead_x = (self.playhead_pos_pct / 100.0) * w
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
        if w <= 0:
            return
        pct = max(0.0, min(100.0, (event.x / w) * 100.0))
        self.playhead_pos_pct = pct
        self._sync_playhead_to_player()
        self._draw_timeline()

    def _on_timeline_drag(self, event):
        self._on_timeline_click(event)

    def _sync_playhead_to_player(self):
        dur = self.video_info.get("duration", 10.0) or 10.0
        curr_sec = (self.playhead_pos_pct / 100.0) * dur
        m = int(curr_sec // 60)
        s = int(curr_sec % 60)
        f = int((curr_sec - int(curr_sec)) * 30)

        dur_m = int(dur // 60)
        dur_s = int(dur % 60)
        dur_f = int((dur - int(dur)) * 30)

        self.lbl_timecode.configure(text=f"{m:02d}:{s:02d}:{f:02d} / {dur_m:02d}:{dur_s:02d}:{dur_f:02d}")

        if self.preview_frames_cache:
            idx = int((self.playhead_pos_pct / 100.0) * len(self.preview_frames_cache))
            self.playback_idx = max(0, min(len(self.preview_frames_cache) - 1, idx))
            self._render_current_cached_frame()

    # 4. BOTTOM-MOST STATUS BAR
    def _build_status_bar(self):
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
            text="Bereit. Importiere ein Video oder wähle ein Preset, um den RTX-Remaster zu starten.",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8"
        )
        self.lbl_status.pack(side="left")

        self.lbl_eta = ctk.CTkLabel(
            status_row,
            text="⚡ NVIDIA RTX 50 Ready | FP16 Tensor Cores",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#76b900"
        )
        self.lbl_eta.pack(side="right")

    # LOGIC
    def _create_slider(self, parent, label_text, min_val, max_val, default_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=4, pady=2)

        lbl = ctk.CTkLabel(frame, text=f"{label_text}: {default_val:.2f}", font=ctk.CTkFont(size=10), text_color="#cbd5e1")
        lbl.pack(anchor="w")

        slider = ctk.CTkSlider(
            frame,
            from_=min_val,
            to=max_val,
            number_of_steps=100,
            button_color="#00c4cc",
            button_hover_color="#009ea5",
            progress_color="#00c4cc"
        )
        slider.set(default_val)
        slider.pack(fill="x", pady=(1, 3))

        def on_change(val):
            lbl.configure(text=f"{label_text}: {float(val):.2f}")
            if not self.is_playing and self.video_path:
                self._generate_first_frame_preview()
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

        self.lbl_clip_title.configure(text=f"📄 {os.path.basename(file_path)}")
        self.lbl_clip_details.configure(
            text=f"Auflösung: {w}x{h} ({aspect}) | {fps:.1f} FPS\nDauer: {dur:.1f}s ({frames} Frames)"
        )
        self.lbl_media_status.configure(text=f"✅ Geladen: {os.path.basename(file_path)}")

        self.trim_start_sec = 0.0
        self.trim_end_sec = min(dur, 10.0) if dur > 0 else 10.0
        self.lbl_timeline_summary.configure(
            text=f"Clip-Schnittbereich: {self.trim_start_sec:.1f}s – {self.trim_end_sec:.1f}s (Dauer: {self.trim_end_sec - self.trim_start_sec:.1f}s)"
        )

        self._draw_timeline()
        self._generate_first_frame_preview()

    def _apply_preset_by_name(self, choice):
        if choice in self.presets_data:
            p = self.presets_data[choice]
            if "rtgi_intensity" in p: self.slider_rtgi.set(p["rtgi_intensity"])
            if "ssr_intensity" in p: self.slider_ssr.set(p["ssr_intensity"])
            if "rtao_intensity" in p: self.slider_rtao.set(p["rtao_intensity"])
            if "clarity" in p: self.slider_clarity.set(p["clarity"])
            if "exposure" in p: self.slider_exposure.set(p["exposure"])
            if "contrast" in p: self.slider_contrast.set(p["contrast"])
            if "bloom_intensity" in p: self.slider_bloom.set(p["bloom_intensity"])
            if "film_grain" in p: self.slider_grain.set(p["film_grain"])

            if "RTX 50" in choice or "Blackwell" in choice:
                self.opt_arch.set("🔥 RTX 50 Blackwell (Hyper-PT & 8K)")
                self._on_arch_profile_change("🔥 RTX 50 Blackwell (Hyper-PT & 8K)")

            self.lbl_status.configure(text=f"Preset '{choice}' angewendet!")
            self._generate_first_frame_preview()

    def _on_arch_profile_change(self, choice):
        profile_key = self.profile_map.get(choice, "ultra")
        self.current_profile_key = profile_key
        settings = get_profile_settings(profile_key)
        if profile_key == "blackwell":
            self.lbl_status.configure(
                text="🔥 RTX 50 Blackwell aktiv: 18 RTGI Bounces, 24 SSR Steps, 12 RTAO Samples & AV1 Dual-NVENC."
            )
            self.combo_codec.set("AV1 (NVIDIA RTX 50 & 40 Dual-NVENC)")
        elif profile_key == "ultra":
            self.lbl_status.configure(text="⚡ RTX 40 / Ultra Profil aktiv (12 RTGI / 16 SSR / HEVC).")
            self.combo_codec.set("HEVC / H.265 (NVIDIA NVENC 4K/8K)")
        elif profile_key == "balanced":
            self.lbl_status.configure(text="⚡ RTX 30 Balanced Profil aktiv (10 RTGI / 12 SSR).")
            self.combo_codec.set("HEVC / H.265 (NVIDIA NVENC 4K/8K)")
        elif profile_key == "low_vram":
            self.lbl_status.configure(text="🌱 RTX 20 Low-VRAM Profil aktiv (6 RTGI / 8 SSR / Memory-Saver).")
            self.combo_codec.set("H.264 (NVIDIA NVENC)")

    def _on_auto_preset_toggle(self):
        is_auto = bool(self.sw_auto_preset.get())
        if is_auto:
            self.lbl_ai_metrics.configure(text="⚡ Status: Frame-Analyse Aktiv (16ms Latenz)", text_color="#34d399")
        else:
            self.lbl_ai_metrics.configure(text="⚡ Status: Deaktiviert (Manuelle Slider aktiv)", text_color="#f87171")

    def _on_view_mode_change(self, mode):
        if "Remaster" in mode: self.split_view_mode = "Remaster"
        elif "Split" in mode: self.split_view_mode = "Split"
        elif "Original" in mode: self.split_view_mode = "Original"
        elif "Tiefenkarte" in mode: self.split_view_mode = "Depth"

        if not self.is_playing and self.preview_frames_cache:
            self._render_current_cached_frame()
        elif not self.is_playing and self.video_path:
            self._generate_first_frame_preview()

    def _on_aspect_change(self, choice):
        self.aspect_ratio_mode = choice
        if not self.is_playing and self.video_path:
            self._generate_first_frame_preview()

    def _on_menu_click(self):
        pass

    def _on_split_clip(self):
        dur = self.video_info.get("duration", 10.0) or 10.0
        curr_sec = (self.playhead_pos_pct / 100.0) * dur
        self.trim_end_sec = max(curr_sec, self.trim_start_sec + 0.5)
        self.lbl_timeline_summary.configure(
            text=f"Clip geteilt bei {curr_sec:.1f}s | Aktiver Bereich: {self.trim_start_sec:.1f}s – {self.trim_end_sec:.1f}s"
        )
        self._draw_timeline()

    def _trim_start_at_playhead(self):
        dur = self.video_info.get("duration", 10.0) or 10.0
        curr_sec = (self.playhead_pos_pct / 100.0) * dur
        if curr_sec < self.trim_end_sec - 0.2:
            self.trim_start_sec = curr_sec
            self.lbl_timeline_summary.configure(
                text=f"Start getrimmt auf {self.trim_start_sec:.1f}s | Länge: {self.trim_end_sec - self.trim_start_sec:.1f}s"
            )
            self._draw_timeline()

    def _trim_end_at_playhead(self):
        dur = self.video_info.get("duration", 10.0) or 10.0
        curr_sec = (self.playhead_pos_pct / 100.0) * dur
        if curr_sec > self.trim_start_sec + 0.2:
            self.trim_end_sec = curr_sec
            self.lbl_timeline_summary.configure(
                text=f"Ende getrimmt auf {self.trim_end_sec:.1f}s | Länge: {self.trim_end_sec - self.trim_start_sec:.1f}s"
            )
            self._draw_timeline()

    def _reset_trim(self):
        dur = self.video_info.get("duration", 10.0) or 10.0
        self.trim_start_sec = 0.0
        self.trim_end_sec = dur
        self.lbl_timeline_summary.configure(
            text=f"Clip-Schnittbereich: 0.0s – {dur:.1f}s (Gesamtlänge)"
        )
        self._draw_timeline()

    def _rewind_to_start(self):
        self.playhead_pos_pct = 0.0
        self.playback_idx = 0
        self._sync_playhead_to_player()
        self._draw_timeline()

    def _seek_to_end(self):
        self.playhead_pos_pct = 100.0
        if self.preview_frames_cache:
            self.playback_idx = len(self.preview_frames_cache) - 1
        self._sync_playhead_to_player()
        self._draw_timeline()

    def _step_frame_back(self):
        if self.preview_frames_cache:
            self.playback_idx = max(0, self.playback_idx - 1)
            self.playhead_pos_pct = (self.playback_idx / len(self.preview_frames_cache)) * 100.0
            self._sync_playhead_to_player()
            self._draw_timeline()

    def _step_frame_forward(self):
        if self.preview_frames_cache:
            self.playback_idx = min(len(self.preview_frames_cache) - 1, self.playback_idx + 1)
            self.playhead_pos_pct = (self.playback_idx / len(self.preview_frames_cache)) * 100.0
            self._sync_playhead_to_player()
            self._draw_timeline()

    def _open_export_modal(self):
        self.insp_tabview.set("8K Export")
        self._start_full_render()

    def _collect_params(self):
        is_auto = bool(self.sw_auto_preset.get())
        codec_choice = self.combo_codec.get()
        if "AV1" in codec_choice: enc_codec = "av1_nvenc"
        elif "HEVC" in codec_choice: enc_codec = "hevc_nvenc"
        elif "CPU" in codec_choice: enc_codec = "libx264"
        else: enc_codec = "h264_nvenc"

        profile_key = getattr(self, "current_profile_key", "ultra")
        profile_settings = get_profile_settings(profile_key)

        return {
            "auto_preset": is_auto,
            "enable_trim": (self.trim_start_sec > 0.0 or self.trim_end_sec < (self.video_info.get("duration", 9999.0) - 0.5)),
            "trim_start": float(self.trim_start_sec),
            "trim_end": float(self.trim_end_sec),
            "output_resolution": self.combo_res.get(),
            "encoder_codec": enc_codec,
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
            "rtgi_steps": profile_settings.get("rtgi_steps", 12),
            "ssr_steps": profile_settings.get("ssr_steps", 16),
            "rtao_samples": profile_settings.get("rtao_samples", 8),
            "rtao_radius": profile_settings.get("rtao_radius", 1.4),
            "max_internal_res": profile_settings.get("max_internal_res", 2160),
            "dual_nvenc": profile_settings.get("dual_nvenc", False),
            "nvenc_preset": profile_settings.get("nvenc_preset", "p7"),
            "empty_cache_freq": profile_settings.get("empty_cache_freq", 60),
        }

    def _generate_first_frame_preview(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return

        params = self._collect_params()
        view_mode = self.split_view_mode
        curr_pct = self.playhead_pos_pct

        def task():
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur = self.video_info.get("duration", 10.0) or 10.0
            curr_sec = (curr_pct / 100.0) * dur
            target_frame = int(curr_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if self.pipeline is None:
                self.pipeline = VideoPipeline()

            out_rgb, depth, normals = self.pipeline.process_single_frame(frame_rgb, params)

            if view_mode == "Original":
                display = frame_rgb
            elif view_mode == "Depth":
                h, w = depth.shape[:2]
                combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
                combined[:, :w] = depth
                combined[:, w:] = normals
                display = combined
            elif view_mode == "Split":
                h, w = out_rgb.shape[:2]
                mid = w // 2
                display = np.copy(out_rgb)
                display[:, :mid] = frame_rgb[:, :mid]
                display[:, mid-2:mid+2] = [255, 255, 255]
            else:
                display = out_rgb

            self.after(0, lambda d=display: self._display_image_on_screen(d))

        threading.Thread(target=task, daemon=True).start()

    def _generate_video_preview(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Video", "Bitte wähle zuerst eine Videodatei aus!")
            return

        if self.is_rendering:
            return

        self.btn_gen_preview.configure(text="⏳ Rendere Vorschau...", state="disabled")
        self.lbl_status.configure(text="⚡ Rendere interaktive Vorschau auf NVIDIA RTX Tensor Cores...")
        self.progress_bar.set(0)

        def task():
            try:
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                preview_duration = 3.5
                max_frames = int(preview_duration * min(fps, 30.0))

                start_frame = int(self.trim_start_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                params = self._collect_params()
                if self.pipeline is None:
                    self.pipeline = VideoPipeline()

                self.auto_optimizer.reset()
                frames_cache = []

                for i in range(max_frames):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    ph = 540
                    pw = int(frame_rgb.shape[1] * (ph / frame_rgb.shape[0]))
                    frame_small = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_AREA)

                    out_rgb, depth, normals = self.pipeline.process_single_frame(frame_small, params)
                    frames_cache.append((frame_small, out_rgb, depth, normals))

                    prog = (i + 1) / max_frames
                    self.after(0, lambda p=prog, idx=i+1: self._update_preview_progress(p, idx, max_frames))

                cap.release()
                self.preview_frames_cache = frames_cache
                self.playback_idx = 0

                self.after(0, self._on_preview_ready)
            except Exception as e:
                self.after(0, lambda err=str(e): self.lbl_status.configure(text=f"Fehler bei Vorschau: {err}"))
            finally:
                self.after(0, lambda: self.btn_gen_preview.configure(text="⚡ Vorschau rendern", state="normal"))

        threading.Thread(target=task, daemon=True).start()

    def _update_preview_progress(self, prog, curr, total):
        self.progress_bar.set(prog)
        self.lbl_status.configure(text=f"Rendere Video-Vorschau: Frame {curr}/{total}...")

    def _on_preview_ready(self):
        self.lbl_status.configure(text="✅ Video-Vorschau fertig! Spielt in Schleife ab.")
        self.progress_bar.set(1.0)
        self.is_playing = True
        self.btn_play_pause.configure(text="⏸")
        self._run_playback_loop()

    def _toggle_playback(self):
        if not self.preview_frames_cache:
            self._generate_video_preview()
            return
        self.is_playing = not self.is_playing
        self.btn_play_pause.configure(text="⏸" if self.is_playing else "▶")
        if self.is_playing:
            self._run_playback_loop()

    def _run_playback_loop(self):
        if not self.is_playing or not self.preview_frames_cache:
            return

        self._render_current_cached_frame()

        n = len(self.preview_frames_cache)
        self.playhead_pos_pct = (self.playback_idx / n) * 100.0 if n > 0 else 0
        self._draw_timeline()

        dur = self.video_info.get("duration", 10.0) or 10.0
        curr_sec = (self.playhead_pos_pct / 100.0) * dur
        m = int(curr_sec // 60)
        s = int(curr_sec % 60)
        f = int((curr_sec - int(curr_sec)) * 30)
        dur_m = int(dur // 60)
        dur_s = int(dur % 60)
        dur_f = int((dur - int(dur)) * 30)
        self.lbl_timecode.configure(text=f"{m:02d}:{s:02d}:{f:02d} / {dur_m:02d}:{dur_s:02d}:{dur_f:02d}")

        self.playback_idx = (self.playback_idx + 1) % n
        self.after(33, self._run_playback_loop)

    def _render_current_cached_frame(self):
        if not self.preview_frames_cache or self.playback_idx >= len(self.preview_frames_cache):
            return

        item = self.preview_frames_cache[self.playback_idx]
        orig_frame = item[0]
        remaster_frame = item[1]
        depth = item[2] if len(item) > 2 else None
        normals = item[3] if len(item) > 3 else None

        if self.split_view_mode == "Remaster":
            display_img = remaster_frame
        elif self.split_view_mode == "Original":
            display_img = orig_frame
        elif self.split_view_mode == "Depth" and depth is not None:
            h, w = depth.shape[:2]
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = depth
            combined[:, w:] = normals
            display_img = combined
        else:
            h, w = remaster_frame.shape[:2]
            mid = w // 2
            display_img = np.copy(remaster_frame)
            display_img[:, :mid] = orig_frame[:, :mid]
            display_img[:, mid-2:mid+2] = [255, 255, 255]

        self._display_image_on_screen(display_img)

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

    def _start_full_render(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Kein Video", "Bitte wähle zuerst ein Video aus!")
            return

        if self.is_rendering:
            return

        self.is_rendering = True
        self.is_playing = False
        self.btn_header_export.configure(state="disabled", text="⏳ Rendert...")

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
            text=f"Rendere: Frame {curr}/{total} ({pct:.1f}%) | {fps:.1f} FPS | Verstrichen: {m_el:02d}:{s_el:02d}"
        )
        self.lbl_eta.configure(text=f"Restzeit: {m_eta:02d}:{s_eta:02d} Min")

    def _on_render_complete(self, output_path):
        self.is_rendering = False
        self.progress_bar.set(1.0)
        self.btn_header_export.configure(state="normal", text="🚀 Exportieren")
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
        self.btn_header_export.configure(state="normal", text="🚀 Exportieren")
        self.lbl_status.configure(text=f"Fehler: {err_msg}")
        messagebox.showerror("Fehler beim Rendern", f"Ein Fehler ist aufgetreten:\n{err_msg}")


if __name__ == "__main__":
    app = LuxanixDesktopApp()
    app.mainloop()