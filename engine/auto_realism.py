"""
Luxanix Autonomous Photorealism & Color Science AI Engine
=========================================================
Performs 100% autonomous real-time scene analysis and computes ALL visual
parameters automatically — no manual presets needed.

Master Intensity Slider Behavior:
  ■ 0.0 (Far Left):   100% Untouched RAW Original image (0% AI, 0% Raytracing, 0% Grading)
  ■ 1.0 (Middle):     Balanced Photorealism & Auto-Grading (Wet SSR, RTAO, Livery Pop)
  ■ 2.0 (Far Right):  Maximum Remaster (Hyper Wet Reflections, Deep Shadows, Vivid Colors)

Automatic Color Grading & Correction:
  ■ Scene Detection (Night, Low Light, Golden Hour, Overcast, Daylight, Bright Day, Pit Lane)
  ■ Auto White Balance (Gray-World Kelvin correction)
  ■ Color Temperature adjustment
  ■ Auto Saturation (boosts dull/muted footage noticeably so racing liveries pop)
  ■ Smart Vibrance (lifts midtones while protecting skin/reds)
  ■ Deep Contrast S-Curve (preserves rich blacks, crisps up highlights)
  ■ SSR / RTGI / RTAO / Bloom / Clarity
  ■ Zero digital film grain / noise
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, Optional


class AutonomousRealismEngine:
    def __init__(self, smoothing_alpha: float = 0.22):
        self.alpha = smoothing_alpha
        self.smoothed_params: Optional[Dict[str, float]] = None

    def reset(self):
        """Resets temporal smoothing history for immediate parameter response."""
        self.smoothed_params = None

    def _detect_scene_type(
        self,
        log_mean_lum: float,
        mean_lum: float,
        sky_blue_bias: float,
        sky_red_bias: float,
        mean_sat: float,
    ) -> str:
        """Classifies scene into a lighting category for targeted color grading."""
        if log_mean_lum < 0.04:
            return "night"
        elif log_mean_lum < 0.12:
            return "low_light"
        elif sky_red_bias > 0.06 and mean_lum < 0.45:
            return "golden_hour"
        elif sky_blue_bias > 0.08 and mean_sat < 0.20:
            return "overcast"
        elif sky_blue_bias > 0.05 and mean_sat > 0.15:
            return "daylight"
        elif log_mean_lum > 0.55:
            return "bright_day"
        else:
            return "indoor"

    def _compute_auto_wb(self, float_rgb: np.ndarray) -> float:
        """Estimates color temperature shift (-1.0 cool/blue … +1.0 warm/orange)."""
        mean_r = float(np.mean(float_rgb[:, :, 0]))
        mean_g = float(np.mean(float_rgb[:, :, 1]))
        mean_b = float(np.mean(float_rgb[:, :, 2]))
        mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-5

        r_bias = (mean_r - mean_all) / mean_all
        b_bias = (mean_b - mean_all) / mean_all
        wb_shift = float(np.clip(-(r_bias - b_bias) * 0.25, -0.30, 0.30))
        return wb_shift

    def analyze_and_compute(
        self,
        frame_rgb: np.ndarray,
        master_intensity: float = 1.0,
        depth_map: Optional[np.ndarray] = None,
        smooth: bool = True,
    ) -> Dict[str, Any]:
        """
        Autonomously analyzes an RGB frame and computes the full suite of
        photorealism, color correction, and cinematic grading parameters.

        master_intensity:
          0.0 = 100% Original (All effects = 0, exact original frame)
          1.0 = Balanced Photorealism & Color Grading
          2.0 = Maximum Photorealism & Hyper-realism
        """
        # Clamp master_intensity to [0.0, 2.0]
        intensity = float(np.clip(master_intensity, 0.0, 2.0))

        # Downscale for ultra-fast telemetry analysis (160×90)
        small_rgb = cv2.resize(frame_rgb, (160, 90), interpolation=cv2.INTER_AREA)
        float_rgb = small_rgb.astype(np.float32) / 255.0

        # Luminance (Rec. 709)
        lum = (0.2126 * float_rgb[:, :, 0]
               + 0.7152 * float_rgb[:, :, 1]
               + 0.0722 * float_rgb[:, :, 2])
        mean_lum = float(np.mean(lum))
        std_lum = float(np.std(lum))
        log_mean_lum = float(np.exp(np.mean(np.log(lum + 1e-4))))

        highlights = float(np.mean(lum > 0.82))

        # Road / Track Surface (lower 45% of frame)
        road_lum = lum[int(0.55 * 90):, :]
        road_std = float(np.std(road_lum))
        road_highlights = float(np.mean(road_lum > 0.75))

        # HSV Color Analysis
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(np.mean(sat))

        # Sky Region Analysis (upper 35%)
        sky = float_rgb[:int(0.35 * 90), :, :]
        sky_blue_bias = float(np.mean(sky[:, :, 2] - sky[:, :, 0]))
        sky_red_bias = float(np.mean(sky[:, :, 0] - sky[:, :, 2]))

        # Scene Classification
        scene = self._detect_scene_type(log_mean_lum, mean_lum, sky_blue_bias, sky_red_bias, mean_sat)

        # -------------------------------------------------------------
        # Physical Parameter Computations (Normalized for intensity = 1.0)
        # -------------------------------------------------------------

        # Exposure EV
        if log_mean_lum < 0.04:
            base_ev = float(np.clip(0.06 - log_mean_lum * 0.5, 0.0, 0.14))
        elif log_mean_lum > 0.60:
            base_ev = float(np.clip(-(log_mean_lum - 0.60) * 0.30, -0.16, 0.0))
        else:
            base_ev = float(np.clip(0.01 * (0.28 - log_mean_lum), -0.05, 0.05))

        # Contrast S-Curve (Noticeable punch at 1.0, high punch at 2.0, 1.0 at 0.0)
        # Flat overcast scenes get more contrast expansion, high dynamic scenes get gentle S-curve
        base_contrast_offset = 0.10 if std_lum < 0.14 else (0.05 if std_lum > 0.28 else 0.08)

        # White Balance & Temperature
        wb_correction = self._compute_auto_wb(float_rgb)
        scene_temp_bias = {
            "night":       -0.08,
            "low_light":   -0.04,
            "golden_hour": +0.18,
            "overcast":    -0.06,
            "daylight":    -0.03,
            "bright_day":  -0.05,
            "indoor":      +0.05,
        }.get(scene, 0.0)
        base_temp = float(np.clip(wb_correction + scene_temp_bias, -0.30, 0.30))

        # Auto Saturation & Vibrance (Rich, vibrant racing liveries!)
        # Gaming footage often suffers from dull, desaturated flat color palettes.
        # We boost saturation & smart vibrance significantly so sponsor decals, paint jobs,
        # brake calipers, curbs, and environments look rich, vibrant, and broadcast-grade.
        if mean_sat < 0.15:
            base_sat_offset = 0.36
            base_vib = 0.32
        elif mean_sat < 0.25:
            base_sat_offset = 0.28
            base_vib = 0.26
        else:
            base_sat_offset = 0.16
            base_vib = 0.15

        # SSR (Screen-Space Reflections on wet asphalt and car paint)
        wetness_score = float(np.clip(road_highlights * 4.5 + road_std * 1.5, 0.10, 0.90))
        base_ssr = float(np.clip(0.30 + wetness_score * 0.40, 0.25, 0.70))

        # RTGI (Indirect bounce light)
        base_rtgi = float(np.clip(0.25 + mean_sat * 0.30, 0.20, 0.50))

        # RTAO (Contact shadows under chassis and tires)
        base_rtao = 0.75

        # Bloom (Headlights & specular highlights)
        base_bloom = float(np.clip(0.10 + highlights * 1.4, 0.08, 0.30))

        # Detail Clarity (GPU Contrast-Adaptive Sharpening for razor-sharp logos & text)
        base_clarity = 0.75

        # Lens Vignette
        base_vignette = 0.15

        # -------------------------------------------------------------
        # SCALE ALL PARAMETERS WITH MASTER INTENSITY (0.0 to 2.0)
        # At intensity = 0.0: EVERYTHING reverts 100% to RAW ORIGINAL!
        # -------------------------------------------------------------
        final_exposure    = float(base_ev * intensity)
        final_contrast    = float(1.0 + base_contrast_offset * intensity)
        final_temperature = float(base_temp * intensity)
        final_saturation  = float(1.0 + base_sat_offset * intensity)
        final_vibrance    = float(base_vib * intensity)
        final_ssr         = float(base_ssr * intensity)
        final_rtgi        = float(base_rtgi * intensity)
        final_rtao        = float(base_rtao * intensity)
        final_bloom       = float(base_bloom * intensity)
        final_clarity     = float(base_clarity * intensity)
        final_vignette    = float(base_vignette * intensity)

        raw_computed = {
            "master_intensity":  intensity,
            "exposure":          final_exposure,
            "contrast":          final_contrast,
            "temperature":       final_temperature,
            "saturation":        final_saturation,
            "vibrance":          final_vibrance,
            "ssr_intensity":     final_ssr,
            "rtgi_intensity":    final_rtgi,
            "rtao_intensity":    final_rtao,
            "bloom_intensity":   final_bloom,
            "clarity":           final_clarity,
            "vignette":          final_vignette,
            "film_grain":        0.0,   # Broadcast-clean: ZERO NOISE
            "wetness_score":     wetness_score,
            "mean_luminance":    mean_lum,
            "scene_type":        scene,
        }

        # Temporal EMA smoothing for video playback (bypassed if smooth=False)
        if not smooth or self.smoothed_params is None:
            self.smoothed_params = {k: v for k, v in raw_computed.items() if isinstance(v, (int, float))}
            self.smoothed_params["scene_type"] = scene
        else:
            for k, val in raw_computed.items():
                if isinstance(val, (int, float)):
                    prev = self.smoothed_params.get(k, val)
                    self.smoothed_params[k] = (1.0 - self.alpha) * prev + self.alpha * val
            self.smoothed_params["scene_type"] = scene

        result = self.smoothed_params.copy()
        result["auto_preset"] = True
        return result
