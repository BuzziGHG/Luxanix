"""
Luxanix Autonomous Photorealism & Color Science AI Engine
=========================================================
Performs 100% autonomous real-time scene analysis and computes ALL visual
parameters automatically — no manual presets needed.

Physics modules computed per-frame:
  ■ Exposure (EV) — atmosphere-preserving, maintains original mood
  ■ Contrast S-Curve — punchy cinematic contrast, protects deep blacks
  ■ Auto White Balance — Gray-World + scene classification (Kelvin shift)
  ■ Color Temperature — detects golden hour / overcast / night / indoor
  ■ Auto Saturation — scene-aware, boosts muted liveries, protects saturated colors
  ■ Smart Vibrance — selectively lifts dull midtones without oversaturating reds
  ■ Cinematic Vignette — automatic lens-falloff
  ■ SSR / RTGI / RTAO / Bloom / Clarity — RTX physics (grounded, additive)
  ■ Zero digital noise / grain — broadcast-clean output
  ■ Temporal EMA smoothing for fluid video, instantaneous response when paused/scrubbing
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, Optional


class AutonomousRealismEngine:
    def __init__(self, smoothing_alpha: float = 0.20):
        self.alpha = smoothing_alpha
        self.smoothed_params: Optional[Dict[str, float]] = None

    def reset(self):
        """Resets temporal smoothing history for immediate parameter response."""
        self.smoothed_params = None

    # ------------------------------------------------------------------
    # SCENE TYPE DETECTION
    # ------------------------------------------------------------------
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
            return "low_light"   # dusk, tunnel, overcast night
        elif sky_red_bias > 0.06 and mean_lum < 0.45:
            return "golden_hour"  # warm sunset / sunrise
        elif sky_blue_bias > 0.08 and mean_sat < 0.20:
            return "overcast"    # grey sky, muted colors
        elif sky_blue_bias > 0.05 and mean_sat > 0.15:
            return "daylight"    # blue sky, vibrant
        elif log_mean_lum > 0.55:
            return "bright_day"  # harsh noon sunlight
        else:
            return "indoor"      # pit lane, garage, neutral

    # ------------------------------------------------------------------
    # AUTO WHITE BALANCE
    # ------------------------------------------------------------------
    def _compute_auto_wb(self, float_rgb: np.ndarray) -> float:
        """Estimates color temperature shift (-1.0 cool/blue … +1.0 warm/orange)."""
        mean_r = float(np.mean(float_rgb[:, :, 0]))
        mean_g = float(np.mean(float_rgb[:, :, 1]))
        mean_b = float(np.mean(float_rgb[:, :, 2]))
        mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-5

        r_bias = (mean_r - mean_all) / mean_all
        b_bias = (mean_b - mean_all) / mean_all
        wb_shift = float(np.clip(-(r_bias - b_bias) * 0.20, -0.25, 0.25))
        return wb_shift

    # ------------------------------------------------------------------
    # MAIN COMPUTATION
    # ------------------------------------------------------------------
    def analyze_and_compute(
        self,
        frame_rgb: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        master_intensity: float = 1.0,
        smooth: bool = True,
    ) -> Dict[str, Any]:
        """
        Autonomously analyzes an RGB frame and computes the full suite of
        photorealism, color correction, and cinematic grading parameters.

        master_intensity scales all enhancement factors (0.2 = subtle, 1.0 = balanced, 1.8 = dramatic).
        smooth=False forces immediate non-smoothed values (crucial when user moves slider).
        """
        # Downscale for ultra-fast telemetry analysis (160×90)
        small_rgb = cv2.resize(frame_rgb, (160, 90), interpolation=cv2.INTER_AREA)
        float_rgb = small_rgb.astype(np.float32) / 255.0

        # Luminance (Rec. 709)
        lum = (0.2126 * float_rgb[:, :, 0]
               + 0.7152 * float_rgb[:, :, 1]
               + 0.0722 * float_rgb[:, :, 2])
        mean_lum  = float(np.mean(lum))
        std_lum   = float(np.std(lum))
        log_mean_lum = float(np.exp(np.mean(np.log(lum + 1e-4))))

        highlights = float(np.mean(lum > 0.82))
        shadows    = float(np.mean(lum < 0.08))

        # Road / Track Surface (lower 45% of frame)
        road_lum  = lum[int(0.55 * 90):, :]
        road_std  = float(np.std(road_lum))
        road_highlights = float(np.mean(road_lum > 0.75))

        # HSV Color Analysis
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(np.mean(sat))

        # Sky Region Analysis (upper 35%)
        sky = float_rgb[:int(0.35 * 90), :, :]
        sky_blue_bias = float(np.mean(sky[:, :, 2] - sky[:, :, 0]))
        sky_red_bias  = float(np.mean(sky[:, :, 0] - sky[:, :, 2]))

        # Scene Classification
        scene = self._detect_scene_type(log_mean_lum, mean_lum, sky_blue_bias, sky_red_bias, mean_sat)

        # -------------------------------------------------------------
        # A. Auto-Exposure EV (Conservative — preserves original mood)
        # -------------------------------------------------------------
        if log_mean_lum < 0.04:
            base_ev = float(np.clip(0.06 - log_mean_lum * 0.5, 0.0, 0.15))
        elif log_mean_lum > 0.60:
            base_ev = float(np.clip(-(log_mean_lum - 0.60) * 0.30, -0.18, 0.0))
        else:
            base_ev = float(np.clip(0.01 * (0.28 - log_mean_lum), -0.06, 0.06))
        auto_exposure = base_ev * min(master_intensity, 1.2)

        # -------------------------------------------------------------
        # B. Contrast S-Curve (Rich punch, preserves deep blacks)
        # -------------------------------------------------------------
        # Boost contrast with master_intensity for punchy broadcast look
        base_contrast = 1.06 if std_lum < 0.15 else (1.03 if std_lum > 0.28 else 1.05)
        auto_contrast = 1.0 + (base_contrast - 1.0) * master_intensity

        # -------------------------------------------------------------
        # C. Auto White Balance / Temperature
        # -------------------------------------------------------------
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
        auto_temperature = float(np.clip((wb_correction + scene_temp_bias) * master_intensity, -0.35, 0.35))

        # -------------------------------------------------------------
        # D. Saturation & Vibrance (Makes liveries and decals pop)
        # -------------------------------------------------------------
        if mean_sat < 0.15:
            base_sat = 1.10
            base_vib = 0.18
        elif mean_sat < 0.25:
            base_sat = 1.05
            base_vib = 0.12
        else:
            base_sat = 1.01
            base_vib = 0.05

        auto_saturation = float(np.clip(1.0 + (base_sat - 1.0) * master_intensity, 0.85, 1.35))
        auto_vibrance   = float(np.clip(base_vib * master_intensity, 0.0, 0.35))

        # -------------------------------------------------------------
        # E. SSR — Screen-Space Reflections (Wet track & car paint gloss)
        # -------------------------------------------------------------
        wetness_score = float(np.clip(road_highlights * 4.5 + road_std * 1.5, 0.10, 0.90))
        # Clear base reflection that scales directly with slider
        auto_ssr = float(np.clip((0.25 + wetness_score * 0.45) * master_intensity, 0.0, 1.2))

        # -------------------------------------------------------------
        # F. RTGI — Ray Traced Global Illumination (Indirect bounce light)
        # -------------------------------------------------------------
        auto_rtgi = float(np.clip((0.25 + mean_sat * 0.30) * master_intensity, 0.0, 0.9))

        # -------------------------------------------------------------
        # G. RTAO — Ray Traced Ambient Occlusion (Contact shadows)
        # -------------------------------------------------------------
        # Gives deep, firm ground contact shadows under car chassis & tires
        auto_rtao = float(np.clip(0.65 * master_intensity, 0.2, 1.2))

        # -------------------------------------------------------------
        # H. Bloom (Headlights & specular reflections)
        # -------------------------------------------------------------
        auto_bloom = float(np.clip((0.08 + highlights * 1.5) * master_intensity, 0.0, 0.40))

        # -------------------------------------------------------------
        # I. Detail Clarity (Anti-TAA Sharpening)
        # -------------------------------------------------------------
        auto_clarity = float(np.clip(0.18 * master_intensity, 0.0, 0.35))

        # -------------------------------------------------------------
        # J. Lens Vignette
        # -------------------------------------------------------------
        auto_vignette = float(np.clip(0.12 * master_intensity, 0.0, 0.30))

        raw_computed = {
            "exposure":          auto_exposure,
            "contrast":          auto_contrast,
            "temperature":       auto_temperature,
            "saturation":        auto_saturation,
            "vibrance":          auto_vibrance,
            "ssr_intensity":     auto_ssr,
            "rtgi_intensity":    auto_rtgi,
            "rtao_intensity":    auto_rtao,
            "bloom_intensity":   auto_bloom,
            "clarity":           auto_clarity,
            "vignette":          auto_vignette,
            "film_grain":        0.0,   # Broadcast-clean: NO NOISE
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
