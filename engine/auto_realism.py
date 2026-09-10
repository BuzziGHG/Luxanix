"""
Luxanix Autonomous Photorealism & Color Science AI Engine
=========================================================
Performs 100% autonomous real-time scene analysis and computes ALL visual
parameters automatically — no manual presets, no sliders needed.

Physics modules computed per-frame:
  ■ Exposure (EV) — atmosphere-preserving, never over-brightens cinematic darks
  ■ Contrast S-Curve — subtle expansion for flat scenes, leaves dynamic scenes alone
  ■ Auto White Balance — Gray-World + weighted scene analysis
  ■ Color Temperature — detects golden hour / overcast / night / indoor and shifts Kelvin
  ■ Auto Saturation — scene-aware, boosts muted colors, protects already-saturated footage
  ■ Smart Vibrance — selectively lifts dull midtones without oversaturating skin/reds
  ■ Shadow Lift — gently lifts crushed blacks for detail recovery (cinematic grade)
  ■ Highlight Rolloff — soft shoulder on blown highlights, preserves speculars
  ■ Cinematic Vignette — automatic lens-falloff, scales with scene brightness
  ■ SSR / RTGI / RTAO / Bloom / Clarity — RTX physics (additive, atmosphere-preserving)
  ■ Film Grain — scene-adaptive (darker = more grain, brighter = less)
  ■ Temporal EMA smoothing — prevents inter-frame flickering
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, Optional


class AutonomousRealismEngine:
    def __init__(self, smoothing_alpha: float = 0.15):
        self.alpha = smoothing_alpha
        self.smoothed_params: Optional[Dict[str, float]] = None

    def reset(self):
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
            return "indoor"      # circuit pit lane, indoors, neutral

    # ------------------------------------------------------------------
    # AUTO WHITE BALANCE — Gray-World + per-channel normalization
    # ------------------------------------------------------------------
    def _compute_auto_wb(
        self, float_rgb: np.ndarray
    ) -> float:
        """
        Estimates color temperature shift (-1.0=cool/blue, +1.0=warm/orange)
        using a Gray-World assumption cross-checked with the sky region.
        """
        mean_r = float(np.mean(float_rgb[:, :, 0]))
        mean_g = float(np.mean(float_rgb[:, :, 1]))
        mean_b = float(np.mean(float_rgb[:, :, 2]))
        mean_all = (mean_r + mean_g + mean_b) / 3.0 + 1e-5

        # Relative channel bias vs. neutral gray
        r_bias = (mean_r - mean_all) / mean_all
        b_bias = (mean_b - mean_all) / mean_all

        # Positive = image is warm/red → slightly cool WB correction
        # Negative = image is cool/blue → slightly warm WB correction
        # We CORRECT very gently — don't fight the artistic intent
        wb_shift = float(np.clip(-(r_bias - b_bias) * 0.25, -0.30, 0.30))
        return wb_shift

    # ------------------------------------------------------------------
    # MAIN COMPUTATION
    # ------------------------------------------------------------------
    def analyze_and_compute(
        self,
        frame_rgb: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        master_intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Autonomously analyzes an RGB frame and computes the full suite of
        photorealism, color correction, and cinematic grading parameters.
        """
        # ── Downscale for ultra-fast analysis (160×90) ─────────────────
        small_rgb = cv2.resize(frame_rgb, (160, 90), interpolation=cv2.INTER_AREA)
        float_rgb = small_rgb.astype(np.float32) / 255.0

        # ── Luminance (Rec. 709) ────────────────────────────────────────
        lum = (0.2126 * float_rgb[:, :, 0]
               + 0.7152 * float_rgb[:, :, 1]
               + 0.0722 * float_rgb[:, :, 2])
        mean_lum  = float(np.mean(lum))
        std_lum   = float(np.std(lum))
        log_mean_lum = float(np.exp(np.mean(np.log(lum + 1e-4))))

        # Highlight / shadow metrics
        highlights = float(np.mean(lum > 0.82))
        shadows    = float(np.mean(lum < 0.06))

        # ── Road / Track Surface (lower 45% of frame) ──────────────────
        road_lum  = lum[int(0.55 * 90):, :]
        road_std  = float(np.std(road_lum))
        road_highlights = float(np.mean(road_lum > 0.75))

        # ── HSV Color Analysis ──────────────────────────────────────────
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        hue = hsv[:, :, 0].astype(np.float32)  # 0–180 in OpenCV
        mean_sat = float(np.mean(sat))
        std_sat  = float(np.std(sat))

        # ── Sky Region Analysis (upper 35%) ────────────────────────────
        sky = float_rgb[:int(0.35 * 90), :, :]
        sky_blue_bias = float(np.mean(sky[:, :, 2] - sky[:, :, 0]))   # blue > red
        sky_red_bias  = float(np.mean(sky[:, :, 0] - sky[:, :, 2]))   # red > blue (golden)

        # ── Scene Type Classification ───────────────────────────────────
        scene = self._detect_scene_type(log_mean_lum, mean_lum, sky_blue_bias, sky_red_bias, mean_sat)

        # ==================================================================
        # A. AUTO-EXPOSURE EV  (very conservative — preserve mood)
        # ==================================================================
        if log_mean_lum < 0.04:
            auto_exposure = float(np.clip(0.08 - log_mean_lum * 0.6, 0.0, 0.20))
        elif log_mean_lum > 0.55:
            auto_exposure = float(np.clip(-(log_mean_lum - 0.55) * 0.35, -0.20, 0.0))
        else:
            auto_exposure = float(np.clip(0.015 * (0.25 - log_mean_lum), -0.08, 0.08))

        # ==================================================================
        # B. CONTRAST S-CURVE  (subtle expansion for flat scenes only)
        # ==================================================================
        if std_lum < 0.12:
            auto_contrast = 1.08
        elif std_lum > 0.28:
            auto_contrast = 1.02
        else:
            auto_contrast = float(1.08 - (std_lum - 0.12) * (0.06 / 0.16))

        # ==================================================================
        # C. AUTO WHITE BALANCE / COLOR TEMPERATURE
        # Scene-based Kelvin shift → temperature param (-1.0 cool … +1.0 warm)
        # ==================================================================
        wb_correction = self._compute_auto_wb(float_rgb)

        # Per-scene bias on top of gray-world correction:
        scene_temp_bias = {
            "night":       -0.10,  # slight blue-cool for night ambiance
            "low_light":   -0.05,  # neutral-cool
            "golden_hour": +0.22,  # warm amber push
            "overcast":    -0.08,  # desaturated cool
            "daylight":    -0.04,  # slightly cool / neutral
            "bright_day":  -0.06,  # cool to counter harsh noon
            "indoor":      +0.06,  # slight warm for artificial light
        }.get(scene, 0.0)

        auto_temperature = float(np.clip(wb_correction + scene_temp_bias, -0.40, 0.40))

        # ==================================================================
        # D. AUTO SATURATION  (scene-aware, protect already-saturated footage)
        # ==================================================================
        # If footage is naturally muted (low mean_sat), gently boost
        # If footage is already vivid, barely touch
        if mean_sat < 0.10:
            # Very desaturated — e.g. overcast, foggy, underexposed
            auto_saturation = 1.14
        elif mean_sat < 0.20:
            # Muted — slight boost
            auto_saturation = 1.07
        elif mean_sat > 0.35:
            # Already vibrant — don't push further
            auto_saturation = 0.98
        else:
            # Normal range
            auto_saturation = float(1.07 - (mean_sat - 0.20) * (0.09 / 0.15))

        # Scene-based saturation modifier
        scene_sat_mult = {
            "night":       0.90,  # night: desaturate slightly for realism
            "low_light":   0.95,
            "golden_hour": 1.10,  # golden hour: pop the warm tones
            "overcast":    0.95,
            "daylight":    1.00,
            "bright_day":  0.98,
            "indoor":      1.02,
        }.get(scene, 1.0)
        auto_saturation = float(np.clip(auto_saturation * scene_sat_mult, 0.85, 1.25))

        # ==================================================================
        # E. SMART VIBRANCE  (lifts muted midtones, protects saturated areas)
        # Low vibrance if scene is already vivid, higher if muted/flat
        # ==================================================================
        if mean_sat < 0.12:
            auto_vibrance = 0.25  # strong vibrance lift for muted footage
        elif mean_sat < 0.22:
            auto_vibrance = 0.14
        elif mean_sat > 0.35:
            auto_vibrance = 0.0   # leave vivid footage alone
        else:
            auto_vibrance = float(0.14 - (mean_sat - 0.22) * (0.14 / 0.13))
        auto_vibrance = float(np.clip(auto_vibrance * master_intensity, 0.0, 0.30))

        # ==================================================================
        # F. SHADOW LIFT  (gentle black-point lift for cinematic detail)
        # Lifts deep shadows slightly to reveal texture — never crushes blacks
        # Range: 0.0 = no lift, 0.05 = professional grade lift
        # ==================================================================
        if scene == "night":
            auto_shadow_lift = 0.02  # very subtle — keep night dark
        elif scene == "low_light":
            auto_shadow_lift = 0.03
        elif shadows > 0.25:
            # Many crushed blacks — lift for detail
            auto_shadow_lift = float(np.clip(0.025 + shadows * 0.06, 0.02, 0.06))
        else:
            auto_shadow_lift = 0.015

        # ==================================================================
        # G. HIGHLIGHT ROLLOFF  (soft shoulder for blown speculars)
        # Range: 0.0 = no rolloff, 1.0 = strong rolloff
        # ==================================================================
        if highlights > 0.15:
            auto_highlight_rolloff = float(np.clip(0.30 + highlights * 1.2, 0.30, 0.80))
        else:
            auto_highlight_rolloff = 0.0

        # ==================================================================
        # H. CINEMATIC VIGNETTE  (automatic lens falloff — scale with brightness)
        # Brighter outdoor scenes: heavier vignette for cinematic framing
        # Dark scenes: almost none (avoid double-darkening corners)
        # ==================================================================
        if scene in ("night", "low_light"):
            auto_vignette = 0.10
        elif scene == "golden_hour":
            auto_vignette = float(np.clip(0.25 + mean_lum * 0.20, 0.20, 0.45))
        else:
            auto_vignette = float(np.clip(0.15 + mean_lum * 0.25, 0.12, 0.40))
        auto_vignette *= master_intensity

        # ==================================================================
        # I. SSR — Screen-Space Reflections (wet road / puddle physics)
        # ==================================================================
        wetness_score = float(np.clip(road_highlights * 4.0 + road_std * 1.2, 0.05, 0.80))
        auto_ssr = float(np.clip(0.15 + wetness_score * 0.40, 0.10, 0.55)) * master_intensity

        # ==================================================================
        # J. RTGI — Ray Traced Global Illumination (indirect bounce light)
        # ==================================================================
        auto_rtgi = float(np.clip(
            0.20 + mean_sat * 0.35 + (0.05 if sky_blue_bias > 0.05 else 0.0),
            0.15, 0.45
        )) * master_intensity

        # ==================================================================
        # K. RTAO — Ray Traced Ambient Occlusion (contact shadows)
        # ==================================================================
        auto_rtao = float(np.clip(0.55 + (0.10 if mean_lum > 0.25 else 0.05), 0.50, 0.70))

        # ==================================================================
        # L. BLOOM  (tight cinematic glow on headlights / sun)
        # ==================================================================
        auto_bloom = float(np.clip(0.08 + highlights * 1.8, 0.08, 0.35))

        # ==================================================================
        # M. DETAIL CLARITY  (TAA sharpening — subtle, no halos)
        # ==================================================================
        auto_clarity = float(np.clip(0.18 + (0.08 if std_lum < 0.18 else 0.02), 0.15, 0.28))

        # ==================================================================
        # N. FILM GRAIN  (adaptive — dark scenes need more grain to hide banding)
        # ==================================================================
        if mean_lum < 0.10:
            auto_grain = 0.06
        elif mean_lum < 0.25:
            auto_grain = 0.04
        else:
            auto_grain = 0.02

        # ==================================================================
        # PACK ALL COMPUTED VALUES
        # ==================================================================
        raw_computed = {
            # Exposure & Tone
            "exposure":           auto_exposure,
            "contrast":           auto_contrast,
            "shadow_lift":        auto_shadow_lift,
            "highlight_rolloff":  auto_highlight_rolloff,

            # Color Correction & Grading
            "temperature":        auto_temperature,
            "saturation":         auto_saturation,
            "vibrance":           auto_vibrance,
            "vignette":           auto_vignette,

            # RTX Physics
            "ssr_intensity":      auto_ssr,
            "rtgi_intensity":     auto_rtgi,
            "rtao_intensity":     auto_rtao * master_intensity,
            "bloom_intensity":    auto_bloom,

            # Detail & Grain
            "clarity":            auto_clarity,
            "film_grain":         auto_grain,

            # Telemetry metadata
            "wetness_score":      wetness_score,
            "mean_luminance":     mean_lum,
            "scene_type":         scene,
        }

        # ==================================================================
        # TEMPORAL EMA SMOOTHING  (prevents flickering between frames)
        # ==================================================================
        if self.smoothed_params is None:
            self.smoothed_params = {k: v for k, v in raw_computed.items() if isinstance(v, (int, float))}
            self.smoothed_params["scene_type"] = scene
        else:
            for k, val in raw_computed.items():
                if isinstance(val, (int, float)):
                    prev = self.smoothed_params.get(k, val)
                    self.smoothed_params[k] = (1.0 - self.alpha) * prev + self.alpha * val
            self.smoothed_params["scene_type"] = scene  # scene_type: instant, no smoothing

        result = self.smoothed_params.copy()
        result["auto_preset"] = True
        return result
