"""
Luxanix Studio — Auto-Adaptive AI Preset Engine
Dynamic per-frame & millisecond scene analyzer and parameter optimizer.
Continuously calculates optimal RTGI, SSR, RTAO, Tonemapping, Exposure, and Wetness.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional

class AutoSceneOptimizer:
    """
    Analyzes video frames in real-time and dynamically computes physically plausible
    and photorealistic raytracing & color grading parameters.
    Uses Exponential Moving Average (EMA) to ensure buttery smooth, flicker-free transitions.
    """
    def __init__(self, smoothing_alpha: float = 0.2):
        self.alpha = smoothing_alpha
        self.state: Optional[Dict[str, float]] = None

    def reset(self):
        self.state = None

    def analyze_and_optimize(
        self,
        frame_rgb: np.ndarray,
        depth_tensor: Optional[torch.Tensor] = None,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a single RGB frame (and optional depth map) and returns
        dynamically optimized parameters.
        """
        if base_params is None:
            base_params = {}

        h, w = frame_rgb.shape[:2]

        # 1. Downsampled analysis for sub-millisecond calculation speed
        small_h = 180
        small_w = int(w * (small_h / h))
        small_rgb = cv2.resize(frame_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Convert to Grayscale & HSV
        gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)

        # Luminance & Dynamic Range Metrics
        mean_lum = float(np.mean(gray)) / 255.0  # 0.0 to 1.0
        std_lum = float(np.std(gray)) / 255.0
        shadow_clip = float(np.mean(gray < 25))
        highlight_clip = float(np.mean(gray > 230))
        mean_sat = float(np.mean(hsv[:, :, 1])) / 255.0

        # 2. Road / Track Wetness & Specular Reflection Detection
        # Lower 45% of the frame is predominantly the track/asphalt and car chassis
        road_y_start = int(small_h * 0.55)
        road_crop = gray[road_y_start:, :]
        road_specular = float(np.mean(road_crop > 185))
        road_contrast = float(np.std(road_crop)) / 255.0

        # Wetness metric: high specular highlights with high local contrast on the road
        wetness_score = np.clip((road_specular * 8.0) + (road_contrast * 1.5) - 0.15, 0.0, 1.0)

        # 3. Ambient & Sky Bounce Analysis
        # Top 30% of the frame is the sky/horizon
        sky_crop = small_rgb[:int(small_h * 0.3), :, :]
        sky_brightness = float(np.mean(sky_crop)) / 255.0

        # 4. Target Parameter Computation based on Scene Dynamics
        # Auto Exposure: Compensate dark scenes or overexposed sunny scenes
        target_exposure = np.clip((0.48 - mean_lum) * 1.6, -0.6, 0.7)

        # Contrast: If scene is naturally hazy/flat, boost contrast; if already punchy, soften
        target_contrast = np.clip(1.05 + (0.28 - std_lum) * 0.8, 0.95, 1.35)

        # RTGI (Global Illumination): Stronger when sky/environment is bright or in shaded areas
        target_rtgi_intensity = np.clip(0.50 + sky_brightness * 0.35 + (1.0 - mean_lum) * 0.2, 0.35, 0.90)
        target_rtgi_range = np.clip(12.0 + sky_brightness * 8.0, 8.0, 24.0)

        # SSR (Screen-Space Reflections):
        # In wet conditions, SSR is dramatically boosted with low roughness
        target_ssr_intensity = np.clip(0.35 + wetness_score * 0.60, 0.25, 0.95)
        target_roughness = np.clip(0.65 - wetness_score * 0.50, 0.12, 0.75)
        is_wet_mode = bool(wetness_score > 0.35)

        # RTAO (Ambient Occlusion): More pronounced in overcast/diffuse lighting, subtle in harsh direct sun
        target_rtao_intensity = np.clip(0.45 + (1.0 - mean_lum) * 0.35, 0.30, 0.80)
        target_rtao_radius = np.clip(1.1 + (1.0 - mean_lum) * 0.4, 0.8, 1.8)

        # Bloom: Triggered by specular reflections or headlight highlights
        target_bloom = np.clip(0.20 + highlight_clip * 4.0, 0.15, 0.65)
        target_bloom_thresh = np.clip(0.85 - highlight_clip * 1.5, 0.68, 0.90)

        # Vibrance & Color Temperature
        target_vibrance = np.clip(0.20 + (0.4 - mean_sat) * 0.5, 0.05, 0.45)
        # Check color balance (warm vs cool)
        mean_r = float(np.mean(small_rgb[:, :, 0]))
        mean_b = float(np.mean(small_rgb[:, :, 2]))
        color_diff = (mean_r - mean_b) / 255.0
        target_temp = np.clip(color_diff * 0.4, -0.3, 0.3)

        raw_targets = {
            "exposure": float(target_exposure),
            "contrast": float(target_contrast),
            "rtgi_intensity": float(target_rtgi_intensity),
            "rtgi_range": float(target_rtgi_range),
            "ssr_intensity": float(target_ssr_intensity),
            "roughness": float(target_roughness),
            "rtao_intensity": float(target_rtao_intensity),
            "rtao_radius": float(target_rtao_radius),
            "bloom_intensity": float(target_bloom),
            "bloom_threshold": float(target_bloom_thresh),
            "vibrance": float(target_vibrance),
            "temperature": float(target_temp),
        }

        # 5. Temporal Smoothing (Exponential Moving Average)
        if self.state is None:
            self.state = raw_targets
        else:
            for k, val in raw_targets.items():
                self.state[k] = (1.0 - self.alpha) * self.state[k] + self.alpha * val

        # 6. Build combined parameter dictionary
        optimized = dict(base_params)
        for k, val in self.state.items():
            optimized[k] = val

        optimized["wet_track_mode"] = is_wet_mode
        optimized["use_aces"] = base_params.get("use_aces", True)
        optimized["clarity"] = base_params.get("clarity", 0.35)
        optimized["film_grain"] = base_params.get("film_grain", 0.08)
        optimized["denoise"] = base_params.get("denoise", True)

        # Attach telemetry data for UI
        optimized["_telemetry"] = {
            "mean_luminance": mean_lum,
            "wetness_score": wetness_score,
            "sky_brightness": sky_brightness,
            "is_wet": is_wet_mode,
        }

        return optimized
