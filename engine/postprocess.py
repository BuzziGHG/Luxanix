"""
Photorealistic Post-Processing and Color Grading for SimRTX Studio.
Features:
- Composite Ray Traced buffers (Base + AO + RTGI + SSR)
- Saturation and Smart Vibrance
- Exposure, Contrast, and Black Level
- Color Temperature & Tint
- Physically-based Multi-Scale Bloom
- ACES Filmic Tone Mapping
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any


class ColorGrader:
    def __init__(self, device: torch.device):
        self.device = device

    def grade(
        self,
        base_color: torch.Tensor,
        rt_buffers: Dict[str, torch.Tensor],
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Full autonomous color science pipeline:
          RTX Composite → Exposure → Contrast → Shadow Lift → Highlight Rolloff
          → Auto White Balance (Temperature) → Auto Saturation → Vibrance
          → Bloom → Tone Mapping → Vignette → Clarity → Film Grain
        """
        ao = rt_buffers.get("rtao", torch.ones_like(base_color[:, :1]))
        rtgi = rt_buffers.get("rtgi", torch.zeros_like(base_color))
        ssr = rt_buffers.get("ssr", torch.zeros_like(base_color))

        # 1. Physically-Based Composite (atmosphere-preserving)
        ao_clamped = torch.clamp(ao, min=0.85, max=1.0)
        composited = base_color * ao_clamped
        rtgi_weight = params.get("rtgi_intensity", 0.25)
        ssr_weight  = params.get("ssr_intensity",  0.15)
        composited = composited + rtgi * rtgi_weight * 0.4 + ssr * ssr_weight * 0.4

        # 2. Exposure (EV stops)
        exposure = params.get("exposure", 0.0)
        if exposure != 0.0:
            composited = composited * (2.0 ** exposure)

        # 3. Contrast S-Curve (around 0.18 middle-grey)
        contrast = params.get("contrast", 1.0)
        if contrast != 1.0:
            composited = torch.clamp((composited - 0.18) * contrast + 0.18, min=0.0)

        # 4. Shadow Lift — raises black point gently for cinematic detail recovery
        shadow_lift = params.get("shadow_lift", 0.0)
        if shadow_lift > 0.0:
            composited = self._apply_shadow_lift(composited, shadow_lift)

        # 5. Highlight Rolloff — soft shoulder on blown highlights / speculars
        highlight_rolloff = params.get("highlight_rolloff", 0.0)
        if highlight_rolloff > 0.0:
            composited = self._apply_highlight_rolloff(composited, highlight_rolloff)

        # 6. Color Temperature (auto white balance / Kelvin shift)
        temp = params.get("temperature", 0.0)  # -1.0 cool/blue … +1.0 warm/orange
        if abs(temp) > 0.005:
            r_scale = 1.0 + temp * 0.15
            b_scale = 1.0 - temp * 0.15
            color_weights = torch.tensor(
                [r_scale, 1.0, b_scale], device=self.device, dtype=composited.dtype
            ).view(1, 3, 1, 1)
            composited = composited * color_weights

        # 7. Auto Saturation & Smart Vibrance
        saturation = params.get("saturation", 1.0)
        vibrance   = params.get("vibrance", 0.0)
        if saturation != 1.0 or vibrance != 0.0:
            composited = self._apply_saturation_vibrance(composited, saturation, vibrance)

        # 8. Bloom (RTX headlight & sun glow)
        bloom_intensity = params.get("bloom_intensity", 0.10)
        bloom_threshold = params.get("bloom_threshold", 0.85)
        if bloom_intensity > 0.0:
            composited = self._apply_bloom(composited, bloom_intensity, bloom_threshold)

        # 9. Tone Mapping (Soft Reinhard default — preserves cinematic darks)
        use_aces = params.get("use_aces", False)
        if use_aces:
            composited = self._aces_tonemap(composited)
        else:
            composited = composited / (composited + 0.5)
            composited = torch.clamp(composited * 1.04, 0.0, 1.0)

        # 10. Cinematic Lens Vignette
        vignette_amount = params.get("vignette", 0.0)
        if vignette_amount > 0.0:
            composited = self._apply_vignette(composited, vignette_amount)

        # 11. Detail Clarity / TAA Sharpening
        clarity = params.get("clarity", 0.3)
        if clarity > 0.0:
            composited = self._apply_clarity(composited, clarity)

        # 12. Adaptive Film Grain
        film_grain = params.get("film_grain", 0.0)
        if film_grain > 0.0:
            composited = self._apply_film_grain(composited, film_grain)

        return torch.clamp(composited, 0.0, 1.0)

    def _apply_shadow_lift(self, rgb: torch.Tensor, lift: float) -> torch.Tensor:
        """
        Gently raises the black point for cinematic shadow detail recovery.
        Uses a toe curve: only affects the dark quarter of the tonal range.
        lift: 0.0 = no change, 0.05 = professional grade lift
        """
        # Shadows are values below 0.20 — apply a smooth lift only there
        shadow_mask = torch.clamp(1.0 - rgb / 0.20, min=0.0, max=1.0)
        return rgb + shadow_mask * lift

    def _apply_highlight_rolloff(self, rgb: torch.Tensor, rolloff: float) -> torch.Tensor:
        """
        Soft-clips bright highlights above 0.75 to prevent harsh clipping of speculars.
        Uses a smooth S-shaped shoulder: highlights are compressed, not clipped hard.
        rolloff: 0.0 = no rolloff, 1.0 = strong compression
        """
        # Only compress above the shoulder threshold
        threshold = 0.75
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        luma = (rgb * weights).sum(dim=1, keepdim=True)
        hi_mask = torch.clamp((luma - threshold) / (1.0 - threshold + 1e-5), min=0.0, max=1.0)
        # Compress highlights: blend toward 1.0 softly
        compressed = rgb - hi_mask * (rgb - 1.0) * rolloff * 0.3
        return torch.clamp(compressed, 0.0, 1.0)

    def _apply_saturation_vibrance(
        self,
        rgb: torch.Tensor,
        saturation: float,
        vibrance: float,
    ) -> torch.Tensor:
        """
        Adjusts saturation and smart vibrance.
        """
        # Rec. 709 Luminance
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        luma = (rgb * weights).sum(dim=1, keepdim=True)

        # Per-pixel max/min channel difference (saturation metric)
        max_c, _ = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        sat_metric = (max_c - min_c) / (max_c + 1e-5)

        # Vibrance scales muted colors more than saturated colors
        vibrance_factor = 1.0 + vibrance * (1.0 - sat_metric)
        total_sat = saturation * vibrance_factor

        # Linear blend between luminance and color
        adjusted = luma + (rgb - luma) * total_sat
        return torch.clamp(adjusted, min=0.0)

    def _apply_bloom(
        self,
        rgb: torch.Tensor,
        intensity: float,
        threshold: float,
    ) -> torch.Tensor:
        """
        Simulates camera lens bloom on high-luminance pixels.
        """
        b, c, h, w = rgb.shape
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        luma = (rgb * weights).sum(dim=1, keepdim=True)

        # Extract highlight areas exceeding threshold
        bright_mask = torch.clamp((luma - threshold) / (1.0 - threshold + 1e-5), min=0.0, max=1.0)
        bright_pixels = rgb * bright_mask

        # Multi-scale downsample and blur
        bloom1 = F.interpolate(bright_pixels, scale_factor=0.5, mode="bilinear", align_corners=False)
        bloom2 = F.interpolate(bloom1, scale_factor=0.5, mode="bilinear", align_corners=False)
        bloom3 = F.interpolate(bloom2, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Upsample back
        up3 = F.interpolate(bloom3, size=(h, w), mode="bilinear", align_corners=False)
        up2 = F.interpolate(bloom2, size=(h, w), mode="bilinear", align_corners=False)
        up1 = F.interpolate(bloom1, size=(h, w), mode="bilinear", align_corners=False)

        combined_bloom = (up1 * 0.5 + up2 * 0.3 + up3 * 0.2) * intensity
        return rgb + combined_bloom

    def _aces_tonemap(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Krzysztof Narkowicz ACES Filmic Tone Mapping approximation.
        Provides beautiful highlights compression, rich contrast, and cinematic color roll-off.
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        tonemapped = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
        return torch.clamp(tonemapped, 0.0, 1.0)

    def _apply_vignette(self, rgb: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Subtle optical camera vignette darkening at corners.
        """
        b, c, h, w = rgb.shape
        y = torch.linspace(-1, 1, h, device=self.device, dtype=rgb.dtype).view(1, 1, h, 1)
        x = torch.linspace(-1, 1, w, device=self.device, dtype=rgb.dtype).view(1, 1, 1, w)
        dist_sq = (x * x + y * y) * 0.5
        vignette_mask = torch.clamp(1.0 - dist_sq * amount, min=0.0, max=1.0)
        return rgb * vignette_mask

    def _apply_clarity(self, rgb: torch.Tensor, clarity: float) -> torch.Tensor:
        """
        Enhances edge crispness and micro-contrast using fast unsharp masking.
        Effectively removes TAA (temporal anti-aliasing) motion blur in racing games.
        """
        kernel = (torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=self.device, dtype=rgb.dtype) / 16.0)
        kernel = kernel.repeat(3, 1, 1, 1)
        blurred = F.conv2d(rgb, kernel, padding=1, groups=3)
        high_pass = rgb - blurred
        return torch.clamp(rgb + high_pass * clarity, min=0.0)

    def _apply_film_grain(self, rgb: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Adds subtle cinematographic film grain to eliminate digital color banding.
        """
        noise = (torch.rand_like(rgb) - 0.5) * 2.0 * amount
        return torch.clamp(rgb + noise, min=0.0, max=1.0)
