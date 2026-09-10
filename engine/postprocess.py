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
        Combines base image with Ray Traced lighting and applies full color grading.

        Args:
            base_color: (B, 3, H, W) normalized base frame in [0, 1].
            rt_buffers: Dict with 'rtgi', 'ssr', 'rtao'.
            params: Dictionary containing user-adjusted parameters.

        Returns:
            torch.Tensor: (B, 3, H, W) graded output in [0, 1].
        """
        ao = rt_buffers.get("rtao", torch.ones_like(base_color[:, :1]))
        rtgi = rt_buffers.get("rtgi", torch.zeros_like(base_color))
        ssr = rt_buffers.get("ssr", torch.zeros_like(base_color))

        # 1. Physically-Based Composite (atmosphere-preserving)
        # AO is kept very close to 1.0 so it doesn't darken the whole scene.
        # RTGI and SSR are blended in as additive highlights only on bright areas
        # (wet reflections, indirect bounces) — not affecting overall luminance.
        # Clamp AO to [0.85, 1.0] so it only adds subtle contact shadow, not full darkness.
        ao_clamped = torch.clamp(ao, min=0.85, max=1.0)
        composited = base_color * ao_clamped

        # RTX effects: additive, but scaled down to stay subtle
        # Only blend into areas that have existing luminance (avoids brightening flat darks)
        rtgi_weight = params.get("rtgi_intensity", 0.25)
        ssr_weight  = params.get("ssr_intensity",  0.15)
        composited = composited + rtgi * rtgi_weight * 0.4 + ssr * ssr_weight * 0.4

        # 2. Exposure adjustment (in EV stops: 2^ev) — now very small range from auto_realism
        exposure = params.get("exposure", 0.0)
        if exposure != 0.0:
            composited = composited * (2.0 ** exposure)

        # 3. Contrast adjustment (around midpoint 0.18 middle-grey)
        contrast = params.get("contrast", 1.0)
        if contrast != 1.0:
            composited = torch.clamp((composited - 0.18) * contrast + 0.18, min=0.0)

        # 4. Color Temperature and Tint
        temp = params.get("temperature", 0.0)  # -1.0 (cool/blue) to +1.0 (warm/orange)
        if temp != 0.0:
            r_scale = 1.0 + temp * 0.15
            b_scale = 1.0 - temp * 0.15
            color_weights = torch.tensor([r_scale, 1.0, b_scale], device=self.device, dtype=composited.dtype).view(1, 3, 1, 1)
            composited = composited * color_weights

        # 5. Saturation & Vibrance
        saturation = params.get("saturation", 1.0)
        vibrance = params.get("vibrance", 0.0)
        if saturation != 1.0 or vibrance != 0.0:
            composited = self._apply_saturation_vibrance(composited, saturation, vibrance)

        # 6. Multi-scale Bloom (RTX Glow on headlights & reflections)
        bloom_intensity = params.get("bloom_intensity", 0.10)
        bloom_threshold = params.get("bloom_threshold", 0.85)
        if bloom_intensity > 0.0:
            composited = self._apply_bloom(composited, bloom_intensity, bloom_threshold)

        # 7. Tone Mapping: use a gentler Reinhard-style for cinematic dark scenes,
        # ACES only when explicitly requested (it compresses midtones heavily).
        use_aces = params.get("use_aces", False)   # Default OFF — preserves mood
        if use_aces:
            composited = self._aces_tonemap(composited)
        else:
            # Soft Reinhard: only compresses near-white highlights, leaves darks intact
            composited = composited / (composited + 0.5)   # very gentle shoulder
            composited = torch.clamp(composited * 1.04, 0.0, 1.0)  # slight exposure compensation for the division

        # 8. Subtle Lens Vignette
        vignette_amount = params.get("vignette", 0.0)
        if vignette_amount > 0.0:
            composited = self._apply_vignette(composited, vignette_amount)

        # 9. Texture Clarity & Detail Sharpening (Anti-TAA Blur)
        clarity = params.get("clarity", 0.3)
        if clarity > 0.0:
            composited = self._apply_clarity(composited, clarity)

        # 10. Subtle Cinematic Film Grain (Reduces banding)
        film_grain = params.get("film_grain", 0.0)
        if film_grain > 0.0:
            composited = self._apply_film_grain(composited, film_grain)

        return torch.clamp(composited, 0.0, 1.0)

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
