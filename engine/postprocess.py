"""
Photorealistic Post-Processing and Color Grading for Luxanix Studio Pro.
Features:
- Composite Ray Traced buffers (Base + AO + RTGI + SSR) with natural physical blending
- Specular highlight preservation (smooth shoulder compression without darkening base image)
- High-fidelity Saturation and Smart Vibrance
- Precise Exposure, Contrast, and Black Level retention (no washed-out blacks)
- Color Temperature & Tint (Kelvin shifting)
- Multi-Scale RTX Bloom for headlights, sun rays & wet road speculars
- Crisp Anti-TAA Texture Clarity (removes game blur)
- Clean broadcast-grade output (no digital noise/film grain artifacts)
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
        Full autonomous color science and raytracing compositing pipeline:
          RTX Composite (RTAO + RTGI + SSR)
          → Exposure EV
          → Cinematic Contrast S-Curve
          → Color Temperature (Kelvin / Auto White Balance)
          → Auto Saturation & Smart Vibrance
          → Multi-scale Bloom (Headlights & Speculars)
          → Soft Shoulder Highlight Compression
          → Subtle Cinematic Lens Vignette
          → Detail Clarity (Anti-TAA Sharpening)
        """
        ao = rt_buffers.get("rtao", torch.ones_like(base_color[:, :1]))
        rtgi = rt_buffers.get("rtgi", torch.zeros_like(base_color))
        ssr = rt_buffers.get("ssr", torch.zeros_like(base_color))

        # -------------------------------------------------------------
        # 1. Physically-Based Composite (Grounded & Natural)
        # -------------------------------------------------------------
        # RTAO: grounds the car chassis and tires onto the tarmac with deep contact shadows.
        # Clamped to [0.45, 1.0] to prevent unnatural black voids while providing rich depth.
        ao_clamped = torch.clamp(ao, min=0.45, max=1.0)
        composited = base_color * ao_clamped

        # Additive Ray-Traced Global Illumination & Reflections
        rtgi_weight = params.get("rtgi_intensity", 0.35)
        ssr_weight  = params.get("ssr_intensity",  0.40)
        if rtgi_weight > 0.0:
            composited = composited + rtgi * rtgi_weight
        if ssr_weight > 0.0:
            composited = composited + ssr * ssr_weight

        # -------------------------------------------------------------
        # 2. Exposure Adjustment (in EV stops: 2^ev)
        # -------------------------------------------------------------
        exposure = params.get("exposure", 0.0)
        if abs(exposure) > 0.001:
            composited = composited * (2.0 ** exposure)

        # -------------------------------------------------------------
        # 3. Contrast S-Curve (Pivoted around 0.18 middle-grey)
        # -------------------------------------------------------------
        contrast = params.get("contrast", 1.0)
        if abs(contrast - 1.0) > 0.005:
            composited = torch.clamp((composited - 0.18) * contrast + 0.18, min=0.0)

        # -------------------------------------------------------------
        # 4. Color Temperature (Auto White Balance / Kelvin Shift)
        # -------------------------------------------------------------
        temp = params.get("temperature", 0.0)  # -1.0 cool/blue … +1.0 warm/orange
        if abs(temp) > 0.005:
            r_scale = 1.0 + temp * 0.14
            b_scale = 1.0 - temp * 0.14
            color_weights = torch.tensor(
                [r_scale, 1.0, b_scale], device=self.device, dtype=composited.dtype
            ).view(1, 3, 1, 1)
            composited = composited * color_weights

        # -------------------------------------------------------------
        # 5. Saturation & Smart Vibrance (Pops racing liveries & decals)
        # -------------------------------------------------------------
        saturation = params.get("saturation", 1.0)
        vibrance   = params.get("vibrance", 0.0)
        if abs(saturation - 1.0) > 0.005 or abs(vibrance) > 0.005:
            composited = self._apply_saturation_vibrance(composited, saturation, vibrance)

        # -------------------------------------------------------------
        # 6. Multi-Scale Bloom (RTX Glow on headlights, taillights & speculars)
        # -------------------------------------------------------------
        bloom_intensity = params.get("bloom_intensity", 0.0)
        bloom_threshold = params.get("bloom_threshold", 0.88)
        if bloom_intensity > 0.01:
            composited = self._apply_bloom(composited, bloom_intensity, bloom_threshold)

        # -------------------------------------------------------------
        # 7. Soft Shoulder Highlight Rolloff (Preserves 100% Base Dynamic Range)
        # -------------------------------------------------------------
        # Unlike old Reinhard which compressed 1.0 to 0.66 and washed out the image,
        # this only compresses highlights above 0.85 that exceeded 1.0 from additive RTX.
        use_aces = params.get("use_aces", False)
        if use_aces:
            composited = self._aces_tonemap(composited)
        else:
            # Soft shoulder: pixels <= 0.85 are 100% untouched
            threshold = 0.85
            over = torch.clamp(composited - threshold, min=0.0)
            composited = torch.where(
                composited > threshold,
                threshold + over / (1.0 + over * 1.8),
                composited
            )

        # -------------------------------------------------------------
        # 8. Subtle Lens Vignette (Cinematic Edge Framing)
        # -------------------------------------------------------------
        vignette_amount = params.get("vignette", 0.0)
        if vignette_amount > 0.01:
            composited = self._apply_vignette(composited, vignette_amount)

        # -------------------------------------------------------------
        # 9. Detail Clarity / Anti-TAA Texture Sharpening
        # -------------------------------------------------------------
        clarity = params.get("clarity", 0.0)
        if clarity > 0.01:
            composited = self._apply_clarity(composited, clarity)

        # -------------------------------------------------------------
        # 10. Film Grain (Disabled by default — clean broadcast quality)
        # -------------------------------------------------------------
        film_grain = params.get("film_grain", 0.0)
        if film_grain > 0.005:
            composited = self._apply_film_grain(composited, film_grain)

        return torch.clamp(composited, 0.0, 1.0)

    def _apply_saturation_vibrance(
        self,
        rgb: torch.Tensor,
        saturation: float,
        vibrance: float,
    ) -> torch.Tensor:
        """
        Adjusts saturation and smart vibrance with chroma preservation.
        Pops racing car liveries, brake calipers, sponsor decals, and environmental colors
        without burning out highlight details.
        """
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device, dtype=rgb.dtype).view(1, 3, 1, 1)
        luma = (rgb * weights).sum(dim=1, keepdim=True)

        max_c, _ = torch.max(rgb, dim=1, keepdim=True)
        min_c, _ = torch.min(rgb, dim=1, keepdim=True)
        sat_metric = (max_c - min_c) / (max_c + 1e-5)

        # Smart vibrance: lifts muted/flat gaming tones while protecting already vivid colors
        vibrance_factor = 1.0 + vibrance * (1.0 - sat_metric * 0.7)
        total_sat = saturation * vibrance_factor

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

        bright_mask = torch.clamp((luma - threshold) / (1.0 - threshold + 1e-5), min=0.0, max=1.0)
        bright_pixels = rgb * bright_mask

        # Multi-scale downsample and blur
        bloom1 = F.interpolate(bright_pixels, scale_factor=0.5, mode="bilinear", align_corners=False)
        bloom2 = F.interpolate(bloom1, scale_factor=0.5, mode="bilinear", align_corners=False)
        bloom3 = F.interpolate(bloom2, scale_factor=0.5, mode="bilinear", align_corners=False)

        up3 = F.interpolate(bloom3, size=(h, w), mode="bilinear", align_corners=False)
        up2 = F.interpolate(bloom2, size=(h, w), mode="bilinear", align_corners=False)
        up1 = F.interpolate(bloom1, size=(h, w), mode="bilinear", align_corners=False)

        combined_bloom = (up1 * 0.5 + up2 * 0.3 + up3 * 0.2) * intensity
        return rgb + combined_bloom

    def _aces_tonemap(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Krzysztof Narkowicz ACES Filmic Tone Mapping approximation.
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
        GPU Contrast-Adaptive Sharpening (CAS) & Texture Edge Enhancer.
        Specially optimized for racing liveries, sponsor lettering (Pirelli, Shell, etc.),
        windshield banners, carbon weave, and asphalt micro-detail.
        Eliminates temporal anti-aliasing (TAA) and motion blur without edge halos.
        """
        if clarity <= 0.005:
            return rgb

        padded = F.pad(rgb, (1, 1, 1, 1), mode="reflect")
        c = padded[:, :, 1:-1, 1:-1]   # Center
        t = padded[:, :, 0:-2, 1:-1]   # Top
        b = padded[:, :, 2:,   1:-1]   # Bottom
        l = padded[:, :, 1:-1, 0:-2]   # Left
        r = padded[:, :, 1:-1, 2:]     # Right

        # Cross neighbor extrema for contrast adaptation
        cross_min = torch.min(torch.min(torch.min(t, b), torch.min(l, r)), c)
        cross_max = torch.max(torch.max(torch.max(t, b), torch.max(l, r)), c)

        # Contrast-adaptive weight calculation: sharpens strong decal edges without ringing
        amp = torch.sqrt(torch.clamp(torch.min(cross_min, 2.0 - cross_max) / (cross_max + 1e-4), min=0.0, max=1.0))
        w = -amp * (clarity * 0.25)

        # High-precision CAS filter
        sharp = (c + w * (t + b + l + r)) / (1.0 + 4.0 * w)

        # Edge-gated decal & lettering micro-boost
        edge_contrast = cross_max - cross_min
        text_mask = torch.clamp(edge_contrast * 2.5, 0.0, 1.0)
        unsharp_kernel = (torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], device=self.device, dtype=rgb.dtype) / 8.0).repeat(3, 1, 1, 1)
        blurred = F.conv2d(rgb, unsharp_kernel, padding=1, groups=3)
        text_boost = (rgb - blurred) * (clarity * 0.35) * text_mask

        return torch.clamp(sharp + text_boost, min=0.0, max=1.0)

    def _apply_film_grain(self, rgb: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Adds micro-fine film grain to eliminate digital banding if explicitly requested.
        """
        noise = (torch.rand_like(rgb) - 0.5) * 2.0 * amount
        return torch.clamp(rgb + noise, min=0.0, max=1.0)
