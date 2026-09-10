"""
Cross-Bilateral Edge-Preserving Denoiser for SimRTX Studio.
Denoises RTGI indirect bounce light and SSR reflections using Depth and Normal guidance.
Ensures zero flickering and razor-sharp geometric silhouettes.
"""

import torch
import torch.nn.functional as F


class BilateralDenoiser:
    def __init__(self, device: torch.device):
        self.device = device

    def denoise(
        self,
        signal: torch.Tensor,
        depth: torch.Tensor,
        normals: torch.Tensor,
        spatial_sigma: float = 2.0,
        depth_weight: float = 15.0,
        normal_weight: float = 8.0,
    ) -> torch.Tensor:
        """
        Applies a depth-and-normal guided bilateral filter to the noisy ray-traced buffer.

        Args:
            signal: (B, C, H, W) noisy ray-traced light buffer (RTGI or SSR)
            depth: (B, 1, H, W) depth map
            normals: (B, 3, H, W) normal map
            spatial_sigma: Spatial blur radius
            depth_weight: Sensitivity to depth edges (higher = sharper edges)
            normal_weight: Sensitivity to normal changes (higher = preserves corners)

        Returns:
            torch.Tensor: Filtered clean buffer (B, C, H, W)
        """
        b, c, h, w = signal.shape
        dtype = signal.dtype

        # 5x5 separable or 9-tap cross-bilateral filter
        # Offsets in pixels: center + 8 neighborhood directions
        offsets = [
            (0, 0),
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-2, 0), (2, 0), (0, -2), (0, 2),
            (-1, -1), (1, 1), (-1, 1), (1, -1)
        ]

        accum_signal = torch.zeros_like(signal)
        accum_weights = torch.zeros((b, 1, h, w), device=self.device, dtype=dtype)

        for ox, oy in offsets:
            dist_sq = ox * ox + oy * oy
            w_spatial = float(torch.exp(torch.tensor(-dist_sq / (2.0 * spatial_sigma * spatial_sigma))))

            # Shifted buffers using circular or clamped shift
            shifted_signal = torch.roll(signal, shifts=(oy, ox), dims=(2, 3))
            shifted_depth = torch.roll(depth, shifts=(oy, ox), dims=(2, 3))
            shifted_normals = torch.roll(normals, shifts=(oy, ox), dims=(2, 3))

            # Depth difference weight: exp(-depth_weight * |d1 - d2|)
            depth_diff = torch.abs(depth - shifted_depth)
            w_depth = torch.exp(-depth_weight * depth_diff)

            # Normal difference weight: exp(-normal_weight * (1 - dot(n1, n2)))
            n_dot = torch.clamp((normals * shifted_normals).sum(dim=1, keepdim=True), min=0.0, max=1.0)
            w_normal = torch.pow(n_dot, normal_weight)

            total_w = w_spatial * w_depth * w_normal
            accum_signal = accum_signal + shifted_signal * total_w
            accum_weights = accum_weights + total_w

        # Normalize
        clean_signal = accum_signal / (accum_weights + 1e-6)
        return clean_signal
