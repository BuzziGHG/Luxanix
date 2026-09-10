"""
Screen-Space Ray Tracing Engine for SimRTX Studio.
Implements:
1. Ray Traced Global Illumination (RTGI) - Multi-bounce indirect bounce lighting
2. Screen-Space Reflections (SSR) - Ray-marched specular reflections with Fresnel
3. Ray Traced Ambient Occlusion (RTAO) - Contact shadows and crevice darkening
Accelerated on NVIDIA RTX Tensor / CUDA Cores via PyTorch.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any
from .geometry import reconstruct_positions, compute_normals


class ScreenSpaceRaytracer:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype

    def trace(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        params: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Executes screen-space ray tracing passes.

        Args:
            color: (B, 3, H, W) normalized RGB tensor in [0.0, 1.0].
            depth: (B, 1, H, W) normalized depth tensor in [0.0, 1.0].
            params: Dictionary of ray tracing parameters.

        Returns:
            Dict containing:
                'rtgi': Indirect bounce lighting buffer
                'ssr': Specular reflections buffer
                'rtao': Ambient occlusion factor [0, 1]
                'normals': Surface normal map for debugging/visualization
        """
        b, c, h, w = color.shape
        fov_deg = params.get("fov_deg", 65.0)

        # 1. Reconstruct 3D camera-space positions and normals
        pos = reconstruct_positions(depth, fov_deg=fov_deg)
        normals = compute_normals(depth, fov_deg=fov_deg)

        # 2. Compute Ray Traced Ambient Occlusion (RTAO)
        rtao_intensity = params.get("rtao_intensity", 0.75)
        rtao_radius = params.get("rtao_radius", 1.2)
        if rtao_intensity > 0.0:
            ao_factor = self._compute_rtao(pos, normals, depth, radius=rtao_radius)
            # Modulate intensity
            ao = 1.0 - (1.0 - ao_factor) * rtao_intensity
        else:
            ao = torch.ones((b, 1, h, w), device=self.device, dtype=self.dtype)

        # 3. Compute Ray Traced Global Illumination (RTGI)
        rtgi_intensity = params.get("rtgi_intensity", 0.65)
        rtgi_range = params.get("rtgi_range", 4.0)
        rtgi_steps = int(params.get("rtgi_steps", 10))
        if rtgi_intensity > 0.0:
            indirect_light = self._compute_rtgi(
                color, pos, normals, depth, max_steps=rtgi_steps, max_dist=rtgi_range
            )
            indirect_light = indirect_light * rtgi_intensity
        else:
            indirect_light = torch.zeros_like(color)

        # 4. Compute Screen-Space Reflections (SSR)
        ssr_intensity = params.get("ssr_intensity", 0.5)
        roughness = params.get("roughness", 0.25)
        wet_track_mode = params.get("wet_track_mode", False)
        if ssr_intensity > 0.0:
            ssr_buffer = self._compute_ssr(
                color, pos, normals, depth, roughness=roughness, wet_track_mode=wet_track_mode
            )
            ssr_buffer = ssr_buffer * ssr_intensity
        else:
            ssr_buffer = torch.zeros_like(color)

        return {
            "rtgi": indirect_light,
            "ssr": ssr_buffer,
            "rtao": ao,
            "normals": normals,
            "positions": pos,
        }

    def _compute_rtao(
        self,
        pos: torch.Tensor,
        normals: torch.Tensor,
        depth: torch.Tensor,
        radius: float = 1.2,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """
        Computes screen-space Ray Traced Ambient Occlusion.
        Samples surrounding hemisphere in 3D camera space.
        """
        b, _, h, w = pos.shape
        occlusion = torch.zeros((b, 1, h, w), device=self.device, dtype=self.dtype)

        # Stratified sample angles around circle
        angles = torch.linspace(0, 2 * math.pi, num_samples + 1, device=self.device, dtype=self.dtype)[:-1]

        # Multi-scale step sizes
        step_fractions = [0.25, 0.5, 0.75, 1.0]

        total_weight = 0.0
        for angle in angles:
            dir_x = torch.cos(angle)
            dir_y = torch.sin(angle)

            for frac in step_fractions:
                r_step = radius * frac
                # Convert 3D radius to screen pixel offset based on camera depth Z
                z_depth = torch.clamp(pos[:, 2:3], min=0.5)
                # Projected pixel delta
                px_offset_x = (dir_x * r_step * 500.0 / z_depth).round()
                px_offset_y = (dir_y * r_step * 500.0 / z_depth).round()

                # Clamp pixel offsets to reasonable range
                max_offset = min(h, w) // 8
                px_offset_x = torch.clamp(px_offset_x, -max_offset, max_offset)
                px_offset_y = torch.clamp(px_offset_y, -max_offset, max_offset)

                # Normalized coordinates for grid_sample [-1, 1]
                norm_dx = px_offset_x * (2.0 / w)
                norm_dy = px_offset_y * (2.0 / h)

                # Generate sampling grid
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, h, device=self.device, dtype=self.dtype),
                    torch.linspace(-1, 1, w, device=self.device, dtype=self.dtype),
                    indexing="ij",
                )
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
                sample_grid = grid + torch.cat([norm_dx.permute(0, 2, 3, 1), norm_dy.permute(0, 2, 3, 1)], dim=-1)

                sampled_pos = F.grid_sample(pos, sample_grid, mode="nearest", padding_mode="border", align_corners=False)

                # Vector from surface point to sample point
                v_sample = sampled_pos - pos
                dist = torch.norm(v_sample, dim=1, keepdim=True) + 1e-5

                # Horizon / cosine angle with surface normal
                v_dir = v_sample / dist
                cos_theta = (v_dir * normals).sum(dim=1, keepdim=True)

                # Occlusion test: is sample within radius and above surface plane?
                is_occluder = (cos_theta > 0.05) & (dist < r_step * 1.8) & (dist > 0.02)
                falloff = torch.clamp(1.0 - (dist / (r_step * 1.8)), min=0.0)

                occlusion = occlusion + is_occluder.to(self.dtype) * cos_theta * falloff
                total_weight += 1.0

        occlusion = occlusion / total_weight
        ao = torch.clamp(1.0 - occlusion * 2.2, min=0.0, max=1.0)
        return ao

    def _compute_rtgi(
        self,
        color: torch.Tensor,
        pos: torch.Tensor,
        normals: torch.Tensor,
        depth: torch.Tensor,
        max_steps: int = 8,
        max_dist: float = 3.5,
    ) -> torch.Tensor:
        """
        Computes Screen-Space Ray Traced Global Illumination.
        Gathers bounced light from adjacent surfaces (curbs, barriers, sky, cars).
        """
        b, _, h, w = pos.shape
        indirect_light = torch.zeros_like(color)

        # Primary bounce directions: Hemisphere sampling along normal
        # 6 stratified directions around normal + normal itself
        angles = torch.linspace(0, 2 * math.pi, 7, device=self.device, dtype=self.dtype)[:-1]

        step_dist = max_dist / max_steps
        step_weights = 0.0

        for angle in angles:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # Build tangent frame around normal
            # Up vector for cross product
            up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
            tangent = torch.cross(normals, up.expand_as(normals), dim=1)
            tan_len = torch.norm(tangent, dim=1, keepdim=True) + 1e-6
            tangent = tangent / tan_len
            bitangent = torch.cross(normals, tangent, dim=1)

            # Ray direction in hemisphere
            ray_dir = normals * 0.7 + (tangent * cos_a + bitangent * sin_a) * 0.7
            ray_dir = ray_dir / (torch.norm(ray_dir, dim=1, keepdim=True) + 1e-6)

            # March rays
            for step_idx in range(1, max_steps + 1):
                t = step_idx * step_dist
                march_pos = pos + ray_dir * t

                # Project 3D march position back to screen UV
                z_proj = torch.clamp(march_pos[:, 2:3], min=0.2)
                focal = (h / 2.0) / math.tan(math.radians(65.0) / 2.0)

                u = (march_pos[:, 0:1] * focal / z_proj) + (w / 2.0)
                v = (march_pos[:, 1:2] * focal / z_proj) + (h / 2.0)

                # Normalized grid coords [-1, 1]
                grid_u = (u / (w - 1.0)) * 2.0 - 1.0
                grid_v = (v / (h - 1.0)) * 2.0 - 1.0

                sample_grid = torch.cat([grid_u.permute(0, 2, 3, 1), grid_v.permute(0, 2, 3, 1)], dim=-1)

                # Screen bounds mask
                in_bounds = (grid_u >= -1.0) & (grid_u <= 1.0) & (grid_v >= -1.0) & (grid_v <= 1.0)

                sampled_pos = F.grid_sample(pos, sample_grid, mode="nearest", padding_mode="border", align_corners=False)
                sampled_col = F.grid_sample(color, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
                sampled_norm = F.grid_sample(normals, sample_grid, mode="nearest", padding_mode="border", align_corners=False)

                # Depth collision test
                depth_diff = march_pos[:, 2:3] - sampled_pos[:, 2:3]
                # Hit if within thickness slab
                hit = in_bounds & (depth_diff >= -0.05) & (depth_diff <= 0.6)

                # Normal facing test: don't collect from surfaces facing the same way
                facing = (ray_dir * sampled_norm).sum(dim=1, keepdim=True)
                hit = hit & (facing < 0.2)

                # Distance falloff
                dist_weight = 1.0 / (1.0 + 0.8 * t * t)
                # Lambertian cosine term: ray_dir dot surface normal
                cos_term = torch.clamp((ray_dir * normals).sum(dim=1, keepdim=True), min=0.0)

                weight = hit.to(self.dtype) * dist_weight * cos_term
                indirect_light = indirect_light + sampled_col * weight
                step_weights += 1.0

        if step_weights > 0.0:
            indirect_light = indirect_light / (len(angles) * 1.5)

        return indirect_light

    def _compute_ssr(
        self,
        color: torch.Tensor,
        pos: torch.Tensor,
        normals: torch.Tensor,
        depth: torch.Tensor,
        roughness: float = 0.25,
        wet_track_mode: bool = False,
        max_steps: int = 16,
    ) -> torch.Tensor:
        """
        Computes Screen-Space Ray Traced Reflections (SSR).
        Reflects view vector off surface normal and marches along depth buffer.
        """
        b, _, h, w = pos.shape
        ssr_color = torch.zeros_like(color)

        # View vector V: from surface point to camera (0, 0, 0)
        view_dir = -pos / (torch.norm(pos, dim=1, keepdim=True) + 1e-6)

        # Specular reflection vector R = 2(N dot V)N - V
        n_dot_v = torch.clamp((normals * view_dir).sum(dim=1, keepdim=True), min=0.01)
        refl_dir = 2.0 * n_dot_v * normals - view_dir
        refl_dir = refl_dir / (torch.norm(refl_dir, dim=1, keepdim=True) + 1e-6)

        # Wet track mask: roads/lower surfaces in simracing are mostly upward facing (Normal Y > 0.5)
        if wet_track_mode:
            # Boost reflections on horizontal surfaces (puddles / wet tarmac)
            horizontal_mask = torch.clamp(normals[:, 1:2] * 2.0, min=0.0, max=1.0)
        else:
            horizontal_mask = torch.ones((b, 1, h, w), device=self.device, dtype=self.dtype)

        # Fresnel reflectance (Schlick approximation)
        # F0 = 0.04 for typical dielectric / asphalt / car paint clearcoat, higher for metals
        f0 = 0.05
        fresnel = f0 + (1.0 - f0) * torch.pow(1.0 - n_dot_v, 5.0)

        # Raymarching along reflection vector
        step_size = 0.25
        hit_mask = torch.zeros((b, 1, h, w), device=self.device, dtype=self.dtype)
        accum_color = torch.zeros_like(color)

        focal = (h / 2.0) / math.tan(math.radians(65.0) / 2.0)

        for step in range(1, max_steps + 1):
            t = step * step_size
            march_pos = pos + refl_dir * t

            z_proj = torch.clamp(march_pos[:, 2:3], min=0.2)
            u = (march_pos[:, 0:1] * focal / z_proj) + (w / 2.0)
            v = (march_pos[:, 1:2] * focal / z_proj) + (h / 2.0)

            grid_u = (u / (w - 1.0)) * 2.0 - 1.0
            grid_v = (v / (h - 1.0)) * 2.0 - 1.0

            sample_grid = torch.cat([grid_u.permute(0, 2, 3, 1), grid_v.permute(0, 2, 3, 1)], dim=-1)
            in_bounds = (grid_u >= -0.98) & (grid_u <= 0.98) & (grid_v >= -0.98) & (grid_v <= 0.98)

            sampled_pos = F.grid_sample(pos, sample_grid, mode="nearest", padding_mode="border", align_corners=False)
            sampled_col = F.grid_sample(color, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)

            depth_diff = march_pos[:, 2:3] - sampled_pos[:, 2:3]
            # Valid intersection check
            is_hit = in_bounds & (depth_diff >= 0.0) & (depth_diff <= 0.4) & (hit_mask < 0.5)

            # Edge fade to smoothly eliminate border artifacts
            edge_fade_x = torch.clamp(1.0 - torch.abs(grid_u), min=0.0)
            edge_fade_y = torch.clamp(1.0 - torch.abs(grid_v), min=0.0)
            edge_weight = torch.clamp(edge_fade_x * edge_fade_y * 10.0, max=1.0)

            hit_float = is_hit.to(self.dtype)
            accum_color = accum_color + sampled_col * hit_float * edge_weight
            hit_mask = torch.clamp(hit_mask + hit_float, max=1.0)

        # Modulate reflection with roughness and Fresnel
        roughness_blur_weight = 1.0 - roughness * 0.6
        ssr_color = accum_color * fresnel * horizontal_mask * roughness_blur_weight
        return ssr_color
