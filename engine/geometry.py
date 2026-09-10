"""
Geometry and Surface Normal Reconstruction for SimRTX Studio.
Reconstructs 3D View-Space positions and surface normals from 2D depth buffers.
"""

import math
import torch
import torch.nn.functional as F


def reconstruct_positions(
    depth: torch.Tensor,
    fov_deg: float = 65.0,
    z_near: float = 0.5,
    z_far: float = 200.0,
) -> torch.Tensor:
    """
    Reconstructs 3D camera-space positions (X, Y, Z) for each pixel.

    Args:
        depth: Normalized depth tensor (B, 1, H, W) in [0, 1], where 0 is near, 1 is far.
        fov_deg: Field of view in degrees (typical simracing cockpit/chase cam is ~50-70 deg).
        z_near: Near clipping plane distance in meters.
        z_far: Far clipping plane distance in meters.

    Returns:
        torch.Tensor: (B, 3, H, W) View-space 3D coordinates.
    """
    b, c, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Map normalized depth [0, 1] to metric distance Z
    # Using non-linear or linear metric depth curve
    z = z_near + depth * (z_far - z_near)

    # Compute focal length based on FOV
    fov_rad = math.radians(fov_deg)
    focal_y = (h / 2.0) / math.tan(fov_rad / 2.0)
    focal_x = focal_y  # Square pixels assumption

    c_x = w / 2.0
    c_y = h / 2.0

    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    x_grid = (x_coords - c_x) / focal_x
    y_grid = (y_coords - c_y) / focal_y

    pos_x = x_grid * z
    pos_y = y_grid * z
    pos_z = z

    return torch.cat([pos_x, pos_y, pos_z], dim=1)


def compute_normals(
    depth: torch.Tensor,
    fov_deg: float = 65.0,
    smooth_normals: bool = True,
) -> torch.Tensor:
    """
    Computes view-space surface normal vectors from depth map gradients.
    Uses edge-aware cross-product differences to avoid artifacts at silhouettes.

    Args:
        depth: (B, 1, H, W) normalized depth tensor in [0, 1].
        fov_deg: Camera field of view.
        smooth_normals: If True, applies an edge-preserving slight smooth.

    Returns:
        torch.Tensor: (B, 3, H, W) normalized normal vectors in [-1, 1].
    """
    pos = reconstruct_positions(depth, fov_deg=fov_deg)
    b, _, h, w = pos.shape

    # Position gradients in X and Y directions
    # Pad to preserve spatial dimensions
    pos_padded = F.pad(pos, (1, 1, 1, 1), mode="replicate")

    # Center differences:
    # dP_dx = pos(x+1, y) - pos(x-1, y)
    # dP_dy = pos(x, y+1) - pos(x, y-1)
    dp_dx_c = pos_padded[:, :, 1:-1, 2:] - pos_padded[:, :, 1:-1, :-2]
    dp_dy_c = pos_padded[:, :, 2:, 1:-1] - pos_padded[:, :, :-2, 1:-1]

    # Forward differences:
    dp_dx_f = pos_padded[:, :, 1:-1, 2:] - pos_padded[:, :, 1:-1, 1:-1]
    dp_dy_f = pos_padded[:, :, 2:, 1:-1] - pos_padded[:, :, 1:-1, 1:-1]

    # Backward differences:
    dp_dx_b = pos_padded[:, :, 1:-1, 1:-1] - pos_padded[:, :, 1:-1, :-2]
    dp_dy_b = pos_padded[:, :, 1:-1, 1:-1] - pos_padded[:, :, :-2, 1:-1]

    # Use smaller magnitude difference to prevent edge bleeding across depth discontinuities
    dist_f_x = torch.abs(dp_dx_f[:, 2:3])
    dist_b_x = torch.abs(dp_dx_b[:, 2:3])
    choose_b_x = (dist_b_x < dist_f_x).to(dtype=depth.dtype)
    dp_dx = choose_b_x * dp_dx_b + (1.0 - choose_b_x) * dp_dx_f

    dist_f_y = torch.abs(dp_dy_f[:, 2:3])
    dist_b_y = torch.abs(dp_dy_b[:, 2:3])
    choose_b_y = (dist_b_y < dist_f_y).to(dtype=depth.dtype)
    dp_dy = choose_b_y * dp_dy_b + (1.0 - choose_b_y) * dp_dy_f

    # Cross product: Normal = dP_dx x dP_dy
    normal = torch.cross(dp_dx, dp_dy, dim=1)

    # Normalize vectors
    norm_len = torch.norm(normal, dim=1, keepdim=True) + 1e-6
    normal = normal / norm_len

    # Ensure normals point toward the camera (Z < 0 in standard view-space or consistent orientation)
    z_flip = torch.where(normal[:, 2:3] > 0, -1.0, 1.0).to(dtype=depth.dtype)
    normal = normal * z_flip

    if smooth_normals:
        # Subtle 3x3 gaussian smoothing for stable lighting
        kernel = (torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=depth.device, dtype=torch.float32) / 16.0).to(dtype=normal.dtype)
        kernel = kernel.repeat(3, 1, 1, 1)
        normal = F.conv2d(normal, kernel, padding=1, groups=3)
        normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-6)

    return normal
