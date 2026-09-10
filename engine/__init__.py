"""
SimRTX Studio - Engine Package
Ray Tracing, Depth Estimation, and Photorealistic Post-Processing for Gaming Videos.
"""

from .depth_estimator import DepthEstimator
from .geometry import compute_normals, reconstruct_positions
from .raytracer import ScreenSpaceRaytracer
from .denoiser import BilateralDenoiser
from .postprocess import ColorGrader
from .video_pipeline import VideoPipeline, get_video_info
from .hardware import detect_gpu_hardware, get_profile_settings, GPUHardwareProfile

__all__ = [
    "DepthEstimator",
    "compute_normals",
    "reconstruct_positions",
    "ScreenSpaceRaytracer",
    "BilateralDenoiser",
    "ColorGrader",
    "VideoPipeline",
    "get_video_info",
    "detect_gpu_hardware",
    "get_profile_settings",
    "GPUHardwareProfile",
]
