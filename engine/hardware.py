"""
Hardware Profile & Architecture Detection Module for SimRTX Studio.
Supports:
- RTX 20-Series (Turing / Compute 7.5, e.g. RTX 2060, 2070, 2080)
- RTX 30-Series (Ampere / Compute 8.6, e.g. RTX 3060, 3070, 3080, 3090)
- RTX 40-Series (Ada Lovelace / Compute 8.9, e.g. RTX 4060, 4070, 4080, 4090)
- RTX 50-Series (Blackwell / Compute 9.0+, future-ready)
Automatically optimizes VRAM memory usage, raymarching sample density, and NVENC encoding presets.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class GPUHardwareProfile:
    device_name: str
    architecture: str
    generation: str
    compute_capability: Tuple[int, int]
    vram_gb: float
    recommended_profile: str
    fp16_supported: bool
    bf16_supported: bool
    description: str


# Preset configurations for different hardware profiles
HARDWARE_PROFILES = {
    "blackwell": {
        "name": "RTX 50-Serie (Blackwell) / 16-32 GB (Hyper-Path Tracing & 8K)",
        "rtgi_steps": 18,
        "ssr_steps": 24,
        "rtao_samples": 12,
        "rtao_radius": 1.6,
        "max_internal_res": 4320,  # Native 8K Tensor-Processing
        "nvenc_preset": "p7",
        "encoder_codec": "av1_nvenc",
        "dual_nvenc": True,
        "empty_cache_freq": 120,
        "batch_size": 8,
        "fp8_supported": True,
        "bf16_supported": True,
    },
    "ultra": {
        "name": "RTX 40-Serie / RTX 3080 Ti / 12-16 GB (Ultra Quality)",
        "rtgi_steps": 12,
        "ssr_steps": 16,
        "rtao_samples": 8,
        "rtao_radius": 1.4,
        "max_internal_res": 2160,  # Up to 4K
        "nvenc_preset": "p7",
        "encoder_codec": "hevc_nvenc",
        "dual_nvenc": False,
        "empty_cache_freq": 60,
        "batch_size": 4,
        "fp8_supported": False,
        "bf16_supported": True,
    },
    "balanced": {
        "name": "RTX 30-Serie / 8-12 GB VRAM (Ausgewogen)",
        "rtgi_steps": 10,
        "ssr_steps": 12,
        "rtao_samples": 6,
        "rtao_radius": 1.3,
        "max_internal_res": 1080,  # Full 1080p
        "nvenc_preset": "p6",
        "encoder_codec": "h264_nvenc",
        "dual_nvenc": False,
        "empty_cache_freq": 30,
        "batch_size": 2,
        "fp8_supported": False,
        "bf16_supported": True,
    },
    "low_vram": {
        "name": "RTX 20-Serie / 6-8 GB VRAM (Performance & Memory-Saver)",
        "rtgi_steps": 6,
        "ssr_steps": 8,
        "rtao_samples": 4,
        "rtao_radius": 1.1,
        "max_internal_res": 720,   # 720p internal scaling
        "nvenc_preset": "p4",
        "encoder_codec": "h264_nvenc",
        "dual_nvenc": False,
        "empty_cache_freq": 10,
        "batch_size": 1,
        "fp8_supported": False,
        "bf16_supported": False,
    },
}


def detect_gpu_hardware() -> GPUHardwareProfile:
    """
    Analyzes installed GPU hardware, architecture, compute capability and VRAM.
    Specifically detects RTX 50-Series (Blackwell), RTX 40 (Ada), RTX 30 (Ampere), and RTX 20 (Turing).
    """
    if not torch.cuda.is_available():
        return GPUHardwareProfile(
            device_name="CPU Modus (Keine CUDA GPU)",
            architecture="CPU",
            generation="CPU",
            compute_capability=(0, 0),
            vram_gb=0.0,
            recommended_profile="low_vram",
            fp16_supported=False,
            bf16_supported=False,
            description="Software-Rendering aktiv (Keine NVIDIA Grafikkarte erkannt)."
        )

    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(0)

    # Blackwell RTX 50 Detection (GB202, GB203, GB205, GB206, GB207, Compute 10.0+ or device string)
    is_blackwell = (
        "5090" in device_name
        or "5080" in device_name
        or "5070" in device_name
        or "5060" in device_name
        or "RTX 50" in device_name
        or "Blackwell" in device_name
        or "GB20" in device_name
        or major >= 10
    )

    # Detect Architecture Generation
    if is_blackwell:
        arch = "Blackwell"
        gen = "NVIDIA RTX 50-Serie (Blackwell)"
    elif major == 8 and minor == 9:
        arch = "Ada Lovelace"
        gen = "NVIDIA RTX 40-Serie (Ada Lovelace)"
    elif major == 8 and minor == 6:
        arch = "Ampere"
        gen = "NVIDIA RTX 30-Serie (Ampere)"
    elif major == 8 and minor == 0:
        arch = "Ampere Data Center"
        gen = "NVIDIA Ampere A100"
    elif major == 7 and minor == 5:
        arch = "Turing"
        gen = "NVIDIA RTX 20-Serie (Turing)"
    elif major == 7 and minor == 0:
        arch = "Volta"
        gen = "NVIDIA Titan V / V100"
    elif major == 6:
        arch = "Pascal"
        gen = "NVIDIA GTX 10-Serie (Pascal)"
    elif major >= 9:
        arch = "Blackwell / Hopper"
        gen = "NVIDIA RTX 50-Serie / Compute 9.x+"
    else:
        arch = f"CUDA SM {major}.{minor}"
        gen = f"NVIDIA GPU (SM {major}.{minor})"

    # Select recommended profile based on architecture and VRAM capacity
    if is_blackwell or vram_gb >= 15.5:
        recommended = "blackwell"
    elif vram_gb < 7.5:
        recommended = "low_vram"
    elif vram_gb <= 11.0:
        recommended = "balanced"
    else:
        recommended = "ultra"

    fp16_supported = (major >= 7)
    bf16_supported = (major >= 8)

    desc = f"{device_name} ({gen}) mit {vram_gb:.1f} GB VRAM"

    return GPUHardwareProfile(
        device_name=device_name,
        architecture=arch,
        generation=gen,
        compute_capability=(major, minor),
        vram_gb=vram_gb,
        recommended_profile=recommended,
        fp16_supported=fp16_supported,
        bf16_supported=bf16_supported,
        description=desc,
    )


def get_profile_settings(profile_key: str) -> Dict[str, Any]:
    """Retrieves settings dict for a given profile key ('blackwell', 'ultra', 'balanced', 'low_vram')."""
    return HARDWARE_PROFILES.get(profile_key, HARDWARE_PROFILES["ultra"])


def get_all_profiles() -> Dict[str, str]:
    """Returns mapping of profile_key -> display_name."""
    return {k: v["name"] for k, v in HARDWARE_PROFILES.items()}
