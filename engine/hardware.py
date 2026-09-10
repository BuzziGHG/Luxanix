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
    "low_vram": {
        "name": "RTX 20-Serie / 6-8 GB VRAM (Performance & Memory-Saver)",
        "rtgi_steps": 6,
        "ssr_steps": 8,
        "rtao_samples": 4,
        "rtao_radius": 1.1,
        "max_internal_res": 720,  # Limits internal tensor render to 720p if video is 1080p/4K to save VRAM
        "nvenc_preset": "p4",     # Balanced fast NVENC
        "empty_cache_freq": 10,   # Clear CUDA cache every 10 frames
    },
    "balanced": {
        "name": "RTX 30-Serie / 8-12 GB VRAM (Ausgewogen)",
        "rtgi_steps": 10,
        "ssr_steps": 12,
        "rtao_samples": 6,
        "rtao_radius": 1.3,
        "max_internal_res": 1080, # Full 1080p
        "nvenc_preset": "p6",     # High quality NVENC
        "empty_cache_freq": 30,
    },
    "ultra": {
        "name": "RTX 3080 Ti / RTX 40/50-Serie / 12GB+ (Maximum Quality)",
        "rtgi_steps": 12,
        "ssr_steps": 16,
        "rtao_samples": 8,
        "rtao_radius": 1.4,
        "max_internal_res": 2160, # Up to 4K
        "nvenc_preset": "p7",     # Highest quality NVENC
        "empty_cache_freq": 60,
    },
}


def detect_gpu_hardware() -> GPUHardwareProfile:
    """
    Analyzes installed GPU hardware, architecture, compute capability and VRAM.
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

    # Detect Architecture Generation
    if major == 7 and minor == 5:
        arch = "Turing"
        gen = "NVIDIA RTX 20-Serie (Turing)"
    elif major == 8 and minor == 6:
        arch = "Ampere"
        gen = "NVIDIA RTX 30-Serie (Ampere)"
    elif major == 8 and minor == 0:
        arch = "Ampere Data Center"
        gen = "NVIDIA Ampere A100"
    elif major == 8 and minor == 9:
        arch = "Ada Lovelace"
        gen = "NVIDIA RTX 40-Serie (Ada Lovelace)"
    elif major >= 9:
        arch = "Blackwell / Next-Gen"
        gen = "NVIDIA RTX 50-Serie (Blackwell)"
    elif major == 7 and minor == 0:
        arch = "Volta"
        gen = "NVIDIA Titan V / V100"
    elif major == 6:
        arch = "Pascal"
        gen = "NVIDIA GTX 10-Serie (Pascal)"
    else:
        arch = f"CUDA SM {major}.{minor}"
        gen = f"NVIDIA GPU (SM {major}.{minor})"

    # Select recommended profile based on VRAM capacity
    if vram_gb < 7.5:
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
    """Retrieves settings dict for a given profile key ('low_vram', 'balanced', 'ultra')."""
    return HARDWARE_PROFILES.get(profile_key, HARDWARE_PROFILES["balanced"])
