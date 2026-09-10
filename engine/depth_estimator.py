"""
AI Depth Estimator Module for SimRTX Studio.
Runs monocular depth estimation on NVIDIA RTX GPUs using CUDA FP16.
Supports Depth Anything V2 and MiDaS architectures.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Union, Tuple


class DepthEstimator:
    def __init__(
        self,
        model_type: str = "depth_anything_v2",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the Depth Estimator.

        Args:
            model_type: 'depth_anything_v2' or 'midas_small' or 'midas_hybrid'
            device: 'cuda' or 'cpu'. Defaults to cuda if available.
            use_fp16: Whether to use half-precision on GPU (recommended for RTX 3080 Ti).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_type = model_type
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.model = None
        self.transform = None
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Loads the selected neural depth model onto the target device."""
        print(f"[SimRTX Depth] Initializing {self.model_type} on {self.device} (FP16={self.use_fp16})...")

        if self.model_type == "depth_anything_v2":
            try:
                # Try loading via Hugging Face Transformers pipeline
                from transformers import pipeline
                torch_dtype = torch.float16 if self.use_fp16 else torch.float32
                device_idx = 0 if self.device.type == "cuda" else -1
                self.pipe = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=device_idx,
                    torch_dtype=torch_dtype,
                )
                print("[SimRTX Depth] Successfully loaded Depth-Anything-V2-Small-hf via transformers.")
                return
            except Exception as e:
                print(f"[SimRTX Depth] Transformers pipeline load note: {e}. Falling back to MiDaS...")
                self.model_type = "midas_small"

        if self.model_type in ["midas_small", "midas_hybrid"]:
            try:
                hub_model = "MiDaS_small" if self.model_type == "midas_small" else "DPT_Hybrid"
                self.model = torch.hub.load("intel-isl/MiDaS", hub_model, trust_repo=True)
                self.model.to(self.device)
                self.model.eval()

                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                if self.model_type == "midas_small":
                    self.transform = midas_transforms.small_transform
                else:
                    self.transform = midas_transforms.dpt_transform

                if self.use_fp16:
                    self.model.half()

                print(f"[SimRTX Depth] Successfully loaded {hub_model} via torch.hub.")
            except Exception as e:
                print(f"[SimRTX Depth] Error loading MiDaS: {e}. Falling back to geometric gradient depth.")
                self.model = None

    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Estimate depth map from an input RGB image or video frame.

        Args:
            image: RGB image as numpy array (H, W, 3) in [0, 255] or PIL Image.
            target_size: Optional (width, height) to resize depth map.

        Returns:
            torch.Tensor: (1, 1, H, W) normalized depth tensor in range [0.0, 1.0] on self.device.
                          0.0 represents close / near plane, 1.0 represents far plane.
        """
        if isinstance(image, Image.Image):
            pil_img = image
            np_img = np.array(image)
        else:
            np_img = image
            pil_img = Image.fromarray(image)

        orig_h, orig_w = np_img.shape[:2]

        if self.pipe is not None:
            # Using Depth Anything V2 pipeline
            result = self.pipe(pil_img)
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil, dtype=np.float32)

            # Invert so 0 = near, 1 = far
            d_min = depth_np.min()
            d_max = depth_np.max()
            if d_max - d_min > 1e-6:
                depth_norm = 1.0 - ((depth_np - d_min) / (d_max - d_min))
            else:
                depth_norm = np.zeros_like(depth_np)

            depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            if self.use_fp16:
                depth_tensor = depth_tensor.half()

            if target_size is not None and (target_size[0] != orig_w or target_size[1] != orig_h):
                depth_tensor = F.interpolate(depth_tensor, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)

            return depth_tensor

        elif self.model is not None:
            # Using MiDaS
            input_batch = self.transform(np_img).to(self.device)
            if self.use_fp16:
                input_batch = input_batch.half()

            prediction = self.model(input_batch)

            # MiDaS outputs inverse disparity (higher = closer)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            )

            p_min = prediction.min()
            p_max = prediction.max()
            if p_max - p_min > 1e-6:
                # Invert disparity to depth: 0 = near, 1 = far
                depth_tensor = 1.0 - ((prediction - p_min) / (p_max - p_min))
            else:
                depth_tensor = torch.zeros_like(prediction)

            if target_size is not None and (target_size[0] != orig_w or target_size[1] != orig_h):
                depth_tensor = F.interpolate(depth_tensor, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)

            return depth_tensor

        else:
            # Fast fallback heuristic if offline: Y-axis linear perspective depth (typical in racing games)
            y_coords = np.linspace(0.0, 1.0, orig_h, dtype=np.float32)[:, None]
            depth_synthetic = np.repeat(y_coords, orig_w, axis=1)
            depth_tensor = torch.from_numpy(depth_synthetic).unsqueeze(0).unsqueeze(0).to(self.device)
            if self.use_fp16:
                depth_tensor = depth_tensor.half()
            return depth_tensor
