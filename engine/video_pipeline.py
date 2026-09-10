"""
Video Processing Pipeline for SimRTX Studio.
Handles:
- Frame reading and preview extraction
- High-throughput GPU batch processing on RTX 3080 Ti
- Preserving original audio tracks
- Hardware-accelerated NVENC encoding via FFmpeg
"""

import os
import subprocess
import time
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable, Tuple
from PIL import Image

from .depth_estimator import DepthEstimator
from .raytracer import ScreenSpaceRaytracer
from .denoiser import BilateralDenoiser
from .postprocess import ColorGrader


class VideoPipeline:
    def __init__(
        self,
        depth_estimator: Optional[DepthEstimator] = None,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        if depth_estimator is not None:
            self.depth_estimator = depth_estimator
        else:
            self.depth_estimator = DepthEstimator(device=str(self.device), use_fp16=self.use_fp16)

        self.raytracer = ScreenSpaceRaytracer(device=self.device, dtype=self.dtype)
        self.denoiser = BilateralDenoiser(device=self.device)
        self.grader = ColorGrader(device=self.device)

    def process_single_frame(
        self,
        frame_rgb: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single RGB frame for live UI preview.

        Returns:
            Tuple of:
                - output_rgb (H, W, 3) in uint8 [0, 255]
                - depth_viz (H, W, 3) in uint8 [0, 255]
                - normals_viz (H, W, 3) in uint8 [0, 255]
        """
        h, w = frame_rgb.shape[:2]

        # 1. Depth Estimation
        depth = self.depth_estimator.estimate_depth(frame_rgb).to(device=self.device, dtype=self.dtype)

        # 2. Prepare color tensor
        color_np = frame_rgb.astype(np.float32) / 255.0
        color_tensor = torch.from_numpy(color_np).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # 3. Ray Tracing
        with torch.no_grad():
            rt_results = self.raytracer.trace(color_tensor, depth, params)

            # 4. Denoise RTGI & SSR
            if params.get("denoise", True):
                if params.get("rtgi_intensity", 0.65) > 0:
                    rt_results["rtgi"] = self.denoiser.denoise(
                        rt_results["rtgi"], depth, rt_results["normals"]
                    )
                if params.get("ssr_intensity", 0.5) > 0:
                    rt_results["ssr"] = self.denoiser.denoise(
                        rt_results["ssr"], depth, rt_results["normals"]
                    )

            # 5. Color Grading & Compositing
            output_tensor = self.grader.grade(color_tensor, rt_results, params)

        # Convert back to uint8 numpy
        out_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Visualizations for Depth & Normals
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_viz = (depth_np * 255.0).clip(0, 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)

        normals_np = rt_results["normals"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        normals_viz = ((normals_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

        return out_np, depth_viz, normals_viz

    def process_video(
        self,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> str:
        """
        Processes an entire video file with Ray Tracing, Color Grading, and Audio preservation.

        Args:
            input_path: Path to input video file.
            output_path: Destination path for output video file.
            params: Parameters dictionary.
            progress_callback: Optional callback(current_frame, total_frames, fps, eta_seconds).

        Returns:
            Path to rendered video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_video_no_audio = output_path.replace(".mp4", "_temp_raw.mp4")
        temp_audio = output_path.replace(".mp4", "_audio.aac")

        # Step 1: Extract audio if present
        has_audio = self._extract_audio(input_path, temp_audio)

        # Step 2: Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))

        print(f"[SimRTX] Rendering video {width}x{height} @ {fps:.1f} FPS, total {total_frames} frames...")

        start_time = time.time()
        frame_idx = 0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Process through RTX Raytracing engine
                out_rgb, _, _ = self.process_single_frame(frame_rgb, params)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

                writer.write(out_bgr)
                frame_idx += 1

                elapsed = time.time() - start_time
                current_fps = frame_idx / (elapsed + 1e-5)
                eta = (total_frames - frame_idx) / (current_fps + 1e-5) if total_frames > 0 else 0.0

                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames, current_fps, eta)

        finally:
            cap.release()
            writer.release()

        # Step 3: Combine with audio & encode via NVENC/FFmpeg
        self._finalize_video(temp_video_no_audio, temp_audio if has_audio else None, output_path, fps)

        # Cleanup temp files
        if os.path.exists(temp_video_no_audio):
            try:
                os.remove(temp_video_no_audio)
            except Exception:
                pass
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass

        print(f"[SimRTX] Render complete! Saved to {output_path}")
        return output_path

    def _extract_audio(self, video_path: str, audio_output: str) -> bool:
        """Extracts audio track using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "copy", audio_output
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return (res.returncode == 0) and os.path.exists(audio_output) and os.path.getsize(audio_output) > 0
        except Exception:
            return False

    def _finalize_video(self, video_path: str, audio_path: Optional[str], output_path: str, fps: float):
        """
        Combines processed video with audio and encodes via NVIDIA NVENC if available.
        """
        # Try NVENC first (RTX 3080 Ti hardware encoder)
        cmd_nvenc = ["ffmpeg", "-y", "-i", video_path]
        if audio_path and os.path.exists(audio_path):
            cmd_nvenc.extend(["-i", audio_path, "-c:a", "aac"])

        cmd_nvenc.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p7",  # Highest quality NVENC preset
            "-rc", "vbr",
            "-cq", "19",
            "-pix_fmt", "yuv420p",
            output_path
        ])

        try:
            res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print("[SimRTX] Successfully encoded with NVIDIA NVENC hardware acceleration.")
                return
        except Exception as e:
            print(f"[SimRTX] NVENC encoding note: {e}. Falling back to standard encoder.")

        # Fallback to libx264 or simple copy
        cmd_cpu = ["ffmpeg", "-y", "-i", video_path]
        if audio_path and os.path.exists(audio_path):
            cmd_cpu.extend(["-i", audio_path, "-c:a", "aac"])
        cmd_cpu.extend([
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ])

        try:
            res = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                return
        except Exception:
            pass

        # Final fallback: if ffmpeg failed or is missing, move the raw mp4
        if os.path.exists(video_path):
            import shutil
            shutil.copyfile(video_path, output_path)
