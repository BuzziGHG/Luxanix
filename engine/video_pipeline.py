"""
Video Processing Pipeline for Luxanix Studio.
Handles:
- Video metadata inspection (resolution, aspect ratio, duration, fps)
- Timeline trimming / cutting (start and end time selection)
- High-throughput GPU batch processing on RTX GPUs
- Preserving original audio tracks (with exact trim synchronization)
- Hardware-accelerated NVENC (H.264 & HEVC) encoding via FFmpeg
- Detailed live progress reporting (Frames, FPS, Elapsed Time, ETA, Memory)
"""

import os
import subprocess
import time
import math
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable, Tuple
from PIL import Image

from .depth_estimator import DepthEstimator
from .raytracer import ScreenSpaceRaytracer
from .denoiser import BilateralDenoiser
from .postprocess import ColorGrader
from .auto_preset import AutoSceneOptimizer


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extracts detailed video metadata for the UI stats card.
    """
    if not os.path.exists(video_path):
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    # Determine Aspect Ratio
    gcd_val = math.gcd(width, height) if (width > 0 and height > 0) else 1
    aspect_w = width // gcd_val
    aspect_h = height // gcd_val

    if abs(width / max(1, height) - 16 / 9) < 0.05:
        aspect_label = "16:9 (Standard Widescreen)"
    elif abs(width / max(1, height) - 21 / 9) < 0.1:
        aspect_label = "21:9 (Ultrawide Gaming)"
    elif abs(width / max(1, height) - 32 / 9) < 0.1:
        aspect_label = "32:9 (Super-Ultrawide / Triple-Screen)"
    elif abs(width / max(1, height) - 4 / 3) < 0.05:
        aspect_label = "4:3 (Classic)"
    elif abs(width / max(1, height) - 9 / 16) < 0.05:
        aspect_label = "9:16 (Vertikal / TikTok / Shorts / Reels)"
    else:
        aspect_label = f"{aspect_w}:{aspect_h}"

    # Determine Resolution Label
    if height >= 2160 or width >= 3840:
        res_label = f"4K Ultra HD ({width} x {height})"
    elif height >= 1440 or width >= 2560:
        res_label = f"1440p 2K QHD ({width} x {height})"
    elif height >= 1080 or width >= 1920:
        res_label = f"1080p Full HD ({width} x {height})"
    elif height >= 720:
        res_label = f"720p HD ({width} x {height})"
    else:
        res_label = f"{width} x {height} SD"

    # Format duration
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins:02d}:{secs:02d} ({duration_sec:.1f}s)"

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "duration_str": duration_str,
        "aspect_ratio": aspect_label,
        "resolution_label": res_label,
    }


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
        self.auto_optimizer = AutoSceneOptimizer()

    def process_single_frame(
        self,
        frame_rgb: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single RGB frame for live UI preview.
        Supports adaptive internal resolution scaling for low VRAM cards (e.g. RTX 2060/2070).
        """
        orig_h, orig_w = frame_rgb.shape[:2]
        max_internal_res = params.get("max_internal_res", None)

        # Adaptive downscaling for low-memory GPUs if necessary
        needs_resize = (max_internal_res is not None) and (orig_h > max_internal_res)
        if needs_resize:
            scale = max_internal_res / orig_h
            proc_w = int(orig_w * scale)
            proc_h = int(max_internal_res)
            frame_proc = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            frame_proc = frame_rgb

        # Auto-Adaptive AI Preset (Dynamic Scene Optimizer per Frame/Millisecond)
        if params.get("auto_preset", False):
            params = self.auto_optimizer.analyze_and_optimize(frame_proc, None, params)

        # 1. Depth Estimation
        depth = self.depth_estimator.estimate_depth(frame_proc).to(device=self.device, dtype=self.dtype)

        # 2. Prepare color tensor
        color_np = frame_proc.astype(np.float32) / 255.0
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

            # 5. Color Grading, Detail Clarity & Compositing
            output_tensor = self.grader.grade(color_tensor, rt_results, params)

        # Convert back to uint8 numpy
        out_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Upscale back to original resolution if scaled
        if needs_resize:
            out_np = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Visualizations for Depth & Normals
        depth_np = depth.squeeze().cpu().float().numpy()
        depth_viz = (depth_np * 255.0).clip(0, 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)
        if needs_resize:
            depth_viz = cv2.resize(depth_viz, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        normals_np = rt_results["normals"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        normals_viz = ((normals_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        if needs_resize:
            normals_viz = cv2.resize(normals_viz, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return out_np, depth_viz, normals_viz

    def process_video(
        self,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, float, float, float], None]] = None,
    ) -> str:
        """
        Processes an entire video file or cut segment with Ray Tracing,
        Color Grading, Audio preservation, and detailed ETA tracking.

        Args:
            input_path: Path to input video file.
            output_path: Destination path for output video file.
            params: Parameters dictionary.
            progress_callback: Optional callback(curr_frame, total_frames, fps, eta_seconds, elapsed_seconds).

        Returns:
            Path to rendered video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Kann Video nicht öffnen: {input_path}")

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_video_frames / fps if fps > 0 else 0.0

        self.auto_optimizer.reset()

        # Trimming Handling
        enable_trim = params.get("enable_trim", False)
        trim_start = float(params.get("trim_start", 0.0))
        trim_end = float(params.get("trim_end", total_duration))

        if enable_trim:
            start_frame = max(0, int(trim_start * fps))
            end_frame = min(total_video_frames, int(trim_end * fps)) if trim_end > trim_start else total_video_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            total_frames_to_process = max(1, end_frame - start_frame)
            print(f"[Luxanix] Trimming aktiv: Sekunde {trim_start:.1f} bis {trim_end:.1f} (Frames {start_frame} bis {end_frame}, Gesamt: {total_frames_to_process} Frames)")
        else:
            start_frame = 0
            end_frame = total_video_frames
            total_frames_to_process = total_video_frames

        # Target Output Resolution (Supports 1080p, 1440p, 4K UHD, and 8K Ultra-Upscaling)
        out_res_choice = str(params.get("output_resolution", "Original"))
        out_w, out_h = width, height
        if "1080p" in out_res_choice and height != 1080:
            out_h = 1080
            out_w = int(width * (1080 / height))
            out_w = out_w - (out_w % 2)
        elif "1440p" in out_res_choice and height != 1440:
            out_h = 1440
            out_w = int(width * (1440 / height))
            out_w = out_w - (out_w % 2)
        elif "4K" in out_res_choice and height != 2160:
            out_h = 2160
            out_w = int(width * (2160 / height))
            out_w = out_w - (out_w % 2)
        elif "8K" in out_res_choice and height != 4320:
            out_h = 4320
            out_w = int(width * (4320 / height))
            out_w = out_w - (out_w % 2)

        # Respect hardware encoder maximum limit (8192 for NVIDIA NVENC)
        if out_w > 8192:
            out_h = int(out_h * (8192 / out_w))
            out_h = out_h - (out_h % 2)
            out_w = 8192
        if out_h > 8192:
            out_w = int(out_w * (8192 / out_h))
            out_w = out_w - (out_w % 2)
            out_h = 8192

        temp_video_no_audio = output_path.replace(".mp4", "_temp_raw.mp4")
        temp_audio = output_path.replace(".mp4", "_audio.aac")

        # Step 1: Extract trimmed audio
        has_audio = self._extract_audio(
            input_path,
            temp_audio,
            start_time=trim_start if enable_trim else None,
            duration=(trim_end - trim_start) if enable_trim else None
        )

        # Step 2: Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (out_w, out_h))

        print(f"[Luxanix] Rendere Video {out_w}x{out_h} @ {fps:.1f} FPS, zu berechnen: {total_frames_to_process} Frames...")

        start_time = time.time()
        processed_count = 0
        empty_cache_freq = int(params.get("empty_cache_freq", 30))

        try:
            current_frame_pos = start_frame
            while current_frame_pos < end_frame:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Process through RTX Raytracing engine
                out_rgb, _, _ = self.process_single_frame(frame_rgb, params)

                # Scale output if custom resolution selected
                if (out_w != width) or (out_h != height):
                    out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    if out_h >= 4320:
                        # 8K Super-Resolution Detail Enhancement (Edge-preserving clarity)
                        blurred = cv2.GaussianBlur(out_rgb, (0, 0), 1.2)
                        detail = cv2.addWeighted(out_rgb, 1.25, blurred, -0.25, 0)
                        out_rgb = np.clip(detail, 0, 255).astype(np.uint8)

                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                processed_count += 1
                current_frame_pos += 1

                # Periodic CUDA cache cleanup
                if processed_count % empty_cache_freq == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                elapsed = time.time() - start_time
                current_fps = processed_count / (elapsed + 1e-5)
                eta = (total_frames_to_process - processed_count) / (current_fps + 1e-5) if total_frames_to_process > 0 else 0.0

                if progress_callback is not None:
                    progress_callback(processed_count, total_frames_to_process, current_fps, eta, elapsed)

        finally:
            cap.release()
            writer.release()

        # Step 3: Combine with audio & encode via NVENC/FFmpeg
        nvenc_preset = params.get("nvenc_preset", "p7")
        codec = params.get("encoder_codec", "hevc_nvenc" if out_h >= 2160 else "h264_nvenc")
        bitrate_mbps = int(params.get("bitrate_mbps", 60 if out_h >= 4320 else (45 if out_h >= 2160 else 30)))
        dual_nvenc = params.get("dual_nvenc", False)

        # 8K resolution (>4096px) exceeds H.264 level limits; auto-switch to HEVC or AV1
        if (out_w > 4096 or out_h > 4096) and "h264" in codec:
            print(f"[Luxanix] 8K-Auflösung ({out_w}x{out_h}) erfordert HEVC oder AV1. Schalte automatisch auf hevc_nvenc um...")
            codec = "hevc_nvenc"

        self._finalize_video(
            temp_video_no_audio,
            temp_audio if has_audio else None,
            output_path,
            fps,
            codec=codec,
            nvenc_preset=nvenc_preset,
            bitrate_mbps=bitrate_mbps,
            dual_nvenc=dual_nvenc,
        )

        # Cleanup temp files
        for tmp in [temp_video_no_audio, temp_audio]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        print(f"[Luxanix] Render vollständig! Gespeichert in: {output_path}")
        return output_path

    def _extract_audio(
        self,
        video_path: str,
        audio_output: str,
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> bool:
        """Extracts audio track with optional trim range."""
        cmd = ["ffmpeg", "-y"]
        if start_time is not None and start_time > 0:
            cmd.extend(["-ss", f"{start_time:.3f}"])
        cmd.extend(["-i", video_path])
        if duration is not None and duration > 0:
            cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend(["-vn", "-acodec", "aac", audio_output])

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return (res.returncode == 0) and os.path.exists(audio_output) and os.path.getsize(audio_output) > 0
        except Exception:
            return False

    def _finalize_video(
        self,
        video_path: str,
        audio_path: Optional[str],
        output_path: str,
        fps: float,
        codec: str = "hevc_nvenc",
        nvenc_preset: str = "p7",
        bitrate_mbps: int = 35,
        dual_nvenc: bool = False,
    ):
        """
        Combines video and audio with NVIDIA NVENC hardware acceleration (AV1, HEVC, H.264).
        Supports Blackwell & Ada Lovelace Dual-NVENC parallel stream encoding.
        """
        # Prioritize requested codec with intelligent hardware fallbacks
        codecs_to_try = [codec]
        if codec == "av1_nvenc":
            codecs_to_try.extend(["hevc_nvenc", "h264_nvenc"])
        elif codec == "hevc_nvenc":
            codecs_to_try.append("h264_nvenc")

        for current_codec in codecs_to_try:
            cmd_nvenc = ["ffmpeg", "-y", "-i", video_path]
            if audio_path and os.path.exists(audio_path):
                cmd_nvenc.extend(["-i", audio_path, "-c:a", "copy"])

            cmd_nvenc.extend([
                "-c:v", current_codec,
                "-preset", nvenc_preset,
                "-rc", "vbr",
                "-cq", "18",
                "-b:v", f"{bitrate_mbps}M",
                "-maxrate", f"{bitrate_mbps * 2}M",
                "-pix_fmt", "yuv420p",
            ])
            if dual_nvenc and current_codec in ["av1_nvenc", "hevc_nvenc"]:
                cmd_nvenc.extend(["-split_encode", "1"])
            cmd_nvenc.append(output_path)

            try:
                res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"[Luxanix] NVENC Export erfolgreich mit {current_codec} ({nvenc_preset}, {bitrate_mbps} Mbps).")
                    return
            except Exception:
                pass

        # Fallback to libx264
        cmd_cpu = ["ffmpeg", "-y", "-i", video_path]
        if audio_path and os.path.exists(audio_path):
            cmd_cpu.extend(["-i", audio_path, "-c:a", "aac"])
        cmd_cpu.extend([
            "-c:v", "libx264",
            "-crf", "17",
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

        if os.path.exists(video_path):
            import shutil
            shutil.copyfile(video_path, output_path)
