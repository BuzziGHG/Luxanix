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
import queue
import threading
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
from .auto_realism import AutonomousRealismEngine
from .upscaler import NeuralUpscaler


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tga"}


def get_ffmpeg_binary() -> str:
    """Locates a working FFmpeg binary with NVENC support, checking imageio_ffmpeg if PATH lacks ffmpeg."""
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return "ffmpeg"


def get_video_info(media_path: str) -> Dict[str, Any]:
    """
    Extracts detailed metadata for videos or photos for the UI stats card.
    """
    if not os.path.exists(media_path):
        return {}

    ext = os.path.splitext(media_path)[1].lower()

    # --- PHOTO / IMAGE MODE ---
    if ext in IMAGE_EXTENSIONS:
        img_bgr = cv2.imread(media_path)
        if img_bgr is not None:
            height, width = img_bgr.shape[:2]
        else:
            try:
                pil_img = Image.open(media_path)
                width, height = pil_img.size
            except Exception:
                width, height = 1920, 1080

        gcd_val = math.gcd(width, height) if (width > 0 and height > 0) else 1
        aspect_w = width // gcd_val
        aspect_h = height // gcd_val

        if abs(width / max(1, height) - 16 / 9) < 0.05:
            aspect_label = "16:9 (Breitbild)"
        elif abs(width / max(1, height) - 21 / 9) < 0.1:
            aspect_label = "21:9 (Ultrawide)"
        elif abs(width / max(1, height) - 9 / 16) < 0.05:
            aspect_label = "9:16 (Vertikal / Story)"
        elif abs(width / max(1, height) - 1.0) < 0.05:
            aspect_label = "1:1 (Quadrat)"
        else:
            aspect_label = f"{aspect_w}:{aspect_h}"

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

        return {
            "media_type": "image",
            "width": width,
            "height": height,
            "fps": 30.0,
            "total_frames": 1,
            "duration_sec": 5.0,
            "duration_str": "Standbild (5.0s Clip)",
            "aspect_ratio": aspect_label,
            "resolution_label": res_label,
        }

    # --- VIDEO MODE ---
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

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

    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins:02d}:{secs:02d} ({duration_sec:.1f}s)"

    return {
        "media_type": "video",
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "duration_str": duration_str,
        "aspect_ratio": aspect_label,
        "resolution_label": res_label,
    }


get_media_info = get_video_info


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

        if self.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        if depth_estimator is not None:
            self.depth_estimator = depth_estimator
        else:
            self.depth_estimator = DepthEstimator(device=str(self.device), use_fp16=self.use_fp16)

        self.raytracer = ScreenSpaceRaytracer(device=self.device, dtype=self.dtype)
        self.denoiser = BilateralDenoiser(device=self.device)
        self.grader = ColorGrader(device=self.device)
        self.auto_optimizer = AutoSceneOptimizer()
        self.auto_realism = AutonomousRealismEngine()
        self.upscaler = NeuralUpscaler(device=self.device, use_fp16=self.use_fp16)

    @torch.inference_mode()
    def process_single_frame(
        self,
        frame_rgb: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single RGB frame for live UI preview.
        Supports adaptive internal resolution scaling for low VRAM cards (e.g. RTX 2060/2070).
        When params['fast_preview'] is True (live playback), skips depth estimation and
        raytracing to achieve near real-time 25-30 fps preview — only applies color grading.
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

        # Autonomous AI Realism Engine (Real-time physical calculation without presets)
        if params.get("auto_realism", True) or params.get("auto_preset", False):
            auto_vals = self.auto_realism.analyze_and_compute(frame_proc, master_intensity=params.get("realism_intensity", 1.0))
            for k, v in auto_vals.items():
                params[k] = v

        # ---------------------------------------------------------------
        # FAST PREVIEW PATH — GPU-accelerated real-time RTX preview
        # Calculates depth on a fast 360p thumbnail (~34ms on GPU) and upscales on GPU.
        # Executes GPU ray-marched SSR and contact shadows (RTAO) with preview samples.
        # Photorealismus slider changes are immediately visible.
        # ---------------------------------------------------------------
        if params.get("fast_preview", False):
            h_fp, w_fp = frame_proc.shape[:2]
            intensity = params.get("realism_intensity", 1.0)

            # 360p thumbnail for ultra-fast GPU depth estimation
            thumb_h = 360
            thumb_w = int(w_fp * (thumb_h / h_fp))
            thumb_w = max(16, thumb_w - (thumb_w % 2))
            thumb_rgb = cv2.resize(frame_proc, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            try:
                depth_thumb = self.depth_estimator.estimate_depth(thumb_rgb).to(device=self.device, dtype=self.dtype)
                depth = F.interpolate(depth_thumb, size=(h_fp, w_fp), mode="bilinear", align_corners=False)
            except Exception:
                depth = torch.ones(1, 1, h_fp, w_fp, device=self.device, dtype=self.dtype) * 0.5

            color_np = frame_proc.astype(np.float32) / 255.0
            color_tensor = torch.from_numpy(color_np).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)

            # Fast GPU raytracing configuration
            preview_params = dict(params)
            preview_params["rtao_samples"] = 4
            preview_params["ssr_steps"]    = 8
            preview_params["rtgi_steps"]   = 0
            preview_params["rtgi_intensity"] = 0.0

            rt_results = self.raytracer.trace(color_tensor, depth, preview_params)
            output_tensor = self.grader.grade(color_tensor, rt_results, preview_params)

            out_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if needs_resize:
                out_np = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

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

        # ---------------------------------------------------------------
        # FULL QUALITY PATH — used for export and single-frame scrub preview
        # ---------------------------------------------------------------

        # 1. Depth Estimation
        depth = self.depth_estimator.estimate_depth(frame_proc).to(device=self.device, dtype=self.dtype)

        # 2. Prepare color tensor
        color_np = frame_proc.astype(np.float32) / 255.0
        color_tensor = torch.from_numpy(color_np).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # 3. Ray Tracing
        rt_results = self.raytracer.trace(color_tensor, depth, params)

        # 4. Denoise RTGI & SSR
        if params.get("denoise", True):
            if params.get("rtgi_intensity", 0.25) > 0:
                rt_results["rtgi"] = self.denoiser.denoise(
                    rt_results["rtgi"], depth, rt_results["normals"]
                )
            if params.get("ssr_intensity", 0.15) > 0:
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


    @torch.inference_mode()
    def process_image(
        self,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Processes a single photo/screenshot with full autonomous realism,
        RTX raytracing, color grading, and optional neural AI upscaling (up to 8K).
        Supports instant cancellation via cancel_event.
        """
        c_evt = cancel_event or params.get("cancel_event", None)
        if c_evt is not None and c_evt.is_set():
            raise InterruptedError("Foto-Export durch Benutzer abgebrochen.")

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            pil_img = Image.open(input_path).convert("RGB")
            img_rgb = np.array(pil_img)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        if c_evt is not None and c_evt.is_set():
            raise InterruptedError("Foto-Export durch Benutzer abgebrochen.")

        # 1. RTX Raytracing Engine
        out_rgb, _, _ = self.process_single_frame(img_rgb, params)

        if c_evt is not None and c_evt.is_set():
            raise InterruptedError("Foto-Export durch Benutzer abgebrochen.")

        # 2. Target Output Resolution (1080p, 1440p, 4K UHD, 8K Ultra-Upscaling)
        out_res_choice = str(params.get("output_resolution", "Original"))
        out_w, out_h = orig_w, orig_h
        if "1080p" in out_res_choice and orig_h != 1080:
            out_h = 1080
            out_w = int(orig_w * (1080 / orig_h))
            out_w = out_w - (out_w % 2)
        elif "1440p" in out_res_choice and orig_h != 1440:
            out_h = 1440
            out_w = int(orig_w * (1440 / orig_h))
            out_w = out_w - (out_w % 2)
        elif "4K" in out_res_choice and orig_h != 2160:
            out_h = 2160
            out_w = int(orig_w * (2160 / orig_h))
            out_w = out_w - (out_w % 2)
        elif "8K" in out_res_choice and orig_h != 4320:
            out_h = 4320
            out_w = int(orig_w * (4320 / orig_h))
            out_w = out_w - (out_w % 2)

        # 3. Neural AI Super-Resolution
        if (out_w != orig_w) or (out_h != orig_h):
            if c_evt is not None and c_evt.is_set():
                raise InterruptedError("Foto-Export durch Benutzer abgebrochen.")
            if params.get("neural_upscale", True) and (out_h > orig_h):
                out_rgb = self.upscaler.upscale_frame(out_rgb, target_height=out_h)
                if out_rgb.shape[1] != out_w or out_rgb.shape[0] != out_h:
                    out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        if c_evt is not None and c_evt.is_set():
            raise InterruptedError("Foto-Export durch Benutzer abgebrochen.")

        # 4. Save with optimal fidelity
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        ext = os.path.splitext(output_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(output_path, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        elif ext == ".webp":
            cv2.imwrite(output_path, out_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), 98])
        else:
            if not ext.endswith(".png"):
                output_path = os.path.splitext(output_path)[0] + ".png"
            cv2.imwrite(output_path, out_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        print(f"[Luxanix] Foto erfolgreich gerendert: {out_w}x{out_h} -> {output_path}")
        return output_path

    def _open_ffmpeg_pipe(
        self,
        output_path: str,
        audio_path: Optional[str],
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc_nvenc",
        preset: str = "p7",
        bitrate_mbps: int = 35,
        dual_nvenc: bool = False,
    ) -> Optional[subprocess.Popen]:
        """
        Opens a high-throughput FFmpeg pipe to encode frames directly from memory.
        Avoids writing gigabytes of uncompressed intermediate video files to disk.
        """
        codecs_to_try = [codec]
        if codec == "av1_nvenc":
            codecs_to_try.extend(["hevc_nvenc", "h264_nvenc"])
        elif codec == "hevc_nvenc":
            codecs_to_try.append("h264_nvenc")

        ffmpeg_bin = get_ffmpeg_binary()
        for cur_codec in codecs_to_try:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", f"{fps:.3f}",
                "-i", "-", # stdin
            ]
            if audio_path and os.path.exists(audio_path):
                cmd.extend(["-i", audio_path, "-c:a", "copy"])

            cmd.extend([
                "-c:v", cur_codec,
                "-preset", preset,
                "-rc", "vbr",
                "-cq", "18",
                "-b:v", f"{bitrate_mbps}M",
                "-maxrate", f"{bitrate_mbps * 2}M",
                "-pix_fmt", "yuv420p",
            ])
            if dual_nvenc and cur_codec in ["av1_nvenc", "hevc_nvenc"]:
                cmd.extend(["-split_encode", "1"])
            cmd.append(output_path)

            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return proc
            except Exception:
                continue

        # CPU fallback pipe
        try:
            cmd_cpu = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", f"{fps:.3f}",
                "-i", "-",
            ]
            if audio_path and os.path.exists(audio_path):
                cmd_cpu.extend(["-i", audio_path, "-c:a", "aac"])
            cmd_cpu.extend([
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                output_path
            ])
            return subprocess.Popen(cmd_cpu, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None

    def _prepare_audio_track(
        self,
        orig_audio_path: Optional[str],
        music_path: Optional[str],
        output_audio_path: str,
        video_volume: float = 1.0,
        music_volume: float = 1.0,
        duration_sec: Optional[float] = None,
        mute_video_audio: bool = False
    ) -> bool:
        """
        Mixes original video audio and optional background music into a single synchronized AAC track.
        Handles volume leveling, trimming, and muting.
        """
        ffmpeg_bin = get_ffmpeg_binary()
        has_orig = bool(orig_audio_path and os.path.exists(orig_audio_path) and not mute_video_audio and video_volume > 0.001)
        has_music = bool(music_path and os.path.exists(music_path) and music_volume > 0.001)

        if not has_orig and not has_music:
            return False

        try:
            # Case 1: Both Original Audio AND Background Music -> Mix with FFmpeg amix
            if has_orig and has_music:
                filter_str = f"[0:a]volume={video_volume:.2f}[a0];[1:a]volume={music_volume:.2f}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", orig_audio_path,
                    "-i", music_path,
                    "-filter_complex", filter_str,
                    "-map", "[aout]",
                    "-c:a", "aac",
                    "-b:a", "192k",
                ]
                if duration_sec is not None and duration_sec > 0:
                    cmd.extend(["-t", f"{duration_sec:.3f}"])
                cmd.append(output_audio_path)

            # Case 2: Only Music (video has no audio or original audio is muted)
            elif has_music:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", music_path,
                    "-af", f"volume={music_volume:.2f}",
                    "-c:a", "aac",
                    "-b:a", "192k",
                ]
                if duration_sec is not None and duration_sec > 0:
                    cmd.extend(["-t", f"{duration_sec:.3f}"])
                cmd.append(output_audio_path)

            # Case 3: Only Original Audio
            else:
                if abs(video_volume - 1.0) < 0.01:
                    cmd = [ffmpeg_bin, "-y", "-i", orig_audio_path, "-c:a", "copy", output_audio_path]
                else:
                    cmd = [
                        ffmpeg_bin, "-y",
                        "-i", orig_audio_path,
                        "-af", f"volume={video_volume:.2f}",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        output_audio_path
                    ]

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return (res.returncode == 0) and os.path.exists(output_audio_path) and (os.path.getsize(output_audio_path) > 0)
        except Exception as e:
            print(f"[Luxanix Audio] Warnung beim Mischen der Audio-Spuren: {e}")
            return False

    def process_video(
        self,
        input_path: str,
        output_path: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, float, float, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Processes an entire video file or cut segment with Ray Tracing,
        Color Grading, Multi-Track Audio preservation, and detailed ETA tracking.
        Leverages direct FFmpeg NVENC streaming, frame prefetching, and instant cancel support.
        """
        c_evt = cancel_event or params.get("cancel_event", None)
        if c_evt is not None and c_evt.is_set():
            raise InterruptedError("Video-Render vor dem Start durch Benutzer abgebrochen.")

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

        # Create isolated temporary directory inside OS Temp folder
        # Never writes temporary raw video or audio files into the user's render folder!
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="luxanix_export_")
        temp_video_no_audio = os.path.join(temp_dir, "temp_raw.mp4")
        temp_extracted_audio = os.path.join(temp_dir, "orig_audio.aac")
        final_audio = os.path.join(temp_dir, "final_audio.aac")

        # Audio options
        music_path = params.get("music_path", None)
        music_volume = float(params.get("music_volume", 1.0))
        video_volume = float(params.get("video_volume", 1.0))
        mute_video_audio = bool(params.get("mute_video_audio", False))
        clip_duration = (trim_end - trim_start) if enable_trim else total_duration

        # Step 1: Extract trimmed audio from original video
        has_orig_audio = False
        if not mute_video_audio and video_volume > 0.001:
            has_orig_audio = self._extract_audio(
                input_path,
                temp_extracted_audio,
                start_time=trim_start if enable_trim else None,
                duration=clip_duration if enable_trim else None
            )

        # Step 1B: Prepare & mix audio tracks (Original + Background Music)
        has_final_audio = self._prepare_audio_track(
            orig_audio_path=temp_extracted_audio if has_orig_audio else None,
            music_path=music_path,
            output_audio_path=final_audio,
            video_volume=video_volume,
            music_volume=music_volume,
            duration_sec=clip_duration,
            mute_video_audio=mute_video_audio
        )

        nvenc_preset = params.get("nvenc_preset", "p7")
        codec = params.get("encoder_codec", "hevc_nvenc" if out_h >= 2160 else "h264_nvenc")
        bitrate_mbps = int(params.get("bitrate_mbps", 60 if out_h >= 4320 else (45 if out_h >= 2160 else 30)))
        dual_nvenc = params.get("dual_nvenc", False)

        if (out_w > 4096 or out_h > 4096) and "h264" in codec:
            codec = "hevc_nvenc"

        # Step 2: Open Direct High-Throughput NVENC Pipe
        proc = self._open_ffmpeg_pipe(
            output_path=output_path,
            audio_path=final_audio if has_final_audio else None,
            width=out_w,
            height=out_h,
            fps=fps,
            codec=codec,
            preset=nvenc_preset,
            bitrate_mbps=bitrate_mbps,
            dual_nvenc=dual_nvenc
        )

        use_pipe = (proc is not None) and (proc.stdin is not None)
        writer = None
        if not use_pipe:
            # Fallback to intermediate writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (out_w, out_h))

        print(f"[Luxanix] Rendere Video {out_w}x{out_h} @ {fps:.1f} FPS (Direct NVENC Pipe={use_pipe}), Frames: {total_frames_to_process}...")

        # Step 3: Asynchronous Prefetch Reader Thread
        frame_queue = queue.Queue(maxsize=16)
        stop_reader = threading.Event()

        def reader_worker():
            try:
                cur_pos = start_frame
                while cur_pos < end_frame and not stop_reader.is_set():
                    ret, f_bgr = cap.read()
                    if not ret:
                        break
                    frame_queue.put(f_bgr)
                    cur_pos += 1
            except Exception:
                pass
            finally:
                frame_queue.put(None)

        t_reader = threading.Thread(target=reader_worker, daemon=True)
        t_reader.start()

        start_time = time.time()
        processed_count = 0
        empty_cache_freq = int(params.get("empty_cache_freq", 60))

        try:
            while processed_count < total_frames_to_process:
                # Check for cancellation
                if c_evt is not None and c_evt.is_set():
                    print("[Luxanix] Render durch Benutzer abgebrochen!")
                    stop_reader.set()
                    if use_pipe and proc is not None:
                        try:
                            proc.stdin.close()
                            proc.kill()
                        except Exception:
                            pass
                    elif writer is not None:
                        try:
                            writer.release()
                        except Exception:
                            pass
                    for p in [output_path, temp_video_no_audio, temp_extracted_audio, final_audio]:
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    raise InterruptedError("Video-Render durch Benutzer abgebrochen.")

                frame_bgr = frame_queue.get()
                if frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Process through RTX Raytracing engine
                out_rgb, _, _ = self.process_single_frame(frame_rgb, params)

                # Neural AI Super-Resolution if custom resolution requested
                if (out_w != width) or (out_h != height):
                    if params.get("neural_upscale", True) and (out_h > height):
                        out_rgb = self.upscaler.upscale_frame(out_rgb, target_height=out_h)
                        if out_rgb.shape[1] != out_w or out_rgb.shape[0] != out_h:
                            out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

                if use_pipe:
                    try:
                        proc.stdin.write(out_bgr.tobytes())
                    except (BrokenPipeError, OSError):
                        pass
                else:
                    writer.write(out_bgr)

                processed_count += 1

                # Periodic CUDA cache cleanup
                if processed_count % empty_cache_freq == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                elapsed = time.time() - start_time
                current_fps = processed_count / (elapsed + 1e-5)
                eta = (total_frames_to_process - processed_count) / (current_fps + 1e-5) if total_frames_to_process > 0 else 0.0

                if progress_callback is not None:
                    progress_callback(processed_count, total_frames_to_process, current_fps, eta, elapsed)

        finally:
            stop_reader.set()
            cap.release()
            while not frame_queue.empty():
                try: frame_queue.get_nowait()
                except Exception: break

            if c_evt is not None and c_evt.is_set():
                if use_pipe and proc is not None:
                    try:
                        proc.stdin.close()
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                elif writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass

                # Clean up incomplete output file if cancelled
                time.sleep(0.15)
                for _ in range(5):
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                            break
                        except Exception:
                            time.sleep(0.2)
            else:
                if use_pipe:
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=30)
                    except Exception:
                        pass
                elif writer is not None:
                    writer.release()
                    self._finalize_video(
                        temp_video_no_audio,
                        final_audio if has_final_audio else None,
                        output_path,
                        fps,
                        codec=codec,
                        nvenc_preset=nvenc_preset,
                        bitrate_mbps=bitrate_mbps,
                        dual_nvenc=dual_nvenc,
                    )

        # Cleanup isolated temp directory completely
        time.sleep(0.15)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
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
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [ffmpeg_bin, "-y"]
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
        ffmpeg_bin = get_ffmpeg_binary()

        # Prioritize requested codec with intelligent hardware fallbacks
        codecs_to_try = [codec]
        if codec == "av1_nvenc":
            codecs_to_try.extend(["hevc_nvenc", "h264_nvenc"])
        elif codec == "hevc_nvenc":
            codecs_to_try.append("h264_nvenc")

        for current_codec in codecs_to_try:
            cmd_nvenc = [ffmpeg_bin, "-y", "-i", video_path]
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
        cmd_cpu = [ffmpeg_bin, "-y", "-i", video_path]
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
