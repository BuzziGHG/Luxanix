"""
Luxanix GPU VRAM Frame Cache & Background Pre-Renderer
======================================================
Leverages high-speed NVIDIA RTX GDDR6/GDDR6X VRAM (up to 12 GB on RTX 3080 Ti,
up to 32 GB on RTX 5090) to pre-compute and store photorealistic enhanced frames.

Features:
- Stores frames as FP16 PyTorch CUDA Tensors directly in GPU VRAM
- Delivers ultra-fast, zero-latency (60 FPS) live playback from GPU memory
- Asynchronous background worker pre-processes upcoming video frames on GPU
- Real-time VRAM telemetry (allocated, cached frame count, cache hit rate)
- Instant cache invalidation on slider or color grading changes
"""

import time
import threading
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import cv2


class GPUFrameCache:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_cache_frames: int = 180,  # ~1.0 GB in FP16 720p
        max_vram_gb: float = 4.0,     # Max VRAM ceiling for frame buffer
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.max_cache_frames = max_cache_frames
        self.max_vram_gb = max_vram_gb

        # LRU Frame Cache: frame_idx -> (torch.Tensor [3, H, W] on CUDA FP16, auto_params dict)
        self.cache: OrderedDict[int, Tuple[torch.Tensor, Dict[str, Any]]] = OrderedDict()
        self.lock = threading.Lock()

        # Background worker state
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_worker = threading.Event()
        self.video_path: Optional[str] = None
        self.pipeline: Any = None
        self.current_frame_target = 0
        self.active_params: Dict[str, Any] = {}
        self.preview_size: Tuple[int, int] = (1280, 720)

        # Performance stats
        self.hits = 0
        self.misses = 0

    def clear(self):
        """Clears all cached frames from GPU VRAM."""
        with self.lock:
            self.cache.clear()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_vram_stats(self) -> Dict[str, Any]:
        """Returns live VRAM utilization and cache stats."""
        with self.lock:
            count = len(self.cache)
            hits = self.hits
            misses = self.misses

        if torch.cuda.is_available() and self.device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            res_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        else:
            alloc_mb, res_mb, total_gb = 0.0, 0.0, 0.0

        hit_rate = (hits / (hits + misses + 1e-5)) * 100.0
        return {
            "cached_frames": count,
            "allocated_mb": alloc_mb,
            "reserved_mb": res_mb,
            "total_gb": total_gb,
            "hit_rate_pct": hit_rate,
        }

    def get(self, frame_idx: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieves a pre-rendered frame from GPU VRAM cache.
        Returns (RGB uint8 numpy array, telemetry params) or None.
        """
        with self.lock:
            if frame_idx in self.cache:
                self.cache.move_to_end(frame_idx)
                tensor_gpu, params = self.cache[frame_idx]
                self.hits += 1
                # Convert CUDA FP16 tensor to host uint8 numpy for display
                out_np = (tensor_gpu.permute(1, 2, 0).cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                return out_np, params
            else:
                self.misses += 1
                return None

    def put(self, frame_idx: int, frame_rgb: np.ndarray, params: Dict[str, Any]):
        """Stores a rendered frame into GPU VRAM as an FP16 CUDA tensor."""
        try:
            # Check VRAM limit
            if torch.cuda.is_available() and self.device.type == "cuda":
                alloc_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                if alloc_gb >= self.max_vram_gb:
                    # Evict oldest 20 frames
                    with self.lock:
                        for _ in range(min(20, len(self.cache))):
                            self.cache.popitem(last=False)

            color_f32 = frame_rgb.astype(np.float32) / 255.0
            tensor_gpu = torch.from_numpy(color_f32).permute(2, 0, 1).to(device=self.device, dtype=torch.float16)

            with self.lock:
                if len(self.cache) >= self.max_cache_frames:
                    self.cache.popitem(last=False)
                self.cache[frame_idx] = (tensor_gpu, params.copy())
        except Exception:
            pass

    def start_background_precache(
        self,
        video_path: str,
        pipeline: Any,
        start_frame: int,
        active_params: Dict[str, Any],
        preview_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Launches an asynchronous background worker to pre-render upcoming video frames
        into GPU VRAM so continuous playback is instantaneous and smooth.
        """
        self.stop_background_precache()
        self.video_path = video_path
        self.pipeline = pipeline
        self.current_frame_target = start_frame
        self.active_params = active_params.copy()
        self.preview_size = preview_size
        self.stop_worker.clear()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop_background_precache(self):
        """Stops the background pre-caching worker thread."""
        self.stop_worker.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
        self.worker_thread = None

    def update_target_position(self, target_frame: int):
        """Notifies the background worker of a seek / playhead shift."""
        self.current_frame_target = target_frame

    def _worker_loop(self):
        """Background loop that fills the GPU VRAM cache ahead of playhead."""
        if not self.video_path or self.pipeline is None:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_pos = -1

        try:
            while not self.stop_worker.is_set():
                target = self.current_frame_target
                lookahead_window = 30  # Pre-cache 30 frames ahead of playhead

                # Find next uncached frame in the lookahead window
                next_to_cache = None
                for f_idx in range(target, min(target + lookahead_window, total_frames)):
                    with self.lock:
                        in_cache = f_idx in self.cache
                    if not in_cache:
                        next_to_cache = f_idx
                        break

                if next_to_cache is None:
                    # All frames in lookahead window are cached in VRAM! Sleep briefly.
                    time.sleep(0.04)
                    continue

                # Seek if necessary
                if last_pos != next_to_cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_to_cache)
                    last_pos = next_to_cache

                ret, frame_bgr = cap.read()
                last_pos += 1
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pw, ph = self.preview_size
                frame_small = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_AREA)

                # Process single frame on GPU
                render_params = dict(self.active_params)
                render_params["fast_preview"] = True

                try:
                    out_rgb, _, _ = self.pipeline.process_single_frame(frame_small, render_params)
                    self.put(next_to_cache, out_rgb, render_params)
                except Exception:
                    pass

                # Short yield so main thread gets priority
                time.sleep(0.002)

        finally:
            cap.release()
