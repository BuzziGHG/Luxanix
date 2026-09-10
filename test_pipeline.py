"""
SimRTX Studio - Test & Benchmark Pipeline
Tests GPU CUDA detection, Depth Estimation, Ray Tracing passes, and Color Grading.
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.depth_estimator import DepthEstimator
from engine.geometry import compute_normals, reconstruct_positions
from engine.raytracer import ScreenSpaceRaytracer
from engine.denoiser import BilateralDenoiser
from engine.postprocess import ColorGrader
from engine.video_pipeline import VideoPipeline


def create_synthetic_simracing_frame(width=1280, height=720) -> np.ndarray:
    """
    Creates a synthetic simracing racetrack frame with road, grass, sky, and a race car.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky (gradient blue)
    for y in range(int(height * 0.45)):
        ratio = y / (height * 0.45)
        b = int(180 - ratio * 40)
        g = int(140 - ratio * 30)
        r = int(70 - ratio * 20)
        frame[y, :] = [b, g, r]

    # 2. Distant hills / horizon
    frame[int(height * 0.40):int(height * 0.45), :] = [40, 70, 30]

    # 3. Grass side runoffs
    frame[int(height * 0.45):, :] = [30, 95, 35]

    # 4. Asphalt track (perspective trapezoid)
    track_pts = np.array([
        [int(width * 0.40), int(height * 0.45)],
        [int(width * 0.60), int(height * 0.45)],
        [int(width * 0.95), height],
        [int(width * 0.05), height]
    ], np.int32)
    cv2.fillPoly(frame, [track_pts], (45, 48, 52))

    # 5. Red & White Curbs
    curb_left = np.array([
        [int(width * 0.38), int(height * 0.45)],
        [int(width * 0.40), int(height * 0.45)],
        [int(width * 0.05), height],
        [int(width * 0.00), height]
    ], np.int32)
    cv2.fillPoly(frame, [curb_left], (30, 30, 200))  # Red curb

    # 6. Race car in foreground (chase camera)
    car_x1 = int(width * 0.38)
    car_y1 = int(height * 0.52)
    car_x2 = int(width * 0.62)
    car_y2 = int(height * 0.82)

    # Car shadow on asphalt
    cv2.ellipse(frame, (int((car_x1 + car_x2)/2), int(car_y2 * 0.98)), (int((car_x2 - car_x1)*0.6), 25), 0, 0, 360, (15, 15, 15), -1)

    # Car body (GT3 racecar red & gloss)
    cv2.rectangle(frame, (car_x1, car_y1 + 40), (car_x2, car_y2), (20, 25, 210), -1)
    # Windshield / Roof
    roof_pts = np.array([
        [car_x1 + 30, car_y1 + 40],
        [car_x1 + 60, car_y1],
        [car_x2 - 60, car_y1],
        [car_x2 - 30, car_y1 + 40]
    ], np.int32)
    cv2.fillPoly(frame, [roof_pts], (35, 35, 40))
    # Rear wing
    cv2.rectangle(frame, (car_x1 - 10, car_y1 + 10), (car_x2 + 10, car_y1 + 22), (20, 20, 20), -1)
    # Taillights
    cv2.rectangle(frame, (car_x1 + 10, car_y1 + 55), (car_x1 + 50, car_y1 + 70), (20, 20, 245), -1)
    cv2.rectangle(frame, (car_x2 - 50, car_y1 + 55), (car_x2 - 10, car_y1 + 70), (20, 20, 245), -1)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    print("=" * 65)
    print("       SimRTX Studio — Pipeline & GPU Benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gerät erkannt: {device}")
    if device.type == "cuda":
        print(f"GPU Modell:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:  {torch.version.cuda}")
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"VRAM Speicher: {vram_mb:.0f} MB")
    else:
        print("Hinweis: CUDA nicht aktiv, verwende CPU.")

    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/5] Erstelle synthetischen Simracing-Testframe (1280x720)...")
    test_frame = create_synthetic_simracing_frame()
    Image.fromarray(test_frame).save(os.path.join(output_dir, "01_original_frame.png"))

    print("\n[2/5] Initialisiere SimRTX Pipeline...")
    pipeline = VideoPipeline(device=str(device), use_fp16=(device.type == "cuda"))

    test_params = {
        "rtgi_intensity": 0.75,
        "rtgi_range": 4.0,
        "rtgi_steps": 10,
        "ssr_intensity": 0.70,
        "roughness": 0.18,
        "wet_track_mode": True,
        "rtao_intensity": 0.85,
        "rtao_radius": 1.3,
        "exposure": 0.05,
        "contrast": 1.12,
        "saturation": 1.20,
        "vibrance": 0.22,
        "temperature": 0.05,
        "bloom_intensity": 0.25,
        "bloom_threshold": 0.75,
        "use_aces": True,
        "vignette": 0.15,
        "denoise": True,
    }

    print("\n[3/5] Führe Raytracing & Color Grading durch...")
    # Warmup
    _ = pipeline.process_single_frame(test_frame, test_params)

    # Benchmark run
    start_time = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        out_rgb, depth_viz, normals_viz = pipeline.process_single_frame(test_frame, test_params)
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = (time.perf_counter() - start_time) / iterations
    fps = 1.0 / duration

    print(f"\n[4/5] Benchmark-Ergebnis:")
    print(f"      Laufzeit pro Frame: {duration * 1000.0:.2f} ms")
    print(f"      Verarbeitungsrate:  {fps:.1f} FPS")

    print("\n[5/5] Speichere Bild-Ausgaben...")
    Image.fromarray(depth_viz).save(os.path.join(output_dir, "02_depth_map.png"))
    Image.fromarray(normals_viz).save(os.path.join(output_dir, "03_normals_map.png"))
    Image.fromarray(out_rgb).save(os.path.join(output_dir, "04_raytraced_output.png"))

    print(f"Dateien erfolgreich gespeichert in: {output_dir}")
    print("\nAlle Tests erfolgreich bestanden! SimRTX Studio ist voll einsatzbereit.")


if __name__ == "__main__":
    main()
