"""
BlindNav — Assistive Navigation System
Raspberry Pi Camera + MiDaS Depth + YOLOv11 + Priority Audio Engine

Hardware: Raspberry Pi 4/5, Pi Camera Module v2/v3, headphones (3.5mm or BT)
"""

import cv2
import torch
import numpy as np
import threading
import queue
import time
import logging
from picamera2 import Picamera2  # Raspberry Pi camera

from ultralytics import YOLO
from audio_engine import AudioEngine
from priority_engine import PriorityEngine, Detection

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("BlindNav")

# ─── Config ─────────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
YOLO_CONF        = 0.45          # detection confidence threshold
DEPTH_SCALE      = 10.0          # MiDaS relative depth → approximate meters
DISPLAY          = True          # set False on headless Pi deployment

# ─── Model Loading ───────────────────────────────────────────────────────────
log.info("Loading MiDaS …")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform  = midas_transforms.small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
log.info(f"MiDaS on {device}")

log.info("Loading YOLOv11n …")
yolo = YOLO("yolo11n.pt")
log.info("Models ready.")

# ─── Camera Init ─────────────────────────────────────────────────────────────
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
log.info("Pi Camera started.")

# ─── Shared State ────────────────────────────────────────────────────────────
frame_lock   = threading.Lock()
latest_frame = None
audio_queue  = queue.PriorityQueue()   # (neg_priority, Detection)
stop_event   = threading.Event()

# ─── Engines ─────────────────────────────────────────────────────────────────
priority_engine = PriorityEngine()
audio_engine    = AudioEngine()

# ─── Thread: Frame Capture ───────────────────────────────────────────────────
def capture_loop():
    global latest_frame
    while not stop_event.is_set():
        frame = picam2.capture_array()          # RGB numpy array
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with frame_lock:
            latest_frame = frame_bgr
        time.sleep(0.01)

# ─── Thread: Inference ───────────────────────────────────────────────────────
def inference_loop():
    while not stop_event.is_set():
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.02)
                continue
            frame = latest_frame.copy()

        h, w = frame.shape[:2]

        # ── MiDaS depth ──────────────────────────────────────────────────────
        img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transform(img_rgb).to(device)
        with torch.no_grad():
            depth_pred = midas(input_batch)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = depth_pred.cpu().numpy()

        # ── YOLO detections ──────────────────────────────────────────────────
        yolo_results = yolo(frame, conf=YOLO_CONF, verbose=False)[0]

        detections = []
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = int(box.cls)
            conf = float(box.conf)
            label = yolo.names[cls]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Relative depth → estimated meters
            raw       = depth_map[cy, cx]
            dmin, dmax = depth_map.min(), depth_map.max()
            rel_depth  = 1.0 - (raw - dmin) / (dmax - dmin + 1e-6)
            est_meters = rel_depth * DEPTH_SCALE

            # Bbox fraction of frame width (larger = more central/big)
            bbox_fraction = (x2 - x1) / w

            det = Detection(
                label=label,
                conf=conf,
                distance=est_meters,
                bbox=(x1, y1, x2, y2),
                bbox_fraction=bbox_fraction,
                frame_shape=(h, w),
            )
            detections.append(det)

        # ── Priority scoring & audio scheduling ──────────────────────────────
        scored = priority_engine.score(detections)
        for det in scored:
            if audio_engine.should_announce(det):
                audio_queue.put((-det.priority_score, det))  # neg = highest first

        # ── Optional visual output (debug / demo) ────────────────────────────
        if DISPLAY:
            depth_norm    = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            vis = frame.copy()
            for det in scored:
                x1, y1, x2, y2 = det.bbox
                c = (0, 255, 0) if det.distance > 4 else (0, 165, 255) if det.distance > 2 else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                txt = f"{det.label} {det.conf:.0%} | {det.distance:.1f}m | P{det.priority_score:.0f}"
                cv2.putText(vis, txt, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            combined = np.hstack([vis, depth_colored])
            cv2.imshow("BlindNav", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()

# ─── Thread: Audio Output ────────────────────────────────────────────────────
def audio_loop():
    while not stop_event.is_set():
        try:
            _, det = audio_queue.get(timeout=0.5)
            audio_engine.speak(det)
        except queue.Empty:
            pass

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_loop,   daemon=True, name="Capture"),
        threading.Thread(target=inference_loop, daemon=True, name="Inference"),
        threading.Thread(target=audio_loop,     daemon=True, name="Audio"),
    ]
    for t in threads:
        t.start()
        log.info(f"Thread '{t.name}' started.")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Shutting down …")
        stop_event.set()

    picam2.stop()
    cv2.destroyAllWindows()
    log.info("BlindNav stopped.")
