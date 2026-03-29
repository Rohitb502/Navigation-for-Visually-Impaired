"""
blindnav_demo.py  —  BlindNav Professor Demo
=============================================
Hardware : Raspberry Pi 4/5  +  Pi Camera Module v2/v3  +  wired headphones
Run      : python blindnav_demo.py
Quit     : press Q in the display window  (or Ctrl-C in terminal)

TIMING DESIGN
─────────────
Three threads run concurrently so no stage ever blocks another:

  Thread-1  CAPTURE    Pi Camera → single-slot frame buffer (always fresh)
  Thread-2  INFERENCE  Reads newest frame → MiDaS + YOLO → scored detections
  Thread-3  AUDIO      Reads newest scored result → pyttsx3 → headphones

Key properties
  • Frame buffer holds exactly ONE frame.  Inference always grabs the newest;
    frames that arrived during inference are silently overwritten (dropped).
    Latency = one inference cycle, never an accumulated queue.
  • Audio runs in its own thread.  Speech never blocks inference.
  • When speech ends the person starts moving.  By then inference has already
    completed 1-2 fresh cycles.  Audio reads that fresh result — not a stale
    queued one.  The person always acts on recent information.
  • Per-class cooldown prevents the same alert repeating every second.
  • Urgency override fires immediately for objects < 1.8 m, cooldown bypassed.

CHEST-LEVEL CAMERA NOTES
─────────────────────────
  • Ground plane (lower 30 % of frame) excluded from depth estimation —
    it reads artificially close at chest height and causes false alerts.
  • Warning threshold tightened to 4 m (less approach time than glasses mount).
  • Frame split into LEFT / CENTRE / RIGHT thirds for spatial audio cues.
"""

import sys
import time
import random
import threading
import logging

import cv2
import numpy as np
import torch
import pyttsx3
from picamera2 import Picamera2
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(threadName)-12s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("blindnav")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all tunable numbers in one place
# ═════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H = 640, 480
CAM_FPS      = 30

YOLO_MODEL = "yolo11n.pt"
YOLO_CONF  = 0.45

MIDAS_MODEL = "MiDaS_small"      # fastest on Pi CPU

# Chest-height ground-plane exclusion: ignore bottom 30 % of frame
DEPTH_ROI_BOTTOM_FRAC = 0.70     # keep rows 0 .. 70 % only

# Depth scale: set object at 3 m, note printed rel_depth, set DEPTH_SCALE = 3/rel_depth
DEPTH_SCALE = 10.0

# Distance thresholds (tighter than glasses mount — less approach warning time)
URGENT_DIST_M  = 1.8
WARN_DIST_M    = 4.0

# Cooldown tiers (seconds before re-announcing same class)
COOLDOWN_URGENT  = 1.2
COOLDOWN_NEAR    = 3.5
COOLDOWN_DISTANT = 8.0

POST_SPEECH_PAUSE = 0.5   # after speaking, pause so person can react

VOICE_RATE = 175
VOICE_VOL  = 1.0

DISPLAY = True    # set False for headless deployment

# ═════════════════════════════════════════════════════════════════════════════
# PRIORITY TABLES
# ═════════════════════════════════════════════════════════════════════════════
CLASS_DANGER = {
    "car": 90, "truck": 95, "bus": 95, "motorcycle": 88, "bicycle": 72,
    "scooter": 72, "train": 100, "tram": 95,
    "pothole": 82, "construction": 75, "barrier": 60,
    "person": 60, "dog": 58, "cat": 30,
    "bench": 25, "chair": 20, "suitcase": 22, "backpack": 12,
    "umbrella": 18, "fire hydrant": 38, "parking meter": 28,
    "pole": 42, "bollard": 45, "cone": 40,
    "stop sign": 30, "traffic light": 25,
}
DEFAULT_DANGER = 20

NAV_HINTS = {
    "car":          "stop and move right",
    "truck":        "stop immediately and move right",
    "bus":          "stop and move right",
    "motorcycle":   "step right",
    "bicycle":      "step right",
    "scooter":      "step right",
    "train":        "stop — do not cross",
    "person":       "bear slightly right",
    "pothole":      "step around carefully",
    "construction": "stop and find another path",
    "barrier":      "stop — path is blocked",
    "dog":          "slow down, bear right",
    "bench":        "move around to the right",
    "fire hydrant": "step left",
    "bollard":      "step around",
    "cone":         "step around",
    "pole":         "move around the pole",
    "traffic light":"check the signal before crossing",
}
DEFAULT_HINT = "proceed with caution"

TEMPLATES = {
    "urgent": [
        "{label} very close, {dist:.1f} metres — {hint} now",
        "Warning — {label} at {dist:.1f} metres — {hint} immediately",
        "Danger — {label} {dist:.1f} metres — {hint}",
    ],
    "near": [
        "{zone}{label} ahead, {dist:.1f} metres — {hint}",
        "{zone}{label} at {dist:.1f} metres — {hint}",
        "Approaching {label}{zone_tail}, {dist:.1f} metres",
    ],
    "far": [
        "{label} ahead, roughly {dist:.1f} metres",
        "{label} detected, {dist:.1f} metres — be aware",
        "Heads up — {label} at {dist:.1f} metres",
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════
log.info("Loading MiDaS (%s) …", MIDAS_MODEL)
_midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL)
_midas.eval()
_midas_tx        = torch.hub.load("intel-isl/MiDaS", "transforms")
_midas_transform = _midas_tx.small_transform
_device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_midas.to(_device)
log.info("MiDaS on %s", _device)

log.info("Loading YOLO (%s) …", YOLO_MODEL)
_yolo = YOLO(YOLO_MODEL)
log.info("Models ready.")

# ═════════════════════════════════════════════════════════════════════════════
# CAMERA
# ═════════════════════════════════════════════════════════════════════════════
_picam2 = Picamera2()
_picam2.configure(_picam2.create_video_configuration(
    main={"size": (CAM_W, CAM_H), "format": "RGB888"},
    controls={"FrameRate": CAM_FPS},
))
_picam2.start()
log.info("Pi Camera %dx%d @ %d fps", CAM_W, CAM_H, CAM_FPS)

# ═════════════════════════════════════════════════════════════════════════════
# SHARED STATE  — two single-slot buffers (no queues)
# ═════════════════════════════════════════════════════════════════════════════
_frame_slot  = {"frame": None, "ts": 0.0}
_frame_lock  = threading.Lock()

_result_slot = {"detections": [], "frame": None, "depth_map": None, "ts": 0.0}
_result_lock = threading.Lock()

_last_spoken  = {}
_urgent_fired = {}
_cooldown_lock = threading.Lock()

_stop = threading.Event()

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def _zone(cx, frame_w):
    third = frame_w // 3
    if cx < third:
        return "on your left — ", " on your left"
    if cx > 2 * third:
        return "on your right — ", " on your right"
    return "", " ahead"


def _dist_multiplier(d):
    if d <= 1.0: return 3.5
    if d <= 2.0: return 2.8
    if d <= 3.5: return 2.0
    if d <= 5.0: return 1.5
    if d <= 8.0: return 1.0
    return 0.4


def _score(label, dist, cx, frame_w, bbox_w):
    base       = CLASS_DANGER.get(label, DEFAULT_DANGER)
    deviation  = abs(cx - frame_w / 2) / (frame_w / 2)
    centre     = 1.0 - 0.5 * deviation
    size_bonus = min((bbox_w / frame_w) * 50, 30)
    return base * _dist_multiplier(dist) * centre + size_bonus


def _build_msg(label, dist, cx, frame_w):
    hint             = NAV_HINTS.get(label, DEFAULT_HINT)
    zone_pfx, zone_sfx = _zone(cx, frame_w)

    if dist < 2.0:   tmpl_key = "urgent"
    elif dist < 5.0: tmpl_key = "near"
    else:            tmpl_key = "far"

    t = random.choice(TEMPLATES[tmpl_key])
    return t.format(label=label, dist=dist, hint=hint,
                    zone=zone_pfx, zone_tail=zone_sfx)


def _should_speak(label, dist):
    now = time.monotonic()
    with _cooldown_lock:
        # Urgency one-shot override
        if dist <= URGENT_DIST_M:
            if now - _urgent_fired.get(label, 0.0) >= COOLDOWN_URGENT:
                _urgent_fired[label] = now
                return True

        # Normal tier
        cd = COOLDOWN_URGENT if dist < 2.0 else \
             COOLDOWN_NEAR    if dist < 4.5 else COOLDOWN_DISTANT
        return now - _last_spoken.get(label, 0.0) >= cd


def _mark_spoken(label):
    with _cooldown_lock:
        _last_spoken[label] = time.monotonic()


def _estimate_dist(depth_map, cx, cy, frame_h):
    roi_bot    = int(frame_h * DEPTH_ROI_BOTTOM_FRAC)
    cy_clamped = min(max(cy, 0), roi_bot - 1)
    r          = 8
    patch = depth_map[
        max(0, cy_clamped - r): cy_clamped + r,
        max(0, cx - r): cx + r,
    ]
    raw  = float(patch.mean()) if patch.size > 0 else float(depth_map[cy_clamped, cx])
    roi  = depth_map[:roi_bot, :]
    dmin, dmax = roi.min(), roi.max()
    rel  = 1.0 - (raw - dmin) / (dmax - dmin + 1e-6)
    return rel * DEPTH_SCALE


# ═════════════════════════════════════════════════════════════════════════════
# THREAD 1 — CAPTURE
# Runs at camera FPS. Writes newest frame into single slot.
# Old frames are silently overwritten — never accumulate.
# ═════════════════════════════════════════════════════════════════════════════
def capture_loop():
    while not _stop.is_set():
        rgb = _picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with _frame_lock:
            _frame_slot["frame"] = bgr
            _frame_slot["ts"]    = time.monotonic()
        # picamera2 paces itself to FrameRate — no sleep needed here


# ═════════════════════════════════════════════════════════════════════════════
# THREAD 2 — INFERENCE
# Always grabs the newest frame. On Pi 4 CPU this runs at ~1 Hz.
# 29 out of 30 frames per second are dropped — that is correct behaviour.
# ═════════════════════════════════════════════════════════════════════════════
def inference_loop():
    last_ts = 0.0

    while not _stop.is_set():
        # Grab newest frame
        with _frame_lock:
            frame    = _frame_slot["frame"]
            frame_ts = _frame_slot["ts"]

        if frame is None or frame_ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = frame_ts
        t0 = time.monotonic()
        h, w = frame.shape[:2]

        # MiDaS
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = _midas_transform(rgb).to(_device)
        with torch.no_grad():
            pred = _midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w),
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth_map = pred.cpu().numpy()
        t_midas = time.monotonic()

        # YOLO
        yolo_out = _yolo(frame, conf=YOLO_CONF, verbose=False)[0]
        t_yolo = time.monotonic()

        # Score
        dets = []
        for box in yolo_out.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = _yolo.names[int(box.cls)]
            conf  = float(box.conf)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist  = _estimate_dist(depth_map, cx, cy, h)
            s     = _score(label, dist, cx, w, x2 - x1)
            dets.append({"label": label, "conf": conf, "dist": dist,
                         "score": s, "bbox": (x1, y1, x2, y2),
                         "cx": cx, "cy": cy})
        dets.sort(key=lambda d: d["score"], reverse=True)

        with _result_lock:
            _result_slot["detections"] = dets
            _result_slot["frame"]      = frame
            _result_slot["depth_map"]  = depth_map
            _result_slot["ts"]         = time.monotonic()

        log.info(
            "frame-age %4.0f ms | MiDaS %4.0f ms | YOLO %4.0f ms | %d objects",
            (t0 - frame_ts) * 1000,
            (t_midas - t0) * 1000,
            (t_yolo - t_midas) * 1000,
            len(dets),
        )


# ═════════════════════════════════════════════════════════════════════════════
# THREAD 3 — AUDIO
# After speech ends the person starts moving. Inference has already run
# 1-2 more cycles by then. Audio reads that fresh result — never a backlog.
# ═════════════════════════════════════════════════════════════════════════════
def audio_loop():
    tts = pyttsx3.init()
    tts.setProperty("rate",   VOICE_RATE)
    tts.setProperty("volume", VOICE_VOL)
    last_ts = 0.0

    while not _stop.is_set():
        with _result_lock:
            dets      = _result_slot.get("detections", [])
            result_ts = _result_slot.get("ts", 0.0)

        if result_ts == last_ts or not dets:
            time.sleep(0.05)
            continue
        last_ts = result_ts

        # Pick highest-priority detection that is due for announcement
        to_speak = next((d for d in dets if _should_speak(d["label"], d["dist"])), None)
        if to_speak is None:
            continue

        msg = _build_msg(to_speak["label"], to_speak["dist"],
                         to_speak["cx"], CAM_W)
        log.info("SPEAKING: %s", msg)
        _mark_spoken(to_speak["label"])

        tts.say(msg)
        tts.runAndWait()

        # Brief pause — person processes audio and begins moving
        # Inference completes another cycle during this pause
        time.sleep(POST_SPEECH_PAUSE)


# ═════════════════════════════════════════════════════════════════════════════
# THREAD 4 — DISPLAY  (optional — debug / demo)
# Left panel:  camera frame with colour-coded bounding boxes
# Right panel: MiDaS depth map (MAGMA colourmap)
# Ground-plane exclusion zone shown as dimmed band at bottom
# ═════════════════════════════════════════════════════════════════════════════
def display_loop():
    last_ts = 0.0

    while not _stop.is_set():
        with _result_lock:
            dets      = _result_slot.get("detections", [])
            frame     = _result_slot.get("frame", None)
            depth_map = _result_slot.get("depth_map", None)
            result_ts = _result_slot.get("ts", 0.0)

        if frame is None or result_ts == last_ts:
            time.sleep(0.03)
            continue
        last_ts = result_ts

        h, w  = frame.shape[:2]
        vis   = frame.copy()
        excl_y = int(h * DEPTH_ROI_BOTTOM_FRAC)

        # Ground-plane exclusion overlay
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, excl_y), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        cv2.putText(vis, "ground-plane excluded", (8, excl_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Horizontal zone dividers
        for x in [w // 3, 2 * w // 3]:
            cv2.line(vis, (x, 0), (x, excl_y), (60, 60, 60), 1)
        for i, lbl in enumerate(["LEFT", "CENTRE", "RIGHT"]):
            cv2.putText(vis, lbl, (w // 3 * i + 6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        # Bounding boxes
        for rank, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            d = det["dist"]
            colour = (0, 0, 220) if d < 2.0 else \
                     (0, 130, 255) if d < 4.5 else (30, 200, 30)
            thick = 3 if rank == 0 else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thick)
            tag = f"#{rank+1} {det['label']} {det['conf']:.0%} | {d:.1f}m | P{det['score']:.0f}"
            cv2.putText(vis, tag, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
            cv2.circle(vis, (det["cx"], det["cy"]), 4, (255, 255, 0), -1)

        # Depth panel
        if depth_map is not None:
            dn  = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            col = cv2.applyColorMap(dn, cv2.COLORMAP_MAGMA)
            cv2.line(col, (0, excl_y), (w, excl_y), (255, 255, 255), 1)
        else:
            col = np.zeros_like(vis)

        combined = np.hstack([vis, col])

        # Legend
        for i, (txt, c) in enumerate([
            ("RED   < 2 m   danger",   (80, 80, 220)),
            ("ORANGE 2-4.5m warning",  (80, 160, 255)),
            ("GREEN > 4.5m  clear",    (80, 200, 80)),
        ]):
            cv2.putText(combined, txt, (w + 8, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        cv2.imshow("BlindNav  |  Camera        Depth", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _stop.set()
            break


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    threads = [
        threading.Thread(target=capture_loop,   name="Capture",   daemon=True),
        threading.Thread(target=inference_loop, name="Inference", daemon=True),
        threading.Thread(target=audio_loop,     name="Audio",     daemon=True),
    ]
    if DISPLAY:
        threads.append(
            threading.Thread(target=display_loop, name="Display", daemon=True)
        )

    for t in threads:
        t.start()
        log.info("Thread '%s' started.", t.name)

    log.info("BlindNav running.  Press Q in window or Ctrl-C to stop.")

    try:
        while not _stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _stop.set()
    finally:
        _picam2.stop()
        cv2.destroyAllWindows()
        log.info("BlindNav stopped.")


if __name__ == "__main__":
    main()
