import cv2
import numpy as np
import threading
import time
import os
import math
import serial
import RPi.GPIO as GPIO
from gtts import gTTS
from picamera2 import Picamera2
from ultralytics import YOLO
import torch
import easyocr
import face_recognition
from flask import Flask, Response, request, jsonify
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── CONFIG ─────────────────────────
DISP_W, DISP_H      = 640, 360
ANNOUNCE_INTERVAL   = 8
SENSOR_HEIGHT_CM    = 125

# ── Pinhole Camera ──
FOCAL_LENGTH_MM  = 3.6
SENSOR_HEIGHT_MM = 2.76
IMAGE_HEIGHT_PX  = DISP_H
FOCAL_LENGTH_PX  = (FOCAL_LENGTH_MM / SENSOR_HEIGHT_MM) * IMAGE_HEIGHT_PX

# ── GPS CONFIG ──
GPS_PORT     = "/dev/ttyAMA0"
GPS_BAUD     = 9600
VJTI_LAT     = 19.0228
VJTI_LON     = 72.8553
VJTI_NAME    = "VJTI, Mumbai"

# ── ULTRASONIC (HC-SR04) ──
TRIG_PIN     = 23
ECHO_PIN     = 24
US_MAX_RANGE = 1.0   # metres — ultrasonic only used within this range
US_MIN_RANGE = 0.02  # metres — below this is sensor self-noise

# ── SERVO (SG90 / MG996R) ──
SERVO_PIN    = 18
SERVO_MIN_DC = 2.5
SERVO_MAX_DC = 12.5
SERVO_FREQ   = 50
SERVO_STEP   = 10    # increased from 5 → faster sweep (36→18 steps per half)
SERVO_DELAY  = 0.12  # seconds per step — allows servo to settle

# ─── KNOWN HEIGHTS (metres) ──────────
KNOWN_HEIGHTS = {
    "person":1.70,"car":1.50,"bus":3.00,"truck":3.50,"bicycle":1.00,
    "motorbike":1.10,"dog":0.50,"cat":0.30,"chair":0.90,"bottle":0.25,
    "dining table":0.75,"potted plant":0.40,"bench":0.50,"suitcase":0.60,
    "backpack":0.50,"umbrella":1.00,"handbag":0.30,"fire hydrant":0.60,
    "stop sign":2.20,"traffic light":3.00,"parking meter":1.20,"bed":0.60,
    "toilet":0.70,"tv":0.60,"laptop":0.30,"mouse":0.05,"keyboard":0.04,
    "cell phone":0.15,"microwave":0.35,"oven":0.60,"refrigerator":1.70,
    "sink":0.25,"clock":0.30,"vase":0.30,"scissors":0.15,"book":0.22,
    "cup":0.12,"fork":0.02,"knife":0.02,"spoon":0.02,"bowl":0.10,
    "banana":0.20,"apple":0.08,"sandwich":0.10,"orange":0.08,
    "broccoli":0.20,"carrot":0.15,"hot dog":0.12,"pizza":0.05,
    "donut":0.07,"cake":0.15,"couch":0.85,"skis":1.60,"snowboard":1.50,
    "sports ball":0.22,"kite":0.50,"baseball bat":1.00,"baseball glove":0.25,
    "skateboard":0.15,"surfboard":1.80,"tennis racket":0.68,"wine glass":0.22,
    "airplane":4.00,"boat":1.50,"train":3.50,"horse":1.60,"cow":1.40,
    "elephant":3.00,"bear":1.20,"zebra":1.40,"giraffe":5.50,"sheep":0.80,
    "bird":0.20,
    "zebra_crossing":0.05,  # flat on ground — used for bbox-height distance estimation
}

HIGH_RISK = {"car","bus","truck","bicycle","motorbike","train","airplane"}
STATIONARY_OBSTACLES = {
    "chair","bench","dining table","potted plant","fire hydrant","stop sign",
    "parking meter","suitcase","backpack","couch","refrigerator","toilet",
    "sink","bed","tv","oven","microwave","umbrella","vase","clock","bottle",
    "cup","bowl","sports ball","skateboard","surfboard","skis","snowboard","boat",
    "zebra_crossing",  # added: announce as a stationary feature
}
ANIMALS = {"dog","cat","horse","cow","elephant","bear","zebra","giraffe","sheep","bird"}

# ─── SHARED STATE ────────────────────
frame     = None
dets      = []
texts     = []
faces     = []
depth_map = None

gps_state = {
    "lat": VJTI_LAT, "lon": VJTI_LON, "fix": False,
    "name": VJTI_NAME, "raw": "No fix — using VJTI default"
}

nav_state = {
    "destination_lat": None, "destination_lon": None,
    "destination_name": "Not set",
    "bearing": None, "distance_m": None,
    "instruction": "No destination set"
}

# ── radar_state: sweep is a list of (angle, dist_m) updated per-point ──
radar_state = {
    "angle":      90,
    "direction":  1,
    "distance_m": None,
    "sweep":      [],        # updated incrementally every step
    "close_zone": None,      # reset at start of each new sweep
    "alert":      None,      # set to a string when something is close
}

locks = {k: threading.Lock()
         for k in ["frame","det","ocr","face","depth","log","gps","nav","radar"]}
stop_event    = threading.Event()
last_announce = 0

# ─── LOG SYSTEM ─────────────────────
log_buffer = deque(maxlen=300)

def add_log(msg):
    ts = time.strftime("%H:%M:%S")
    full = f"[{ts}] {msg}"
    with locks["log"]:
        log_buffer.append(full)
    print(full)

# ─── FACE LOADING ───────────────────
known_encodings, known_names = [], []
known_names_set = set()   # O(1) lookup

def load_faces():
    if not os.path.exists("faces"):
        add_log("No 'faces' dir — skipping")
        return
    for f in os.listdir("faces"):
        try:
            img = face_recognition.load_image_file(f"faces/{f}")
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(f)[0])
        except Exception as e:
            add_log(f"Face load error {f}: {e}")
    known_names_set.update(known_names)
    add_log(f"Loaded {len(known_names)} faces: {known_names}")

# ─── TTS ────────────────────────────
_tts_lock   = threading.Lock()
_tts_active = False

def speak(text):
    def run():
        global _tts_active
        with _tts_lock:
            _tts_active = True
        try:
            import tempfile
            tmp = tempfile.mktemp(suffix=".mp3")
            gTTS(text=text, lang='en', slow=False).save(tmp)
            os.system(f"mpg123 -q {tmp}")
            try:
                os.remove(tmp)
            except Exception:
                pass
            add_log(f"Spoke: {text}")
        except Exception as e:
            add_log(f"TTS Error: {e}")
        finally:
            with _tts_lock:
                _tts_active = False
    threading.Thread(target=run, daemon=True).start()

# ─── CAMERA ─────────────────────────
def camera_thread():
    global frame
    try:
        cam = Picamera2()
        cam.preview_configuration.main.size = (DISP_W, DISP_H)
        cam.configure("preview")
        cam.start()
        add_log("Camera started")
        while not stop_event.is_set():
            img = cam.capture_array()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with locks["frame"]:
                frame = img
            time.sleep(0.01)
    except Exception as e:
        add_log(f"Camera Error: {e}")

# ─── YOLO (dual-model, alternating frames) ───────────────────────────────────
def yolo_thread():
    """
    Loads two models:
      • yolov8n.pt  — general COCO classes (runs on even frame counts)
      • zebra_crossing.pt — YOLOv5 custom model (runs on odd frame counts)

    Detections from both models are merged into `dets` on every cycle so the
    rest of the pipeline (announcements, video overlay, Flask) is unchanged.
    The alternating strategy halves the per-frame CPU load on the Pi.
    """
    global dets
    try:
        # ── Load YOLOv8 (ultralytics) ──
        model_v8 = YOLO("yolov8n.pt")
        add_log(f"YOLOv8 loaded | f_px={FOCAL_LENGTH_PX:.1f}")

        # ── Load YOLOv5 zebra-crossing model (torch.hub) ──
        model_v5 = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path='zebra_crossing.pt',
            force_reload=False,
            verbose=False,
        )
        model_v5.conf = 0.5   # confidence threshold
        model_v5.eval()
        add_log("YOLOv5 zebra_crossing model loaded")

    except Exception as e:
        add_log(f"YOLO load error: {e}")
        return

    frame_count  = 0
    v8_dets      = []   # last results from YOLOv8
    v5_dets      = []   # last results from YOLOv5

    while not stop_event.is_set():
        with locks["frame"]:
            f = frame
        if f is None:
            time.sleep(0.05)
            continue

        if frame_count % 2 == 0:
            # ── Even frame → run YOLOv8 ──────────────────────────────────
            try:
                res    = model_v8(f, imgsz=320, conf=0.5)
                v8_dets = []
                for r in res:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        label = model_v8.names[int(b.cls[0])]
                        v8_dets.append({"label": label, "bbox": (x1, y1, x2, y2)})
            except Exception as e:
                add_log(f"YOLOv8 inference error: {e}")

        else:
            # ── Odd frame → run YOLOv5 zebra-crossing ────────────────────
            try:
                # YOLOv5 hub expects RGB; convert once
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = model_v5(rgb, size=320)
                v5_dets = []
                for *box, conf, cls in res.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    label = model_v5.names[int(cls)]
                    # Normalise label: replace spaces/hyphens with underscore,
                    # lowercase — keeps it consistent with KNOWN_HEIGHTS key
                    label = label.lower().replace(" ", "_").replace("-", "_")
                    v5_dets.append({"label": label, "bbox": (x1, y1, x2, y2)})
            except Exception as e:
                add_log(f"YOLOv5 inference error: {e}")

        # ── Merge and publish ─────────────────────────────────────────────
        with locks["det"]:
            dets = v8_dets + v5_dets

        frame_count += 1

# ─── OCR ────────────────────────────
def ocr_thread():
    global texts
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        add_log("OCR loaded")
        c = 0
        while not stop_event.is_set():
            c = (c + 1) % 100
            if c % 10 != 0:
                time.sleep(0.05); continue
            with locks["frame"]:
                f = frame
            if f is None: continue
            res  = reader.readtext(f)
            temp = []
            for (bbox, text, conf) in res:
                if conf > 0.4:
                    x1=int(bbox[0][0]); x2=int(bbox[2][0])
                    y1=int(bbox[0][1]); y2=int(bbox[2][1])
                    temp.append({"label": text, "bbox": (x1,y1,x2,y2)})
            with locks["ocr"]:
                texts = temp
    except Exception as e:
        add_log(f"OCR Error: {e}")

# ─── FACE RECOGNITION ───────────────
def face_thread():
    global faces
    try:
        add_log("Face thread started")
        while not stop_event.is_set():
            # Only run face recognition when YOLO sees a person — saves CPU
            with locks["det"]:
                has_person = any(d["label"] == "person" for d in dets)
            if not has_person:
                time.sleep(0.5); continue

            with locks["frame"]:
                f = frame
            if f is None:
                time.sleep(0.1); continue
            rgb  = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            temp = []
            for (t,r,b,l), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(known_encodings, enc, 0.5)
                name    = "unknown"
                if True in matches:
                    name = known_names[matches.index(True)]
                temp.append({"label": name, "bbox": (l,t,r,b)})
            with locks["face"]:
                faces = temp
            time.sleep(1.0)   # throttle: face recog is expensive on RPi
    except Exception as e:
        add_log(f"Face Error: {e}")

# ─── DEPTH (visual blur only) ────────
def depth_thread():
    global depth_map
    while not stop_event.is_set():
        with locks["frame"]:
            f = frame
        if f is None:
            time.sleep(0.05); continue
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        depth = cv2.GaussianBlur(gray, (11,11), 0)
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        with locks["depth"]:
            depth_map = depth

# ─── GPS THREAD ─────────────────────
def _parse_nmea_coord(val, hemi):
    if not val:
        return None
    dot = val.index('.')
    deg = float(val[:dot-2])
    mn  = float(val[dot-2:])
    dec = deg + mn / 60.0
    if hemi in ('S','W'):
        dec = -dec
    return dec

def gps_thread():
    add_log(f"GPS: using hardcoded VJTI ({VJTI_LAT},{VJTI_LON}) as fallback")
    try:
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
        add_log(f"GPS serial open: {GPS_PORT}")
    except Exception as e:
        add_log(f"GPS serial error: {e} — staying on VJTI default")
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('$GPRMC'):
                parts = line.split(',')
                if len(parts) >= 7 and parts[2] == 'A':
                    lat = _parse_nmea_coord(parts[3], parts[4])
                    lon = _parse_nmea_coord(parts[5], parts[6])
                    if lat and lon:
                        with locks["gps"]:
                            gps_state["lat"]  = lat
                            gps_state["lon"]  = lon
                            gps_state["fix"]  = True
                            gps_state["name"] = f"{lat:.5f}, {lon:.5f}"
                            gps_state["raw"]  = line
                        add_log(f"GPS fix: {lat:.5f}, {lon:.5f}")
                else:
                    with locks["gps"]:
                        gps_state["fix"] = False
                        gps_state["raw"] = line
        except Exception as e:
            add_log(f"GPS read error: {e}")
            time.sleep(1)

# ─── GPS HELPERS ────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    y  = math.sin(Δλ)*math.cos(φ2)
    x  = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    return round(dist, 1), round(bearing, 1)

def bearing_to_words(bearing):
    dirs = ["north","northeast","east","southeast",
            "south","southwest","west","northwest"]
    idx  = int((bearing + 22.5) / 45) % 8
    return dirs[idx]

def nav_instruction(dist_m, bearing):
    direction_word = bearing_to_words(bearing)
    if dist_m < 10:
        return "You have arrived at your destination."
    elif dist_m < 50:
        return f"Destination is {int(dist_m)} metres ahead. Keep going {direction_word}."
    elif dist_m < 200:
        return f"Head {direction_word} for about {int(dist_m)} metres to reach your destination."
    elif dist_m < 1000:
        return f"Walk {direction_word} for roughly {int(dist_m)} metres."
    else:
        km = dist_m / 1000.0
        return f"Destination is {km:.1f} kilometres {direction_word}. Keep walking {direction_word}."

def update_nav():
    with locks["gps"]:
        cur_lat = gps_state["lat"]
        cur_lon = gps_state["lon"]
    with locks["nav"]:
        if nav_state["destination_lat"] is None:
            nav_state["instruction"] = "No destination set."
            return
        dst_lat = nav_state["destination_lat"]
        dst_lon = nav_state["destination_lon"]
    dist, brng = haversine(cur_lat, cur_lon, dst_lat, dst_lon)
    instr      = nav_instruction(dist, brng)
    with locks["nav"]:
        nav_state["bearing"]     = brng
        nav_state["distance_m"]  = dist
        nav_state["instruction"] = instr

def nav_update_thread():
    while not stop_event.is_set():
        update_nav()
        time.sleep(3)

# ═══════════════════════════════════════════════════════════════
#  ULTRASONIC + SERVO  (fixed)
# ═══════════════════════════════════════════════════════════════

def _us_measure():
    try:
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        pulse_start = None
        t0 = time.time()
        while GPIO.input(ECHO_PIN) == 0:
            if time.time() - t0 > 0.05:
                return None
            pulse_start = time.time()

        if pulse_start is None:
            return None

        pulse_end = pulse_start
        t0 = time.time()
        while GPIO.input(ECHO_PIN) == 1:
            if time.time() - t0 > 0.05:
                return None
            pulse_end = time.time()

        duration = pulse_end - pulse_start
        dist     = (duration * 343.0) / 2.0

        if dist < US_MIN_RANGE or dist > US_MAX_RANGE:
            return None

        return round(dist, 3)

    except Exception as e:
        add_log(f"US measure error: {e}")
        return None


def _angle_to_zone(angle):
    if angle < 60:
        return "right"
    if angle > 120:
        return "left"
    return "center"


def _angle_to_dc(angle):
    return SERVO_MIN_DC + (angle / 180.0) * (SERVO_MAX_DC - SERVO_MIN_DC)


def ultrasonic_servo_thread():
    try:
        pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
        pwm.start(0)
        time.sleep(0.1)
        pwm.ChangeDutyCycle(_angle_to_dc(90))
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)
        add_log("Ultrasonic + Servo thread started")
    except Exception as e:
        add_log(f"Servo init error: {e}")
        return

    angle     = 90
    direction = 1
    sweep_pts = []

    while not stop_event.is_set():

        try:
            pwm.ChangeDutyCycle(_angle_to_dc(angle))
            time.sleep(SERVO_DELAY)
            pwm.ChangeDutyCycle(0)
            time.sleep(0.03)
        except Exception as e:
            add_log(f"Servo move error @ {angle}°: {e}")
            break

        dist = _us_measure()
        zone = _angle_to_zone(angle)

        sweep_pts.append((angle, dist if dist is not None else -1))

        with locks["radar"]:
            radar_state["angle"]      = angle
            radar_state["distance_m"] = dist
            radar_state["sweep"] = list(sweep_pts[-36:])

            if dist is not None:
                radar_state["close_zone"] = zone
                cm = int(dist * 100)
                radar_state["alert"] = (
                    f"Obstacle detected {cm} centimetres to your {zone}"
                )
                add_log(f"US hit: {angle}° ({zone}) → {dist:.3f} m")
            else:
                if radar_state["close_zone"] == zone:
                    radar_state["close_zone"] = None
                    radar_state["alert"]      = None

        next_angle = angle + direction * SERVO_STEP
        if next_angle >= 180 or next_angle <= 0:
            direction = -direction
            sweep_pts = []
            with locks["radar"]:
                radar_state["direction"] = direction
                radar_state["close_zone"] = None
                radar_state["alert"]      = None

        angle = max(0, min(180, angle + direction * SERVO_STEP))

    pwm.stop()


# ─── HELPERS ────────────────────────
def direction_zone(cx):
    if cx < DISP_W * 0.35: return "left"
    if cx > DISP_W * 0.65: return "right"
    return "center"

def estimate_distance(bbox, label):
    if label not in KNOWN_HEIGHTS:
        return None
    _, y1, _, y2 = bbox
    h_px   = max(1, y2 - y1)
    Z      = (KNOWN_HEIGHTS[label] * FOCAL_LENGTH_PX) / h_px
    return round(Z, 2)

def get_ultrasonic_override(dir_zone):
    with locks["radar"]:
        close = radar_state.get("close_zone")
        dist  = radar_state.get("distance_m")
    if close == dir_zone and dist is not None:
        return dist
    return None

def navigation_advice(label, dir_text, dist):
    us_dist = get_ultrasonic_override(dir_text)
    if us_dist is not None and us_dist < (dist or 999):
        dist = us_dist

    steer = "right" if dir_text == "left" else ("left" if dir_text == "right" else None)

    if dist is not None and dist < 0.5:
        if label in HIGH_RISK:
            return "Stop! Vehicle very close."
        if label in ANIMALS:
            return "Stop! Animal right in front of you."
        if label in known_names_set:
            return f"Stop. {label} is right in front of you."
        return "Stop! Obstacle very close."

    if dist is not None and dist < 1.0:
        if dir_text == "center":
            if label in HIGH_RISK:
                return "Vehicle directly ahead. Wait for it to pass."
            if label in ANIMALS:
                return "Animal directly ahead. Back away slowly."
            return "Obstacle right ahead. Stop or step aside."
        if label in HIGH_RISK:
            return f"Vehicle on your {dir_text}. Move sharply {steer}."
        return f"Obstacle on your {dir_text}. Move {steer} now."

    if dist is not None and dist < 2.0:
        if dir_text == "center":
            if label in HIGH_RISK:
                return "Moving vehicle ahead. Slow down and wait."
            if label in ANIMALS:
                return "Animal ahead. Slow down and give it space."
            if label in known_names_set:
                return f"{label} is ahead. Slow down."
            if label == "person":
                return "Person ahead. Walk around them."
            if label in STATIONARY_OBSTACLES:
                return f"{label} blocking your path. Go around it."
            return "Obstacle ahead. Slow down."
        else:
            if label in HIGH_RISK:
                return f"Vehicle on your {dir_text}. Shift {steer}."
            if label in known_names_set:
                return f"{label} on your {dir_text}."
            if label == "person":
                return f"Person on your {dir_text}. Bear {steer}."
            if label in STATIONARY_OBSTACLES:
                return f"{label} on your {dir_text}. Keep {steer}."
            return f"Object on your {dir_text}. Move {steer}."

    if label in HIGH_RISK:
        if dir_text == "center":
            return "Vehicle ahead. Proceed carefully."
        return f"Vehicle on your {dir_text}. Bear {steer}."
    if label in ANIMALS:
        if dir_text == "center":
            return "Animal ahead. Approach slowly."
        return f"Animal on your {dir_text}. Bear {steer}."
    if label in STATIONARY_OBSTACLES:
        if dir_text == "center":
            return f"{label} ahead. Plan to go around it."
        return f"{label} on your {dir_text}. Keep {steer}."
    if label in known_names_set:
        return f"{label} is {'ahead' if dir_text=='center' else 'on your '+dir_text}."
    if label == "person":
        if dir_text == "center":
            return "Person ahead. Proceed with caution."
        return f"Person on your {dir_text}."
    if dir_text == "center":
        return "Text visible ahead."
    return f"Sign on your {dir_text}."

def priority_score(d):
    x1,y1,x2,y2 = d["bbox"]
    area = (x2-x1)*(y2-y1)
    if d["label"] in known_names_set: return area * 3
    if d["label"] in HIGH_RISK:       return area * 2
    if d["label"] in ANIMALS:         return area * 1.5
    return area

# ─── ANNOUNCEMENT THREAD ─────────────
def announcement_thread():
    global last_announce
    while not stop_event.is_set():
        now = time.time()

        with _tts_lock:
            tts_busy = _tts_active
        if tts_busy:
            time.sleep(1); continue

        if now - last_announce < ANNOUNCE_INTERVAL:
            time.sleep(1); continue

        with locks["det"]:  objects   = list(dets)
        with locks["face"]: fs        = list(faces)
        with locks["ocr"]:  ocr_texts = list(texts)
        with locks["nav"]:
            nav_instr = nav_state["instruction"]
            nav_dist  = nav_state["distance_m"]
        with locks["radar"]:
            radar_alert = radar_state.get("alert")

        messages = []

        def build_msg(label, display, bbox):
            x1,y1,x2,y2 = bbox
            cx      = (x1+x2)//2
            dir_txt = direction_zone(cx)
            dist    = estimate_distance(bbox, label)
            advice  = navigation_advice(label, dir_txt, dist)
            dist_str = f", {dist:.1f} metres" if dist else ""
            return f"{display} on your {dir_txt}{dist_str}. {advice}"

        # 1. Ultrasonic alert takes highest priority
        if radar_alert:
            messages.append(radar_alert)

        # 2. Known people
        for p in [f for f in fs if f["label"] in known_names_set]:
            messages.append(build_msg("person", p["label"], p["bbox"]))
        # 3. High-risk
        for obj in sorted([o for o in objects if o["label"] in HIGH_RISK],
                           key=priority_score, reverse=True):
            messages.append(build_msg(obj["label"], obj["label"], obj["bbox"]))
        # 4. Animals
        for obj in sorted([o for o in objects if o["label"] in ANIMALS],
                           key=priority_score, reverse=True):
            messages.append(build_msg(obj["label"], obj["label"], obj["bbox"]))
        # 5. Unknown people
        for p in [f for f in fs if f["label"] == "unknown"]:
            messages.append(build_msg("person", "Unknown person", p["bbox"]))
        # 6. Stationary obstacles
        for obj in sorted([o for o in objects if o["label"] in STATIONARY_OBSTACLES],
                           key=priority_score, reverse=True):
            messages.append(build_msg(obj["label"], obj["label"], obj["bbox"]))
        # 7. Remaining YOLO objects
        classified = HIGH_RISK | ANIMALS | STATIONARY_OBSTACLES | {"person"}
        for obj in sorted([o for o in objects if o["label"] not in classified],
                           key=priority_score, reverse=True):
            messages.append(build_msg(obj["label"], obj["label"], obj["bbox"]))
        # 8. OCR
        for t in ocr_texts:
            x1,y1,x2,y2 = t["bbox"]
            cx      = (x1+x2)//2
            dir_txt = direction_zone(cx)
            advice  = navigation_advice("sign", dir_txt, None)
            messages.append(f"Sign reads: {t['label']}, on your {dir_txt}. {advice}")
        # 9. Navigation
        if nav_dist is not None and nav_dist > 5:
            messages.append(nav_instr)

        if messages:
            full = ". ".join(messages)
            speak(full)
            add_log(f"ANNOUNCE: {full}")
        else:
            speak("Path is clear. You may proceed.")
            add_log("Announce: path clear")

        last_announce = now
        time.sleep(1)

# ─── FLASK ──────────────────────────
app = Flask(__name__)

def generate_video():
    while True:
        with locks["frame"]:
            f = frame
        if f is None:
            time.sleep(0.02); continue

        vis = f.copy()

        with locks["det"]:   objects = list(dets)
        with locks["ocr"]:   txts    = list(texts)
        with locks["face"]:  fs      = list(faces)

        for d in objects + txts + fs:
            x1,y1,x2,y2 = d["bbox"]
            label   = d["label"]
            cx      = (x1+x2)//2
            dir_txt = direction_zone(cx)
            dist    = estimate_distance((x1,y1,x2,y2), label)

            us = get_ultrasonic_override(dir_txt)
            if us is not None:
                dist = us

            if label in HIGH_RISK:                color = (0,0,255)
            elif label in known_names_set:         color = (255,80,80)
            elif label in ANIMALS:                 color = (0,165,255)
            elif label in STATIONARY_OBSTACLES:    color = (0,220,255)
            else:                                  color = (0,200,80)

            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            dist_str = f" {dist:.1f}m" if dist else ""
            cv2.putText(vis, f"{label}|{dir_txt}{dist_str}",
                        (x1, max(y1-6,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        lx = int(DISP_W*0.35); rx = int(DISP_W*0.65)
        cv2.line(vis,(lx,0),(lx,DISP_H),(170,170,170),1)
        cv2.line(vis,(rx,0),(rx,DISP_H),(170,170,170),1)
        cv2.putText(vis,"LEFT",  (8,16),    cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),1)
        cv2.putText(vis,"CENTER",(lx+8,16), cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),1)
        cv2.putText(vis,"RIGHT", (rx+8,16), cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),1)

        with locks["radar"]:
            alert = radar_state.get("alert")
        if alert:
            cv2.putText(vis, f"RADAR: {alert}", (8, DISP_H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    with locks["log"]:
        return jsonify({"logs": list(log_buffer)})

@app.route('/gps_status')
def gps_status():
    with locks["gps"]:
        g = dict(gps_state)
    with locks["nav"]:
        n = dict(nav_state)
    return jsonify({"gps": g, "nav": n})

@app.route('/radar_status')
def radar_status():
    with locks["radar"]:
        r = dict(radar_state)
    return jsonify(r)

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json(force=True)
    lat  = data.get("lat")
    lon  = data.get("lon")
    name = data.get("name","Custom destination")
    if lat is None or lon is None:
        return jsonify({"status":"error","msg":"lat/lon required"}), 400
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"status":"error","msg":"lat/lon out of range"}), 400
    with locks["nav"]:
        nav_state["destination_lat"]  = lat
        nav_state["destination_lon"]  = lon
        nav_state["destination_name"] = name
    add_log(f"Destination set: {name} ({lat},{lon})")
    update_nav()
    with locks["nav"]:
        instr = nav_state["instruction"]
    speak(f"Destination set to {name}. {instr}")
    return jsonify({"status":"ok","instruction": instr})

@app.route('/clear_destination', methods=['POST'])
def clear_destination():
    with locks["nav"]:
        nav_state["destination_lat"]  = None
        nav_state["destination_lon"]  = None
        nav_state["destination_name"] = "Not set"
        nav_state["instruction"]      = "No destination set."
    speak("Destination cleared.")
    return jsonify({"status":"ok"})

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Assistive Vision System</title>
<style>
  body{margin:0;font-family:monospace;background:#0d0d0d;color:#ccc;padding:12px;}
  h1{font-size:1.1rem;color:#fff;margin-bottom:4px;}
  .sub{font-size:.7rem;color:#555;margin-bottom:16px;}
  .row{display:flex;gap:12px;flex-wrap:wrap;}
  .card{background:#111;border:1px solid #222;border-radius:4px;padding:12px;flex:1;min-width:260px;}
  .card h2{font-size:.75rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}
  img#feed{width:100%;border:1px solid #222;border-radius:4px;display:block;margin-bottom:12px;}
  label{font-size:.72rem;color:#888;display:block;margin-bottom:3px;}
  input,select{width:100%;background:#0a0a0a;border:1px solid #333;color:#ccc;
    padding:6px 8px;font-family:monospace;font-size:.75rem;border-radius:3px;box-sizing:border-box;margin-bottom:8px;}
  button{background:#1a1a1a;border:1px solid #333;color:#aaa;padding:7px 14px;
    font-family:monospace;font-size:.72rem;border-radius:3px;cursor:pointer;width:100%;margin-bottom:6px;}
  button:hover{border-color:#00e5ff;color:#00e5ff;}
  .status{font-size:.72rem;line-height:1.7;color:#666;}
  .status span{color:#00e5ff;}
  .ok{color:#00e676!important;} .warn{color:#ff4f4f!important;}
  #logBox{height:180px;overflow-y:auto;font-size:.65rem;line-height:1.8;color:#444;}
  .ll{border-bottom:1px solid #161616;} .la{color:#00e5ff;} .lw{color:#ff4f4f;} .lo{color:#00e676;}
  canvas#radar{display:block;width:100%;border:1px solid #1a1a1a;border-radius:4px;}
  #navBox{background:#0a0a0a;border:1px solid #1e3a1e;border-radius:3px;padding:8px;
    font-size:.75rem;color:#00e676;min-height:32px;margin-top:6px;}
  #alertBox{background:#1a0000;border:1px solid #3a0000;border-radius:3px;padding:6px 8px;
    font-size:.75rem;color:#ff4f4f;min-height:24px;margin-top:6px;display:none;}
  #mapWrap{position:relative;width:100%;padding-bottom:56%;background:#111;border:1px solid #222;border-radius:4px;overflow:hidden;margin-top:8px;}
  #mapWrap iframe{position:absolute;top:0;left:0;width:100%;height:100%;border:0;}
  .badge{display:inline-block;font-size:.6rem;padding:2px 6px;border-radius:2px;margin-left:6px;}
  .gps-ok{background:#0d2b0d;color:#00e676;} .gps-no{background:#2b0d0d;color:#ff4f4f;}
</style>
</head>
<body>
<h1>Assistive Vision System</h1>
<p class="sub">VJTI · YOLOv8 + YOLOv5 Zebra · EasyOCR · Face Recognition · NEO-6M GPS · HC-SR04 Radar</p>

<img id="feed" src="/video_feed" alt="Live camera feed">

<div class="row">
  <div class="card">
    <h2>GPS &amp; Navigation</h2>
    <div class="status" id="gpsStatus">Loading…</div>
    <div id="navBox">No destination set.</div>
    <br>
    <label>Quick destinations (VJTI area)</label>
    <select id="presetSel">
      <option value="">— select preset —</option>
      <option value="19.0255,72.8558,VJTI Main Gate">VJTI Main Gate</option>
      <option value="19.0228,72.8565,VJTI Library">VJTI Library</option>
      <option value="19.0220,72.8548,VJTI Canteen">VJTI Canteen</option>
      <option value="19.0210,72.8530,Matunga Station">Matunga Station (West)</option>
      <option value="19.0185,72.8472,King Circle Station">King Circle Station</option>
    </select>
    <label>Or enter custom destination</label>
    <input id="destName" type="text" placeholder="Destination name">
    <input id="destLat"  type="number" step="0.00001" placeholder="Latitude">
    <input id="destLon"  type="number" step="0.00001" placeholder="Longitude">
    <button onclick="setDest()">Set Destination &amp; Speak</button>
    <button onclick="clearDest()">Clear Destination</button>
    <div id="mapWrap">
      <iframe id="mapFrame"
        src="https://www.openstreetmap.org/export/embed.html?bbox=72.848,19.018,72.862,19.030&layer=mapnik&marker=19.0228,72.8553"
        allowfullscreen></iframe>
    </div>
    <p style="font-size:.6rem;color:#444;margin-top:4px;">Map © <a href="https://openstreetmap.org" style="color:#444;">OpenStreetMap</a> contributors</p>
  </div>

  <div class="card">
    <h2>Ultrasonic Radar (HC-SR04 + Servo)</h2>
    <canvas id="radar" width="300" height="160"></canvas>
    <div class="status" id="radarStatus" style="margin-top:6px;">Initialising…</div>
    <div id="alertBox"></div>
  </div>

  <div class="card">
    <h2>System Log</h2>
    <div id="logBox"></div>
    <button onclick="document.getElementById('logBox').innerHTML=''" style="margin-top:6px;">Clear</button>
  </div>
</div>

<script>
function pollGPS(){
  fetch('/gps_status').then(r=>r.json()).then(d=>{
    const g=d.gps, n=d.nav;
    const fix = g.fix ? '<span class="badge gps-ok">GPS FIX</span>'
                      : '<span class="badge gps-no">NO FIX (VJTI default)</span>';
    document.getElementById('gpsStatus').innerHTML=
      `<span>${g.name}</span>${fix}<br>`+
      `Lat: <span>${g.lat.toFixed(5)}</span> &nbsp; Lon: <span>${g.lon.toFixed(5)}</span><br>`+
      `Destination: <span>${n.destination_name}</span><br>`+
      (n.distance_m !== null ? `Distance: <span>${(n.distance_m/1000).toFixed(2)} km</span> &nbsp; Bearing: <span>${n.bearing}°</span>` : '');
    document.getElementById('navBox').textContent = n.instruction || 'No instruction.';
  }).catch(()=>{});
}
setInterval(pollGPS, 3000);
pollGPS();

document.getElementById('presetSel').addEventListener('change',function(){
  if(!this.value) return;
  const parts = this.value.split(',');
  document.getElementById('destLat').value  = parts[0];
  document.getElementById('destLon').value  = parts[1];
  document.getElementById('destName').value = parts[2];
});

function setDest(){
  const lat  = parseFloat(document.getElementById('destLat').value);
  const lon  = parseFloat(document.getElementById('destLon').value);
  const name = document.getElementById('destName').value || 'Destination';
  if(isNaN(lat)||isNaN(lon)){alert('Enter valid lat/lon'); return;}
  fetch('/set_destination',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lat,lon,name})
  }).then(r=>r.json()).then(d=>{
    document.getElementById('navBox').textContent = d.instruction;
    const url=`https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.01},${lat-0.01},${lon+0.01},${lat+0.01}&layer=mapnik&marker=${lat},${lon}`;
    document.getElementById('mapFrame').src = url;
  });
}

function clearDest(){
  fetch('/clear_destination',{method:'POST'}).then(()=>pollGPS());
}

const canvas = document.getElementById('radar');
const ctx    = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CX = W/2, CY = H;
const MAX_R = H - 10;

function drawRadar(sweep, curAngle, curDist, alert){
  ctx.fillStyle='#080808';
  ctx.fillRect(0,0,W,H);

  [0.25,0.5,0.75,1.0].forEach((d)=>{
    const r = (d/1.0)*MAX_R;
    ctx.beginPath();
    ctx.arc(CX,CY,r,Math.PI,0);
    ctx.strokeStyle='#1a2a1a';
    ctx.lineWidth=1;
    ctx.stroke();
    ctx.fillStyle='#2a3a2a';
    ctx.font='9px monospace';
    ctx.fillText(`${d*100}cm`, CX+r+2, CY-2);
  });

  [[60,'#1a1a2a'],[120,'#1a1a2a']].forEach(([a,c])=>{
    const rad = (a/180)*Math.PI;
    ctx.beginPath();
    ctx.moveTo(CX,CY);
    ctx.lineTo(CX+Math.cos(Math.PI-rad)*MAX_R, CY-Math.sin(Math.PI-rad)*MAX_R);
    ctx.strokeStyle=c; ctx.lineWidth=1; ctx.stroke();
  });
  ctx.fillStyle='#555'; ctx.font='9px monospace';
  ctx.fillText('RIGHT',4,H-4);
  ctx.fillText('CENTER',W/2-18,H-4);
  ctx.fillText('LEFT',W-28,H-4);

  sweep.forEach(([a,d])=>{
    if(d<=0 || d>1.0) return;
    const r   = (d/1.0)*MAX_R;
    const rad = (a/180)*Math.PI;
    const x   = CX + Math.cos(Math.PI-rad)*r;
    const y   = CY - Math.sin(Math.PI-rad)*r;
    const g   = d < 0.5 ? '#ff3333' : '#ff9933';
    ctx.beginPath(); ctx.arc(x,y,4,0,2*Math.PI);
    ctx.fillStyle=g; ctx.fill();
    ctx.fillStyle='#fff'; ctx.font='8px monospace';
    ctx.fillText(`${Math.round(d*100)}cm`, x+5, y-3);
  });

  if(curAngle !== null){
    const rad = (curAngle/180)*Math.PI;
    ctx.beginPath();
    ctx.moveTo(CX,CY);
    ctx.lineTo(CX+Math.cos(Math.PI-rad)*MAX_R, CY-Math.sin(Math.PI-rad)*MAX_R);
    ctx.strokeStyle= alert ? 'rgba(255,50,50,0.6)' : 'rgba(0,200,80,0.35)';
    ctx.lineWidth=2; ctx.stroke();
  }
}

function pollRadar(){
  fetch('/radar_status').then(r=>r.json()).then(d=>{
    drawRadar(d.sweep||[], d.angle, d.distance_m, d.alert);
    const dist = d.distance_m;
    const zone = d.close_zone;
    const alert = d.alert;

    let msg = `Servo: ${d.angle}°`;
    if(dist !== null && dist > 0) msg += `  |  Distance: ${(dist*100).toFixed(0)} cm`;
    if(zone) msg += `  |  Zone: <span class="warn">${zone}</span>`;
    document.getElementById('radarStatus').innerHTML = msg;

    const alertBox = document.getElementById('alertBox');
    if(alert){
      alertBox.style.display='block';
      alertBox.textContent = alert;
    } else {
      alertBox.style.display='none';
    }
  }).catch(()=>{});
}
setInterval(pollRadar, 200);
pollRadar();

let autoScroll = true;
function pollLogs(){
  fetch('/logs').then(r=>r.json()).then(d=>{
    const box = document.getElementById('logBox');
    box.innerHTML='';
    d.logs.forEach(l=>{
      const div=document.createElement('div');
      div.className='ll '+(l.includes('ANNOUNCE')?'la':l.includes('Error')||l.includes('WARN')?'lw':l.includes('loaded')||l.includes('clear')||l.includes('started')||l.includes('US hit')?'lo':'');
      div.textContent=l;
      box.appendChild(div);
    });
    if(autoScroll) box.scrollTop=box.scrollHeight;
  }).catch(()=>{});
}
setInterval(pollLogs,1000);
pollLogs();
</script>
</body>
</html>"""

# ─── MAIN ────────────────────────────
def main():
    add_log("System starting…")
    add_log(f"Pinhole f_px={FOCAL_LENGTH_PX:.2f} | sensor_h={SENSOR_HEIGHT_CM}cm")

    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.1)
        add_log("GPIO pre-initialised")
    except Exception as e:
        add_log(f"GPIO pre-init error: {e}")

    load_faces()

    threads = [
        threading.Thread(target=camera_thread,           daemon=True, name="camera"),
        threading.Thread(target=yolo_thread,             daemon=True, name="yolo"),
        threading.Thread(target=ocr_thread,              daemon=True, name="ocr"),
        threading.Thread(target=face_thread,             daemon=True, name="face"),
        threading.Thread(target=depth_thread,            daemon=True, name="depth"),
        threading.Thread(target=gps_thread,              daemon=True, name="gps"),
        threading.Thread(target=nav_update_thread,       daemon=True, name="nav"),
        threading.Thread(target=ultrasonic_servo_thread, daemon=True, name="radar"),
        threading.Thread(target=announcement_thread,     daemon=True, name="announce"),
    ]
    for t in threads:
        t.start()
        add_log(f"Thread: {t.name}")

    add_log("Flask → http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        stop_event.set()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        add_log("System stopped")

if __name__ == "__main__":
    main()