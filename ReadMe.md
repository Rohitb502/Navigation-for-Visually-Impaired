# BlindNav — Requirements & Setup

## Hardware
- Raspberry Pi 4 or 5 (4 GB RAM recommended)
- Pi Camera Module v2 or v3
- Wired headphones (3.5mm jack) or Bluetooth headphones
- Optional: battery bank for portable deployment

## Python dependencies

```
# Install system deps first (Raspberry Pi OS)
sudo apt-get update
sudo apt-get install -y python3-picamera2 espeak libespeak-dev portaudio19-dev

# Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install pyttsx3
pip install opencv-python-headless   # use 'opencv-python' if you want display
pip install timm                     # required by MiDaS
```

## Model files
- `yolo11n.pt` — download via: `yolo export model=yolo11n.pt` or let ultralytics auto-download on first run
- MiDaS weights are auto-downloaded by `torch.hub.load`

## File structure
```
blind_nav/
├── main.py            # Entry point
├── priority_engine.py # Danger scoring logic
├── audio_engine.py    # TTS + smart throttling
└── requirements.txt
```

## Running
```bash
python main.py
```
Set `DISPLAY = False` in `main.py` for headless/production deployment.
Press `q` in the OpenCV window (if display enabled) to quit.

## Tuning guide

### Priority weights (`priority_engine.py → CLASS_WEIGHTS`)
Increase weights for objects common in your city (e.g. raise `"bicycle"` in
Amsterdam, `"motorcycle"` in Mumbai).

### Audio cooldowns (`audio_engine.py`)
| Setting | Default | Meaning |
|---|---|---|
| COOLDOWN_NORMAL | 6 s | Quiet period before repeating a distant object |
| COOLDOWN_NEAR | 3 s | Repeat interval for objects 2–4 m away |
| COOLDOWN_URGENT | 1 s | Repeat interval for objects < 2 m (danger zone) |
| URGENCY_THRESHOLD | 1.8 m | Distance that triggers the one-shot urgency override |
| POST_SPEECH_PAUSE | 0.6 s | Silence after each spoken message |

### Depth scale (`main.py → DEPTH_SCALE`)
MiDaS produces relative depth (not metric). `DEPTH_SCALE = 10.0` maps the
relative 0–1 range to 0–10 metres. Calibrate by placing a known object at
3 m and adjusting until the reading matches.

## Architecture diagram

```
Pi Camera
    │
    ▼
[Capture Thread] ──── latest_frame (shared, lock-protected)
                              │
              ┌───────────────┘
              ▼
       [Inference Thread]
        ├── MiDaS → depth_map
        └── YOLOv11 → bboxes + labels
              │
              ▼
       PriorityEngine.score()
        → priority_score, nav_hint per Detection
              │
              ▼ (filtered by AudioEngine.should_announce)
       audio_queue (PriorityQueue, highest score first)
              │
              ▼
       [Audio Thread]
        └── AudioEngine.speak() → pyttsx3 → headphones
```
