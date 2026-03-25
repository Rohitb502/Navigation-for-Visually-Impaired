"""
audio_engine.py — Smart Audio Throttle + TTS for BlindNav

Core ideas:
  • Per-class cooldown: avoid repeating "car ahead" every 100 ms
  • Urgency override: critical objects (< 1.5 m) bypass the cooldown
  • Post-speech pause: after speaking, a brief silence lets the user act
  • Priority queue draining: if a higher-priority item arrives while speaking,
    it interrupts (or the lower-priority item is dropped)
  • Sentence templates vary by distance zone to avoid robotic repetition
"""

import pyttsx3
import time
import threading
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from priority_engine import Detection

# ─── Config ──────────────────────────────────────────────────────────────────
COOLDOWN_NORMAL   = 6.0    # seconds before repeating same class at safe distance
COOLDOWN_NEAR     = 3.0    # seconds for near objects (2–4 m)
COOLDOWN_URGENT   = 1.0    # seconds for very close objects (< 2 m)
URGENCY_THRESHOLD = 1.8    # metres — below this, cooldown is overridden once
POST_SPEECH_PAUSE = 0.6    # silence after each utterance

VOICE_RATE  = 175           # words per minute (150–190 is natural)
VOICE_VOL   = 1.0

# ─── Message Templates ───────────────────────────────────────────────────────
# Three distance zones: urgent (<2m), near (2–5m), far (>5m)
# {label}, {dist:.1f}, {hint} are injected at runtime.

TEMPLATES = {
    "urgent": [
        "{label} very close at {dist:.1f} metres — {hint} now",
        "Warning! {label} just {dist:.1f} metres away — {hint}",
        "Danger — {label} ahead, {dist:.1f} metres — {hint} immediately",
    ],
    "near": [
        "{label} ahead, about {dist:.1f} metres — {hint}",
        "{label} at {dist:.1f} metres — please {hint}",
        "Approaching {label}, {dist:.1f} metres away — {hint}",
    ],
    "far": [
        "{label} detected, roughly {dist:.1f} metres — be aware",
        "{label} in your path at {dist:.1f} metres",
        "Heads up — {label} at about {dist:.1f} metres",
    ],
}

def _zone(dist: float) -> str:
    if dist < 2.0:
        return "urgent"
    if dist < 5.0:
        return "near"
    return "far"

def _build_message(det: Detection) -> str:
    zone = _zone(det.distance)
    template = random.choice(TEMPLATES[zone])
    return template.format(label=det.label, dist=det.distance, hint=det.nav_hint)


# ─── Audio Engine ─────────────────────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        self._tts = pyttsx3.init()
        self._tts.setProperty("rate",   VOICE_RATE)
        self._tts.setProperty("volume", VOICE_VOL)

        # last-announced timestamps per class label
        self._last_spoken: Dict[str, float] = {}
        # track whether an urgent one-shot override has fired
        self._urgent_fired: Dict[str, float] = {}

        self._lock = threading.Lock()
        self._speaking = False

    # ── Cooldown logic ────────────────────────────────────────────────────────
    def _cooldown_for(self, det: Detection) -> float:
        if det.distance < 2.0:
            return COOLDOWN_URGENT
        if det.distance < 4.5:
            return COOLDOWN_NEAR
        return COOLDOWN_NORMAL

    def should_announce(self, det: Detection) -> bool:
        """
        Returns True if this detection should enter the audio queue.
        Implements per-class cooldown with an urgency override.
        """
        with self._lock:
            now = time.monotonic()
            last = self._last_spoken.get(det.label, 0.0)
            cooldown = self._cooldown_for(det)

            # Urgency override: if object is critically close and we haven't
            # fired the override recently, bypass cooldown once.
            if det.distance <= URGENCY_THRESHOLD:
                last_urgent = self._urgent_fired.get(det.label, 0.0)
                if now - last_urgent >= COOLDOWN_URGENT:
                    self._urgent_fired[det.label] = now
                    return True

            # Normal cooldown gate
            return (now - last) >= cooldown

    def speak(self, det: Detection):
        """Synthesise and speak the message for a detection, then pause."""
        msg = _build_message(det)

        with self._lock:
            self._speaking = True
            self._last_spoken[det.label] = time.monotonic()

        try:
            self._tts.say(msg)
            self._tts.runAndWait()
            time.sleep(POST_SPEECH_PAUSE)
        finally:
            with self._lock:
                self._speaking = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking
