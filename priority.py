"""
priority_engine.py — Obstacle Priority Scoring for BlindNav

Scoring factors:
  1. Object class danger weight    (pothole > car > person > bicycle …)
  2. Estimated distance            (closer = higher score, non-linear)
  3. Position in frame             (central objects score higher)
  4. Relative size / closing speed (large bbox = likely approaching)
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# ─── Danger Weights ──────────────────────────────────────────────────────────
# Scale: 0–100.  Unlisted classes get DEFAULT_WEIGHT.
DEFAULT_WEIGHT = 20

CLASS_WEIGHTS: dict[str, int] = {
    # ── Vehicles (moving, lethal) ─────────────────────────
    "car":          90,
    "truck":        95,
    "bus":          95,
    "motorcycle":   85,
    "bicycle":      70,
    "scooter":      70,
    "train":       100,
    # ── Road hazards ──────────────────────────────────────
    "pothole":      80,
    "stop sign":    30,
    "traffic light":25,
    # ── Pedestrians ───────────────────────────────────────
    "person":       60,
    # ── Animals ───────────────────────────────────────────
    "dog":          55,
    "cat":          30,
    # ── Static obstacles ──────────────────────────────────
    "bench":        25,
    "chair":        20,
    "suitcase":     20,
    "backpack":     10,
    "umbrella":     15,
    "fire hydrant": 35,
    "parking meter":25,
    "pole":         40,
    # ── Dropped objects on footpath ───────────────────────
    "bottle":       10,
    "cup":          10,
    "skateboard":   30,
}

# Suggested navigation actions per class
NAVIGATION_HINTS: dict[str, str] = {
    "car":          "move to the right and wait",
    "truck":        "stop immediately and move right",
    "bus":          "stop and move to the right",
    "motorcycle":   "step right",
    "bicycle":      "step right",
    "scooter":      "step right",
    "train":        "stop — do not cross",
    "pothole":      "step around carefully",
    "person":       "bear slightly right",
    "dog":          "slow down and bear right",
    "bench":        "move around to the right",
    "fire hydrant": "step to the left",
    "pole":         "move around the pole",
}

DEFAULT_HINT = "proceed with caution"


# ─── Detection Dataclass ─────────────────────────────────────────────────────
@dataclass
class Detection:
    label:          str
    conf:           float
    distance:       float           # estimated meters
    bbox:           Tuple[int,int,int,int]
    bbox_fraction:  float           # bbox width / frame width
    frame_shape:    Tuple[int,int]  # (H, W)

    # Filled by PriorityEngine.score()
    priority_score: float = 0.0
    nav_hint:       str   = ""


# ─── Priority Engine ─────────────────────────────────────────────────────────
class PriorityEngine:
    """
    Scores each Detection and attaches a priority_score (0–300+) and nav_hint.
    Higher score = announce first.
    """

    # Distance breakpoints (metres) → score multiplier
    _DIST_CURVE = [
        (1.0, 3.0),   # ≤ 1 m  → ×3.0  (immediate danger)
        (2.0, 2.5),   # ≤ 2 m  → ×2.5
        (3.5, 2.0),   # ≤ 3.5 m→ ×2.0
        (5.0, 1.5),   # ≤ 5 m  → ×1.5
        (8.0, 1.0),   # ≤ 8 m  → ×1.0
        (float("inf"), 0.5),  # > 8 m  → ×0.5
    ]

    def _distance_multiplier(self, d: float) -> float:
        for threshold, mult in self._DIST_CURVE:
            if d <= threshold:
                return mult
        return 0.5

    def _center_bias(self, det: Detection) -> float:
        """Returns 1.0 if bbox centre is at frame centre, 0.5 at edge."""
        _, _, x2, _ = det.bbox
        x1 = det.bbox[0]
        cx = (x1 + x2) / 2
        _, W = det.frame_shape
        deviation = abs(cx - W / 2) / (W / 2)   # 0 = centre, 1 = edge
        return 1.0 - 0.5 * deviation

    def score(self, detections: List[Detection]) -> List[Detection]:
        for det in detections:
            base      = CLASS_WEIGHTS.get(det.label, DEFAULT_WEIGHT)
            dist_mult = self._distance_multiplier(det.distance)
            center    = self._center_bias(det)
            size_bonus = min(det.bbox_fraction * 50, 30)  # up to +30 for large bbox

            det.priority_score = base * dist_mult * center + size_bonus
            det.nav_hint = NAVIGATION_HINTS.get(det.label, DEFAULT_HINT)

        # Sort highest priority first
        detections.sort(key=lambda d: d.priority_score, reverse=True)
        return detections
