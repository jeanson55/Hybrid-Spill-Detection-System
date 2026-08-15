"""
Multi-Frame Temporal Confirmation Mechanism
Implements the temporal persistence logic from thesis Section 3.5.

Parameters (optimal from Table 4.5):
    T_w   = 3.0 seconds  — confirmation window
    N_min = 7            — minimum detections within window
    T_c   = 60 seconds   — cooldown period after confirmed alert
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TemporalEvent:
    timestamp: float
    bbox: tuple
    confidence: float
    residual: float
    fluid_class: str = "unknown"
    uncertainty: float = 0.0


@dataclass
class ConfirmedAlert:
    confirmed_at: float
    bbox: tuple
    avg_confidence: float
    avg_residual: float
    detection_count: int
    fluid_class: str = "unknown"       # most common fluid_class across the window
    avg_uncertainty: float = 0.0
    alert_id: str = field(default_factory=lambda: f"ALERT-{int(time.time()*1000)%100000:05d}")


class TemporalConfirmation:
    """
    Tracks detection events across frames and issues a confirmed alert
    only when N_min detections occur within T_w seconds.
    Suppresses repeated alerts via a cooldown period T_c.
    """

    T_W: float = 3.0       # confirmation window (seconds)
    N_MIN: int = 7         # minimum detections in window
    T_C: float = 60.0      # cooldown after confirmed alert (seconds)

    def __init__(self):
        self._buffer: deque = deque()
        self._last_alert_time: float = 0.0
        self._confirmed_alerts: list[ConfirmedAlert] = []

    def update(
        self,
        timestamp: float,
        bbox: tuple,
        confidence: float,
        residual: float,
        fluid_class: str = "unknown",
        uncertainty: float = 0.0,
    ) -> Optional[ConfirmedAlert]:
        """
        Push a new validated detection into the temporal buffer.
        Returns a ConfirmedAlert if the confirmation criteria are met,
        or None if still accumulating.
        """
        # Add to buffer
        self._buffer.append(TemporalEvent(timestamp, bbox, confidence, residual,
                                           fluid_class, uncertainty))

        # Prune events outside the window
        cutoff = timestamp - self.T_W
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

        # Check cooldown
        if (timestamp - self._last_alert_time) < self.T_C:
            return None

        # Check confirmation threshold
        if len(self._buffer) >= self.N_MIN:
            events = list(self._buffer)
            fluid_counts = {}
            for e in events:
                fluid_counts[e.fluid_class] = fluid_counts.get(e.fluid_class, 0) + 1
            majority_fluid = max(fluid_counts, key=fluid_counts.get)

            alert = ConfirmedAlert(
                confirmed_at=timestamp,
                bbox=self._consensus_bbox(events),
                avg_confidence=sum(e.confidence for e in events) / len(events),
                avg_residual=sum(e.residual for e in events) / len(events),
                detection_count=len(events),
                fluid_class=majority_fluid,
                avg_uncertainty=sum(e.uncertainty for e in events) / len(events),
            )
            self._last_alert_time = timestamp
            self._confirmed_alerts.append(alert)
            self._buffer.clear()
            return alert

        return None

    def reset_cooldown(self):
        """Allow next alert immediately (for testing/manual override)."""
        self._last_alert_time = 0.0

    def get_buffer_fill(self) -> dict:
        """Returns current buffer state for dashboard display."""
        return {
            "count": len(self._buffer),
            "required": self.N_MIN,
            "window_s": self.T_W,
            "pct": min(100, int(len(self._buffer) / self.N_MIN * 100)),
        }

    def get_all_alerts(self) -> list[ConfirmedAlert]:
        return list(reversed(self._confirmed_alerts))

    @staticmethod
    def _consensus_bbox(events: list[TemporalEvent]) -> tuple:
        """Median bounding box across the window."""
        x1s = [e.bbox[0] for e in events]
        y1s = [e.bbox[1] for e in events]
        x2s = [e.bbox[2] for e in events]
        y2s = [e.bbox[3] for e in events]
        return (
            int(sorted(x1s)[len(x1s)//2]),
            int(sorted(y1s)[len(y1s)//2]),
            int(sorted(x2s)[len(x2s)//2]),
            int(sorted(y2s)[len(y2s)//2]),
        )