"""
Hybrid Spill Detection Pipeline
Integrates: YOLO → Geometric Gate → PINN Residual → Temporal Confirmation
"""

import time
import json
import base64
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional

from modules.geometric_validator import GeometricValidator, Detection
from modules.pinn_validator import PINNValidator
from modules.temporal_confirmation import TemporalConfirmation, ConfirmedAlert
from modules.gradcam_overlay import draw_detection_overlay, try_gradcam
from modules.fluid_classifier import FluidClassifier, EnvironmentalConditions


# ── Pipeline State ────────────────────────────────────────────────────────────

class PipelineStats:
    def __init__(self, maxlen=60):
        self.fps_history = deque(maxlen=maxlen)
        self.residual_history = deque(maxlen=maxlen)
        self.uncertainty_history = deque(maxlen=maxlen)
        self.total_frames = 0
        self.total_detections = 0
        self.geo_rejections = 0
        self.pinn_rejections = 0
        self.confirmed_alerts = 0
        self.decision_method = "fixed_threshold"
        self.pde_mode = "thin_film"
        self._last_frame_time = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        self.total_frames += 1

    @property
    def fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return round(sum(self.fps_history) / len(self.fps_history), 1)

    @property
    def avg_residual(self) -> float:
        if not self.residual_history:
            return 0.0
        return round(sum(self.residual_history) / len(self.residual_history), 4)

    @property
    def avg_uncertainty(self) -> float:
        if not self.uncertainty_history:
            return 0.0
        return round(sum(self.uncertainty_history) / len(self.uncertainty_history), 5)

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "geo_rejections": self.geo_rejections,
            "pinn_rejections": self.pinn_rejections,
            "confirmed_alerts": self.confirmed_alerts,
            "avg_residual": self.avg_residual,
            "avg_uncertainty": self.avg_uncertainty,
            "decision_method": self.decision_method,
            "pde_mode": self.pde_mode,
        }


# ── Spill Image Saver ─────────────────────────────────────────────────────────

class SpillImageSaver:
    """
    Saves annotated spill images to disk whenever a confirmed alert fires.

    Directory layout:
        spill_captures/
            YYYY-MM-DD/
                ALERT-XXXXX_HH-MM-SS_annotated.jpg   ← full frame with bbox
                ALERT-XXXXX_HH-MM-SS_crop.jpg         ← cropped spill region
                ALERT-XXXXX_HH-MM-SS_meta.json        ← alert metadata

    A running index file `spill_captures/capture_log.json` accumulates
    every saved event for easy review.
    """

    JPEG_QUALITY: int = 92
    CROP_PADDING: int = 30      # pixels added around bbox for the crop

    def __init__(self, base_dir: str = "spill_captures"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.base_dir / "capture_log.json"
        self._log: list = self._load_log()
        print(f"[Saver] Spill captures will be saved to: {self.base_dir.resolve()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def save(
        self,
        frame: np.ndarray,
        alert: "ConfirmedAlert",
        fluid_class: str = "auto",
        probability: Optional[float] = None,
        uncertainty: Optional[float] = None,
        detected_fluid_class: Optional[str] = None,
    ) -> dict:
        """
        Save annotated frame + crop + metadata for a confirmed alert.
        Returns a dict with the saved file paths.
        """
        now = datetime.now()
        date_folder = self.base_dir / now.strftime("%Y-%m-%d")
        date_folder.mkdir(exist_ok=True)

        ts_str = now.strftime("%H-%M-%S")
        stem = f"{alert.alert_id}_{ts_str}"

        # 1. Annotated full frame
        annotated = self._draw_save_overlay(frame.copy(), alert)
        ann_path = date_folder / f"{stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])

        # 2. Cropped spill region
        crop = self._crop_bbox(frame, alert.bbox)
        crop_path = date_folder / f"{stem}_crop.jpg"
        cv2.imwrite(str(crop_path), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])

        # 3. Metadata JSON
        meta = {
            "alert_id":       alert.alert_id,
            "timestamp":      now.isoformat(),
            "fluid_class":    fluid_class,
            "detected_fluid_class": detected_fluid_class,
            "bbox":           list(alert.bbox),
            "avg_confidence": round(alert.avg_confidence, 4),
            "avg_residual":   round(alert.avg_residual, 4),
            "probability":    round(probability, 3) if probability is not None else None,
            "uncertainty":    round(uncertainty, 5) if uncertainty is not None else None,
            "detection_count": alert.detection_count,
            "frame_shape":    list(frame.shape),
            "files": {
                "annotated": str(ann_path.relative_to(self.base_dir)),
                "crop":      str(crop_path.relative_to(self.base_dir)),
            },
        }
        meta_path = date_folder / f"{stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Append to running log
        self._log.append(meta)
        self._log_path.write_text(json.dumps(self._log, indent=2))

        print(f"[Saver] Saved → {ann_path.name}  |  crop → {crop_path.name}")
        return meta

    def get_log(self) -> list:
        return list(reversed(self._log))   # newest first

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_save_overlay(self, frame: np.ndarray, alert: "ConfirmedAlert") -> np.ndarray:
        """Draw a prominent confirmed-alert box + metadata banner onto the frame."""
        x1, y1, x2, y2 = [int(v) for v in alert.bbox]
        h, w = frame.shape[:2]

        # Red bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)

        # Corner tick marks for clarity
        tick = 12
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*tick, cy), (0, 0, 220), 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy*tick), (0, 0, 220), 3)

        # Top info banner (semi-transparent)
        banner_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, f"CONFIRMED SPILL  |  {alert.alert_id}",
                    (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 220), 1)
        cv2.putText(frame, f"Conf: {alert.avg_confidence:.3f}   Residual: {alert.avg_residual:.4f}   {now_str}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 255), 1)

        return frame

    def _crop_bbox(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Return padded crop of the spill region."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        p = self.CROP_PADDING
        cx1 = max(0, int(x1) - p)
        cy1 = max(0, int(y1) - p)
        cx2 = min(w, int(x2) + p)
        cy2 = min(h, int(y2) + p)
        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size > 0 else frame

    def _load_log(self) -> list:
        if self._log_path.exists():
            try:
                return json.loads(self._log_path.read_text())
            except Exception:
                pass
        return []


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class SpillDetectionPipeline:
    """
    Full hybrid spill detection pipeline.
    Thread-safe: frame is set externally, processed frame is read externally.
    """

    CONF_THRESHOLD: float = 0.25
    CLASS_INDEX: int = 0          # spill class index in YOLO model

    def __init__(
        self,
        yolo_model_path: str,
        pinn_weights_path: Optional[str] = None,
        device: str = "cpu",
        fluid_class: str = "auto",
        threshold_calibrator_path: Optional[str] = None,
        pde_mode: str = "thin_film",
        floor_slope: tuple = (0.0, 0.0),
        adaptive: bool = True,
        weights_save_path: Optional[str] = None,
        fluid_classifier_method: str = "random_forest",
        fluid_classifier_path: Optional[str] = None,
    ):
        self.fluid_class = fluid_class
        self.device = device
        self._lock = threading.Lock()
        self._running = False

        # Wind is environmental/live, unlike floor_slope (fixed per camera) —
        # set via set_environmental_conditions() from a weather feed or
        # operator input, defaults to calm/no-wind. wind_speed/direction feed
        # PINNValidator's advection_diffusion mode (needs a vector); the full
        # EnvironmentalConditions below feeds FluidClassifier's tabular
        # features (wind speed as scalar + current + temperature) — same
        # underlying readings, two different consumers with different needs.
        self.wind_speed: float = 0.0
        self.wind_direction_deg: float = 0.0
        self.environmental_conditions = EnvironmentalConditions()

        # Load YOLO
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_model_path)
            print(f"[Pipeline] YOLO loaded from {yolo_model_path}")
        except Exception as e:
            self.yolo = None
            print(f"[Pipeline] YOLO load failed: {e}")

        # Initialise sub-modules
        self.geo_validator = GeometricValidator()
        self.fluid_classifier = FluidClassifier(
            method=fluid_classifier_method,
            model_path=fluid_classifier_path,
            device=device,
        )
        # Adapted weights default to a SEPARATE sibling file
        # (pinn_thinfilm.pt -> pinn_thinfilm_adapted.pt), never silently
        # overwriting the checkpoint you pointed pinn_weights_path at.
        # Only when no pretrained weights were given at all does it fall
        # back to a plain "pinn_adapted.pt", since there's no original to
        # protect in that case.
        if weights_save_path:
            resolved_save_path = weights_save_path
        elif pinn_weights_path:
            p = Path(pinn_weights_path)
            resolved_save_path = str(p.with_name(p.stem + "_adapted" + p.suffix))
        else:
            resolved_save_path = "pinn_adapted.pt"

        if pinn_weights_path and resolved_save_path == pinn_weights_path:
            print(f"[Pipeline] WARNING: weights_save_path is the SAME file as "
                  f"pinn_weights_path ({pinn_weights_path}) — adaptation will "
                  f"overwrite your original trained checkpoint in place. Pass a "
                  f"different weights_save_path if you want to keep it untouched.")

        self.pinn_validator = PINNValidator(
            pinn_weights_path,
            device=device,
            threshold_calibrator_path=threshold_calibrator_path,
            pde_mode=pde_mode,
            floor_slope=floor_slope,
            adaptive=adaptive,
            weights_save_path=resolved_save_path,
        )
        self.temporal = TemporalConfirmation()
        self.stats = PipelineStats()
        self.stats.decision_method = (
            "learned_threshold" if self.pinn_validator.calibrator.is_fitted
            else "adaptive_threshold" if adaptive
            else "fixed_threshold"
        )
        self.stats.pde_mode = pde_mode
        self.saver = SpillImageSaver(base_dir="spill_captures")

        # Fluid classes the PINN can distinguish physically (the YOLO model
        # itself is single-class: spill vs not-spill — see fluid_class="auto"
        # in PINNValidator.validate for why this matters).
        self.FLUID_CLASSES = list(self.pinn_validator.VISCOSITY_MAP.keys())

        # Shared state for streaming
        self._raw_frame: Optional[np.ndarray] = None
        self._processed_frame: Optional[np.ndarray] = None
        self._latest_detections: list = []
        self._pending_residuals: list = []
        self._pending_fluid_predictions: list = []

    def set_environmental_conditions(
        self,
        wind_speed: float = None,
        wind_direction_deg: float = None,
        current_velocity_mps: float = None,
        temperature_c: float = None,
    ):
        """
        Update live environmental conditions. wind_speed/wind_direction_deg
        feed PINNValidator's advection_diffusion mode (needs the vector);
        all four fields feed FluidClassifier's tabular environmental
        features. Call this from a weather-feed poller or an operator
        control; safe to call at any rate since it only updates plain
        floats read by the next process_frame() call. Any field left as
        None keeps its current value — this is a partial update, not a
        full replace.
        """
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if wind_direction_deg is not None:
            self.wind_direction_deg = wind_direction_deg

        ec = self.environmental_conditions
        self.environmental_conditions = EnvironmentalConditions(
            wind_speed_mps=wind_speed if wind_speed is not None else ec.wind_speed_mps,
            current_velocity_mps=(
                current_velocity_mps if current_velocity_mps is not None
                else ec.current_velocity_mps
            ),
            temperature_c=temperature_c if temperature_c is not None else ec.temperature_c,
        )

    # ── Frame I/O ─────────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray):
        with self._lock:
            self._raw_frame = frame.copy()

    def get_processed_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._processed_frame

    def get_state(self) -> dict:
        """Returns JSON-serialisable state for the dashboard."""
        buf = self.temporal.get_buffer_fill()
        alerts = self.temporal.get_all_alerts()
        alert_list = [
            {
                "id": a.alert_id,
                "time": time.strftime("%H:%M:%S", time.localtime(a.confirmed_at)),
                "confidence": round(a.avg_confidence, 3),
                "residual": round(a.avg_residual, 4),
                "uncertainty": round(a.avg_uncertainty, 5),
                "fluid_class": a.fluid_class,
                "detections": a.detection_count,
                "bbox": list(a.bbox),
            }
            for a in alerts[:20]   # cap at 20 for dashboard
        ]
        return {
            "stats": self.stats.to_dict(),
            "buffer": buf,
            "alerts": alert_list,
            "pending_residuals": list(self._pending_residuals)[-10:],
            "pending_fluid_predictions": list(self._pending_fluid_predictions)[-10:],
            "current_threshold": (
                self.pinn_validator.DECISION_PROB_THRESHOLD
                if self.pinn_validator.calibrator.is_fitted
                else self.pinn_validator.adaptive_thresh.epsilon
                if self.pinn_validator.adaptive
                else self.pinn_validator.RESIDUAL_THRESHOLD
            ),
        }

    # ── Core Processing ───────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
        """
        Run full pipeline on a single frame.
        Returns annotated BGR frame.
        """
        self.stats.tick()
        output = frame.copy()

        if self.yolo is None:
            cv2.putText(output, "YOLO MODEL NOT LOADED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return output

        # ── Stage 1: YOLO Detection ──────────────────────────────────────────
        results = self.yolo(frame, conf=self.CONF_THRESHOLD, verbose=False)
        raw_dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != self.CLASS_INDEX:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_dets.append(((x1, y1, x2, y2), conf))

        self.stats.total_detections += len(raw_dets)

        for (bbox, conf) in raw_dets:
            det = Detection(bbox=bbox, confidence=conf, frame_id=self.stats.total_frames)

            # ── Stage 2: Geometric Validation ───────────────────────────────
            geo_ok, geo_reason = self.geo_validator.validate(det)
            if not geo_ok:
                self.stats.geo_rejections += 1
                draw_detection_overlay(output, bbox, conf, 0.0, "REJECTED",
                                       alert_id=f"GEO:{geo_reason[:12]}")
                continue

            # ── Stage 2.5: Fluid Identification ─────────────────────────────
            # Separate model from YOLO/PINN — classifies *what* fluid this
            # detection is from visual/geometric/environmental features,
            # feeding the PINN's viscosity lookup as an explicit fluid_class
            # rather than leaving it on "auto" (PINNValidator's own
            # residual-comparison auto-detect, which we've confirmed doesn't
            # apply to a single-learned-viscosity checkpoint — see
            # PINNValidator.validate). Falls back to the pipeline's
            # fluid_class default whenever the classifier isn't confident
            # (or isn't trained yet), so an untrained classifier just
            # reverts to prior behaviour instead of forcing a bad guess into
            # the physics check.
            fluid_pred = self.fluid_classifier.predict(
                frame, bbox, env=self.environmental_conditions
            )
            detection_fluid_class = (
                fluid_pred.label if fluid_pred.label != "unknown" else self.fluid_class
            )
            self._pending_fluid_predictions.append({
                "label": fluid_pred.label,
                "confidence": round(fluid_pred.confidence, 3),
                "used": detection_fluid_class,
            })
            if len(self._pending_fluid_predictions) > 60:
                self._pending_fluid_predictions.pop(0)

            # ── Stage 3: PINN Residual Validation ───────────────────────────
            pinn_result = self.pinn_validator.validate(
                bbox, frame_time, detection_fluid_class,
                wind_speed=self.wind_speed,
                wind_direction_deg=self.wind_direction_deg,
            )
            self._pending_residuals.append({
                "residual": round(pinn_result.residual, 4),
                "uncertainty": round(pinn_result.uncertainty, 5),
                "probability": (
                    round(pinn_result.probability, 3)
                    if pinn_result.probability is not None else None
                ),
                "detected_fluid_class": pinn_result.detected_fluid_class,
                "candidate_residuals": pinn_result.candidate_residuals,
            })
            if len(self._pending_residuals) > 60:
                self._pending_residuals.pop(0)
            self.stats.residual_history.append(pinn_result.residual)
            self.stats.uncertainty_history.append(pinn_result.uncertainty)
            self.stats.decision_method = pinn_result.decision_method

            if not pinn_result.passed:
                self.stats.pinn_rejections += 1
                draw_detection_overlay(
                    output, bbox, conf, pinn_result.residual, "REJECTED",
                    alert_id=f"PINN:{pinn_result.residual:.3f}",
                    probability=pinn_result.probability,
                    uncertainty=pinn_result.uncertainty,
                    fluid_class=pinn_result.detected_fluid_class,
                )
                continue

            # ── Stage 4: Temporal Confirmation ──────────────────────────────
            alert: Optional[ConfirmedAlert] = self.temporal.update(
                frame_time, bbox, conf, pinn_result.residual,
                fluid_class=pinn_result.detected_fluid_class or "unknown",
                uncertainty=pinn_result.uncertainty,
            )

            status = "CONFIRMED" if alert else "PENDING"
            alert_id = alert.alert_id if alert else ""
            if alert:
                self.stats.confirmed_alerts += 1
                # ── Save annotated image to disk ─────────────────────────────
                self.saver.save(
                    frame.copy(), alert, fluid_class=self.fluid_class,
                    probability=pinn_result.probability,
                    uncertainty=pinn_result.uncertainty,
                    detected_fluid_class=pinn_result.detected_fluid_class,
                )
                # ── Feed the online adaptation system (no-op if disabled) ────
                self.pinn_validator.record_confirmation(
                    bbox, frame_time, pinn_result.detected_fluid_class,
                    pinn_result.residual, pinn_result.uncertainty,
                    wind_speed=self.wind_speed,
                    wind_direction_deg=self.wind_direction_deg,
                )

            draw_detection_overlay(
                output, bbox, conf, pinn_result.residual, status, alert_id,
                probability=pinn_result.probability,
                uncertainty=pinn_result.uncertainty,
                fluid_class=pinn_result.detected_fluid_class,
            )

        # HUD overlay
        self._draw_hud(output)
        return output

    def analyze_image(self, image_bytes: bytes, fluid_class: str = "auto") -> dict:
        """
        Run Stages 1–3 (YOLO → Geometric → PINN) on a single uploaded
        still image and return the annotated result. Used by an
        "Upload Image" feature for one-off testing, distinct from the
        live camera/RTSP/video stream.

        Deliberately does NOT touch self.stats, self.temporal, or
        self._pending_residuals — those track the live monitoring
        session, and an ad-hoc test image shouldn't get folded into
        that history (a single frame can't satisfy the temporal
        confirmation window anyway, N_MIN detections across T_w
        seconds, so Stage 4 is skipped here by design, not by mistake).
        Uses a fresh GeometricValidator so the single image isn't
        judged against unrelated frame-to-frame history either.

        Returns a dict:
            {"image_base64": <annotated JPEG, base64>,
             "detections": [ {bbox, confidence, status, residual,
                               uncertainty, probability,
                               detected_fluid_class, reason}, ... ],
             "pde_mode": "thin_film" | "advection_diffusion"}
        or {"error": "..."} if the image can't be decoded or YOLO
        isn't loaded.
        """
        if self.yolo is None:
            return {"error": "YOLO model not loaded"}

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode image"}

        output = frame.copy()
        frame_time = time.time()
        local_geo_validator = GeometricValidator()
        detections_out = []

        results = self.yolo(frame, conf=self.CONF_THRESHOLD, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != self.CLASS_INDEX:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1, y1, x2, y2)
                det = Detection(bbox=bbox, confidence=conf, frame_id=0)

                geo_ok, geo_reason = local_geo_validator.validate(det)
                if not geo_ok:
                    draw_detection_overlay(output, bbox, conf, 0.0, "REJECTED",
                                            alert_id=f"GEO:{geo_reason[:12]}")
                    detections_out.append({
                        "bbox": list(bbox), "confidence": round(conf, 4),
                        "status": "REJECTED", "reason": geo_reason,
                    })
                    continue

                # Only run the fluid classifier when the caller left this as
                # "auto" — an explicit fluid_class request (e.g. testing a
                # known-substance image) is a deliberate override, same as
                # in process_frame.
                if fluid_class == "auto":
                    fluid_pred = self.fluid_classifier.predict(
                        frame, bbox, env=self.environmental_conditions
                    )
                    detection_fluid_class = (
                        fluid_pred.label if fluid_pred.label != "unknown" else "auto"
                    )
                else:
                    fluid_pred = None
                    detection_fluid_class = fluid_class

                pinn_result = self.pinn_validator.validate(
                    bbox, frame_time, detection_fluid_class,
                    wind_speed=self.wind_speed,
                    wind_direction_deg=self.wind_direction_deg,
                )
                status = "PASSED" if pinn_result.passed else "REJECTED"
                draw_detection_overlay(
                    output, bbox, conf, pinn_result.residual, status,
                    probability=pinn_result.probability,
                    uncertainty=pinn_result.uncertainty,
                    fluid_class=pinn_result.detected_fluid_class,
                )
                detections_out.append({
                    "bbox": list(bbox),
                    "confidence": round(conf, 4),
                    "status": status,
                    "residual": round(pinn_result.residual, 4),
                    "uncertainty": round(pinn_result.uncertainty, 5),
                    "probability": (
                        round(pinn_result.probability, 3)
                        if pinn_result.probability is not None else None
                    ),
                    "fluid_classifier_prediction": (
                        {"label": fluid_pred.label, "confidence": round(fluid_pred.confidence, 3)}
                        if fluid_pred is not None else None
                    ),
                    "detected_fluid_class": pinn_result.detected_fluid_class,
                    "reason": pinn_result.reason,
                })

        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None

        return {
            "image_base64": image_base64,
            "detections": detections_out,
            "pde_mode": self.pinn_validator.pde_mode,
        }

    def _draw_hud(self, frame: np.ndarray):
        """Burn pipeline status into frame."""
        h, w = frame.shape[:2]
        buf = self.temporal.get_buffer_fill()
        lines = [
            f"FPS: {self.stats.fps:.1f}",
            f"Frames: {self.stats.total_frames}",
            f"Alerts: {self.stats.confirmed_alerts}",
            f"Buf: {buf['count']}/{buf['required']}",
        ]
        if self.pinn_validator.pde_mode == "advection_diffusion":
            lines.append(f"Wind: {self.wind_speed:.1f} @ {self.wind_direction_deg:.0f}°")
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (w - 160, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)

    # ── Background Worker ─────────────────────────────────────────────────────

    def start(self):
        self._running = True
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        self.pinn_validator.stop()

    def _worker(self):
        while self._running:
            with self._lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.01)
                continue
            t = time.time()
            processed = self.process_frame(frame, t)
            with self._lock:
                self._processed_frame = processed
                self._raw_frame = None


# ── Video Source Manager ──────────────────────────────────────────────────────

class VideoSource:
    """Wraps webcam, RTSP stream, or video file into a unified interface."""

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._source = None

    def open(self, source) -> bool:
        """
        source: 0 (webcam), "rtsp://...", or "/path/to/file.mp4"
        """
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(source)
        self._source = source
        return self._cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            # Loop video files
            if isinstance(self._source, str) and self._source.endswith(
                ('.mp4', '.avi', '.mov', '.mkv')
            ):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()