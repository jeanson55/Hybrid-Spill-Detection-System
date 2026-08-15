"""
Flask Backend — Spill Detection Dashboard
Serves live MJPEG video stream + SSE state updates + REST API
"""

import os
import io
import time
import json
import tempfile
import threading
import cv2
import numpy as np
from pathlib import Path
from flask import (Flask, render_template, Response, request,
                   jsonify, stream_with_context)

from modules.pipeline import SpillDetectionPipeline, VideoSource

# ── Configuration ─────────────────────────────────────────────────────────────

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "spill_best.pt")
PINN_WEIGHTS_PATH = os.environ.get("PINN_WEIGHTS_PATH", "pinn_thinfilm.pt")
DEVICE = os.environ.get("DEVICE", "cpu")   # set "cuda" if GPU available
FLUID_CLASSIFIER_METHOD = os.environ.get("FLUID_CLASSIFIER_METHOD", "random_forest")
FLUID_CLASSIFIER_PATH = os.environ.get("FLUID_CLASSIFIER_PATH", "fluid_rf.joblib")
# Optional — falls back to the fixed ε=0.05 threshold if this file doesn't
# exist yet (see calibrate_threshold.py / label_alerts.py to build one).
THRESHOLD_CALIBRATOR_PATH = os.environ.get("THRESHOLD_CALIBRATOR_PATH", "threshold_calibrator.pt")

app = Flask(__name__)

# ── Global State ──────────────────────────────────────────────────────────────

pipeline = SpillDetectionPipeline(
    yolo_model_path=YOLO_MODEL_PATH,
    pinn_weights_path=PINN_WEIGHTS_PATH,
    device=DEVICE,
    fluid_classifier_method=FLUID_CLASSIFIER_METHOD,
    fluid_classifier_path=FLUID_CLASSIFIER_PATH,
    threshold_calibrator_path=THRESHOLD_CALIBRATOR_PATH,
)
pipeline.start()

video_source = VideoSource()
_capture_thread: threading.Thread = None
_capture_running = False


def _capture_loop():
    """Continuously reads frames from the video source and feeds the pipeline."""
    global _capture_running
    while _capture_running:
        frame = video_source.read()
        if frame is None:
            time.sleep(0.05)
            continue
        pipeline.set_frame(frame)
        time.sleep(0.01)   # ~100 fps cap; pipeline itself throttles via processing time


# ── Video Streaming ───────────────────────────────────────────────────────────

def _generate_mjpeg():
    """Generator yielding MJPEG frames for the live feed endpoint."""
    while True:
        frame = pipeline.get_processed_frame()
        if frame is None:
            # Send a blank frame while waiting
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video source...", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            frame = blank

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )
        time.sleep(0.033)   # ~30 fps output cap


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    """Snapshot of pipeline state for polling."""
    return jsonify(pipeline.get_state())


@app.route("/api/events")
def api_events():
    """Server-Sent Events stream for real-time dashboard updates."""
    def _sse():
        while True:
            data = json.dumps(pipeline.get_state())
            yield f"data: {data}\n\n"
            time.sleep(0.5)

    return Response(
        stream_with_context(_sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/source", methods=["POST"])
def api_set_source():
    """
    Switch video source.
    Body JSON: {"type": "webcam"} | {"type": "rtsp", "url": "rtsp://..."} |
               {"type": "file", "path": "/path/to/video.mp4"}
    """
    global _capture_thread, _capture_running

    data = request.get_json(force=True)
    src_type = data.get("type", "webcam")

    if src_type == "webcam":
        source = 0
    elif src_type == "rtsp":
        source = data.get("url", "")
    elif src_type == "file":
        source = data.get("path", "")
    else:
        return jsonify({"error": "Unknown source type"}), 400

    # Stop existing capture
    _capture_running = False
    if _capture_thread and _capture_thread.is_alive():
        _capture_thread.join(timeout=2)
    video_source.release()
    pipeline.geo_validator.reset()
    pipeline.temporal.reset_cooldown()

    # Open new source
    ok = video_source.open(source)
    if not ok:
        return jsonify({"error": f"Could not open source: {source}"}), 400

    _capture_running = True
    _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    _capture_thread.start()

    return jsonify({"status": "ok", "source": str(source)})


@app.route("/api/model_info")
def api_model_info():
    """Return the fluid classes the loaded model can detect."""
    return jsonify({
        "fluid_classes": pipeline.FLUID_CLASSES,
        "multi_class": len(pipeline.FLUID_CLASSES) > 1,
    })


@app.route("/api/reset_alerts", methods=["POST"])
def api_reset_alerts():
    """Clear confirmed alert log and reset cooldown."""
    pipeline.temporal._confirmed_alerts.clear()
    pipeline.temporal.reset_cooldown()
    pipeline.stats.confirmed_alerts = 0
    return jsonify({"status": "ok"})


@app.route("/api/captures")
def api_captures():
    """Return the list of saved spill capture metadata (newest first)."""
    return jsonify(pipeline.saver.get_log())


@app.route("/api/adaptation")
def api_adaptation():
    """Full adaptation state snapshot for polling."""
    return jsonify(pipeline.pinn_validator.get_adaptation_state())


@app.route("/api/adaptation/reset", methods=["POST"])
def api_adaptation_reset():
    """
    Reset adaptive thresholds back to ε_global = 0.05 and clear replay buffer.
    Weights are NOT reset — only the threshold history and replay buffer.
    """
    from modules.pinn_validator import AdaptiveThreshold
    pipeline.pinn_validator.adaptive_thresh = AdaptiveThreshold()
    pipeline.pinn_validator.replay_buffer._buf.clear()
    pipeline.pinn_validator.adapt_state = \
        pipeline.pinn_validator.adapt_state.__class__()
    return jsonify({"status": "ok", "message": "Thresholds and replay buffer reset."})


@app.route("/api/adaptation/save", methods=["POST"])
def api_adaptation_save():
    """Manually trigger a weight save to disk."""
    pipeline.pinn_validator._save_weights()
    return jsonify({"status": "ok", "path": pipeline.pinn_validator.weights_save_path})


@app.route("/api/environment", methods=["POST"])
def api_set_environment():
    """
    Update live environmental readings (wind, current, temperature).
    Body JSON (all fields optional, partial updates allowed):
        {"wind_speed": 4.2, "wind_direction_deg": 90,
         "current_velocity_mps": 0.3, "temperature_c": 22.0}
    Feeds PINNValidator's advection_diffusion mode (wind_speed/direction)
    and FluidClassifier's environmental features (all four).
    """
    data = request.get_json(force=True) or {}
    pipeline.set_environmental_conditions(
        wind_speed=data.get("wind_speed"),
        wind_direction_deg=data.get("wind_direction_deg"),
        current_velocity_mps=data.get("current_velocity_mps"),
        temperature_c=data.get("temperature_c"),
    )
    return jsonify({
        "status": "ok",
        "wind_speed": pipeline.wind_speed,
        "wind_direction_deg": pipeline.wind_direction_deg,
        "current_velocity_mps": pipeline.environmental_conditions.current_velocity_mps,
        "temperature_c": pipeline.environmental_conditions.temperature_c,
    })


@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    """
    Analyze a single uploaded still image (YOLO → Geometric → PINN),
    separate from the live video/RTSP/webcam stream. Does not touch
    live pipeline stats or temporal confirmation state.
    Body: multipart form, field "file" = image, optional "fluid_class"
    (defaults to "auto" — see PINNValidator.validate for why auto-detect
    is the right default rather than pinning every image to one fluid).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    fluid_class = request.form.get("fluid_class", "auto")
    image_bytes = f.read()

    result = pipeline.analyze_image(image_bytes, fluid_class=fluid_class)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    """Accept a video file upload and switch to it as source."""
    global _capture_thread, _capture_running

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    tmp_dir = Path(tempfile.gettempdir()); save_path = tmp_dir / f.filename
    f.save(str(save_path))

    # Stop existing capture
    _capture_running = False
    if _capture_thread and _capture_thread.is_alive():
        _capture_thread.join(timeout=2)
    video_source.release()

    ok = video_source.open(str(save_path))
    if not ok:
        return jsonify({"error": "Could not open uploaded video"}), 400

    _capture_running = True
    _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    _capture_thread.start()

    return jsonify({"status": "ok", "filename": f.filename})


# ── Entry Point ───────────────────────────────────────────────────────────────

import atexit

def _on_exit():
    print("\n[Shutdown] Saving adapted PINN weights before exit...")
    pipeline.pinn_validator.stop()
    print("[Shutdown] Done.")

atexit.register(_on_exit)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Spill Detection Dashboard  (Adaptive PINN)")
    print("  Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)