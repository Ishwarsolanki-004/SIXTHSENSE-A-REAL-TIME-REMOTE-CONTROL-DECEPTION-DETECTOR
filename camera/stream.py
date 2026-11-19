from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

import config
from deception_detection import MAX_FRAMES, get_confidence_score, new_tell
from detection_pipeline import DetectionWorker, FrameGrabber
from landmark_detector import FaceHandDetector, draw_landmarks_and_hands
from voice_analysis import AudioPitchMonitor

_lock = threading.Lock()
_cap: Optional[cv2.VideoCapture] = None
_detector: Optional[FaceHandDetector] = None
_grabber: Optional[FrameGrabber] = None
_worker: Optional[DetectionWorker] = None
_pitch_monitor: Optional[AudioPitchMonitor] = None
_prev_calibrated = False


def _ensure_started() -> None:
    """Starts the webcam + detection + voice pipeline on first use.
    Subsequent calls are no-ops while the pipeline is already running."""
    global _cap, _detector, _grabber, _worker, _pitch_monitor

    with _lock:
        if _worker is not None:
            return

        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX}).")

        detector = FaceHandDetector()

        grabber = FrameGrabber(cap)
        grabber.start()
        for _ in range(100):  # wait briefly for the first real frame
            ret, _frame = grabber.read()
            if ret:
                break
            time.sleep(0.01)

        worker = DetectionWorker(grabber, detector, get_fps=lambda: cap.get(cv2.CAP_PROP_FPS) or 30)
        worker.start()

        pitch_monitor = AudioPitchMonitor()
        pitch_monitor.start()

        _cap, _detector, _grabber, _worker, _pitch_monitor = cap, detector, grabber, worker, pitch_monitor


def stop() -> None:
    """Shuts the whole pipeline down (used on app exit)."""
    global _cap, _detector, _grabber, _worker, _pitch_monitor
    with _lock:
        if _worker is not None:
            _worker.stop()
        if _grabber is not None:
            _grabber.stop()
        if _pitch_monitor is not None:
            _pitch_monitor.stop()
        if _cap is not None:
            _cap.release()
        if _detector is not None:
            _detector.close()
        _cap = _detector = _grabber = _worker = _pitch_monitor = None


def reset_calibration() -> None:
    """Equivalent of the desktop app's "Recalibrate" button - called when
    a new recording session starts, so thresholds are re-learned for
    whoever is now in front of the camera."""
    global _prev_calibrated
    _ensure_started()
    _worker.reset_calibration()
    _pitch_monitor.reset_calibration()
    _prev_calibrated = False


def get_state() -> dict:
    """Returns the current detection state as a JSON-serializable dict -
    mirrors what the browser's live view renders, just as data instead
    of pixels."""
    global _prev_calibrated
    _ensure_started()

    face_landmarks, hands_landmarks, tells, calibrated, calibration_frames, confidence = _worker.get_latest()

    if calibrated and not _prev_calibrated:
        _pitch_monitor.finish_calibration()
    _prev_calibrated = calibrated

    if calibrated:
        voice_tell = _pitch_monitor.get_tell()
        if voice_tell:
            tells = dict(tells)
            tells["voice"] = new_tell(voice_tell, 30)
            confidence = get_confidence_score(tells)

    tells_list = [
        {"key": key, "text": tell["text"], "ttl": tell["ttl"]}
        for key, tell in tells.items()
        if key != "avg_bpms"  # BPM has its own dedicated readout, not a "tell"
    ]
    bpm_text = tells.get("avg_bpms", {}).get("text", "BPM: ...")

    return {
        "calibrated": calibrated,
        "calibration_frames": calibration_frames,
        "calibration_total": MAX_FRAMES,
        "confidence": confidence,
        "tells": tells_list,
        "bpm": bpm_text,
        "mic_available": _pitch_monitor.available,
        "mic_pitch": _pitch_monitor.get_current_pitch(),
        "mic_baseline": _pitch_monitor.baseline_pitch,
    }


def generate_mjpeg(draw_landmarks: bool = True, target_fps: int = 15):
    """Generator yielding multipart/x-mixed-replace JPEG frames for
    Flask's Response(). Deliberately capped at `target_fps` (independent
    of the detector's own throughput) - the browser doesn't benefit from
    more than that for a live preview, and capping it keeps bandwidth and
    JPEG-encoding CPU cost down on low-end hardware."""
    _ensure_started()
    frame_interval = 1.0 / target_fps

    while True:
        loop_start = time.time()
        ret, frame = _grabber.read()
        if ret and frame is not None:
            face_landmarks, hands_landmarks, _tells, _cal, _cf, _conf = _worker.get_latest()
            if draw_landmarks:
                draw_landmarks_and_hands(frame, face_landmarks, hands_landmarks)
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
