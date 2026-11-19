from __future__ import annotations

import threading
import time

import cv2

from deception_detection import MAX_FRAMES, finish_calibration, get_confidence_score, process_frame
from detection_lock import LOCK
from landmark_detector import FaceHandDetector


class FrameGrabber(threading.Thread):
    """Continuously reads frames from a cv2.VideoCapture in the background
    so the main/render thread is never blocked on camera I/O."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = True

    def run(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                break

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self) -> None:
        self._running = False


class DetectionWorker(threading.Thread):
    """Runs face/hand detection + tells processing on whatever the latest
    available frame is, on its own thread. On slower hardware detection may
    lag a few frames behind the live camera feed - the camera preview
    itself (FrameGrabber + whatever renders it) stays smooth regardless,
    which is the trade-off that actually matters for perceived
    responsiveness on a low-end CPU."""

    def __init__(self, frame_grabber: FrameGrabber, detector: FaceHandDetector, get_fps):
        super().__init__(daemon=True)
        self.frame_grabber = frame_grabber
        self.detector = detector
        self.get_fps = get_fps
        self._lock = threading.Lock()
        self._running = True

        self.calibrated = False
        self.calibration_frames = 0
        self.face_landmarks = None
        self.hands_landmarks = None
        self.tells: dict = {}
        self.confidence = 0

        # Measured (not nominal-camera) processing rate - see run() below.
        # Seeded with the nominal camera FPS as a reasonable first guess
        # before real measurements have accumulated.
        self._measured_fps = get_fps()
        self._last_process_time = None

    def run(self) -> None:
        while self._running:
            ret, frame = self.frame_grabber.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Detection runs on its own thread, decoupled from the
            # camera's capture rate - on a slow CPU this thread's real
            # throughput can be well below the camera's nominal FPS (e.g.
            # a 30 FPS camera feeding a detector that only manages 10-15
            # FPS on an Intel i3). BPM estimation depends on knowing the
            # true time between samples, so we measure our own actual
            # processing rate here (smoothed with an EMA to reduce frame-
            # to-frame jitter) instead of trusting cap.get(CAP_PROP_FPS),
            # which reflects the camera, not this thread.
            now = time.monotonic()
            if self._last_process_time is not None:
                dt = now - self._last_process_time
                if dt > 0:
                    instantaneous_fps = 1.0 / dt
                    self._measured_fps = 0.8 * self._measured_fps + 0.2 * instantaneous_fps
            self._last_process_time = now

            face_landmarks, hands_landmarks = self.detector.process(frame)
            with self._lock:
                calibrated = self.calibrated
            # LOCK (detection_lock.py) serializes access to
            # deception_detection.py's module-level rolling state against
            # any concurrent video-file analysis (video_upload.py) - see
            # that module's docstring for why this is needed and its limits.
            with LOCK:
                tells = process_frame(frame, face_landmarks, hands_landmarks, calibrated, fps=self._measured_fps)

            with self._lock:
                if not self.calibrated:
                    self.calibration_frames += 1
                    if self.calibration_frames >= MAX_FRAMES:
                        finish_calibration()
                        self.calibrated = True
                self.face_landmarks = face_landmarks
                self.hands_landmarks = hands_landmarks
                self.tells = tells
                self.confidence = get_confidence_score(tells)

    def get_latest(self):
        with self._lock:
            return (
                self.face_landmarks, self.hands_landmarks, dict(self.tells),
                self.calibrated, self.calibration_frames, self.confidence,
            )

    def reset_calibration(self) -> None:
        from deception_detection import reset_calibration as _reset_calibration
        with self._lock:
            self.calibrated = False
            self.calibration_frames = 0
        with LOCK:
            _reset_calibration()

    def stop(self) -> None:
        self._running = False
