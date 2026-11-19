from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Tuple

import cv2

from deception_detection import (
    MAX_FRAMES,
    finish_calibration,
    get_confidence_score,
    new_tell,
    process_frame,
    reset_calibration,
)
from detection_lock import LOCK
from landmark_detector import FaceHandDetector
from voice_analysis import VOICE_TELL_TEXT, analyze_video_audio_pitch

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}

TellEvent = Tuple[float, str, str, int]  # (video_time_seconds, tell_key, tell_text, confidence)


def is_allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def analyze_video(file_path: str, fps_sample_limit: float = 30.0):
   
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {file_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, round(native_fps / fps_sample_limit))
    effective_fps = native_fps / frame_skip
    calibration_seconds = MAX_FRAMES / effective_fps

    # Voice-stress analysis is independent of the (lock-protected) visual
    # detection state, so it deliberately runs *outside* `with LOCK:`
    # below - no need to hold that lock while moviepy decodes the audio
    # track, which can take a little while on longer clips.
    voice_rising_times = analyze_video_audio_pitch(file_path, calibration_seconds)
    voice_idx = 0

    events: List[TellEvent] = []
    confidence_samples: List[int] = []
    seen_keys: set = set()

    with LOCK:
        reset_calibration()
        detector = FaceHandDetector()
        try:
            calibrated = False
            calibration_frames = 0
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1
                if frame_index % frame_skip != 0:
                    continue

                face_landmarks, hands_landmarks = detector.process(frame)
                tells = process_frame(
                    frame, face_landmarks, hands_landmarks, calibrated, fps=effective_fps,
                )
                video_time = frame_index / native_fps

                if not calibrated:
                    calibration_frames += 1
                    if calibration_frames >= MAX_FRAMES:
                        finish_calibration()
                        calibrated = True
                    continue  # don't log events collected during calibration

                # Merge in any voice-stress rising edge that happened at or
                # before this frame's timestamp, so it's combined into the
                # same confidence-score calculation as the visual tells
                # (get_confidence_score() sums whichever tell keys are
                # present in `tells`, "voice" included).
                if voice_idx < len(voice_rising_times) and voice_rising_times[voice_idx] <= video_time:
                    tells = dict(tells)
                    tells["voice"] = new_tell(VOICE_TELL_TEXT, 30)
                    voice_idx += 1

                confidence = get_confidence_score(tells)
                confidence_samples.append(confidence)

                current_keys = {k for k in tells if k != "avg_bpms"}
                newly_active = current_keys - seen_keys
                for key in newly_active:
                    events.append((video_time, key, tells[key]["text"], confidence))
                seen_keys = current_keys
        finally:
            detector.close()
            cap.release()

    avg_confidence = sum(confidence_samples) / len(confidence_samples) if confidence_samples else 0
    max_confidence = max(confidence_samples) if confidence_samples else 0
    return events, avg_confidence, max_confidence


def events_to_absolute_times(events: List[TellEvent], session_start: datetime):
    
    return [
        (session_start + timedelta(seconds=video_time), tell_key, tell_text, confidence)
        for video_time, tell_key, tell_text, confidence in events
    ]
