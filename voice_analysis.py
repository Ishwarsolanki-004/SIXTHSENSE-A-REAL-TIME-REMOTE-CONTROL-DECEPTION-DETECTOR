from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except Exception:  # pragma: no cover - environment without a working audio backend
    _SOUNDDEVICE_AVAILABLE = False

VOICE_TELL_TEXT = "Voice pitch rising (possible stress)"


def estimate_pitch(signal: np.ndarray, sample_rate: int, fmin: float = 75.0, fmax: float = 400.0) -> float:
    """Estimates the fundamental frequency (F0) of a mono audio block using
    autocorrelation. Returns 0.0 for silence / unvoiced audio.

    `fmin`/`fmax` (75-400 Hz) cover typical human speech F0 range.
    """
    signal = signal.astype(np.float64) - np.mean(signal)
    if np.max(np.abs(signal)) < 1e-4:
        return 0.0  # effectively silent

    corr = np.correlate(signal, signal, mode="full")
    corr = corr[len(corr) // 2:]

    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return 0.0

    segment = corr[min_lag:max_lag]
    if segment.size == 0:
        return 0.0

    peak_idx = int(np.argmax(segment)) + min_lag
    if corr[peak_idx] <= 0:
        return 0.0

    return sample_rate / peak_idx


class AudioPitchMonitor:
    """Runs a background microphone stream and continuously tracks pitch.

    Fails soft (prints a warning, becomes a no-op) if no microphone/audio
    backend is available, so the app never crashes because of a missing mic.
    """

    def __init__(self, sample_rate: int = 16000, block_duration: float = 0.5,
                 device: Optional[int] = None, rising_threshold: float = 0.15):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)
        self.device = device
        self.rising_threshold = rising_threshold

        self._stream = None
        self._lock = threading.Lock()
        self._pitch_history: deque = deque(maxlen=100)
        self._calibration_pitches: list = []
        self._calibrated = False
        self.baseline_pitch: Optional[float] = None
        self.available = _SOUNDDEVICE_AVAILABLE

    def _callback(self, indata, _frames, _time_info, _status) -> None:
        mono = indata[:, 0] if indata.ndim > 1 else indata
        pitch = estimate_pitch(mono, self.sample_rate)
        if pitch <= 0:
            return
        with self._lock:
            self._pitch_history.append(pitch)
            if not self._calibrated:
                self._calibration_pitches.append(pitch)

    def start(self) -> None:
        if not self.available:
            print("Voice stress monitor: sounddevice/PortAudio not available - skipping.")
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=self._callback,
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - no mic, permissions denied, etc.
            print(f"Voice stress monitor unavailable (no microphone?): {exc}")
            self._stream = None
            self.available = False

    def reset_calibration(self) -> None:
        """Call this alongside your video "Recalibrate" action."""
        with self._lock:
            self._calibration_pitches = []
            self._calibrated = False
            self.baseline_pitch = None
            self._pitch_history.clear()

    def finish_calibration(self) -> None:
        with self._lock:
            if self._calibration_pitches:
                self.baseline_pitch = float(np.median(self._calibration_pitches))
            self._calibrated = True

    def get_tell(self) -> Optional[str]:
        """Returns a short tell string if recent pitch is significantly
        above the speaker's own calibrated baseline, else None."""
        if not self.available:
            return None
        with self._lock:
            if not self._calibrated or not self.baseline_pitch:
                return None
            recent = list(self._pitch_history)[-10:]
        if len(recent) < 5:
            return None
        recent_avg = float(np.mean(recent))
        deviation = (recent_avg - self.baseline_pitch) / self.baseline_pitch
        if deviation > self.rising_threshold:
            return VOICE_TELL_TEXT
        return None

    def get_recent_pitches(self, n: int = 60) -> list:
        """Recent pitch (Hz) samples, most recent last - used for the
        in-app voice mini chart. Empty list while silent/unavailable."""
        with self._lock:
            return list(self._pitch_history)[-n:]

    def get_current_pitch(self) -> float:
        """Latest single pitch reading in Hz, or 0.0 if silent/no data yet -
        used for a live 'listening' readout in the UI."""
        with self._lock:
            return self._pitch_history[-1] if self._pitch_history else 0.0

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __enter__(self) -> "AudioPitchMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def analyze_video_audio_pitch(
    video_path: str,
    calibration_seconds: float,
    block_duration: float = 0.5,
    rising_threshold: float = 0.15,
) -> List[float]:
    """Batch/offline equivalent of `AudioPitchMonitor` for an uploaded
    video file: extracts the video's audio track, splits it into
    `block_duration`-second blocks, estimates pitch per block with the
    same `estimate_pitch()` used for the live microphone, and returns a
    sorted list of video-relative timestamps (seconds) at which pitch
    newly rises above this speaker's own baseline (a "rising edge" -
    only the moment stress *starts*, matching how the live view logs a
    tell once when it first becomes active rather than continuously).

    `calibration_seconds` marks where the same calibration window the
    visual detector used ends (see video_upload.py) - audio from before
    that point establishes the baseline; audio after it is compared
    against that baseline.

    Never raises: returns an empty list if the video has no audio track,
    moviepy isn't installed, or extraction fails for any other reason -
    voice analysis is a bonus signal, not something that should block the
    rest of an uploaded video's analysis.
    """
    try:
        from moviepy.editor import VideoFileClip
    except Exception:
        print("Voice analysis for uploaded video skipped: moviepy not available.")
        return []

    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return []  # silent video / no audio track

        sample_rate = 16000
        audio_array = clip.audio.to_soundarray(fps=sample_rate, nbytes=2)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # mix down to mono

        block_size = int(sample_rate * block_duration)
        n_blocks = len(audio_array) // block_size
        if n_blocks == 0:
            return []

        pitches = []  # list of (block_time_seconds, pitch_hz)
        for i in range(n_blocks):
            block = audio_array[i * block_size:(i + 1) * block_size]
            pitch = estimate_pitch(block, sample_rate)
            pitches.append((i * block_duration, pitch))

        calibration_pitches = [p for t, p in pitches if t < calibration_seconds and p > 0]
        if not calibration_pitches:
            return []  # no usable voiced audio during calibration - can't set a baseline
        baseline_pitch = float(np.median(calibration_pitches))

        rising_edges: List[float] = []
        recent: deque = deque(maxlen=10)
        flagged = False
        for block_time, pitch in pitches:
            if block_time < calibration_seconds:
                continue
            if pitch > 0:
                recent.append(pitch)
            if len(recent) < 5:
                continue
            recent_avg = float(np.mean(recent))
            deviation = (recent_avg - baseline_pitch) / baseline_pitch
            is_high = deviation > rising_threshold
            if is_high and not flagged:
                rising_edges.append(block_time)
                flagged = True
            elif not is_high:
                flagged = False

        return rising_edges
    except Exception as exc:  # noqa: BLE001 - never let voice analysis break video upload
        print(f"Voice analysis for uploaded video failed, continuing without it: {exc}")
        return []
    finally:
        if clip is not None:
            clip.close()
