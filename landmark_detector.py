from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Model locations
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# ---------------------------------------------------------------------------
# Drawing constants (replaces mp.solutions.face_mesh.FACEMESH_* /
# mp.solutions.hands.HAND_CONNECTIONS)
# ---------------------------------------------------------------------------
FACE_LANDMARKS_TESSELATION = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
FACE_LANDMARKS_CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
# The new API splits the old single "FACEMESH_IRISES" constant into a left
# and a right eye; we recombine them so calling code doesn't need to care.
FACE_LANDMARKS_IRISES = (
    list(vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
    + list(vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
)
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

drawing_utils = vision.drawing_utils
drawing_styles = vision.drawing_styles

NormalizedLandmark = mp.Image  # re-exported only for type-hint convenience


def _require_models() -> None:
    missing = [p for p in (FACE_MODEL_PATH, HAND_MODEL_PATH) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing MediaPipe model bundle(s):\n  "
            + "\n  ".join(missing)
            + "\n\nRun:\n    python download_models.py\n"
            "once (requires an internet connection) before starting the app."
        )


class FaceHandDetector:
    """Drop-in replacement for the old pair of
    ``mp.solutions.face_mesh.FaceMesh`` + ``mp.solutions.hands.Hands``
    objects, built on the MediaPipe Tasks API running in VIDEO mode.

    Usage mirrors the previous helper function ``find_face_and_hands``::

        with FaceHandDetector() as detector:
            face_landmarks, hands_landmarks = detector.process(frame_bgr)
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        max_num_hands: int = 2,
        min_face_detection_confidence: float = 0.5,
        min_face_tracking_confidence: float = 0.5,
        min_hand_detection_confidence: float = 0.7,
        min_hand_tracking_confidence: float = 0.5,
    ) -> None:
        _require_models()

        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_hand_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        self._t0 = time.monotonic()
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        # VIDEO-mode tasks require strictly increasing timestamps per call.
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def process(
        self, frame_bgr
    ) -> Tuple[Optional[list], Optional[List[list]]]:
        """Runs both landmarkers on a single BGR frame.

        Returns ``(face_landmarks, hands_landmarks)`` where:
          * ``face_landmarks`` is a list of 478 landmarks for the first
            detected face (or ``None``).
          * ``hands_landmarks`` is a list of per-hand landmark lists
            (or ``None``).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = self._next_timestamp_ms()

        face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_landmarks = (
            face_result.face_landmarks[0] if face_result.face_landmarks else None
        )
        hands_landmarks = (
            hand_result.hand_landmarks if hand_result.hand_landmarks else None
        )
        return face_landmarks, hands_landmarks

    def close(self) -> None:
        self._face_landmarker.close()
        self._hand_landmarker.close()

    def __enter__(self) -> "FaceHandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_landmarks_and_hands(image_bgr, face_landmarks, hands_landmarks) -> None:
    """Draws the face mesh (tesselation + contours + irises) and hand
    skeletons onto ``image_bgr`` in place. Replaces the old function of
    the same name that relied on ``mp.solutions.drawing_utils``.
    """
    if face_landmarks:
        drawing_utils.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=image_bgr,
            landmark_list=face_landmarks,
            connections=FACE_LANDMARKS_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    if hands_landmarks:
        for hand_landmarks in hands_landmarks:
            drawing_utils.draw_landmarks(
                image=image_bgr,
                landmark_list=hand_landmarks,
                connections=HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
            )
