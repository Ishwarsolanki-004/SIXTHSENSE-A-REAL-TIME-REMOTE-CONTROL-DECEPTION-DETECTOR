from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from fer.fer import FER
from scipy.signal import find_peaks
from scipy.spatial import distance as dist

from landmark_detector import draw_landmarks_and_hands

logging.basicConfig(
    filename="deception_detection.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FRAMES = 120
RECENT_FRAMES = int(MAX_FRAMES / 10)
EYE_BLINK_HEIGHT = 0.15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = 0.35
TEXT_HEIGHT = 30

# MediaPipe FaceMesh face-oval landmark indices (unchanged - the Tasks API
# uses the same 478-point topology as the legacy refine_landmarks=True mesh).
FACEMESH_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109, 10,
]

EPOCH = time.time()

# Fraction of the calibration window's samples to discard as outliers before
# computing a personalized baseline (e.g. blink frames during eye-ratio
# calibration) - see finish_calibration().
CALIBRATION_OUTLIER_FRACTION = 0.10

# Multipliers applied to a person's own calibrated baseline to derive their
# personalized thresholds. Tuned to behave like the original fixed
# constants (EYE_BLINK_HEIGHT / LIP_COMPRESSION_RATIO) for an "average"
# face, while adapting up/down for people whose resting eye or lip ratios
# differ from that average.
EYE_BLINK_BASELINE_RATIO = 0.65
LIP_COMPRESSION_BASELINE_RATIO = 0.75

# Weights used by get_confidence_score() - tuned by how strongly each tell
# is generally associated with deception in the literature this project is
# based on; heart-rate change is weighted highest since it's involuntary.
CONFIDENCE_WEIGHTS = {
    "blinking": 20,
    "hand": 15,
    "gaze": 15,
    "lips": 20,
    "bpm_change": 30,
    "voice": 20,
}

# ---------------------------------------------------------------------------
# Rolling state
# ---------------------------------------------------------------------------
blinks = [False] * MAX_FRAMES
hand_on_face = [False] * MAX_FRAMES
face_area_size = 0
hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES
gaze_values = [0] * MAX_FRAMES

# Per-person baseline calibration state
calibration_ear_values: list = []
calibration_lip_values: list = []
personalized_eye_threshold: float = EYE_BLINK_HEIGHT
personalized_lip_threshold: float = LIP_COMPRESSION_RATIO

emotion_detector = FER(mtcnn=True)
calculating_mood = False
mood = ""
tells: dict = {}


def get_recent_hr_values(n: int = 60) -> list:
    """Raw cheek-brightness rPPG signal, most recent `n` samples - not
    currently consumed by the Flask UI, kept ready for a future chart."""
    return hr_values[-n:]


def get_recent_gaze_values(n: int = 60) -> list:
    """Raw relative-gaze signal, most recent `n` samples."""
    return gaze_values[-n:]


def reset_calibration() -> None:
    """Call when the user hits "Recalibrate" so a new baseline is captured
    for whoever is now in frame, instead of reusing the previous person's."""
    global calibration_ear_values, calibration_lip_values
    global personalized_eye_threshold, personalized_lip_threshold
    calibration_ear_values = []
    calibration_lip_values = []
    personalized_eye_threshold = EYE_BLINK_HEIGHT
    personalized_lip_threshold = LIP_COMPRESSION_RATIO


def finish_calibration() -> None:
    """Derives this person's personalized blink / lip-compression
    thresholds from the samples collected during the calibration window.
    Falls back to the original fixed constants if not enough data was
    collected (e.g. face wasn't visible during calibration)."""
    global personalized_eye_threshold, personalized_lip_threshold

    if calibration_ear_values:
        sorted_ear = sorted(calibration_ear_values)
        # Drop the lowest slice (likely blink frames) so the baseline
        # reflects "eyes normally open", not an average dragged down by
        # blinks that happened to occur during calibration.
        cutoff = int(len(sorted_ear) * CALIBRATION_OUTLIER_FRACTION)
        open_eye_samples = sorted_ear[cutoff:] or sorted_ear
        baseline_ear = float(np.mean(open_eye_samples))
        personalized_eye_threshold = baseline_ear * EYE_BLINK_BASELINE_RATIO

    if calibration_lip_values:
        baseline_lip = float(np.mean(calibration_lip_values))
        personalized_lip_threshold = baseline_lip * LIP_COMPRESSION_BASELINE_RATIO


def decrement_tells(tells_dict: dict) -> dict:
    for key, tell in tells_dict.copy().items():
        if "ttl" in tell:
            tell["ttl"] -= 1
            if tell["ttl"] <= 0:
                del tells_dict[key]
    return tells_dict


def new_tell(result: str, ttl_for_tells: int) -> dict:
    return {"text": result, "ttl": ttl_for_tells}


def smooth(signal, window_size: int):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode="same")


def calculate_bpm(signal, fps, min_bpm=50, max_bpm=150):
    if not fps:
        return None
    signal = smooth(signal, window_size=5)
    peaks, _ = find_peaks(signal, distance=max(fps / 2.5, 1), height=0.05)
    if len(peaks) < 2:
        return None
    peak_intervals = np.diff(peaks) / fps * 60
    valid_peaks = peak_intervals[(peak_intervals >= min_bpm) & (peak_intervals <= max_bpm)]
    if len(valid_peaks) == 0:
        return None
    return float(np.mean(valid_peaks))


def draw_on_frame(image, face_landmarks, hands_landmarks) -> None:
    """Kept for parity with the original module (used by CLI-style
    entry points such as intercept.py). Delegates to the new drawing
    helper built on the Tasks API."""
    draw_landmarks_and_hands(image, face_landmarks, hands_landmarks)


def add_text(image, tells_dict: dict, calibrated: bool) -> None:
    global mood
    text_y = TEXT_HEIGHT
    if mood:
        write(f"Mood: {mood}", image, int(0.75 * image.shape[1]), TEXT_HEIGHT)
    if calibrated:
        for tell in tells_dict.values():
            write(tell["text"], image, 10, text_y)
            text_y += TEXT_HEIGHT


def write(text: str, image, x: int, y: int) -> None:
    cv2.putText(
        img=image, text=text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=4,
    )
    cv2.putText(
        img=image, text=text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=2,
    )


def get_aspect_ratio(top, bottom, right, left) -> float:
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width if width else 0.0


def get_area(image, draw, topL, topR, bottomR, bottomL):
    topY = int((topR.y + topL.y) / 2 * image.shape[0])
    botY = int((bottomR.y + bottomL.y) / 2 * image.shape[0])
    leftX = int((topL.x + bottomL.x) / 2 * image.shape[1])
    rightX = int((topR.x + bottomR.x) / 2 * image.shape[1])
    return image[topY:botY, rightX:leftX]


def get_eye_aspect_ratio(face) -> float:
    eyeR = [face[p] for p in [159, 145, 133, 33]]
    eyeR_ar = get_aspect_ratio(*eyeR)
    eyeL = [face[p] for p in [386, 374, 362, 263]]
    eyeL_ar = get_aspect_ratio(*eyeL)
    return (eyeR_ar + eyeL_ar) / 2


def is_blinking(face, eye_ar: float | None = None) -> bool:
    if eye_ar is None:
        eye_ar = get_eye_aspect_ratio(face)
    return eye_ar < personalized_eye_threshold


def get_blink_tell(blinks_list) -> str | None:
    # `blinks_list` is updated as `blinks[1:] + [new_value]`, so the most
    # recent sample is at the END of the list. The guard below must look
    # at the same recent window used for `recent_closed`, not the oldest
    # frames - otherwise the tell would be gated on stale data.
    if sum(blinks_list[-RECENT_FRAMES:]) < 3:
        return None
    recent_closed = 1.0 * sum(blinks_list[-RECENT_FRAMES:]) / RECENT_FRAMES
    avg_closed = 1.0 * sum(blinks_list) / MAX_FRAMES
    if avg_closed > 0 and recent_closed > (20 * avg_closed):
        return "Increased blinking"
    if recent_closed > 0 and avg_closed > (20 * recent_closed):
        return "Decreased blinking"
    return None


def check_hand_on_face(hands_landmarks, face) -> bool:
    """`hands_landmarks` is `list[list[NormalizedLandmark]]` under the
    Tasks API - each hand entry is already the flat list of 21 points
    (no more `.landmark` wrapper), so we iterate directly."""
    if hands_landmarks:
        face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
        face_points = [[[p.x, p.y] for p in face_landmarks]]
        face_contours = np.array(face_points).astype(np.single)
        for hand_landmarks in hands_landmarks:
            hand = [(point.x, point.y) for point in hand_landmarks]
            for finger in [4, 8, 20]:
                overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
                if overlap != -1:
                    return True
    return False


def get_avg_gaze(face) -> float:
    gaze_left = get_gaze(face, 476, 474, 263, 362)
    gaze_right = get_gaze(face, 471, 469, 33, 133)
    return round((gaze_left + gaze_right) / 2, 1)


def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner) -> float:
    iris = (face[iris_L_side].x + face[iris_R_side].x, face[iris_L_side].y + face[iris_R_side].y)
    eye_center = (face[eye_L_corner].x + face[eye_R_corner].x, face[eye_L_corner].y + face[eye_R_corner].y)
    gaze_dist = dist.euclidean(iris, eye_center)
    eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
    gaze_relative = gaze_dist / eye_width if eye_width else 0.0
    if (eye_center[0] - iris[0]) < 0:
        gaze_relative *= -1
    return gaze_relative


# How many standard deviations the recent-window gaze average must be
# from the longer-window baseline average to count as a real "change in
# gaze" (rather than normal micro-jitter). Replaces the previous
# exact-value-repeat-count check, which mostly measured float-rounding
# luck instead of actual gaze movement.
GAZE_CHANGE_STD_MULTIPLIER = 2.0


def detect_gaze_change(avg_gaze: float) -> bool:
    global gaze_values
    gaze_values = gaze_values[1:] + [avg_gaze]

    baseline_window = gaze_values[:-RECENT_FRAMES]
    recent_window = gaze_values[-RECENT_FRAMES:]

    baseline_mean = float(np.mean(baseline_window))
    baseline_std = float(np.std(baseline_window))
    recent_mean = float(np.mean(recent_window))

    if baseline_std < 1e-6:
        return False  # not enough variation yet to judge a "change" against

    deviation = abs(recent_mean - baseline_mean) / baseline_std
    return deviation > GAZE_CHANGE_STD_MULTIPLIER


def get_lip_ratio(face) -> float:
    return get_aspect_ratio(face[0], face[17], face[61], face[291])


def get_mood(image):
    global emotion_detector, calculating_mood, mood
    try:
        detected_mood, score = emotion_detector.top_emotion(image)
    except Exception:
        logging.exception("Emotion detection failed")
        calculating_mood = False
        return None
    calculating_mood = False
    if score and (score > 0.4 or detected_mood == "neutral"):
        mood = detected_mood
        return mood
    return None


def get_emotions(image) -> dict:
    global emotion_detector
    emotion_data = {
        "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
        "sad": 0, "surprise": 0, "neutral": 0,
    }
    emotions = emotion_detector.detect_emotions(image)
    if emotions:
        for emotion in emotions:
            for key in emotion["emotions"]:
                emotion_data[key] += emotion["emotions"][key]
    return emotion_data


def get_face_relative_area(face) -> float:
    face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
    face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
    return face_width * face_height


def get_confidence_score(tells_dict: dict) -> int:
    """Combines the currently active tells into a single weighted 0-100
    "deception confidence" score. `avg_bpms` is an always-on readout (not
    itself a tell), so it's excluded from scoring."""
    score = sum(
        CONFIDENCE_WEIGHTS.get(key, 0)
        for key in tells_dict
        if key != "avg_bpms"
    )
    return min(score, 100)


def find_face_and_hands(image_original, detector):
    """`detector` is a `landmark_detector.FaceHandDetector` instance.
    Kept as a thin wrapper (rather than inlining `detector.process`
    everywhere) so existing call sites / tests barely change."""
    return detector.process(image_original)


def process_frame(image, face_landmarks, hands_landmarks, calibrated=False, fps=None, ttl_for_tells=30):
    global tells, calculating_mood
    global blinks, hand_on_face, face_area_size
    global calibration_ear_values, calibration_lip_values

    tells = decrement_tells(tells)

    if face_landmarks:
        face = face_landmarks  # Tasks API: already a flat landmark list
        face_area_size = get_face_relative_area(face)

        if not calculating_mood:
            calculating_mood = True
            emothread = threading.Thread(target=get_mood, args=(image,), daemon=True)
            emothread.start()

        bpm = get_bpm_change_value(image, False, face_landmarks, hands_landmarks, fps)
        bpm_display = f"BPM: {bpm:.2f}" if bpm else "BPM: ..."
        tells["avg_bpms"] = new_tell(bpm_display, ttl_for_tells)
        if bpm:
            bpm_delta = bpm - avg_bpms[-1]
            if abs(bpm_delta) > SIGNIFICANT_BPM_CHANGE:
                change_desc = "Heart rate increasing" if bpm_delta > 0 else "Heart rate decreasing"
                tells["bpm_change"] = new_tell(change_desc, ttl_for_tells)

        eye_ar = get_eye_aspect_ratio(face)
        lip_ratio = get_lip_ratio(face)

        if not calibrated:
            # Collect this person's own baseline instead of relying only on
            # the fixed EYE_BLINK_HEIGHT / LIP_COMPRESSION_RATIO constants.
            calibration_ear_values.append(eye_ar)
            calibration_lip_values.append(lip_ratio)

        blinks = blinks[1:] + [is_blinking(face, eye_ar)]
        recent_blink_tell = get_blink_tell(blinks)
        if recent_blink_tell:
            tells["blinking"] = new_tell(recent_blink_tell, ttl_for_tells)

        recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
        hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
        if recent_hand_on_face:
            tells["hand"] = new_tell("Hand covering face", ttl_for_tells)

        avg_gaze = get_avg_gaze(face)
        if detect_gaze_change(avg_gaze):
            tells["gaze"] = new_tell("Change in gaze", ttl_for_tells)

        if lip_ratio < personalized_lip_threshold:
            tells["lips"] = new_tell("Lip compression", ttl_for_tells)

    return tells


def get_bpm_change_value(image, draw, face_landmarks, hands_landmarks, fps):
    global hr_values
    if face_landmarks:
        face = face_landmarks  # Tasks API: already a flat landmark list
        cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
        cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])
        if cheekL.size and cheekR.size:
            cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
            cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
            hr_values = hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]
    bpm = calculate_bpm(hr_values, fps)
    return bpm
