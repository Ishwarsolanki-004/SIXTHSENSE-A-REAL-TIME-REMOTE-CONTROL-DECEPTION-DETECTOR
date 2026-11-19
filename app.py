from __future__ import annotations

import atexit
import os
from datetime import datetime

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

import camera.stream as stream
import config
import database.db as db
import video_upload

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# Runs at import time (not just inside `if __name__ == "__main__":`) so the
# tables also get created when the app is served by a WSGI server such as
# gunicorn ("gunicorn app:app"), not only via `python app.py`. Safe to run
# every time the process starts - schema.sql uses CREATE ... IF NOT EXISTS
# throughout.
db.init_db()

# In-memory state for whichever recording session is currently active.
# This is a single-process, single-active-session design - correct and
# simple for a demo/student project running on one machine. A
# multi-user production version would move this into the DB itself or a
# proper server-side session store instead of module-level globals.
_active_session_id: int | None = None
_seen_tell_keys: set[str] = set()
_confidence_samples: list[int] = []


@app.route("/")
def index():
    return render_template("index.html", session_active=_active_session_id is not None)


@app.route("/video_feed")
def video_feed():
    return Response(
        stream.generate_mjpeg(draw_landmarks=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/tells")
def api_tells():
    global _seen_tell_keys

    state = stream.get_state()

    # Only log a tell to the database the moment it *first* becomes
    # active (not on every 0.5s poll while it's still showing), so the
    # tell_events table stays meaningful instead of flooding with
    # duplicates of the same still-active tell.
    if _active_session_id is not None and state["calibrated"]:
        _confidence_samples.append(state["confidence"])
        current_keys = {t["key"] for t in state["tells"]}
        newly_active = current_keys - _seen_tell_keys
        for tell in state["tells"]:
            if tell["key"] in newly_active:
                db.log_tell_event(_active_session_id, tell["key"], tell["text"], state["confidence"])
        _seen_tell_keys = current_keys

    state["session_active"] = _active_session_id is not None
    return jsonify(state)


@app.route("/session/start", methods=["POST"])
def session_start():
    global _active_session_id, _seen_tell_keys, _confidence_samples

    data = request.get_json(silent=True) or {}
    subject_name = data.get("subject_name") or request.form.get("subject_name") or "Unknown"

    _active_session_id = db.create_session(subject_name)
    _seen_tell_keys = set()
    _confidence_samples = []
    stream.reset_calibration()

    return jsonify({"session_id": _active_session_id, "subject_name": subject_name})


@app.route("/session/stop", methods=["POST"])
def session_stop():
    global _active_session_id, _confidence_samples

    if _active_session_id is None:
        return jsonify({"error": "No active session"}), 400

    avg_conf = sum(_confidence_samples) / len(_confidence_samples) if _confidence_samples else 0
    max_conf = max(_confidence_samples) if _confidence_samples else 0
    db.end_session(_active_session_id, avg_conf, max_conf)

    ended_id = _active_session_id
    _active_session_id = None
    _confidence_samples = []

    return jsonify({"session_id": ended_id, "avg_confidence": avg_conf, "max_confidence": max_conf})


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    uploaded_file = request.files.get("video_file")
    subject_name = request.form.get("subject_name") or "Uploaded video"

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("upload.html", error="Please choose a video file.")
    if not video_upload.is_allowed_file(uploaded_file.filename):
        return render_template(
            "upload.html",
            error="Unsupported file type. Allowed: .mp4 .avi .mkv .mov .wmv",
        )

    os.makedirs(video_upload.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.filename}"
    save_path = os.path.join(video_upload.UPLOAD_DIR, safe_name)
    uploaded_file.save(save_path)

    try:
        events, avg_conf, max_conf = video_upload.analyze_video(save_path)
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        return render_template("upload.html", error=f"Could not process video: {exc}")

    session_start = datetime.now()
    session_id = db.create_session(subject_name)
    for event_time, tell_key, tell_text, confidence in video_upload.events_to_absolute_times(events, session_start):
        db.log_tell_event(session_id, tell_key, tell_text, confidence, event_time=event_time)
    db.end_session(session_id, avg_conf, max_conf)

    return redirect(url_for("history", session_id=session_id))


@app.route("/history")
def history():
    sessions = db.get_all_sessions()
    selected_id = request.args.get("session_id", type=int)
    events = db.get_session_events(selected_id) if selected_id else []
    return render_template("history.html", sessions=sessions, events=events, selected_id=selected_id)


atexit.register(stream.stop)

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
