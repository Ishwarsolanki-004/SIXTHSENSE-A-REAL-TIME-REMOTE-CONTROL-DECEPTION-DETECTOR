-- schema.sql
-- Run automatically by database/db.py's init_db() on every app startup
-- (CREATE ... IF NOT EXISTS everywhere, so this is safe to re-run).

CREATE DATABASE IF NOT EXISTS sixthsense;
USE sixthsense;

-- One row per "Start Session" / "Stop Session" recording.
CREATE TABLE IF NOT EXISTS sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(100) DEFAULT 'Unknown',
    start_time DATETIME NOT NULL,
    end_time DATETIME NULL,
    avg_confidence FLOAT NULL,
    max_confidence FLOAT NULL
);

-- One row per tell that became active during a session (e.g. "Hand
-- covering face", "Change in gaze") - only logged the moment a tell
-- first appears, not on every poll, so this doesn't flood the table.
CREATE TABLE IF NOT EXISTS tell_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    tell_type VARCHAR(50) NOT NULL,
    tell_text VARCHAR(255) NOT NULL,
    event_time DATETIME NOT NULL,
    confidence_at_event INT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
