from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import mysql.connector

import config

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def get_connection():
    return mysql.connector.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
    )


def init_db() -> None:
    """Creates the database + tables if they don't exist yet. Safe to call
    every time the app starts - connects without selecting a database
    first, since the database itself might not exist yet."""
    conn = mysql.connector.connect(
        host=config.DB_HOST, port=config.DB_PORT,
        user=config.DB_USER, password=config.DB_PASSWORD,
    )
    try:
        cursor = conn.cursor()
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            schema_sql = fh.read()
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def create_session(subject_name: str) -> int:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (subject_name, start_time) VALUES (%s, %s)",
            (subject_name or "Unknown", datetime.now()),
        )
        conn.commit()
        session_id = cursor.lastrowid
        cursor.close()
        return session_id
    finally:
        conn.close()


def end_session(session_id: int, avg_confidence: float, max_confidence: float) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET end_time=%s, avg_confidence=%s, max_confidence=%s WHERE id=%s",
            (datetime.now(), avg_confidence, max_confidence, session_id),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def log_tell_event(
    session_id: int, tell_type: str, tell_text: str, confidence: int,
    event_time: Optional[datetime] = None,
) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tell_events (session_id, tell_type, tell_text, event_time, confidence_at_event) "
            "VALUES (%s, %s, %s, %s, %s)",
            (session_id, tell_type, tell_text, event_time or datetime.now(), confidence),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def get_all_sessions() -> list:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC")
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()


def get_session(session_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sessions WHERE id=%s", (session_id,))
        row = cursor.fetchone()
        cursor.close()
        return row
    finally:
        conn.close()


def get_session_events(session_id: int) -> list:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM tell_events WHERE session_id=%s ORDER BY event_time ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()
