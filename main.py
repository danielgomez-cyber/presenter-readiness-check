from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import google.auth
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydantic import BaseModel, EmailStr, Field, field_validator


# =============================================================================
# Application configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("presenter-readiness-check")

app = FastAPI(
    title="Presenter Readiness Check",
    version="3.0.0",
)


# =============================================================================
# CORS
# =============================================================================
# The form that submits to this API (e.g. the Lovable-hosted front end) is
# served from a different origin than this API, so the browser requires
# CORS headers on the response or the fetch() call is blocked silently.
# Configure ALLOWED_ORIGINS as a comma-separated list in .env; falls back to
# the known Lovable site and localhost for local development.

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
] or [
    "https://lawlinepresentercheck.lovable.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# =============================================================================
# Environment helpers
# =============================================================================

def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()

    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()

    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


# =============================================================================
# Google Sheets configuration
# =============================================================================

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# =============================================================================
# Remote Filming Test booking lookup (Google Calendar)
# =============================================================================
# The "Book your remote filming test" step sends presenters to a Google
# Calendar Appointment Schedule page. Booking there creates a real Calendar
# event with the presenter listed as an attendee (by the email they typed
# into the booking page). To surface "did this presenter book their test"
# on the Trello card, this backend looks up that calendar and matches
# events by attendee email against the readiness-check submitter's email.
#
# Setup required (one-time, by the calendar owner):
#   1. Share the calendar behind the booking page with this backend's
#      service account email (see service-account.json's "client_email")
#      -- "See all event details" access is enough, no edit access needed.
#   2. If the booking page isn't on that account's PRIMARY calendar, set
#      GOOGLE_CALENDAR_ID to the calendar's ID (Calendar Settings -> that
#      calendar -> "Integrate calendar" -> Calendar ID). Defaults to
#      "primary".
GOOGLE_CALENDAR_ID = (
    os.getenv("GOOGLE_CALENDAR_ID", "primary").strip() or "primary"
)

# Only events whose title contains this text are considered -- guards
# against matching an unrelated event on the same calendar.
FILMING_TEST_CALENDAR_KEYWORD = (
    os.getenv("FILMING_TEST_CALENDAR_KEYWORD", "Remote Filming Test").strip()
    or "Remote Filming Test"
)

# How far back/forward to search for a matching booking. Wide enough to
# catch a test the presenter already completed, or one they booked for
# later, without querying the calendar's entire history.
FILMING_TEST_LOOKUP_DAYS_PAST = env_int("FILMING_TEST_LOOKUP_DAYS_PAST", 30)
FILMING_TEST_LOOKUP_DAYS_FUTURE = env_int("FILMING_TEST_LOOKUP_DAYS_FUTURE", 120)

# Shown on the Trello card when no booking is found, so whoever reads the
# card can immediately send the presenter to book.
FILMING_TEST_BOOKING_URL = os.getenv(
    "FILMING_TEST_BOOKING_URL",
    "https://calendar.google.com/calendar/u/0/appointments/schedules/"
    "AcZssZ1Jaj1s5_aXkicZ3nec_QXe7cfW4Z_GeKPVhppnpUbKficbjUsQYKopR0KwdrDgMkiVa77vUJk5",
).strip()

SPREADSHEET_ID = os.getenv(
    "GOOGLE_SPREADSHEET_ID",
    "",
).strip()

SHEET_NAME = (
    os.getenv("GOOGLE_SHEET_NAME", "Speed Tests").strip()
    or "Speed Tests"
)


# =============================================================================
# Trello configuration
# =============================================================================

TRELLO_KEY = os.getenv("TRELLO_KEY", "").strip()
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN", "").strip()
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID", "").strip()

# New two-list setup.
TRELLO_READY_LIST_ID = os.getenv(
    "TRELLO_READY_LIST_ID",
    "",
).strip()

TRELLO_REVIEW_LIST_ID = os.getenv(
    "TRELLO_REVIEW_LIST_ID",
    "",
).strip()

# Backward compatibility with the original single-list setup.
TRELLO_LIST_ID = os.getenv(
    "TRELLO_LIST_ID",
    "",
).strip()


# =============================================================================
# Readiness thresholds
# =============================================================================

MIN_DOWNLOAD_MBPS = env_float("MIN_DOWNLOAD_MBPS", 15.0)
MIN_UPLOAD_MBPS = env_float("MIN_UPLOAD_MBPS", 8.0)
MAX_LATENCY_MS = env_float("MAX_LATENCY_MS", 120.0)
MAX_JITTER_MS = env_float("MAX_JITTER_MS", 30.0)

MIN_CAMERA_WIDTH = env_int("MIN_CAMERA_WIDTH", 960)
MIN_CAMERA_HEIGHT = env_int("MIN_CAMERA_HEIGHT", 540)
MIN_CAMERA_FPS = env_float("MIN_CAMERA_FPS", 20.0)

MIN_LIGHTING_SCORE = env_float("MIN_LIGHTING_SCORE", 60.0)
MIN_FRAMING_SCORE = env_float("MIN_FRAMING_SCORE", 60.0)
MIN_BLUR_SCORE = env_float("MIN_BLUR_SCORE", 50.0)
MIN_BACKGROUND_SCORE = env_float("MIN_BACKGROUND_SCORE", 45.0)
MIN_OVERALL_SCORE = env_float("MIN_OVERALL_SCORE", 75.0)

# Target band for the microphone's peak level, in dBFS. Standard guidance
# is an average speaking level of -18 to -12 dBFS with peaks landing
# around -6 dBFS; since the browser only reports a single peak reading
# (see static/index.html's state.mic.peak), that peak is flagged directly
# against the outer bounds of that guidance (-18 too quiet, -6 too loud /
# clipping risk).
MIC_PEAK_MIN_DBFS = env_float("MIC_PEAK_MIN_DBFS", -18.0)
MIC_PEAK_MAX_DBFS = env_float("MIC_PEAK_MAX_DBFS", -6.0)

MAX_SNAPSHOT_BYTES = env_int(
    "MAX_SNAPSHOT_BYTES",
    4_000_000,
)


# =============================================================================
# Google Sheets columns
# =============================================================================

HEADERS = [
    "Timestamp UTC",
    "Submission ID",
    "Presenter Name",
    "Email",
    "Phone Number",
    "Title",
    "Location",
    "Connection Type",

    "Download Mbps",
    "Download Grade",
    "Upload Mbps",
    "Upload Grade",
    "Latency ms",
    "Latency Grade",
    "Jitter ms",
    "Jitter Grade",
    "Internet Score",
    "Test Duration Seconds",

    "Camera Permission",
    "Camera Device",
    "Camera Width",
    "Camera Height",
    "Camera FPS",
    "Camera Video Active",
    "Camera Score",

    "Microphone Permission",
    "Microphone Device",
    "Microphone Signal Detected",
    "Microphone Peak Level",
    "Microphone Score",

    "Speaker Confirmed",
    "Speaker Score",
    "Audio Score",

    "Snapshot Captured",
    "Snapshot Filename",
    "Snapshot Width",
    "Snapshot Height",
    "Snapshot Analysis Error",

    "Face Detected",
    "Face Count",
    "Multiple Faces",
    "Face Centered",
    "Face Too Close",
    "Face Too Far",
    "Headroom Acceptable",

    "Average Brightness",
    "Contrast",
    "Lighting Score",
    "Backlighting Detected",

    "Blur Value",
    "Blur Score",
    "Blur Detected",

    "Background Edge Density",
    "Background Score",
    "Background Clutter Detected",

    "Glare Ratio",
    "Glasses Glare Warning",

    "Framing Score",
    "Environment Score",
    "Visual Score",
    "Visual Pass",

    "Internet Pass",
    "Camera Pass",
    "Microphone Pass",
    "Speaker Pass",

    "Overall Score",
    "Overall Grade",
    "Recommendation",
    "Recommendation Detail",
    "Fix Recommendations",

    "Time Zone",
    "User Agent",
    "Effective Network Type",
    "Browser Downlink Estimate Mbps",

    "Trello Card URL",
    "Snapshot Attachment URL",
    "Integration Status",
    "Notes",

    # Added at the END, not inserted alongside the related presenter
    # fields above -- inserting a column in the middle would misalign
    # every row already written to the sheet, since existing rows keep
    # their values in their original column positions.
    "Lawline Subscriber",
    "Using Virtual Background",
    "Testing On Intended Device/Location",
    "Device/Location Note",
    "Filming Test Booked",
    "Filming Test Date",
]


def excel_column_name(number: int) -> str:
    result = ""

    while number > 0:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result

    return result


SHEET_LAST_COLUMN = excel_column_name(len(HEADERS))

TRELLO_URL_COLUMN = excel_column_name(
    HEADERS.index("Trello Card URL") + 1
)

SNAPSHOT_URL_COLUMN = excel_column_name(
    HEADERS.index("Snapshot Attachment URL") + 1
)

INTEGRATION_STATUS_COLUMN = excel_column_name(
    HEADERS.index("Integration Status") + 1
)


# =============================================================================
# Submission model
# =============================================================================

PermissionState = Literal[
    "granted",
    "denied",
    "prompt",
    "unavailable",
    "unknown",
]


class ReadinessSubmission(BaseModel):
    presenter_name: str = Field(
        min_length=2,
        max_length=120,
    )

    email: EmailStr

    phone_number: str = Field(
        default="",
        max_length=25,
    )

    presenter_title: str = Field(
        default="",
        max_length=200,
    )

    location: str = Field(
        default="",
        max_length=200,
    )

    # Required radio-button questions on the form (empty string only if a
    # future frontend build somehow bypasses the "required" gate).
    is_subscriber: Literal["yes", "no", ""] = ""
    using_virtual_background: Literal["yes", "no", ""] = ""

    # "Before you begin" gate: is this check running on the actual
    # presentation-day device/location? device_note is only filled in
    # when the answer is "no".
    testing_on_intended: Literal["yes", "no", ""] = ""

    device_note: str = Field(
        default="",
        max_length=300,
    )

    connection_type: Literal[
        "Ethernet",
        "Wi-Fi",
        "Hotspot",
        "Other",
    ]

    download_mbps: float = Field(
        ge=0,
        le=100000,
    )

    upload_mbps: float = Field(
        ge=0,
        le=100000,
    )

    latency_ms: float = Field(
        ge=0,
        le=100000,
    )

    jitter_ms: float = Field(
        ge=0,
        le=100000,
    )

    test_duration_seconds: float = Field(
        ge=0,
        le=3600,
    )

    camera_permission: PermissionState = "unknown"

    camera_device: str = Field(
        default="",
        max_length=500,
    )

    camera_width: int = Field(
        default=0,
        ge=0,
        le=16384,
    )

    camera_height: int = Field(
        default=0,
        ge=0,
        le=16384,
    )

    camera_fps: float = Field(
        default=0,
        ge=0,
        le=1000,
    )

    camera_video_active: bool = False

    microphone_permission: PermissionState = "unknown"

    microphone_device: str = Field(
        default="",
        max_length=500,
    )

    microphone_signal_detected: bool = False

    microphone_peak_level: float = Field(
        default=0,
        ge=0,
        le=1,
    )

    speaker_confirmed: bool = False

    # This is optional until the replacement index.html is installed.
    snapshot_data_url: str = Field(
        default="",
        max_length=6_000_000,
    )

    snapshot_filename: str = Field(
        default="presenter-snapshot.jpg",
        max_length=200,
    )

    timezone: str = Field(
        default="",
        max_length=100,
    )

    user_agent: str = Field(
        default="",
        max_length=1000,
    )

    effective_network_type: str = Field(
        default="",
        max_length=100,
    )

    browser_downlink_mbps: float | None = Field(
        default=None,
        ge=0,
        le=100000,
    )

    notes: str = Field(
        default="",
        max_length=2000,
    )

    consent: bool

    # Honeypot field.
    website: str = Field(
        default="",
        max_length=200,
    )

    @field_validator(
        "presenter_name",
        "phone_number",
        "presenter_title",
        "location",
        "camera_device",
        "microphone_device",
        "snapshot_filename",
        "timezone",
        "user_agent",
        "effective_network_type",
        "notes",
        "website",
        "device_note",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class TrelloIntegrationError(RuntimeError):
    pass


class SnapshotAnalysisError(RuntimeError):
    pass


# =============================================================================
# General helpers
# =============================================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(minimum, min(maximum, value))


def yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def status_icon(value: Any) -> str:
    return "✅" if bool(value) else "❌"


def safe_filename(value: str) -> str:
    original_name = Path(
        value or "presenter-snapshot.jpg"
    ).name

    cleaned_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        original_name,
    )

    if not cleaned_name:
        cleaned_name = "presenter-snapshot.jpg"

    if not cleaned_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        cleaned_name += ".jpg"

    return cleaned_name[:180]


def device_name(value: str) -> str:
    return value or "Browser did not provide a device name"


def testing_on_intended_line(payload: "ReadinessSubmission") -> str:
    """One-line summary of the "Before you begin" gate question, for the
    top of the Trello card. "no" surfaces the presenter's free-text note
    about what they're actually testing on."""
    if payload.testing_on_intended == "yes":
        return (
            f"{status_icon(True)} **Testing on presentation-day "
            f"device/location:** Yes"
        )

    if payload.testing_on_intended == "no":
        note = payload.device_note or "no detail given"
        return (
            f"{status_icon(False)} **Testing on presentation-day "
            f"device/location:** No — {note}"
        )

    return (
        f"{status_icon(False)} **Testing on presentation-day "
        f"device/location:** Not specified"
    )


def yes_no_field(value: str) -> str:
    """Render a required yes/no radio-button answer (is_subscriber,
    using_virtual_background) for display. "" only happens if a future
    frontend build bypasses the form's required gate; falls back to
    "Not specified" there, matching the wording already used in the
    presenter-facing results email (readiness-email.server.ts)."""
    if value == "yes":
        return "Yes"
    if value == "no":
        return "No"
    return "Not specified"


def microphone_peak_dbfs(peak_level: float) -> float:
    """Convert the 0-1 linear peak amplitude (microphone_peak_level) to
    dBFS. 0 dBFS is full scale (peak_level == 1.0); quieter signals are
    negative. A peak of 0 (silence / no signal) has no finite dB value,
    so this returns -inf rather than raising on log10(0).
    """
    if peak_level <= 0:
        return float("-inf")

    return 20 * math.log10(peak_level)


def peak_level_dbfs(peak_level: float) -> str:
    """Render the 0-1 linear peak amplitude as a dBFS string for display
    (Trello card, etc.)."""
    db = microphone_peak_dbfs(peak_level)

    if db == float("-inf"):
        return "-inf dBFS"

    return f"{db:.1f} dBFS"


def microphone_peak_in_range(peak_level: float) -> bool:
    """True if the peak level falls within the target dBFS band
    (MIC_PEAK_MIN_DBFS to MIC_PEAK_MAX_DBFS)."""
    db = microphone_peak_dbfs(peak_level)
    return MIC_PEAK_MIN_DBFS <= db <= MIC_PEAK_MAX_DBFS


def score_grade(score: float) -> str:
    if score >= 90:
        return "Excellent"

    if score >= 80:
        return "Good"

    if score >= 70:
        return "Fair"

    if score >= 60:
        return "Needs Improvement"

    return "Poor"


def letter_grade(score: float) -> str:
    if score >= 93:
        return "A"

    if score >= 90:
        return "A-"

    if score >= 87:
        return "B+"

    if score >= 83:
        return "B"

    if score >= 80:
        return "B-"

    if score >= 77:
        return "C+"

    if score >= 73:
        return "C"

    if score >= 70:
        return "C-"

    if score >= 60:
        return "D"

    return "F"


def value_score(
    value: float,
    ideal_minimum: float,
    ideal_maximum: float,
    absolute_minimum: float,
    absolute_maximum: float,
) -> float:
    if ideal_minimum <= value <= ideal_maximum:
        return 100.0

    if value < ideal_minimum:
        denominator = max(
            ideal_minimum - absolute_minimum,
            0.001,
        )

        return clamp(
            100
            * (value - absolute_minimum)
            / denominator
        )

    denominator = max(
        absolute_maximum - ideal_maximum,
        0.001,
    )

    return clamp(
        100
        * (absolute_maximum - value)
        / denominator
    )


# =============================================================================
# Google Sheets helpers
# =============================================================================

def quoted_sheet_name(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def google_credentials():
    raw_json = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "",
    ).strip()

    credentials_file = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "",
    ).strip()

    if raw_json:
        try:
            account_info = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON"
            ) from exc

        return service_account.Credentials.from_service_account_info(
            account_info,
            scopes=GOOGLE_SCOPES,
        )

    if credentials_file:
        credential_path = Path(credentials_file)

        if not credential_path.is_absolute():
            credential_path = BASE_DIR / credential_path

        if not credential_path.exists():
            raise RuntimeError(
                f"Google credential file not found: "
                f"{credential_path}"
            )

        return service_account.Credentials.from_service_account_file(
            credential_path,
            scopes=GOOGLE_SCOPES,
        )

    credentials, _ = google.auth.default(
        scopes=GOOGLE_SCOPES
    )

    return credentials


def sheets_service():
    return build(
        "sheets",
        "v4",
        credentials=google_credentials(),
        cache_discovery=False,
    )


def calendar_service():
    return build(
        "calendar",
        "v3",
        credentials=google_credentials(),
        cache_discovery=False,
    )


def format_calendar_start(value: str) -> str:
    """Render a Calendar API event start value ("dateTime" or all-day
    "date") for display on the Trello card."""
    if not value:
        return "unknown time"

    try:
        if "T" in value:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%b %-d, %Y, %-I:%M %p")
        return datetime.fromisoformat(value).strftime("%b %-d, %Y")
    except ValueError:
        return value


def find_filming_test_booking(
    email: str,
) -> tuple[Optional[dict[str, str]], bool]:
    """Look up whether this presenter has a booking on the Remote Filming
    Tests calendar, matched by email against the event's attendees.

    Returns (booking, lookup_ok):
      - booking: {"start", "summary", "html_link"} for the soonest
        matching event within the search window, or None if nothing
        matched.
      - lookup_ok: False only if the lookup itself failed (missing
        calendar share, network error, etc.) -- kept distinct from "no
        booking found" so a broken lookup is never reported to the
        reader as "they didn't book."
    """
    if not email:
        return None, True

    try:
        service = calendar_service()

        now = datetime.now(timezone.utc)
        time_min = (now - timedelta(days=FILMING_TEST_LOOKUP_DAYS_PAST)).isoformat()
        time_max = (now + timedelta(days=FILMING_TEST_LOOKUP_DAYS_FUTURE)).isoformat()

        events_result = (
            service.events()
            .list(
                calendarId=GOOGLE_CALENDAR_ID,
                timeMin=time_min,
                timeMax=time_max,
                q=FILMING_TEST_CALENDAR_KEYWORD,
                singleEvents=True,
                orderBy="startTime",
                maxResults=250,
            )
            .execute()
        )

        target = email.strip().lower()

        for event in events_result.get("items", []):
            attendee_emails = {
                str(a.get("email", "")).strip().lower()
                for a in event.get("attendees", [])
            }

            if target in attendee_emails:
                start = event.get("start", {})
                return (
                    {
                        "start": start.get("dateTime") or start.get("date") or "",
                        "summary": event.get("summary", ""),
                        "html_link": event.get("htmlLink", ""),
                    },
                    True,
                )

        return None, True

    except Exception:
        logger.exception("Filming test calendar lookup failed")
        return None, False


def filming_test_booking_line(
    booking: Optional[dict[str, str]],
    lookup_ok: bool,
) -> str:
    """One-line summary of the filming-test booking check, for the top of
    the Trello card."""
    if not lookup_ok:
        return (
            f"{status_icon(False)} **Filming test booked:** Could not "
            f"check (calendar lookup failed) — verify manually: "
            f"{FILMING_TEST_BOOKING_URL}"
        )

    if booking:
        return (
            f"{status_icon(True)} **Filming test booked:** Yes — "
            f"{format_calendar_start(booking['start'])}"
        )

    return (
        f"{status_icon(False)} **Filming test booked:** Not booked yet "
        f"— {FILMING_TEST_BOOKING_URL}"
    )


def ensure_sheet_headers(service) -> None:
    sheet = quoted_sheet_name(SHEET_NAME)

    (
        service.spreadsheets()
        .values()
        .update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet}!A1:{SHEET_LAST_COLUMN}1",
            valueInputOption="RAW",
            body={
                "values": [HEADERS],
            },
        )
        .execute()
    )


# =============================================================================
# Snapshot decoding
# =============================================================================

def decode_snapshot(
    data_url: str,
) -> tuple[bytes, np.ndarray]:
    if not data_url:
        raise SnapshotAnalysisError(
            "No snapshot was supplied."
        )

    match = re.fullmatch(
        r"data:image/(jpeg|jpg|png);base64,(.+)",
        data_url,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        raise SnapshotAnalysisError(
            "The snapshot format is invalid."
        )

    encoded_data = re.sub(
        r"\s+",
        "",
        match.group(2),
    )

    try:
        image_bytes = base64.b64decode(
            encoded_data,
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise SnapshotAnalysisError(
            "The snapshot could not be decoded."
        ) from exc

    if len(image_bytes) > MAX_SNAPSHOT_BYTES:
        raise SnapshotAnalysisError(
            "The snapshot exceeds the allowed file size."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise SnapshotAnalysisError(
            "The snapshot is not a readable image."
        )

    return image_bytes, image


# =============================================================================
# Visual analysis
# =============================================================================

def empty_visual_analysis(
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "snapshot_width": 0,
        "snapshot_height": 0,
        "analysis_error": error_message,

        "face_detected": False,
        "face_count": 0,
        "multiple_faces": False,
        "face_centered": False,
        "face_too_close": False,
        "face_too_far": False,
        "headroom_acceptable": False,

        "average_brightness": 0.0,
        "contrast": 0.0,
        "lighting_score": 0.0,
        "backlighting_detected": False,

        "blur_value": 0.0,
        "blur_score": 0.0,
        "blur_detected": False,

        "background_edge_density": 0.0,
        "background_score": 0.0,
        "background_clutter_detected": False,

        "glare_ratio": 0.0,
        "glasses_glare_warning": False,

        "framing_score": 0.0,
        "environment_score": 0.0,
        "visual_score": 0.0,
        "visual_pass": False,
    }


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        str(cascade_path)
    )

    if detector.empty():
        raise SnapshotAnalysisError(
            "The OpenCV face detector could not be loaded."
        )

    return detector


def analyze_snapshot(
    original_image: np.ndarray,
) -> dict[str, Any]:
    original_height, original_width = (
        original_image.shape[:2]
    )

    image = original_image

    # Shrink large images to keep processing quick.
    if original_width > 1280:
        scale = 1280 / original_width

        image = cv2.resize(
            original_image,
            (
                1280,
                max(
                    1,
                    round(original_height * scale),
                ),
            ),
            interpolation=cv2.INTER_AREA,
        )

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # -------------------------------------------------------------------------
    # Lighting
    # -------------------------------------------------------------------------

    average_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    brightness_score = value_score(
        average_brightness,
        ideal_minimum=90,
        ideal_maximum=185,
        absolute_minimum=35,
        absolute_maximum=235,
    )

    contrast_score = value_score(
        contrast,
        ideal_minimum=35,
        ideal_maximum=80,
        absolute_minimum=10,
        absolute_maximum=120,
    )

    border_size = max(
        5,
        round(min(width, height) * 0.12),
    )

    border_mask = np.zeros(
        gray.shape,
        dtype=np.uint8,
    )

    border_mask[:border_size, :] = 255
    border_mask[-border_size:, :] = 255
    border_mask[:, :border_size] = 255
    border_mask[:, -border_size:] = 255

    center_mask = np.zeros(
        gray.shape,
        dtype=np.uint8,
    )

    center_mask[
        round(height * 0.25):
        round(height * 0.75),
        round(width * 0.25):
        round(width * 0.75),
    ] = 255

    border_brightness = float(
        cv2.mean(
            gray,
            mask=border_mask,
        )[0]
    )

    center_brightness = float(
        cv2.mean(
            gray,
            mask=center_mask,
        )[0]
    )

    backlighting_detected = (
        border_brightness - center_brightness > 30
        and border_brightness > 145
    )

    lighting_score = clamp(
        brightness_score * 0.65
        + contrast_score * 0.35
        - (
            25
            if backlighting_detected
            else 0
        )
    )

    # -------------------------------------------------------------------------
    # Blur
    # -------------------------------------------------------------------------

    blur_value = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    blur_score = clamp(
        blur_value / 180 * 100
    )

    blur_detected = (
        blur_score < MIN_BLUR_SCORE
    )

    # -------------------------------------------------------------------------
    # Face and framing
    # -------------------------------------------------------------------------

    face_detector = load_face_detector()

    minimum_face_size = max(
        40,
        round(min(width, height) * 0.08),
    )

    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(
            minimum_face_size,
            minimum_face_size,
        ),
    )

    faces = [
        tuple(int(value) for value in face)
        for face in detected_faces
    ]

    face_count = len(faces)
    face_detected = face_count > 0
    multiple_faces = face_count > 1

    face_centered = False
    face_too_close = False
    face_too_far = False
    headroom_acceptable = False
    framing_score = 0.0
    glare_ratio = 0.0
    glasses_glare_warning = False

    primary_face: tuple[int, int, int, int] | None = None

    if faces:
        primary_face = max(
            faces,
            key=lambda face: face[2] * face[3],
        )

        x, y, face_width, face_height = primary_face

        face_center_x = (
            x + face_width / 2
        ) / width

        face_center_y = (
            y + face_height / 2
        ) / height

        horizontal_offset = abs(
            face_center_x - 0.5
        )

        vertical_offset = abs(
            face_center_y - 0.43
        )

        face_area_ratio = (
            face_width * face_height
        ) / (width * height)

        face_centered = (
            horizontal_offset <= 0.16
            and vertical_offset <= 0.20
        )

        face_too_close = (
            face_area_ratio > 0.34
        )

        face_too_far = (
            face_area_ratio < 0.035
        )

        headroom_ratio = y / height

        headroom_acceptable = (
            0.03 <= headroom_ratio <= 0.23
        )

        centering_score = clamp(
            100
            - horizontal_offset * 260
            - vertical_offset * 170
        )

        face_size_score = value_score(
            face_area_ratio,
            ideal_minimum=0.07,
            ideal_maximum=0.24,
            absolute_minimum=0.015,
            absolute_maximum=0.50,
        )

        headroom_score = value_score(
            headroom_ratio,
            ideal_minimum=0.06,
            ideal_maximum=0.18,
            absolute_minimum=0,
            absolute_maximum=0.40,
        )

        framing_score = clamp(
            centering_score * 0.50
            + face_size_score * 0.30
            + headroom_score * 0.20
            - (
                20
                if multiple_faces
                else 0
            )
        )

        face_region = gray[
            max(0, y):
            min(height, y + face_height),
            max(0, x):
            min(width, x + face_width),
        ]

        if face_region.size:
            eye_region = face_region[
                round(face_region.shape[0] * 0.18):
                round(face_region.shape[0] * 0.52),
                :,
            ]

            if eye_region.size:
                glare_pixels = np.count_nonzero(
                    eye_region >= 245
                )

                glare_ratio = float(
                    glare_pixels / eye_region.size
                )

                glasses_glare_warning = (
                    glare_ratio > 0.035
                )

    # -------------------------------------------------------------------------
    # Background
    # -------------------------------------------------------------------------

    edge_image = cv2.Canny(
        gray,
        threshold1=80,
        threshold2=170,
    )

    background_mask = np.full(
        gray.shape,
        255,
        dtype=np.uint8,
    )

    if primary_face is not None:
        x, y, face_width, face_height = primary_face

        horizontal_padding = round(
            face_width * 0.65
        )

        vertical_padding = round(
            face_height * 0.45
        )

        x1 = max(
            0,
            x - horizontal_padding,
        )

        y1 = max(
            0,
            y - vertical_padding,
        )

        x2 = min(
            width,
            x + face_width + horizontal_padding,
        )

        y2 = min(
            height,
            y + face_height + vertical_padding,
        )

        background_mask[
            y1:y2,
            x1:x2,
        ] = 0

    background_pixel_count = np.count_nonzero(
        background_mask
    )

    background_edge_count = np.count_nonzero(
        cv2.bitwise_and(
            edge_image,
            background_mask,
        )
    )

    if background_pixel_count:
        background_edge_density = float(
            background_edge_count
            / background_pixel_count
        )
    else:
        background_edge_density = 0.0

    background_score = clamp(
        100
        - max(
            0,
            background_edge_density - 0.035,
        )
        * 900
    )

    background_clutter_detected = (
        background_score < MIN_BACKGROUND_SCORE
    )

    environment_score = clamp(
        lighting_score * 0.55
        + background_score * 0.35
        + (
            0
            if glasses_glare_warning
            else 10
        )
    )

    visual_score = clamp(
        lighting_score * 0.30
        + blur_score * 0.20
        + framing_score * 0.30
        + environment_score * 0.20
    )

    visual_pass = (
        face_detected
        and not multiple_faces
        and lighting_score >= MIN_LIGHTING_SCORE
        and blur_score >= MIN_BLUR_SCORE
        and framing_score >= MIN_FRAMING_SCORE
        and background_score >= MIN_BACKGROUND_SCORE
    )

    return {
        "snapshot_width": original_width,
        "snapshot_height": original_height,
        "analysis_error": "",

        "face_detected": face_detected,
        "face_count": face_count,
        "multiple_faces": multiple_faces,
        "face_centered": face_centered,
        "face_too_close": face_too_close,
        "face_too_far": face_too_far,
        "headroom_acceptable": headroom_acceptable,

        "average_brightness": average_brightness,
        "contrast": contrast,
        "lighting_score": lighting_score,
        "backlighting_detected": backlighting_detected,

        "blur_value": blur_value,
        "blur_score": blur_score,
        "blur_detected": blur_detected,

        "background_edge_density": (
            background_edge_density
        ),
        "background_score": background_score,
        "background_clutter_detected": (
            background_clutter_detected
        ),

        "glare_ratio": glare_ratio,
        "glasses_glare_warning": (
            glasses_glare_warning
        ),

        "framing_score": framing_score,
        "environment_score": environment_score,
        "visual_score": visual_score,
        "visual_pass": visual_pass,
    }


# =============================================================================
# Internet scoring
# =============================================================================

def download_score(value: float) -> float:
    if value >= 50:
        return 100

    if value >= 25:
        return 85 + (value - 25) * 0.6

    if value >= MIN_DOWNLOAD_MBPS:
        available_range = max(
            25 - MIN_DOWNLOAD_MBPS,
            1,
        )

        return (
            70
            + (value - MIN_DOWNLOAD_MBPS)
            * (15 / available_range)
        )

    return clamp(
        value
        / max(MIN_DOWNLOAD_MBPS, 1)
        * 60
    )


def upload_score(value: float) -> float:
    if value >= 25:
        return 100

    if value >= 15:
        return 85 + (value - 15) * 1.5

    if value >= MIN_UPLOAD_MBPS:
        available_range = max(
            15 - MIN_UPLOAD_MBPS,
            1,
        )

        return (
            70
            + (value - MIN_UPLOAD_MBPS)
            * (15 / available_range)
        )

    return clamp(
        value
        / max(MIN_UPLOAD_MBPS, 1)
        * 60
    )


def latency_score(value: float) -> float:
    if value <= 30:
        return 100

    if value <= 60:
        return 100 - (value - 30) * 0.5

    if value <= MAX_LATENCY_MS:
        available_range = max(
            MAX_LATENCY_MS - 60,
            1,
        )

        return (
            85
            - (value - 60)
            * (25 / available_range)
        )

    return clamp(
        60
        - (value - MAX_LATENCY_MS) * 0.5
    )


def jitter_score(value: float) -> float:
    if value <= 5:
        return 100

    if value <= 15:
        return 100 - (value - 5) * 1.5

    if value <= MAX_JITTER_MS:
        available_range = max(
            MAX_JITTER_MS - 15,
            1,
        )

        return (
            85
            - (value - 15)
            * (25 / available_range)
        )

    return clamp(
        60
        - (value - MAX_JITTER_MS)
    )


def calculate_internet_metrics(
    payload: ReadinessSubmission,
) -> dict[str, Any]:
    download_result = download_score(
        payload.download_mbps
    )

    upload_result = upload_score(
        payload.upload_mbps
    )

    latency_result = latency_score(
        payload.latency_ms
    )

    jitter_result = jitter_score(
        payload.jitter_ms
    )

    internet_result = clamp(
        download_result * 0.25
        + upload_result * 0.40
        + latency_result * 0.20
        + jitter_result * 0.15
    )

    return {
        "download_score": download_result,
        "download_grade": score_grade(
            download_result
        ),

        "upload_score": upload_result,
        "upload_grade": score_grade(
            upload_result
        ),

        "latency_score": latency_result,
        "latency_grade": score_grade(
            latency_result
        ),

        "jitter_score": jitter_result,
        "jitter_grade": score_grade(
            jitter_result
        ),

        "internet_score": internet_result,
        "internet_grade": score_grade(
            internet_result
        ),
    }


# =============================================================================
# Individual hardware scores
# =============================================================================

def calculate_camera_score(
    payload: ReadinessSubmission,
    visual: dict[str, Any],
) -> float:
    if (
        payload.camera_permission != "granted"
        or not payload.camera_video_active
    ):
        return 0.0

    pixel_count = (
        payload.camera_width
        * payload.camera_height
    )

    if pixel_count >= 1920 * 1080:
        resolution_score = 100.0
    elif pixel_count >= 1280 * 720:
        resolution_score = 90.0
    elif pixel_count >= 960 * 540:
        resolution_score = 75.0
    elif pixel_count > 0:
        resolution_score = 45.0
    else:
        resolution_score = 0.0

    if payload.camera_fps >= 29:
        fps_score = 100.0
    elif payload.camera_fps >= 24:
        fps_score = 85.0
    elif payload.camera_fps >= MIN_CAMERA_FPS:
        fps_score = 70.0
    elif payload.camera_fps > 0:
        fps_score = 45.0
    else:
        fps_score = 0.0

    # Until a snapshot is supplied, do not punish the camera score
    # for missing visual analysis.
    if visual["analysis_error"]:
        sharpness_score = 70.0
    else:
        sharpness_score = float(
            visual["blur_score"]
        )

    return clamp(
        resolution_score * 0.40
        + fps_score * 0.25
        + sharpness_score * 0.25
        + 10
    )


def calculate_microphone_score(
    payload: ReadinessSubmission,
) -> float:
    if payload.microphone_permission != "granted":
        return 0.0

    if not payload.microphone_signal_detected:
        return 20.0

    peak_level = payload.microphone_peak_level

    if 0.08 <= peak_level <= 0.75:
        level_score = 100.0
    elif peak_level < 0.08:
        level_score = clamp(
            peak_level / 0.08 * 100
        )
    else:
        level_score = clamp(
            100
            - (peak_level - 0.75) * 250
        )

    return clamp(
        35 + level_score * 0.65
    )


# =============================================================================
# Readiness assessment
# =============================================================================

def assess_submission(
    payload: ReadinessSubmission,
    visual: dict[str, Any],
) -> dict[str, Any]:
    internet = calculate_internet_metrics(
        payload
    )

    internet_issues: list[str] = []
    camera_issues: list[str] = []
    microphone_issues: list[str] = []
    speaker_issues: list[str] = []
    visual_issues: list[str] = []
    fixes: list[str] = []

    # -------------------------------------------------------------------------
    # Internet checks
    # -------------------------------------------------------------------------

    if payload.download_mbps < MIN_DOWNLOAD_MBPS:
        internet_issues.append(
            f"Download speed is "
            f"{payload.download_mbps:.1f} Mbps; "
            f"at least {MIN_DOWNLOAD_MBPS:.1f} Mbps is required"
        )

    if payload.upload_mbps < MIN_UPLOAD_MBPS:
        internet_issues.append(
            f"Upload speed is "
            f"{payload.upload_mbps:.1f} Mbps; "
            f"at least {MIN_UPLOAD_MBPS:.1f} Mbps is required"
        )

    if payload.latency_ms > MAX_LATENCY_MS:
        internet_issues.append(
            f"Latency is {payload.latency_ms:.0f} ms; "
            f"it should be no higher than "
            f"{MAX_LATENCY_MS:.0f} ms"
        )

    if payload.jitter_ms > MAX_JITTER_MS:
        internet_issues.append(
            f"Jitter is {payload.jitter_ms:.0f} ms; "
            f"it should be no higher than "
            f"{MAX_JITTER_MS:.0f} ms"
        )

    # -------------------------------------------------------------------------
    # Camera checks
    # -------------------------------------------------------------------------

    if payload.camera_permission != "granted":
        camera_issues.append(
            "Camera permission was not granted"
        )

    if not payload.camera_video_active:
        camera_issues.append(
            "An active camera picture was not confirmed"
        )

    if (
        payload.camera_width <= 0
        or payload.camera_height <= 0
    ):
        camera_issues.append(
            "Camera resolution could not be detected"
        )
    elif (
        payload.camera_width < MIN_CAMERA_WIDTH
        or payload.camera_height < MIN_CAMERA_HEIGHT
    ):
        camera_issues.append(
            f"Camera resolution is below "
            f"{MIN_CAMERA_WIDTH} × {MIN_CAMERA_HEIGHT}"
        )

    if (
        payload.camera_fps > 0
        and payload.camera_fps < MIN_CAMERA_FPS
    ):
        camera_issues.append(
            f"Camera frame rate is below "
            f"{MIN_CAMERA_FPS:.0f} FPS"
        )

    # -------------------------------------------------------------------------
    # Microphone and speaker checks
    # -------------------------------------------------------------------------

    if payload.microphone_permission != "granted":
        microphone_issues.append(
            "Microphone permission was not granted"
        )

    if not payload.microphone_signal_detected:
        microphone_issues.append(
            "No microphone signal was detected"
        )

    mic_peak_db = microphone_peak_dbfs(
        payload.microphone_peak_level
    )
    mic_peak_in_range = (
        MIC_PEAK_MIN_DBFS <= mic_peak_db <= MIC_PEAK_MAX_DBFS
    )

    # Only flag the level itself if a signal was actually detected --
    # otherwise this would just duplicate the "no signal" issue above.
    if payload.microphone_signal_detected and not mic_peak_in_range:
        mic_peak_display = (
            "-inf"
            if mic_peak_db == float("-inf")
            else f"{mic_peak_db:.1f}"
        )

        if mic_peak_db < MIC_PEAK_MIN_DBFS:
            microphone_issues.append(
                f"Microphone peak level is {mic_peak_display} dBFS "
                f"(too quiet); target range is {MIC_PEAK_MIN_DBFS:.0f} "
                f"to {MIC_PEAK_MAX_DBFS:.0f} dBFS"
            )
        else:
            microphone_issues.append(
                f"Microphone peak level is {mic_peak_display} dBFS "
                f"(too loud / clipping risk); target range is "
                f"{MIC_PEAK_MIN_DBFS:.0f} to {MIC_PEAK_MAX_DBFS:.0f} dBFS"
            )

    if not payload.speaker_confirmed:
        speaker_issues.append(
            "Speaker playback was not confirmed"
        )

    # -------------------------------------------------------------------------
    # Snapshot checks
    # -------------------------------------------------------------------------

    snapshot_was_supplied = bool(
        payload.snapshot_data_url
    )

    if snapshot_was_supplied:
        if visual["analysis_error"]:
            visual_issues.append(
                str(visual["analysis_error"])
            )
        else:
            if not visual["face_detected"]:
                visual_issues.append(
                    "No face was detected in the camera snapshot"
                )

            if visual["multiple_faces"]:
                visual_issues.append(
                    "Multiple faces were detected"
                )

            if not visual["face_centered"]:
                visual_issues.append(
                    "The presenter is not centered in the frame"
                )

            if visual["face_too_close"]:
                visual_issues.append(
                    "The presenter appears too close to the camera"
                )

            if visual["face_too_far"]:
                visual_issues.append(
                    "The presenter appears too far from the camera"
                )

            if not visual["headroom_acceptable"]:
                visual_issues.append(
                    "The amount of headroom should be adjusted"
                )

            if visual["lighting_score"] < MIN_LIGHTING_SCORE:
                visual_issues.append(
                    "Lighting quality needs improvement"
                )

            if visual["backlighting_detected"]:
                visual_issues.append(
                    "Strong backlighting was detected"
                )

            if visual["blur_detected"]:
                visual_issues.append(
                    "The camera image appears blurry"
                )

            if visual["background_clutter_detected"]:
                visual_issues.append(
                    "The background may be visually distracting"
                )

            if visual["glasses_glare_warning"]:
                visual_issues.append(
                    "Possible glare was detected near the eyes"
                )

    # -------------------------------------------------------------------------
    # Fix recommendations
    # -------------------------------------------------------------------------

    if payload.connection_type in {
        "Wi-Fi",
        "Hotspot",
    }:
        fixes.append(
            "Use a wired Ethernet connection when possible."
        )

    if payload.download_mbps < MIN_DOWNLOAD_MBPS:
        fixes.append(
            "Stop downloads, cloud syncing, and other streaming activity."
        )

    if payload.upload_mbps < MIN_UPLOAD_MBPS:
        fixes.append(
            "Stop uploads and cloud backups before joining the webcast."
        )

    if payload.latency_ms > MAX_LATENCY_MS:
        fixes.append(
            "Disable VPN software and move closer to the router."
        )

    if payload.jitter_ms > MAX_JITTER_MS:
        fixes.append(
            "Use Ethernet and ask other users to reduce network activity."
        )

    if payload.camera_permission != "granted":
        fixes.append(
            "Allow camera access in the browser and reload the page."
        )

    if (
        payload.camera_width < MIN_CAMERA_WIDTH
        or payload.camera_height < MIN_CAMERA_HEIGHT
    ):
        fixes.append(
            "Choose a camera setting of at least 960 × 540."
        )

    if payload.microphone_permission != "granted":
        fixes.append(
            "Allow microphone access in the browser and reload the page."
        )

    if not payload.microphone_signal_detected:
        fixes.append(
            "Select the intended microphone and speak closer to it."
        )

    if payload.microphone_signal_detected and not mic_peak_in_range:
        if mic_peak_db < MIC_PEAK_MIN_DBFS:
            fixes.append(
                "Increase microphone gain or move closer to the mic — "
                "the level is too quiet."
            )
        else:
            fixes.append(
                "Reduce microphone gain or move back from the mic — "
                "the level is too loud and risks clipping."
            )

    if not payload.speaker_confirmed:
        fixes.append(
            "Check the selected output device and speaker volume."
        )

    if snapshot_was_supplied and not visual["analysis_error"]:
        if visual["backlighting_detected"]:
            fixes.append(
                "Move the brightest light or window in front of you."
            )

        if visual["lighting_score"] < MIN_LIGHTING_SCORE:
            fixes.append(
                "Add a soft light in front of you at face level."
            )

        if visual["face_too_close"]:
            fixes.append(
                "Move the camera farther away."
            )

        if visual["face_too_far"]:
            fixes.append(
                "Move closer to the camera."
            )

        if not visual["face_centered"]:
            fixes.append(
                "Center yourself horizontally in the camera frame."
            )

        if not visual["headroom_acceptable"]:
            fixes.append(
                "Adjust the camera so there is a small amount of space above your head."
            )

        if visual["blur_detected"]:
            fixes.append(
                "Clean the camera lens and verify that the image is in focus."
            )

        if visual["background_clutter_detected"]:
            fixes.append(
                "Remove distracting objects or use a simpler background."
            )

        if visual["glasses_glare_warning"]:
            fixes.append(
                "Raise or move the light source to reduce glare on your glasses."
            )

    if not fixes:
        fixes.append(
            "No corrective action is currently required."
        )

    # Remove duplicate recommendations while preserving order.
    fixes = list(dict.fromkeys(fixes))

    internet_pass = not internet_issues
    camera_pass = not camera_issues
    microphone_pass = not microphone_issues
    speaker_pass = not speaker_issues

    if snapshot_was_supplied:
        visual_pass = (
            not visual_issues
            and bool(visual["visual_pass"])
        )
    else:
        # Phase 1 can still run before index.html adds snapshots.
        visual_pass = True

    camera_result = calculate_camera_score(
        payload,
        visual,
    )

    microphone_result = calculate_microphone_score(
        payload
    )

    speaker_result = (
        100.0
        if payload.speaker_confirmed
        else 0.0
    )

    audio_result = clamp(
        microphone_result * 0.70
        + speaker_result * 0.30
    )

    if snapshot_was_supplied and not visual["analysis_error"]:
        lighting_result = float(
            visual["lighting_score"]
        )

        framing_result = float(
            visual["framing_score"]
        )

        environment_result = float(
            visual["environment_score"]
        )
    else:
        # Neutral values until the frontend sends a snapshot.
        lighting_result = 75.0
        framing_result = 75.0
        environment_result = 75.0

    overall_score = clamp(
        float(internet["internet_score"]) * 0.25
        + camera_result * 0.20
        + audio_result * 0.20
        + lighting_result * 0.15
        + framing_result * 0.10
        + environment_result * 0.10
    )

    all_issues = [
        *internet_issues,
        *camera_issues,
        *microphone_issues,
        *speaker_issues,
        *visual_issues,
    ]

    all_required_checks_pass = (
        internet_pass
        and camera_pass
        and microphone_pass
        and speaker_pass
        and visual_pass
    )

    if (
        all_required_checks_pass
        and overall_score >= MIN_OVERALL_SCORE
    ):
        recommendation = "Ready"

        if snapshot_was_supplied:
            recommendation_detail = (
                "Internet, camera, microphone, speaker, "
                "lighting, framing, and environment checks passed."
            )
        else:
            recommendation_detail = (
                "Internet, camera, microphone, and speaker checks passed. "
                "Visual snapshot analysis was not included."
            )
    else:
        recommendation = "Needs review"

        if all_issues:
            recommendation_detail = (
                "; ".join(
                    dict.fromkeys(all_issues)
                )
                + "."
            )
        else:
            recommendation_detail = (
                "The overall readiness score is below "
                "the required level."
            )

    return {
        **internet,

        "camera_score": camera_result,
        "microphone_score": microphone_result,
        "speaker_score": speaker_result,
        "audio_score": audio_result,

        "internet_pass": internet_pass,
        "camera_pass": camera_pass,
        "microphone_pass": microphone_pass,
        "speaker_pass": speaker_pass,
        "visual_pass": visual_pass,

        "microphone_peak_db": mic_peak_db,
        "microphone_peak_in_range": mic_peak_in_range,

        "overall_score": overall_score,
        "overall_grade": letter_grade(
            overall_score
        ),

        "recommendation": recommendation,
        "recommendation_detail": (
            recommendation_detail
        ),

        "fix_recommendations": fixes,

        "internet_issues": internet_issues,
        "camera_issues": camera_issues,
        "microphone_issues": microphone_issues,
        "speaker_issues": speaker_issues,
        "visual_issues": visual_issues,
    }


# =============================================================================
# Google Sheets writing
# =============================================================================

def append_sheet_row(
    service,
    payload: ReadinessSubmission,
    submission_id: str,
    visual: dict[str, Any],
    assessment: dict[str, Any],
    snapshot_filename: str,
    integration_status: str,
    booking: Optional[dict[str, str]] = None,
    booking_lookup_ok: bool = True,
) -> int | None:
    timestamp = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    values = [
        timestamp,
        submission_id,
        payload.presenter_name,
        str(payload.email),
        payload.phone_number,
        payload.presenter_title,
        payload.location,
        payload.connection_type,

        round(payload.download_mbps, 2),
        assessment["download_grade"],
        round(payload.upload_mbps, 2),
        assessment["upload_grade"],
        round(payload.latency_ms, 2),
        assessment["latency_grade"],
        round(payload.jitter_ms, 2),
        assessment["jitter_grade"],
        round(
            float(assessment["internet_score"]),
            2,
        ),
        round(
            payload.test_duration_seconds,
            2,
        ),

        payload.camera_permission,
        payload.camera_device,
        payload.camera_width,
        payload.camera_height,
        round(payload.camera_fps, 2),
        payload.camera_video_active,
        round(
            float(assessment["camera_score"]),
            2,
        ),

        payload.microphone_permission,
        payload.microphone_device,
        payload.microphone_signal_detected,
        round(
            payload.microphone_peak_level,
            4,
        ),
        round(
            float(
                assessment["microphone_score"]
            ),
            2,
        ),

        payload.speaker_confirmed,
        round(
            float(assessment["speaker_score"]),
            2,
        ),
        round(
            float(assessment["audio_score"]),
            2,
        ),

        bool(payload.snapshot_data_url),
        (
            snapshot_filename
            if payload.snapshot_data_url
            else ""
        ),
        visual["snapshot_width"],
        visual["snapshot_height"],
        visual["analysis_error"],

        visual["face_detected"],
        visual["face_count"],
        visual["multiple_faces"],
        visual["face_centered"],
        visual["face_too_close"],
        visual["face_too_far"],
        visual["headroom_acceptable"],

        round(
            float(visual["average_brightness"]),
            2,
        ),
        round(
            float(visual["contrast"]),
            2,
        ),
        round(
            float(visual["lighting_score"]),
            2,
        ),
        visual["backlighting_detected"],

        round(
            float(visual["blur_value"]),
            2,
        ),
        round(
            float(visual["blur_score"]),
            2,
        ),
        visual["blur_detected"],

        round(
            float(
                visual[
                    "background_edge_density"
                ]
            ),
            5,
        ),
        round(
            float(visual["background_score"]),
            2,
        ),
        visual[
            "background_clutter_detected"
        ],

        round(
            float(visual["glare_ratio"]),
            5,
        ),
        visual["glasses_glare_warning"],

        round(
            float(visual["framing_score"]),
            2,
        ),
        round(
            float(visual["environment_score"]),
            2,
        ),
        round(
            float(visual["visual_score"]),
            2,
        ),
        assessment["visual_pass"],

        assessment["internet_pass"],
        assessment["camera_pass"],
        assessment["microphone_pass"],
        assessment["speaker_pass"],

        round(
            float(assessment["overall_score"]),
            2,
        ),
        assessment["overall_grade"],
        assessment["recommendation"],
        assessment["recommendation_detail"],
        "\n".join(
            assessment["fix_recommendations"]
        ),

        payload.timezone,
        payload.user_agent,
        payload.effective_network_type,

        (
            round(
                payload.browser_downlink_mbps,
                2,
            )
            if payload.browser_downlink_mbps
            is not None
            else ""
        ),

        "",
        "",
        integration_status,
        payload.notes,

        payload.is_subscriber,
        payload.using_virtual_background,
        payload.testing_on_intended,
        payload.device_note,

        (
            "Unknown"
            if not booking_lookup_ok
            else ("Yes" if booking else "No")
        ),
        (
            format_calendar_start(booking["start"])
            if (booking_lookup_ok and booking)
            else ""
        ),
    ]

    response = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=SPREADSHEET_ID,
            range=(
                f"{quoted_sheet_name(SHEET_NAME)}!"
                f"A:{SHEET_LAST_COLUMN}"
            ),
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={
                "values": [values],
            },
        )
        .execute()
    )

    updated_range = response.get(
        "updates",
        {},
    ).get(
        "updatedRange",
        "",
    )

    match = re.search(
        rf"!A(\d+):{SHEET_LAST_COLUMN}\d+$",
        updated_range,
    )

    if not match:
        return None

    return int(match.group(1))


def update_integration_result(
    service,
    row_number: int,
    trello_url: str,
    snapshot_url: str,
    integration_status: str,
) -> None:
    (
        service.spreadsheets()
        .values()
        .update(
            spreadsheetId=SPREADSHEET_ID,
            range=(
                f"{quoted_sheet_name(SHEET_NAME)}!"
                f"{TRELLO_URL_COLUMN}{row_number}:"
                f"{INTEGRATION_STATUS_COLUMN}{row_number}"
            ),
            valueInputOption="RAW",
            body={
                "values": [
                    [
                        trello_url,
                        snapshot_url,
                        integration_status,
                    ]
                ],
            },
        )
        .execute()
    )


# =============================================================================
# Trello helpers
# =============================================================================

def trello_list_id_for_result(
    recommendation: str,
) -> str:
    if recommendation == "Ready":
        return (
            TRELLO_READY_LIST_ID
            or TRELLO_LIST_ID
        )

    return (
        TRELLO_REVIEW_LIST_ID
        or TRELLO_LIST_ID
        or TRELLO_READY_LIST_ID
    )


def trello_is_configured() -> bool:
    list_available = bool(
        TRELLO_READY_LIST_ID
        or TRELLO_REVIEW_LIST_ID
        or TRELLO_LIST_ID
    )

    return bool(
        TRELLO_KEY
        and TRELLO_TOKEN
        and list_available
    )


def trello_board_is_configured() -> bool:
    return bool(
        trello_is_configured()
        and TRELLO_BOARD_ID
    )


def trello_error_message(
    response: requests.Response,
    action: str,
) -> str:
    response_message = response.text.strip()

    if len(response_message) > 300:
        response_message = (
            response_message[:300]
        )

    if response_message:
        return (
            f"{action} returned HTTP "
            f"{response.status_code}: "
            f"{response_message}"
        )

    return (
        f"{action} returned HTTP "
        f"{response.status_code}"
    )


def get_board_labels() -> list[dict[str, Any]]:
    if not trello_board_is_configured():
        return []

    try:
        response = requests.get(
            (
                "https://api.trello.com/1/boards/"
                f"{TRELLO_BOARD_ID}/labels"
            ),
            params={
                "key": TRELLO_KEY,
                "token": TRELLO_TOKEN,
                "limit": 1000,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise TrelloIntegrationError(
            "Could not retrieve Trello labels "
            f"({exc.__class__.__name__})"
        ) from exc

    if not response.ok:
        raise TrelloIntegrationError(
            trello_error_message(
                response,
                "Trello label lookup",
            )
        )

    try:
        labels = response.json()
    except ValueError as exc:
        raise TrelloIntegrationError(
            "Trello label lookup returned invalid JSON"
        ) from exc

    if not isinstance(labels, list):
        return []

    return labels


def create_board_label(
    name: str,
    color: str,
) -> str:
    try:
        response = requests.post(
            "https://api.trello.com/1/labels",
            params={
                "key": TRELLO_KEY,
                "token": TRELLO_TOKEN,
                "idBoard": TRELLO_BOARD_ID,
                "name": name,
                "color": color,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise TrelloIntegrationError(
            "Could not create a Trello label "
            f"({exc.__class__.__name__})"
        ) from exc

    if not response.ok:
        raise TrelloIntegrationError(
            trello_error_message(
                response,
                "Trello label creation",
            )
        )

    try:
        label_data = response.json()
    except ValueError as exc:
        raise TrelloIntegrationError(
            "Trello label creation returned invalid JSON"
        ) from exc

    label_id = label_data.get("id")

    if not label_id:
        raise TrelloIntegrationError(
            "Trello did not return a label ID"
        )

    return str(label_id)


def get_or_create_label(
    current_labels: list[dict[str, Any]],
    name: str,
    color: str,
) -> str:
    for label in current_labels:
        existing_name = str(
            label.get("name", "")
        ).strip()

        if existing_name.casefold() == name.casefold():
            label_id = label.get("id")

            if label_id:
                return str(label_id)

    label_id = create_board_label(
        name,
        color,
    )

    current_labels.append(
        {
            "id": label_id,
            "name": name,
            "color": color,
        }
    )

    return label_id


def desired_trello_labels(
    payload: ReadinessSubmission,
    visual: dict[str, Any],
    assessment: dict[str, Any],
) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []

    if assessment["recommendation"] == "Ready":
        labels.append(
            ("Ready", "green")
        )
    else:
        labels.append(
            ("Needs Review", "orange")
        )

    connection_color = {
        "Ethernet": "blue",
        "Wi-Fi": "yellow",
        "Hotspot": "purple",
        "Other": "sky",
    }.get(
        payload.connection_type,
        "sky",
    )

    labels.append(
        (
            payload.connection_type,
            connection_color,
        )
    )

    checks = [
        (
            assessment["internet_pass"],
            "Internet Passed",
            "Internet Issue",
        ),
        (
            assessment["camera_pass"],
            "Camera Passed",
            "Camera Issue",
        ),
        (
            assessment["microphone_pass"],
            "Microphone Passed",
            "Microphone Issue",
        ),
        (
            assessment["speaker_pass"],
            "Speaker Passed",
            "Speaker Issue",
        ),
    ]

    if payload.snapshot_data_url:
        checks.append(
            (
                assessment["visual_pass"],
                "Visual Passed",
                "Visual Issue",
            )
        )

    for passed, pass_name, issue_name in checks:
        labels.append(
            (
                pass_name if passed else issue_name,
                "green" if passed else "red",
            )
        )

    if visual["backlighting_detected"]:
        labels.append(
            ("Backlighting", "red")
        )

    if visual["blur_detected"]:
        labels.append(
            ("Blur Warning", "red")
        )

    if visual["background_clutter_detected"]:
        labels.append(
            ("Background Warning", "yellow")
        )

    if visual["glasses_glare_warning"]:
        labels.append(
            ("Glare Warning", "yellow")
        )

    return labels


def get_trello_label_ids(
    payload: ReadinessSubmission,
    visual: dict[str, Any],
    assessment: dict[str, Any],
) -> list[str]:
    if not trello_board_is_configured():
        return []

    existing_labels = get_board_labels()
    label_ids: list[str] = []

    for name, color in desired_trello_labels(
        payload,
        visual,
        assessment,
    ):
        label_id = get_or_create_label(
            existing_labels,
            name,
            color,
        )

        if label_id not in label_ids:
            label_ids.append(label_id)

    return label_ids


def build_camera_section(
    payload: ReadinessSubmission,
    visual: dict[str, Any],
    camera_score: float,
) -> list[str]:
    """Combined Camera + visual-assessment block for the Trello card.

    Replaces the old separate "## Camera" and "## Visual assessment"
    headings with one "## Camera" section that also explains WHY the
    camera score came out the way it did. Face detected / face count /
    face centered / headroom acceptable are intentionally left out here —
    they're still in the Google Sheet, just not on the card.
    """
    deductions: list[str] = []

    if visual.get("blur_detected"):
        deductions.append("blurry/soft image")

    if visual.get("backlighting_detected"):
        deductions.append("backlighting")

    if visual.get("background_clutter_detected"):
        deductions.append("cluttered background")

    # NOTE: the dict key is "glasses_glare_warning" everywhere else in
    # this file (see analyze_snapshot / empty_visual_analysis) — using
    # "glasses_glare" here would silently never match.
    if visual.get("glasses_glare_warning"):
        deductions.append("glasses glare")

    if not visual.get("face_detected"):
        deductions.append("no face detected")
    elif not visual.get("face_centered"):
        deductions.append("face off-center")

    if visual.get("face_detected") and not visual.get("headroom_acceptable"):
        deductions.append("poor headroom")

    width = payload.camera_width
    height = payload.camera_height
    fps = payload.camera_fps

    if width and height and (width < 1280 or height < 720):
        deductions.append("resolution below 720p")

    if fps and fps < 24:
        deductions.append("low frame rate")

    if deductions:
        reason = "Score reduced by: " + ", ".join(deductions) + "."
    elif camera_score >= 90:
        reason = "No issues detected."
    else:
        reason = "No specific visual issues detected."

    resolution = f"{width} × {height}" if width and height else "Unknown"
    frame_rate = f"{fps:.1f} FPS" if fps else "Unknown"

    return [
        "## Camera",
        f"**Device:** {device_name(payload.camera_device)}",
        f"**Resolution:** {resolution}",
        f"**Frame rate:** {frame_rate}",
        f"**Camera score:** {camera_score:.0f}/100",
        f"**Why:** {reason}",
        "",
        f"**Backlighting detected:** {yes_no(visual.get('backlighting_detected'))}",
        f"**Blur detected:** {yes_no(visual.get('blur_detected'))}",
        f"**Background clutter detected:** {yes_no(visual.get('background_clutter_detected'))}",
        f"**Possible glasses glare:** {yes_no(visual.get('glasses_glare_warning'))}",
    ]


def trello_description(
    payload: ReadinessSubmission,
    submission_id: str,
    visual: dict[str, Any],
    assessment: dict[str, Any],
    booking: Optional[dict[str, str]] = None,
    booking_lookup_ok: bool = True,
) -> str:
    recommendation = str(
        assessment["recommendation"]
    )

    heading_icon = (
        "🟢"
        if recommendation == "Ready"
        else "🟠"
    )

    lines = [
        f"# {heading_icon} {recommendation.upper()}",
        "",
        (
            f"**Overall score:** "
            f"{float(assessment['overall_score']):.0f}/100 "
            f"({assessment['overall_grade']})"
        ),
        (
            f"**Overall quality:** "
            f"{score_grade(float(assessment['overall_score']))}"
        ),
        testing_on_intended_line(payload),
        filming_test_booking_line(booking, booking_lookup_ok),
        "",
        "## Readiness summary",
        (
            f"{status_icon(assessment['internet_pass'])} "
            f"Internet — "
            f"{float(assessment['internet_score']):.0f}/100"
        ),
        (
            f"{status_icon(assessment['camera_pass'])} "
            f"Camera — "
            f"{float(assessment['camera_score']):.0f}/100"
        ),
        (
            f"{status_icon(assessment['microphone_pass'])} "
            f"Microphone — "
            f"{float(assessment['microphone_score']):.0f}/100"
        ),
        (
            f"{status_icon(assessment['speaker_pass'])} "
            f"Speaker — "
            f"{float(assessment['speaker_score']):.0f}/100"
        ),
    ]

    if payload.snapshot_data_url:
        lines.extend(
            [
                (
                    f"{status_icon(assessment['visual_pass'])} "
                    f"Visual assessment — "
                    f"{float(visual['visual_score']):.0f}/100"
                ),
                (
                    f"{status_icon(float(visual['lighting_score']) >= MIN_LIGHTING_SCORE)} "
                    f"Lighting — "
                    f"{float(visual['lighting_score']):.0f}/100"
                ),
                (
                    f"{status_icon(float(visual['framing_score']) >= MIN_FRAMING_SCORE)} "
                    f"Framing — "
                    f"{float(visual['framing_score']):.0f}/100"
                ),
                (
                    f"{status_icon(not visual['blur_detected'])} "
                    f"Sharpness — "
                    f"{float(visual['blur_score']):.0f}/100"
                ),
                (
                    f"{status_icon(not visual['background_clutter_detected'])} "
                    f"Background — "
                    f"{float(visual['background_score']):.0f}/100"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Presenter",
            f"**Name:** {payload.presenter_name}",
            f"**Email:** {payload.email}",
            (
                f"**Phone number:** "
                f"{payload.phone_number or 'Not supplied'}"
            ),
            (
                f"**Title:** "
                f"{payload.presenter_title or 'Not supplied'}"
            ),
            (
                f"**Location:** "
                f"{payload.location or 'Not supplied'}"
            ),
            (
                f"**Lawline subscriber:** "
                f"{yes_no_field(payload.is_subscriber)}"
            ),
            (
                f"**Using virtual background:** "
                f"{yes_no_field(payload.using_virtual_background)}"
            ),
            "",
            "## Internet",
            (
                f"**Connection:** "
                f"{payload.connection_type}"
            ),
            (
                f"**Download:** "
                f"{payload.download_mbps:.2f} Mbps — "
                f"{assessment['download_grade']}"
            ),
            (
                f"**Upload:** "
                f"{payload.upload_mbps:.2f} Mbps — "
                f"{assessment['upload_grade']}"
            ),
            (
                f"**Latency:** "
                f"{payload.latency_ms:.0f} ms — "
                f"{assessment['latency_grade']}"
            ),
            "",
        ]
    )

    lines.extend(
        build_camera_section(
            payload,
            visual,
            float(assessment["camera_score"]),
        )
    )

    lines.extend(
        [
            "",
            "## Microphone and speaker",
            (
                f"**Microphone:** "
                f"{device_name(payload.microphone_device)}"
            ),
            (
                f"**Signal detected:** "
                f"{yes_no(payload.microphone_signal_detected)}"
            ),
            (
                f"{status_icon(bool(assessment['microphone_peak_in_range']))} "
                f"**Peak level:** "
                f"{peak_level_dbfs(payload.microphone_peak_level)} "
                f"(target: {MIC_PEAK_MIN_DBFS:.0f} to "
                f"{MIC_PEAK_MAX_DBFS:.0f} dBFS)"
            ),
            (
                f"**Speaker playback confirmed:** "
                f"{yes_no(payload.speaker_confirmed)}"
            ),
            "",
            "## Assessment",
            str(
                assessment[
                    "recommendation_detail"
                ]
            ),
            "",
            "## Fix recommendations",
        ]
    )

    for recommendation_item in assessment[
        "fix_recommendations"
    ]:
        lines.append(
            f"- {recommendation_item}"
        )

    lines.extend(
        [
            "",
            (
                f"**Notes:** "
                f"{payload.notes or 'None'}"
            ),
        ]
    )

    return "\n".join(lines)


def create_trello_card(
    payload: ReadinessSubmission,
    submission_id: str,
    visual: dict[str, Any],
    assessment: dict[str, Any],
    booking: Optional[dict[str, str]] = None,
    booking_lookup_ok: bool = True,
) -> tuple[str, str]:
    list_id = trello_list_id_for_result(
        str(assessment["recommendation"])
    )

    if not list_id:
        raise TrelloIntegrationError(
            "No Trello list ID is configured"
        )

    label_ids = get_trello_label_ids(
        payload,
        visual,
        assessment,
    )

    card_name = (
        f"[{assessment['recommendation']}] "
        f"{payload.presenter_name}"
        + (
            f" - {payload.presenter_title}"
            if payload.presenter_title
            else ""
        )
    )

    card_data: dict[str, Any] = {
        "idList": list_id,
        "name": card_name[:16384],
        "desc": trello_description(
            payload,
            submission_id,
            visual,
            assessment,
            booking=booking,
            booking_lookup_ok=booking_lookup_ok,
        ),
        "pos": "top",
    }

    if label_ids:
        card_data["idLabels"] = ",".join(
            label_ids
        )

    try:
        response = requests.post(
            "https://api.trello.com/1/cards",
            params={
                "key": TRELLO_KEY,
                "token": TRELLO_TOKEN,
            },
            json=card_data,
            headers={
                "Accept": "application/json",
            },
            timeout=25,
        )
    except requests.RequestException as exc:
        raise TrelloIntegrationError(
            "Trello card request failed "
            f"({exc.__class__.__name__})"
        ) from exc

    if not response.ok:
        raise TrelloIntegrationError(
            trello_error_message(
                response,
                "Trello card creation",
            )
        )

    try:
        card = response.json()
    except ValueError as exc:
        raise TrelloIntegrationError(
            "Trello card creation returned invalid JSON"
        ) from exc

    card_id = card.get("id")
    card_url = (
        card.get("url")
        or card.get("shortUrl")
    )

    if not card_id or not card_url:
        raise TrelloIntegrationError(
            "Trello did not return a card ID and URL"
        )

    return str(card_id), str(card_url)


def attach_snapshot_to_trello(
    card_id: str,
    image_bytes: bytes,
    filename: str,
) -> str:
    try:
        response = requests.post(
            (
                "https://api.trello.com/1/cards/"
                f"{card_id}/attachments"
            ),
            params={
                "key": TRELLO_KEY,
                "token": TRELLO_TOKEN,
                "name": filename,
                "setCover": "true",
            },
            files={
                "file": (
                    filename,
                    image_bytes,
                    "image/jpeg",
                ),
            },
            headers={
                "Accept": "application/json",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise TrelloIntegrationError(
            "Snapshot attachment request failed "
            f"({exc.__class__.__name__})"
        ) from exc

    if not response.ok:
        raise TrelloIntegrationError(
            trello_error_message(
                response,
                "Trello snapshot attachment",
            )
        )

    try:
        attachment = response.json()
    except ValueError as exc:
        raise TrelloIntegrationError(
            "Trello snapshot attachment returned invalid JSON"
        ) from exc

    return str(
        attachment.get("url")
        or attachment.get("previewUrl")
        or ""
    )


# =============================================================================
# Routes
# =============================================================================

@app.get(
    "/",
    include_in_schema=False,
)
def home():
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "presenter-readiness-check",
        "version": app.version,
        "opencv_version": cv2.__version__,
        "spreadsheet_configured": bool(
            SPREADSHEET_ID
        ),
        "trello_configured": (
            trello_is_configured()
        ),
    }


@app.post("/api/submissions")
def record_submission(
    payload: ReadinessSubmission,
):
    if payload.website:
        raise HTTPException(
            status_code=400,
            detail="Invalid submission",
        )

    if not payload.consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required",
        )

    if not SPREADSHEET_ID:
        raise HTTPException(
            status_code=500,
            detail=(
                "Server is missing "
                "GOOGLE_SPREADSHEET_ID"
            ),
        )

    submission_id = str(uuid.uuid4())

    snapshot_filename = safe_filename(
        payload.snapshot_filename
    )

    snapshot_bytes = b""
    visual = empty_visual_analysis()

    if payload.snapshot_data_url:
        try:
            snapshot_bytes, snapshot_image = (
                decode_snapshot(
                    payload.snapshot_data_url
                )
            )

            visual = analyze_snapshot(
                snapshot_image
            )

        except SnapshotAnalysisError as exc:
            logger.warning(
                "Snapshot analysis failed: %s",
                exc,
            )

            visual = empty_visual_analysis(
                str(exc)
            )

        except Exception:
            logger.exception(
                "Unexpected snapshot analysis failure"
            )

            visual = empty_visual_analysis(
                "The snapshot could not be analyzed."
            )
    else:
        visual = empty_visual_analysis(
            "No snapshot was supplied."
        )

    assessment = assess_submission(
        payload,
        visual,
    )

    # Best-effort: never let a calendar lookup failure block the actual
    # submission. booking_lookup_ok distinguishes "checked, not booked"
    # from "couldn't check" so the card never falsely reports "not
    # booked" when the real issue is a broken calendar share/auth.
    booking, booking_lookup_ok = find_filming_test_booking(str(payload.email))

    initial_status = (
        "Pending"
        if trello_is_configured()
        else "Trello not configured"
    )

    try:
        service = sheets_service()

        ensure_sheet_headers(service)

        row_number = append_sheet_row(
            service=service,
            payload=payload,
            submission_id=submission_id,
            visual=visual,
            assessment=assessment,
            snapshot_filename=snapshot_filename,
            integration_status=initial_status,
            booking=booking,
            booking_lookup_ok=booking_lookup_ok,
        )

    except Exception as exc:
        logger.exception(
            "Google Sheets write failed"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The result could not be saved "
                "to Google Sheets"
            ),
        ) from exc

    trello_url = ""
    snapshot_attachment_url = ""
    integration_status = initial_status

    if trello_is_configured():
        try:
            card_id, trello_url = create_trello_card(
                payload=payload,
                submission_id=submission_id,
                visual=visual,
                assessment=assessment,
                booking=booking,
                booking_lookup_ok=booking_lookup_ok,
            )

            integration_status = (
                "Trello card created"
            )

            if snapshot_bytes:
                try:
                    snapshot_attachment_url = (
                        attach_snapshot_to_trello(
                            card_id=card_id,
                            image_bytes=snapshot_bytes,
                            filename=snapshot_filename,
                        )
                    )

                    integration_status += (
                        "; snapshot attached"
                    )

                except TrelloIntegrationError as exc:
                    logger.warning(
                        "Trello snapshot attachment failed: %s",
                        exc,
                    )

                    integration_status += (
                        "; snapshot attachment failed"
                    )

        except TrelloIntegrationError as exc:
            logger.warning(
                "Trello card creation failed: %s",
                exc,
            )

            integration_status = (
                f"Trello error: {exc}"
            )

    if row_number is not None:
        try:
            update_integration_result(
                service=service,
                row_number=row_number,
                trello_url=trello_url,
                snapshot_url=(
                    snapshot_attachment_url
                ),
                integration_status=(
                    integration_status
                ),
            )

        except Exception:
            logger.exception(
                "Could not update the integration "
                "result in Google Sheets"
            )

            integration_status += (
                "; sheet integration-status update failed"
            )

    return {
        "ok": True,
        "submission_id": submission_id,

        "recommendation": assessment[
            "recommendation"
        ],

        "recommendation_detail": assessment[
            "recommendation_detail"
        ],

        "overall_score": round(
            float(assessment["overall_score"]),
            1,
        ),

        "overall_grade": assessment[
            "overall_grade"
        ],

        "fix_recommendations": assessment[
            "fix_recommendations"
        ],

        "scores": {
            "internet": round(
                float(
                    assessment["internet_score"]
                ),
                1,
            ),
            "camera": round(
                float(
                    assessment["camera_score"]
                ),
                1,
            ),
            "microphone": round(
                float(
                    assessment["microphone_score"]
                ),
                1,
            ),
            "speaker": round(
                float(
                    assessment["speaker_score"]
                ),
                1,
            ),
            "audio": round(
                float(
                    assessment["audio_score"]
                ),
                1,
            ),
            "lighting": round(
                float(
                    visual["lighting_score"]
                ),
                1,
            ),
            "framing": round(
                float(
                    visual["framing_score"]
                ),
                1,
            ),
            "background": round(
                float(
                    visual["background_score"]
                ),
                1,
            ),
            "environment": round(
                float(
                    visual["environment_score"]
                ),
                1,
            ),
            "visual": round(
                float(
                    visual["visual_score"]
                ),
                1,
            ),
        },

        "internet_grades": {
            "download": assessment[
                "download_grade"
            ],
            "upload": assessment[
                "upload_grade"
            ],
            "latency": assessment[
                "latency_grade"
            ],
            "jitter": assessment[
                "jitter_grade"
            ],
        },

        "checks": {
            "internet": assessment[
                "internet_pass"
            ],
            "camera": assessment[
                "camera_pass"
            ],
            "microphone": assessment[
                "microphone_pass"
            ],
            "speaker": assessment[
                "speaker_pass"
            ],
            "visual": assessment[
                "visual_pass"
            ],
            "face_detected": visual[
                "face_detected"
            ],
            "lighting": (
                float(
                    visual["lighting_score"]
                )
                >= MIN_LIGHTING_SCORE
                if payload.snapshot_data_url
                else None
            ),
            "framing": (
                float(
                    visual["framing_score"]
                )
                >= MIN_FRAMING_SCORE
                if payload.snapshot_data_url
                else None
            ),
            "sharpness": (
                not visual["blur_detected"]
                if payload.snapshot_data_url
                else None
            ),
            "background": (
                not visual[
                    "background_clutter_detected"
                ]
                if payload.snapshot_data_url
                else None
            ),
        },

        "visual_analysis": {
            "snapshot_received": bool(
                payload.snapshot_data_url
            ),
            "analysis_error": visual[
                "analysis_error"
            ],
            "face_count": visual[
                "face_count"
            ],
            "face_centered": visual[
                "face_centered"
            ],
            "face_too_close": visual[
                "face_too_close"
            ],
            "face_too_far": visual[
                "face_too_far"
            ],
            "headroom_acceptable": visual[
                "headroom_acceptable"
            ],
            "backlighting_detected": visual[
                "backlighting_detected"
            ],
            "blur_detected": visual[
                "blur_detected"
            ],
            "background_clutter_detected": visual[
                "background_clutter_detected"
            ],
            "glasses_glare_warning": visual[
                "glasses_glare_warning"
            ],
        },

        "trello_url": (
            trello_url or None
        ),

        "snapshot_attachment_url": (
            snapshot_attachment_url or None
        ),

        "integration_status": integration_status,
    }