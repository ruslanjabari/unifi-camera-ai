#!/usr/bin/env python3
"""Capture RTSP segments and send them to Gemini for mail-person detection."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

LOGGER = logging.getLogger("stream_monitor")

DEFAULT_PROMPT = (
    "You are monitoring security footage for package and mail delivery events. "
    "Explain whether the clip shows a postal worker or delivery courier who is trying to enter the building or deliver an item. "
    "Respond strictly in JSON with fields: "
    "'mail_person_detected' (boolean), "
    "'confidence' (float between 0 and 1), "
    "and 'rationale' (short string)."
)


@dataclass
class MonitorConfig:
    rtsp_url: str
    segment_seconds: int
    poll_interval: float
    model_name: str
    prompt: str
    keep_clips: bool
    once: bool
    gemini_timeout: int
    ffmpeg_path: str = "ffmpeg"


@dataclass
class MonitorResult:
    mail_person_detected: bool
    confidence: float
    rationale: str
    raw_response: str


def parse_args() -> MonitorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtsp-url", required=True, help="Camera RTSP(S) URL")
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=12,
        help="Length of each captured clip before analysis",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=20.0,
        help="Seconds to wait between clip captures (may overlap with segment length)",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-2.5-flash",
        help="Gemini model capable of video understanding",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Analysis instructions for the model",
    )
    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help="Persist captured clips to disk for later review",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single capture/analysis cycle and exit",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=int,
        default=180,
        help="Seconds to wait for Gemini API responses",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary if not on PATH",
    )
    args = parser.parse_args()
    prompt = args.prompt or DEFAULT_PROMPT
    return MonitorConfig(
        rtsp_url=args.rtsp_url,
        segment_seconds=args.segment_seconds,
        poll_interval=args.poll_interval,
        model_name=args.model_name,
        prompt=prompt,
        keep_clips=args.keep_clips,
        once=args.once,
        gemini_timeout=args.gemini_timeout,
        ffmpeg_path=args.ffmpeg_path,
    )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def ensure_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is required for Gemini access."
        )
    return api_key


def capture_clip(cfg: MonitorConfig, output_path: Path) -> None:
    cmd = [
        cfg.ffmpeg_path,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        cfg.rtsp_url,
        "-t",
        str(cfg.segment_seconds),
        "-c:v",
        "copy",
        "-an",
        str(output_path),
    ]
    LOGGER.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore")
        LOGGER.error("ffmpeg failed: %s", stderr)
        raise RuntimeError("Failed to capture clip from RTSP stream") from exc


def analyze_clip(cfg: MonitorConfig, clip_path: Path) -> MonitorResult:
    LOGGER.info("Preparing %s for analysis", clip_path.name)
    video_bytes = clip_path.read_bytes()
    contents = [
        {
            "role": "user",
            "parts": [
                {"mime_type": "video/mp4", "data": video_bytes},
                cfg.prompt,
            ],
        }
    ]

    def candidate_models(name: str):
        yield name
        if name.startswith("models/"):
            yield name.split("/", 1)[1]
        else:
            yield f"models/{name}"

    last_exc: Exception | None = None
    response = None
    for model_name in dict.fromkeys(candidate_models(cfg.model_name)):
        try:
            LOGGER.info("Requesting analysis from %s", model_name)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": cfg.gemini_timeout},
            )
            break
        except google_exceptions.NotFound as exc:
            LOGGER.warning("Model %s unavailable: %s", model_name, exc)
            last_exc = exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            break

    if response is None:
        if last_exc:
            raise RuntimeError("Gemini analysis failed for all candidate models") from last_exc
        raise RuntimeError("Gemini analysis failed with unknown error")
    raw_text = response.text.strip()
    LOGGER.debug("Raw model response: %s", raw_text)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to decode JSON response: %s", raw_text)
        raise RuntimeError("Gemini returned non-JSON response") from exc

    def _bool_field(name: str) -> bool:
        value = payload.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Unexpected boolean field {name}: {value}")

    mail_person = _bool_field("mail_person_detected")

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    rationale = str(payload.get("rationale", ""))
    return MonitorResult(
        mail_person_detected=mail_person,
        confidence=confidence,
        rationale=rationale,
        raw_response=raw_text,
    )


def signal_handler(signum, frame):  # type: ignore[override]
    LOGGER.info("Received signal %s, shutting down.", signum)
    sys.exit(0)


def main() -> None:
    configure_logging()
    load_dotenv()
    cfg = parse_args()
    ensure_api_key()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    LOGGER.info("Starting monitor; capturing %ss clips.", cfg.segment_seconds)
    with tempfile.TemporaryDirectory(prefix="mail-monitor-") as tmpdir:
        temp_root = Path(tmpdir)

        while True:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            clip_path = temp_root / f"clip_{timestamp}.mp4"
            capture_clip(cfg, clip_path)

            result: Optional[MonitorResult] = None
            try:
                result = analyze_clip(cfg, clip_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Analysis failed: %s", exc)

            detected = bool(result and result.mail_person_detected)

            if result:
                LOGGER.info(
                    "Mail person detected: %s | confidence=%.2f | %s",
                    result.mail_person_detected,
                    result.confidence,
                    result.rationale,
                )
            else:
                LOGGER.info("No structured result returned; defaulting to False.")

            print(detected)

            if cfg.keep_clips:
                destination = Path.cwd() / clip_path.name
                clip_path.replace(destination)
                LOGGER.debug("Saved clip to %s", destination)

            if cfg.once:
                break

            delay = max(cfg.poll_interval, 0.0)
            LOGGER.debug("Sleeping for %.1fs before next capture.", delay)
            time.sleep(delay)


if __name__ == "__main__":
    main()
