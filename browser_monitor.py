#!/usr/bin/env python3
"""Record a remote Protect web session and ask Gemini to detect mail delivery events."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import google.generativeai as genai
from playwright.async_api import Error as PlaywrightError, async_playwright
from dotenv import load_dotenv

from stream_monitor import (
    DEFAULT_PROMPT,
    LOGGER as BASE_LOGGER,
    MonitorResult,
    analyze_clip,
    ensure_api_key,
)

LOGGER = logging.getLogger("browser_monitor")


@dataclass
class BrowserConfig:
    url: str
    record_seconds: int
    prep_seconds: int
    headless: bool
    auto_reload: bool
    model_name: str
    prompt: str
    gemini_timeout: int
    keep_video: bool
    ffmpeg_path: str
    chromium_path: str | None
    wait_until: str
    viewport_width: int
    viewport_height: int


def parse_args() -> BrowserConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(auto_reload=True)
    parser.add_argument("--url", required=True, help="Public UniFi monitor share link")
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=15,
        help="Duration to record once the page is ready.",
    )
    parser.add_argument(
        "--prep-seconds",
        type=int,
        default=20,
        help="Time to give yourself to log in / start playback before recording.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless (default shows the browser window).",
    )
    parser.add_argument(
        "--no-auto-reload",
        dest="auto_reload",
        action="store_false",
        help="Skip automatic page reload after first load.",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-2.5-flash",
        help="Gemini model capable of video understanding.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instructions for the model; defaults to mail carrier detection.",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=int,
        default=180,
        help="Seconds to wait for Gemini responses and uploads.",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Persist the captured clip in the current directory.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to ffmpeg binary for format conversion.",
    )
    parser.add_argument(
        "--chromium-path",
        default=None,
        help="Custom Chromium/Chrome executable if needed.",
    )
    parser.add_argument(
        "--wait-until",
        default="domcontentloaded",
        choices=["domcontentloaded", "load", "networkidle", "commit"],
        help="Playwright wait condition before the prep timer starts.",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=1280,
        help="Viewport width for the recording context.",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=720,
        help="Viewport height for the recording context.",
    )
    args = parser.parse_args()
    prompt = args.prompt or DEFAULT_PROMPT
    return BrowserConfig(
        url=args.url,
        record_seconds=args.record_seconds,
        prep_seconds=args.prep_seconds,
        headless=args.headless,
        auto_reload=args.auto_reload,
        model_name=args.model_name,
        prompt=prompt,
        gemini_timeout=args.gemini_timeout,
        keep_video=args.keep_video,
        ffmpeg_path=args.ffmpeg_path,
        chromium_path=args.chromium_path,
        wait_until=args.wait_until,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
    )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    BASE_LOGGER.addHandler(handler)
    BASE_LOGGER.setLevel(logging.INFO)


async def capture_with_playwright(cfg: BrowserConfig, temp_root: Path) -> Path:
    async with async_playwright() as playwright:
        launch_kwargs = {"headless": cfg.headless}
        if cfg.chromium_path:
            launch_kwargs["executable_path"] = cfg.chromium_path

        LOGGER.info("Launching Chromium (headless=%s).", cfg.headless)
        try:
            browser = await playwright.chromium.launch(**launch_kwargs)
        except PlaywrightError as exc:
            if "Executable doesn't exist" in str(exc):
                raise RuntimeError(
                    "Playwright Chromium binary is missing. Run "
                    "`python3 -m playwright install chromium` with the same interpreter."
                ) from exc
            raise
        context = await browser.new_context(
            record_video_dir=str(temp_root),
            record_video_size={"width": cfg.viewport_width, "height": cfg.viewport_height},
            viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
        )
        page = await context.new_page()
        LOGGER.info("Opening %s", cfg.url)
        await page.goto(cfg.url, wait_until=cfg.wait_until)
        if cfg.auto_reload:
            LOGGER.info("Reloading page to ensure stream starts.")
            await page.reload(wait_until=cfg.wait_until)

        if not cfg.headless:
            LOGGER.info(
                "You have %s seconds to complete login/start playback before recording.",
                cfg.prep_seconds,
            )
        if cfg.prep_seconds > 0:
            await page.wait_for_timeout(cfg.prep_seconds * 1000)

        LOGGER.info("Recording for %s seconds...", cfg.record_seconds)
        await page.wait_for_timeout(cfg.record_seconds * 1000)

        video_recorder = page.video
        if video_recorder is None:
            raise RuntimeError("Playwright did not produce a video recording handle.")

        await page.close()
        video_path = Path(await video_recorder.path())

        await context.close()
        await browser.close()

        LOGGER.info("Captured session to %s", video_path)
        return video_path


def convert_to_mp4(src: Path, ffmpeg_path: str) -> Path:
    if src.suffix.lower() == ".mp4":
        return src

    target = src.with_suffix(".mp4")
    LOGGER.info("Converting %s to %s", src.name, target.name)

    import subprocess

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(target),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore")
        LOGGER.error("ffmpeg conversion failed: %s", stderr)
        raise RuntimeError("Failed to convert recording to MP4.")

    return target


async def run_once(cfg: BrowserConfig) -> None:
    ensure_api_key()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    with tempfile.TemporaryDirectory(prefix="browser-monitor-") as tmpdir:
        temp_root = Path(tmpdir)
        raw_clip = await capture_with_playwright(cfg, temp_root)
        mp4_clip = convert_to_mp4(raw_clip, cfg.ffmpeg_path)

        gemini_cfg = SimpleNamespace(
            model_name=cfg.model_name,
            prompt=cfg.prompt,
            gemini_timeout=cfg.gemini_timeout,
        )
        result: MonitorResult | None = None
        try:
            result = analyze_clip(gemini_cfg, mp4_clip)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Gemini analysis failed: %s", exc)

        detected = bool(
            result and result.mail_person_detected and result.door_unlocked
        )
        if result:
            LOGGER.info(
                "Mail person: %s | Door unlocked: %s | confidence=%.2f | %s",
                result.mail_person_detected,
                result.door_unlocked,
                result.confidence,
                result.rationale,
            )
        else:
            LOGGER.info("No structured result returned; defaulting to False.")

        if cfg.keep_video:
            destination = Path.cwd() / mp4_clip.name
            mp4_clip.replace(destination)
            LOGGER.info("Saved recording to %s", destination)

        print(detected)


def signal_handler(signum, frame):  # type: ignore[override]
    LOGGER.info("Received signal %s, exiting.", signum)
    sys.exit(0)


def main() -> None:
    configure_logging()
    load_dotenv()
    cfg = parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(run_once(cfg))


if __name__ == "__main__":
    main()
