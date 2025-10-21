#!/usr/bin/env python3
"""Continuous UniFi monitor watcher that unlocks a door when Gemini detects mail delivery."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from playwright.async_api import Error as PlaywrightError, async_playwright

from stream_monitor import DEFAULT_PROMPT, analyze_clip, ensure_api_key
from unifi_unlock import Door, build_session, login, unlock_door

LOGGER = logging.getLogger("automation_loop")


@dataclass
class MonitorConfig:
    monitor_url: Optional[str]
    controller_url: str
    door_id: str
    door_name: Optional[str]
    site_id: str
    record_seconds: int
    poll_interval: float
    prep_seconds: float
    model_name: str
    prompt: str
    gemini_timeout: int
    ffmpeg_path: str
    viewport_width: int
    viewport_height: int
    auto_reload: bool
    unlock_cooldown: float
    clip_dir: Path
    verify_tls: bool
    chromium_path: Optional[str]
    wait_until: str
    capture_modes: list[str]
    rtsp_url: Optional[str]
    rtsp_transport: str
    save_all_dir: Optional[Path]


def load_bool(env_name: str, default: bool = False) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> MonitorConfig:
    monitor_url = os.environ.get("MONITOR_URL")
    controller_url = os.environ.get("UNIFI_CONTROLLER")
    door_id = os.environ.get("UNIFI_DOOR_ID")
    if not controller_url or not door_id:
        raise ValueError("UNIFI_CONTROLLER and UNIFI_DOOR_ID environment variables are required.")

    door_name = os.environ.get("UNIFI_DOOR_NAME")
    site_id = os.environ.get("UNIFI_SITE_ID", "default")

    record_seconds = int(os.environ.get("RECORD_SECONDS", "15"))
    poll_interval = float(os.environ.get("POLL_INTERVAL", "15"))
    prep_seconds = float(os.environ.get("PREP_SECONDS", "5"))

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = os.environ.get("GEMINI_PROMPT", DEFAULT_PROMPT)
    gemini_timeout = int(os.environ.get("GEMINI_TIMEOUT", "180"))
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")

    viewport_width = int(os.environ.get("VIEWPORT_WIDTH", "1280"))
    viewport_height = int(os.environ.get("VIEWPORT_HEIGHT", "720"))
    auto_reload = load_bool("AUTO_RELOAD", True)
    unlock_cooldown = float(os.environ.get("UNLOCK_COOLDOWN_SECONDS", "20"))
    clip_dir = Path(os.environ.get("EVENT_CLIP_DIR", "events")).resolve()
    verify_tls = not load_bool("UNIFI_SKIP_TLS_VERIFY", True)
    chromium_path = os.environ.get("CHROMIUM_PATH")
    wait_until = os.environ.get("PLAYWRIGHT_WAIT_UNTIL", "domcontentloaded")
    capture_order = os.environ.get("CAPTURE_ORDER", "browser")
    capture_modes = [mode.strip().lower() for mode in capture_order.split(",") if mode.strip()]
    if not capture_modes:
        raise ValueError("CAPTURE_ORDER must specify at least one capture mode (browser or rtsp).")

    rtsp_url = os.environ.get("RTSP_URL")
    rtsp_transport = os.environ.get("RTSP_TRANSPORT", "tcp")
    save_all_dir_env = os.environ.get("SAVE_ALL_CLIPS_DIR")
    save_all_dir = Path(save_all_dir_env).resolve() if save_all_dir_env else None

    if "browser" in capture_modes and not monitor_url:
        raise ValueError("MONITOR_URL environment variable is required for browser capture mode.")
    if "rtsp" in capture_modes and not rtsp_url:
        raise ValueError("RTSP_URL environment variable is required for rtsp capture mode.")

    return MonitorConfig(
        monitor_url=monitor_url,
        controller_url=controller_url,
        door_id=door_id,
        door_name=door_name,
        site_id=site_id,
        record_seconds=record_seconds,
        poll_interval=poll_interval,
        prep_seconds=prep_seconds,
        model_name=model_name,
        prompt=prompt,
        gemini_timeout=gemini_timeout,
        ffmpeg_path=ffmpeg_path,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        auto_reload=auto_reload,
        unlock_cooldown=unlock_cooldown,
        clip_dir=clip_dir,
        verify_tls=verify_tls,
        chromium_path=chromium_path,
        wait_until=wait_until,
        capture_modes=capture_modes,
        rtsp_url=rtsp_url,
        rtsp_transport=rtsp_transport,
        save_all_dir=save_all_dir,
    )


def configure_logging() -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


async def _prepare_stream(page, cfg: MonitorConfig) -> None:
    max_attempts = 4
    for attempt in range(max_attempts):
        await page.wait_for_timeout(int(cfg.prep_seconds * 1000))
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)
        except PlaywrightError:
            pass

        error_locator = page.locator("text=Unable to stream")
        if await error_locator.count() > 0 and await error_locator.first.is_visible():
            LOGGER.warning("Monitor shows 'Unable to stream'; refreshing (attempt %s).", attempt + 1)
            await page.reload(wait_until=cfg.wait_until)
            continue

        has_video = await page.evaluate("document.querySelector('video') !== null")
        if has_video:
            LOGGER.debug("Video element detected on monitor page.")
            return

        LOGGER.warning("No video element detected; reloading monitor (attempt %s).", attempt + 1)
        await page.reload(wait_until=cfg.wait_until)

    raise RuntimeError("Unable to prepare monitor stream after multiple retries.")


async def capture_clip_browser(
    browser, cfg: MonitorConfig, temp_root: Path
) -> Path:
    context = await browser.new_context(
        record_video_dir=str(temp_root),
        record_video_size={"width": cfg.viewport_width, "height": cfg.viewport_height},
        viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
    )
    page = await context.new_page()
    LOGGER.debug("Opening monitor URL: %s", cfg.monitor_url)
    await page.goto(cfg.monitor_url, wait_until=cfg.wait_until)
    if cfg.auto_reload:
        LOGGER.debug("Reloading monitor page to ensure stream playback.")
        await page.reload(wait_until=cfg.wait_until)

    await _prepare_stream(page, cfg)

    LOGGER.debug("Recording for %ss.", cfg.record_seconds)
    await page.wait_for_timeout(cfg.record_seconds * 1000)

    video_recorder = page.video
    if video_recorder is None:
        await context.close()
        raise RuntimeError("Playwright failed to provide a video recording.")

    await page.close()
    video_path = Path(await video_recorder.path())
    await context.close()
    LOGGER.debug("Captured clip to %s", video_path)
    return video_path


def capture_clip_rtsp(cfg: MonitorConfig, temp_root: Path) -> Path:
    if not cfg.rtsp_url:
        raise RuntimeError("RTSP_URL is not configured.")
    output = temp_root / "clip.mp4"

    cmd = [
        cfg.ffmpeg_path,
        "-y",
        "-rtsp_transport",
        cfg.rtsp_transport,
        "-i",
        cfg.rtsp_url,
        "-t",
        str(cfg.record_seconds),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(output),
    ]

    import subprocess

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore")
        LOGGER.error("ffmpeg RTSP capture failed: %s", stderr)
        raise RuntimeError("Failed to capture clip from RTSP stream.")
    return output


def convert_to_mp4(src: Path, ffmpeg_path: str) -> Path:
    if src.suffix.lower() == ".mp4":
        return src

    target = src.with_suffix(".mp4")
    LOGGER.debug("Converting %s to %s", src.name, target.name)

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


def process_clip(
    mp4_clip: Path,
    gemini_cfg,
    cfg: MonitorConfig,
    session,
    door: Door,
    last_unlock_ts: float,
) -> float:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result = None
    detected = False
    try:
        result = analyze_clip(gemini_cfg, mp4_clip)
        if result:
            LOGGER.info(
                "Gemini: mail=%s door=%s confidence=%.2f reason=%s",
                result.mail_person_detected,
                result.door_unlocked,
                result.confidence,
                result.rationale,
            )
            detected = bool(result.mail_person_detected and result.door_unlocked)
        else:
            LOGGER.info("Gemini returned no structured result.")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Gemini analysis failed: %s", exc)

    should_unlock = False
    if detected:
        elapsed = time.time() - last_unlock_ts
        if elapsed >= cfg.unlock_cooldown:
            should_unlock = True
        else:
            LOGGER.info(
                "Detection within cooldown (%.1fs remaining); skipping unlock.",
                cfg.unlock_cooldown - elapsed,
            )

    if should_unlock:
        try:
            unlock_door(session, cfg.controller_url, door, cfg.record_seconds)
            last_unlock_ts = time.time()
            LOGGER.info("Door unlock command sent.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Door unlock failed: %s", exc)
    else:
        # No unlock; clip only saved if we triggered action
        pass

    if should_unlock:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dest = cfg.clip_dir / f"event_{timestamp}.mp4"
        try:
            mp4_clip.replace(dest)
            LOGGER.info("Saved event clip to %s", dest)
            mp4_clip = dest
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to save event clip: %s", exc)
        final_path = mp4_clip
    else:
        final_path = mp4_clip

    if cfg.save_all_dir and final_path.exists():
        cfg.save_all_dir.mkdir(parents=True, exist_ok=True)
        debug_dest = cfg.save_all_dir / f"clip_{timestamp}.mp4"
        try:
            shutil.copy2(final_path, debug_dest)
            LOGGER.debug("Saved debug clip to %s", debug_dest)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to copy clip to %s: %s", debug_dest, exc)

    if not should_unlock:
        if final_path.exists():
            try:
                final_path.unlink()
            except FileNotFoundError:
                pass

    return last_unlock_ts


async def browser_capture_loop(
    cfg: MonitorConfig,
    session,
    door: Door,
    gemini_cfg,
    state: SimpleNamespace,
) -> None:
    async with async_playwright() as playwright:
        launch_kwargs = {"headless": True}
        if cfg.chromium_path:
            launch_kwargs["executable_path"] = cfg.chromium_path

        try:
            browser = await playwright.chromium.launch(**launch_kwargs)
        except PlaywrightError as exc:
            LOGGER.error("Unable to launch Chromium: %s", exc)
            raise

        try:
            while True:
                loop_start = time.time()
                with tempfile.TemporaryDirectory(prefix="monitor-clip-") as tmpdir:
                    temp_root = Path(tmpdir)
                    mp4_clip: Optional[Path] = None
                    try:
                        raw_clip = await capture_clip_browser(browser, cfg, temp_root)
                        mp4_clip = convert_to_mp4(raw_clip, cfg.ffmpeg_path)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.exception("Browser capture failed: %s", exc)
                        continue

                    state.last_unlock_ts = process_clip(
                        mp4_clip, gemini_cfg, cfg, session, door, state.last_unlock_ts
                    )

                elapsed_total = time.time() - loop_start
                sleep_for = max(cfg.poll_interval - elapsed_total, 0)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        finally:
            await browser.close()


async def rtsp_capture_loop(
    cfg: MonitorConfig,
    session,
    door: Door,
    gemini_cfg,
    state: SimpleNamespace,
) -> None:
    while True:
        loop_start = time.time()
        with tempfile.TemporaryDirectory(prefix="rtsp-clip-") as tmpdir:
            temp_root = Path(tmpdir)
            try:
                raw_clip = await asyncio.to_thread(capture_clip_rtsp, cfg, temp_root)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("RTSP capture failed: %s", exc)
                await asyncio.sleep(cfg.poll_interval)
                continue

            mp4_clip = convert_to_mp4(raw_clip, cfg.ffmpeg_path)

            state.last_unlock_ts = process_clip(
                mp4_clip, gemini_cfg, cfg, session, door, state.last_unlock_ts
            )

        elapsed_total = time.time() - loop_start
        sleep_for = max(cfg.poll_interval - elapsed_total, 0)
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


def build_door(cfg: MonitorConfig) -> Door:
    return Door(
        door_id=cfg.door_id,
        name=cfg.door_name or cfg.door_id,
        site_id=cfg.site_id,
        device_mac=None,
        status=None,
        unlock_path=f"/proxy/access/api/v2/location/{cfg.door_id}/unlock",
        unlock_method="PUT",
        supports_duration=False,
    )


async def automation_loop(cfg: MonitorConfig) -> None:
    ensure_api_key()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    cfg.clip_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(verify_tls=cfg.verify_tls)
    try:
        login(session, cfg.controller_url, os.environ["UNIFI_USERNAME"], os.environ["UNIFI_PASSWORD"])
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to authenticate with UniFi controller: %s", exc)
        raise

    door = build_door(cfg)
    state = SimpleNamespace(last_unlock_ts=0.0)

    gemini_cfg = SimpleNamespace(
        model_name=cfg.model_name,
        prompt=cfg.prompt or DEFAULT_PROMPT,
        gemini_timeout=cfg.gemini_timeout,
    )

    for mode in cfg.capture_modes:
        LOGGER.info("Starting capture mode: %s", mode)
        try:
            if mode == "browser":
                await browser_capture_loop(cfg, session, door, gemini_cfg, state)
            elif mode == "rtsp":
                await rtsp_capture_loop(cfg, session, door, gemini_cfg, state)
            else:
                LOGGER.warning("Unknown capture mode '%s'; skipping.", mode)
                continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Capture mode %s failed: %s", mode, exc)
            continue
        # the chosen capture loop only exits on exception or cancellation
        break
    else:
        raise RuntimeError("All capture modes failed to start.")


def main() -> None:
    load_dotenv()
    configure_logging()

    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Configuration error: %s", exc)
        sys.exit(1)

    if not cfg.verify_tls:
        try:
            import urllib3

            urllib3.disable_warnings()
        except Exception:  # noqa: BLE001
            pass

    required_creds = ("GOOGLE_API_KEY", "UNIFI_USERNAME", "UNIFI_PASSWORD")
    missing = [key for key in required_creds if not os.environ.get(key)]
    if missing:
        LOGGER.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)

    try:
        loop.run_until_complete(automation_loop(cfg))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()
