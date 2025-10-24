# Mail Carrier Monitor

This folder contains a small Python utility that samples a UniFi Protect camera stream and feeds short clips to a Gemini video model. The model is prompted to identify when a mail carrier unlocks the associated door.

## Prerequisites

- Python 3.10+
- `ffmpeg` compiled with RTSP/RTSPS support
- Google Gemini API access with a key stored locally (set `GOOGLE_API_KEY`)
- Outbound network access from the machine to both your camera and the Gemini endpoint

## Setup

1. (Optional) Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export your Gemini API key:

   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

4. (Optional) Store credentials in a `.env` file so all scripts pick them up automatically:
   ```bash
   cat <<'EOF' > .env
   GOOGLE_API_KEY=your-gemini-key
   MONITOR_URL=https://monitor.ui.com/your-share-link
   UNIFI_CONTROLLER=https://your-controller.local
   UNIFI_DOOR_ID=your-door-id-here
   CAPTURE_ORDER=rtsp,browser
   RTSP_URL=rtsps://your-controller.local:7441/your-stream-path?enableSrtp
   RTSP_TRANSPORT=tcp
   UNIFI_USERNAME=your-username
   UNIFI_PASSWORD=your-password
   UNIFI_SITE_ID=default
   UNIFI_SKIP_TLS_VERIFY=true
   EOF
   ```
   The scripts load `.env` on startup, so you can omit these flags/env exports later.

## Quick commands

The most common commands, ready to copy:

```bash
# Install dependencies (once per environment)
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium

# Analyze a single RTSP clip
python3 stream_monitor.py --rtsp-url "rtsps://your-controller.local:7441/your-stream-path?enableSrtp" --once --keep-clips

# Record and analyze a UniFi monitor share
python3 browser_monitor.py --url "https://monitor.ui.com/your-share-link" --prep-seconds 30 --record-seconds 20 --keep-video

# List UniFi Access doors and unlock one for 8 seconds
python3 unifi_unlock.py --controller "https://your-controller.local" --list-doors
python3 unifi_unlock.py --controller "https://your-controller.local" --door-name "Front Door" --unlock-seconds 8

# Run the continuous monitor/unlocker loop (reads config from .env)
python3 automation_loop.py
```

## Running the monitor

Capture a single 12-second clip from the camera and send it to Gemini:

```bash
python stream_monitor.py \
  --rtsp-url "rtsps://your-controller.local:7441/your-stream-path?enableSrtp" \
  --once
```

The script prints whether a mail person and an unlocked door were detected, a confidence score, and Gemini's rationale. By default, clips are stored in a temporary directory and removed after analysis. Add `--keep-clips` to retain them for manual review.

Each cycle also prints a bare `True`/`False` value: `True` only when Gemini reports both a mail person and a door unlock; failures or negative detections emit `False`.

## Using a remote UniFi Monitor share

If you only have access to a UniFi Protect camera through `https://monitor.ui.com/...`, you can record the remote session in Chromium and analyze that clip:

1. Install Playwright dependencies (one-time):

   ```bash
   pip install -r requirements.txt
   python3 -m playwright install chromium
   ```

2. Record and analyze a session:

   ```bash
   GOOGLE_API_KEY="your-key" python browser_monitor.py \
     --url "https://monitor.ui.com/your-share-link" \
     --prep-seconds 30 \
     --record-seconds 20 \
     --keep-video
   ```

   The script launches Chromium (visible by default), reloads the page once to kick off the feed, gives you `--prep-seconds` to authenticate or press play, then records the stream for `--record-seconds` and forwards the resulting clip to Gemini. Use `--no-auto-reload` if the extra refresh causes issues. The captured WebM is re-encoded to MP4 (H.264) before upload so Gemini accepts it.

The program prints a final `True`/`False` line indicating whether Gemini saw both a mail carrier and an unlocked door; if analysis fails, the output is `False`. As above, the default Gemini model is `gemini-2.5-flash`; pass `--model-name` to match the models available to your Gemini project.

## Continuous automation loop

`automation_loop.py` keeps Chromium headless, captures 5-second clips every 5 seconds, runs them through Gemini, and, when both a mail carrier and an unlocked door are detected, triggers a UniFi Access unlock (and saves the clip under `events/`). Configuration is controlled entirely through environment variables (the `.env` example above covers the required values). Run it with:

```bash
python3 automation_loop.py
```

Optional variables include:

- `RECORD_SECONDS` / `POLL_INTERVAL` (defaults 5 / 5)
- `CAPTURE_ORDER` (comma-separated modes, e.g. `rtsp,browser` to try RTSP first with a monitor fallback)
- `RTSP_URL`, `RTSP_TRANSPORT` (required if `rtsp` appears in `CAPTURE_ORDER`; transport defaults to `tcp`)
- `GEMINI_MODEL`, `GEMINI_TIMEOUT`, `FFMPEG_PATH`
- `UNLOCK_COOLDOWN_SECONDS` (default 20 to avoid spamming the door)
- `EVENT_CLIP_DIR` (defaults to `events`)
- `SAVE_ALL_CLIPS_DIR` (optional folder to save every processed clip for debugging; otherwise only events are kept)
- `UNIFI_SKIP_TLS_VERIFY=true` to ignore self-signed controller certs

The loop runs headless; logs surface detections, unlock attempts, and saved clips.

## Unlocking a UniFi Access door

Use `unifi_unlock.py` to authenticate with your UniFi console (UDM/CloudKey) and send an unlock command to a door controller that the camera monitors. This requires UniFi Access to be enabled on the controller.

```bash
python3 unifi_unlock.py \
  --controller "https://your-controller.local" \
  --username "$UNIFI_USERNAME" \
  --password "$UNIFI_PASSWORD" \
  --door-name "Front Door" \
  --unlock-seconds 8
```

- Point `--controller` to the console URL or IP; use `--insecure` if it has a self-signed certificate.
- Pass either `--door-name` (resolved via `/proxy/ui/access/api/v1/doors`) or `--door-id`. Add `--list-doors` to print the available door IDs first. If the console only exposes the Access v2 APIs, the script falls back to retrieving metadata from `/proxy/access/api/v2/access_policies/by_door/<id>` so a door ID is sufficient even when listing is not available.
- The script prints `Door unlock command issued successfully.` when the API call succeeds; otherwise it reports the HTTP error returned by the controller.
- For automation, export `UNIFI_USERNAME`, `UNIFI_PASSWORD`, and optionally `UNIFI_SITE_ID` so they do not need to be provided on the command line.
- Authentication happens through the UniFi Console’s REST endpoints (`POST /api/auth/login`). A successful login sets the session cookies (and CSRF token when present), which the script then reuses for subsequent calls, including the door listing and the unlock request. If your console uses the newer Access v2 API, unlocks happen via `PUT /proxy/access/api/v2/location/<id>/unlock` and respect the controller’s default unlock duration (custom durations are not exposed in that endpoint).

### Continuous monitoring

Run in a loop, capturing 12-second clips every 20 seconds:

```bash
python stream_monitor.py \
  --rtsp-url "rtsps://your-controller.local:7441/your-stream-path?enableSrtp" \
  --segment-seconds 12 \
  --poll-interval 20
```

Adjust the segment length, poll interval, prompt, or Gemini model with the corresponding CLI arguments (`python stream_monitor.py --help`). The default model is `gemini-2.5-flash`; override `--model-name` if your account exposes different names.

## Integrating with UniFi access events

- The script currently infers delivery presence purely from visual cues in the clip. If you track access events from UniFi Protect/Access, you can merge those events by calling the script with `--once` when an unlock is detected, or by augmenting the script to push its conclusions to your automation pipeline.
- For real-time alerts, wrap the script with a small supervisor that posts to Slack, email, or Home Assistant when `mail_person_detected` is `True` and the confidence exceeds your threshold.

## Troubleshooting

- Use `--ffmpeg-path` if `ffmpeg` is not on your `PATH`.
- Increase `--gemini-timeout` for larger clips or slower uploads.
- Run with `export GOOGLE_API_KEY=...` before launching; the script fails fast if the key is missing.
- To inspect raw JSON from Gemini, bump the log level by exporting `PYTHONLOGLEVEL=DEBUG` or editing the script to suit your logging preferences.
# unifi-camera-ai
