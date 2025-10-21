#!/usr/bin/env python3
"""Authenticate with a UniFi console and trigger a UniFi Access door unlock."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger("unifi_unlock")

DEFAULT_SITE = "default"
DEFAULT_UNLOCK_SECONDS = 5


@dataclass
class Door:
    door_id: str
    name: str
    site_id: str
    device_mac: Optional[str]
    status: Optional[str]
    unlock_path: str
    unlock_method: str = "POST"
    supports_duration: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        required=True,
        help="Base URL to the UniFi Console (e.g. https://192.168.1.1 or https://unifi.local)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("UNIFI_USERNAME"),
        help="UniFi username (defaults to UNIFI_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("UNIFI_PASSWORD"),
        help="UniFi password (defaults to UNIFI_PASSWORD env var)",
    )
    parser.add_argument(
        "--site-id",
        default=os.environ.get("UNIFI_SITE_ID", DEFAULT_SITE),
        help=f"Target site ID (default: {DEFAULT_SITE})",
    )
    parser.add_argument(
        "--door-id",
        help="Door ID to unlock (skips lookup). Overrides --door-name.",
    )
    parser.add_argument(
        "--door-name",
        help="Human-friendly door name; resolved via /proxy/ui/access/api/v1/doors.",
    )
    parser.add_argument(
        "--unlock-seconds",
        type=int,
        default=DEFAULT_UNLOCK_SECONDS,
        help=f"How long to unlock the door (seconds). Default: {DEFAULT_UNLOCK_SECONDS}",
    )
    parser.add_argument(
        "--list-doors",
        action="store_true",
        help="List known doors for the specified site instead of unlocking.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (useful for self-signed controller certs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and dump HTTP errors.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if debug else logging.INFO)


def normalize_base_url(raw: str) -> str:
    parsed = urlparse(raw)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    normalized = urlunparse((scheme, netloc, path.rstrip("/"), "", "", ""))
    return normalized.rstrip("/")


def build_session(verify_tls: bool) -> requests.Session:
    session = requests.Session()
    session.verify = verify_tls
    session.headers.update({"Accept": "application/json"})
    return session


def login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    login_url = f"{base_url}/api/auth/login"
    LOGGER.debug("Logging in at %s", login_url)
    payload = {"username": username, "password": password, "rememberMe": True}
    resp = session.post(login_url, json=payload, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text}")
    csrf_token = resp.headers.get("x-csrf-token")
    if csrf_token:
        session.headers["x-csrf-token"] = csrf_token
        LOGGER.debug("Captured CSRF token.")


def _parse_json_response(resp: requests.Response) -> List[Dict[str, Any]]:
    try:
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "locations", "doors", "items", "entries"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError as exc:
        snippet = resp.text[:200]
        raise RuntimeError(
            f"Expected JSON but received {len(resp.content)} bytes: {snippet}"
        ) from exc
    raise RuntimeError("Unexpected API response structure.")


def fetch_doors(session: requests.Session, base_url: str, site_id: str) -> List[Door]:
    candidate_paths = [
        ("door_v1", "/proxy/ui/access/api/v1/doors"),
        ("door_v1", "/proxy/access/api/v1/doors"),
        ("location_v2", "/proxy/access/api/v2/location"),
        ("location_v2", "/proxy/access/api/v2/locations"),
    ]
    params = {"siteId": site_id}
    last_error: Exception | None = None
    for kind, path in candidate_paths:
        url = f"{base_url}{path}"
        LOGGER.debug("Fetching doors from %s", url)
        try:
            resp = session.get(url, params=params, timeout=15)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.debug("Request to %s failed: %s", url, exc)
            continue
        if resp.status_code >= 400:
            last_error = RuntimeError(f"Door list failed ({resp.status_code}): {resp.text}")
            LOGGER.debug("Request to %s returned %s", url, last_error)
            continue
        try:
            data = _parse_json_response(resp)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.debug("Failed to parse response from %s: %s", url, exc)
            continue
        doors: List[Door] = []
        for item in data:
            door_id = item.get("id") or item.get("_id") or ""
            name = item.get("name") or item.get("label") or ""
            site = item.get("siteId") or site_id
            device_mac = item.get("deviceMac") or item.get("device_mac")
            status = item.get("status")

            if kind == "door_v1":
                unlock_base = path
                if unlock_base.endswith("/doors"):
                    unlock_base = unlock_base
                unlock_path = f"{unlock_base}/{door_id}/unlock"
                door = Door(
                    door_id=door_id,
                    name=name,
                    site_id=site,
                    device_mac=device_mac,
                    status=status,
                    unlock_path=unlock_path,
                    unlock_method="POST",
                    supports_duration=True,
                )
            else:  # location_v2
                unlock_path = f"/proxy/access/api/v2/location/{door_id}/unlock"
                door = Door(
                    door_id=door_id,
                    name=name or item.get("displayName", ""),
                    site_id=site,
                    device_mac=device_mac,
                    status=status or item.get("doorState"),
                    unlock_path=unlock_path,
                    unlock_method="PUT",
                    supports_duration=False,
                )
            doors.append(door)
        return doors
    if last_error:
        raise last_error
    raise RuntimeError("Unable to reach any Access API endpoint.")


def resolve_door(
    session: requests.Session,
    base_url: str,
    site_id: str,
    door_id: Optional[str],
    door_name: Optional[str],
    doors: Optional[List[Door]],
) -> Door:
    if not doors:
        doors = []
    if door_id:
        for door in doors:
            if door.door_id == door_id:
                return door
        LOGGER.debug("Door ID %s not found in cached list; attempting direct lookup.", door_id)
        location_url = f"{base_url}/proxy/access/api/v2/location/{door_id}"
        LOGGER.debug("Fetching location directly from %s", location_url)
        resp = session.get(location_url, params={"siteId": site_id}, timeout=15)
        if resp.status_code == 200:
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                record = payload.get("data") or payload
                if isinstance(record, list):
                    record = record[0] if record else None
                if isinstance(record, dict):
                    return Door(
                        door_id=door_id,
                        name=record.get("name")
                        or record.get("label")
                        or record.get("displayName")
                        or door_id,
                        site_id=record.get("siteId") or site_id,
                        device_mac=record.get("deviceMac"),
                        status=record.get("doorState"),
                        unlock_path=f"/proxy/access/api/v2/location/{door_id}/unlock",
                        unlock_method="PUT",
                        supports_duration=False,
                    )
        if resp.status_code != 404 and resp.status_code >= 400:
            LOGGER.debug(
                "Location lookup returned %s: %s", resp.status_code, resp.text[:200]
            )

        policy_url = f"{base_url}/proxy/access/api/v2/access_policies/by_door/{door_id}"
        LOGGER.debug("Falling back to policy metadata from %s", policy_url)
        resp_policy = session.get(policy_url, timeout=15)
        if resp_policy.status_code == 404:
            raise RuntimeError(f"Door/location ID '{door_id}' not found.")
        if resp_policy.status_code >= 400:
            raise RuntimeError(
                f"Door lookup failed ({resp_policy.status_code}): {resp_policy.text}"
            )
        try:
            policy_payload = resp_policy.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Unexpected response for door {door_id}: {resp_policy.text[:200]}"
            ) from exc
        policies = policy_payload.get("data")
        display_name = None
        if isinstance(policies, list):
            for policy in policies:
                resources = policy.get("resources") or []
                for resource in resources:
                    subs = resource.get("sub_resources") or []
                    for sub in subs:
                        if sub.get("resource_value") == door_id:
                            display_name = sub.get("resource_name")
                            break
                    if display_name:
                        break
                if display_name:
                    break
        if not display_name:
            display_name = door_id
        return Door(
            door_id=door_id,
            name=display_name,
            site_id=site_id,
            device_mac=None,
            status=None,
            unlock_path=f"/proxy/access/api/v2/location/{door_id}/unlock",
            unlock_method="PUT",
            supports_duration=False,
        )
        matches = [door for door in doors if door.door_id == door_id]
        if matches:
            return matches[0]
        raise RuntimeError(f"Door with ID '{door_id}' not found among {len(doors)} doors.")
    if door_name:
        matches = [door for door in doors if door.name.lower() == door_name.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            available = ", ".join(d.door_id for d in matches)
            raise RuntimeError(
                f"Door name '{door_name}' is ambiguous ({len(matches)} matches: {available}). "
                "Use --door-id instead."
            )
        raise RuntimeError(f"Door named '{door_name}' not found.")
    raise RuntimeError("Either --door-id or --door-name must be provided unless --list-doors is set.")


def unlock_door(
    session: requests.Session,
    base_url: str,
    door: Door,
    seconds: int,
) -> Dict[str, Any]:
    unlock_url = f"{base_url}{door.unlock_path}"
    LOGGER.info(
        "Unlocking door '%s' (%s) for %ss.",
        door.name or door.door_id,
        door.door_id,
        seconds,
    )
    payload = {"duration": int(max(seconds, 0))}
    json_payload: Optional[Dict[str, Any]] = payload if door.supports_duration else None
    data_payload: Optional[bytes] = None
    if not door.supports_duration:
        data_payload = b""
    if not door.supports_duration and seconds != DEFAULT_UNLOCK_SECONDS:
        LOGGER.warning(
            "Door endpoint does not support custom durations; default console duration will be used."
        )
    method = door.unlock_method.upper()
    if method == "POST":
        resp = session.post(unlock_url, json=json_payload or payload, timeout=15)
    elif method == "PUT":
        resp = session.put(unlock_url, json=json_payload, data=data_payload, timeout=15)
    else:
        resp = session.request(
            method, unlock_url, json=json_payload, data=data_payload, timeout=15
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Unlock request failed ({resp.status_code}): {resp.text}")
    if not resp.content:
        return {}
    try:
        return resp.json()
    except json.JSONDecodeError:
        return {"raw_response": resp.text}


def main() -> None:
    load_dotenv()
    args = parse_args()
    configure_logging(args.debug)

    if not args.username:
        args.username = input("UniFi username: ")
    if not args.password:
        args.password = getpass.getpass("UniFi password: ")

    base_url = normalize_base_url(args.controller)
    session = build_session(verify_tls=not args.insecure)

    try:
        login(session, base_url, args.username, args.password)
    except Exception as exc:  # noqa: BLE001
        if args.debug and isinstance(exc, RuntimeError):
            LOGGER.exception("Login error: %s", exc)
        else:
            LOGGER.error("Login failed: %s", exc)
        sys.exit(2)

    try:
        doors = fetch_doors(session, base_url, args.site_id)
    except Exception as exc:  # noqa: BLE001
        if args.list_doors:
            LOGGER.error("Unable to fetch doors: %s", exc)
            sys.exit(3)
        if args.door_id:
            LOGGER.warning(
                "Door catalog lookup failed (%s); attempting direct unlock via door ID.",
                exc,
            )
            doors = []
        else:
            LOGGER.error("Unable to fetch doors: %s", exc)
            sys.exit(3)

    if args.list_doors:
        if not doors:
            print("No doors found.")
            return
        print(f"{'Door ID':36}  {'Site':8}  {'Name':25}  MAC")
        print("-" * 80)
        for door in doors:
            print(
                f"{door.door_id:36}  {door.site_id:8}  {door.name[:25]:25}  {door.device_mac or ''}"
            )
        return

    try:
        door = resolve_door(session, base_url, args.site_id, args.door_id, args.door_name, doors)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Door resolution failed: %s", exc)
        sys.exit(4)

    try:
        response = unlock_door(session, base_url, door, args.unlock_seconds)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unlock failed: %s", exc)
        sys.exit(5)

    if response:
        LOGGER.debug("Unlock response: %s", response)
    print("Door unlock command issued successfully.")


if __name__ == "__main__":
    main()
