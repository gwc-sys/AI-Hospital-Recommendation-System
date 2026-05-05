from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.firebase_bridge import FirebaseEmergencySync


def format_runtime_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()

    firebase_connection_markers = (
        "oauth2.googleapis.com",
        "failed to establish a new connection",
        "max retries exceeded",
        "temporarily failed in name resolution",
        "transporterror",
        "connection aborted",
        "connection reset",
        "socket",
        "winerror 10013",
    )
    if any(marker in lowered for marker in firebase_connection_markers):
        return (
            "Could not connect to Firebase. Check that this machine has internet access, "
            "that firewall/antivirus/proxy settings allow Python HTTPS requests, and that "
            "the Firebase service account and Realtime Database URL in `.env` are valid."
        )

    return message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default=None, help="Deprecated. No longer needed for node-based RTDB sync.")
    parser.add_argument("--collection", default=None, help="Deprecated. No longer needed for node-based RTDB sync.")
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Watch Firebase for SOS changes and automatically reconnect if the listener drops."
        ),
    )
    parser.add_argument(
        "--sync-on-startup",
        action="store_true",
        help="Also sync the current latest SOS once on startup before waiting for new button presses.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between watch heartbeats and reconnect attempts.",
    )
    args = parser.parse_args()

    if args.user_id or args.collection:
        raise SystemExit(
            "This script now reads directly from the RTDB nodes "
            "`Ai-based-smart-vehicle-health`, `alerts`, and `mahesh_Raskar`. "
            "Run it as: python scripts/sync_firebase_emergency.py"
        )

    sync = FirebaseEmergencySync()
    if args.watch:
        print(
            "Listening to Firebase for new SOS alerts. "
            "Press Ctrl+C to stop."
        )
        try:
            sync.watch_sos(
                poll_interval_seconds=args.poll_interval,
                sync_on_startup=args.sync_on_startup,
            )
        except Exception as exc:
            raise SystemExit(format_runtime_error(exc)) from exc
        return

    try:
        payload = sync.sync_current_emergency()
    except Exception as exc:
        raise SystemExit(format_runtime_error(exc)) from exc

    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
