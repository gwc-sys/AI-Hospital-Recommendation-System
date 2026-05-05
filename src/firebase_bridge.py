from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any

from config.firebase_config import settings
from src.emergency_response import EmergencyHospitalRecommender


class FirebaseEmergencySync:
    """Read emergency input from Realtime Database and write React Native friendly AI results."""

    def __init__(self, dataset_path: str | None = None) -> None:
        self.recommender = EmergencyHospitalRecommender(dataset_path=dataset_path)
        self.db = None

    def initialize(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials, db
        except ImportError as exc:
            raise ImportError(
                "firebase-admin is required for Firebase Realtime Database sync. Install dependencies first."
            ) from exc

        if not settings.service_account_path:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH is required for Firebase sync.")
        if not settings.database_url:
            raise ValueError("FIREBASE_DATABASE_URL is required for Firebase Realtime Database sync.")

        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.service_account_path)
            firebase_admin.initialize_app(
                cred,
                {
                    "projectId": settings.project_id or None,
                    "databaseURL": settings.database_url,
                },
            )

        self.db = db

    @staticmethod
    def _entry_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int]:
        key, value = item
        timestamp = value.get("timestamp", -1)
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError):
            timestamp_value = -1.0

        try:
            key_value = int(key)
        except (TypeError, ValueError):
            key_value = -1

        return (timestamp_value, key_value)

    def _latest_entry(
        self,
        records: dict[str, Any] | None,
        predicate: Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(records, dict):
            return ("", {})

        candidates: list[tuple[str, dict[str, Any]]] = []
        for key, value in records.items():
            if not isinstance(value, dict):
                continue
            if predicate is not None and not predicate(value):
                continue
            candidates.append((key, value))

        if not candidates:
            return ("", {})

        return max(candidates, key=self._entry_sort_key)

    def _latest_valid_health_entry(
        self,
        records: dict[str, Any] | None,
        metric_key: str,
        valid_key: str,
    ) -> dict[str, Any]:
        _, latest = self._latest_entry(
            records,
            predicate=lambda entry: bool(entry.get(valid_key))
            and self._valid_number(entry.get(metric_key)) is not None,
        )
        return latest

    def _latest_valid_gps_entry(self, records: dict[str, Any] | None) -> dict[str, Any]:
        _, latest = self._latest_entry(
            records,
            predicate=lambda entry: self._valid_number(entry.get("gps_lat")) is not None
            and self._valid_number(entry.get("gps_lon")) is not None,
        )
        return latest

    @staticmethod
    def _valid_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number == -999:
            return None
        return number

    def build_payload(self, document: dict[str, Any]) -> dict[str, Any]:
        preferences = document.get("preferences", {})
        return self.recommender.build_react_native_payload(
            emergency_event=document.get("sos_alert", {}),
            require_emergency=bool(preferences.get("require_emergency", False)),
            only_open=bool(preferences.get("only_open", False)),
        )

    def build_payload_from_nodes(
        self,
        vehicle_root: dict[str, Any],
        alerts_root: dict[str, Any],
        health_root: dict[str, Any],
    ) -> dict[str, Any]:
        latest_sos_key, latest_sos = self._latest_entry(
            alerts_root,
            predicate=lambda alert: str(alert.get("type", "")).lower() == "sos",
        )
        if not latest_sos:
            raise ValueError("No SOS alerts found under the alerts node.")

        vehicle_readings = vehicle_root.get("readings")
        health_readings = health_root.get("readings")
        latest_vehicle_key, latest_vehicle = self._latest_entry(vehicle_readings)
        latest_vehicle_gps = self._latest_valid_gps_entry(vehicle_readings)
        latest_health_key, latest_health = self._latest_entry(health_readings)
        latest_valid_heart_rate = self._latest_valid_health_entry(
            health_readings,
            metric_key="heart_rate_bpm",
            valid_key="heart_rate_valid",
        )
        latest_valid_spo2 = self._latest_valid_health_entry(
            health_readings,
            metric_key="oxygen_saturation_spo2",
            valid_key="spo2_valid",
        )

        alert_latitude = self._valid_number(latest_sos.get("latitude"))
        alert_longitude = self._valid_number(latest_sos.get("longitude"))
        vehicle_latitude = self._valid_number(latest_vehicle_gps.get("gps_lat"))
        vehicle_longitude = self._valid_number(latest_vehicle_gps.get("gps_lon"))

        latest_location_source = "alert"
        latitude = alert_latitude
        longitude = alert_longitude

        # Prefer the SOS alert coordinates as the real trigger location.
        # Only fall back to vehicle GPS if SOS alert coordinates are missing.
        if (latitude is None or longitude is None) and vehicle_latitude is not None and vehicle_longitude is not None:
            latitude = vehicle_latitude
            longitude = vehicle_longitude
            latest_location_source = "vehicle_gps"

        if latitude is None or longitude is None:
            raise ValueError("No usable GPS coordinates found in alerts or vehicle readings.")

        health_payload = {
            "device_id": health_root.get("status", {}).get("device_id", settings.health_node),
            "device_name": health_root.get("status", {}).get("device_name", settings.health_node),
            "heart_rate_bpm": (
                int(latest_valid_heart_rate["heart_rate_bpm"])
                if latest_valid_heart_rate
                else None
            ),
            "spo2": (
                int(latest_valid_spo2["oxygen_saturation_spo2"])
                if latest_valid_spo2
                else None
            ),
            "finger_detected": bool(latest_health.get("finger_detected", False)),
            "timestamp": latest_health.get("timestamp"),
        }
        refresh_metadata = {
            "refresh_trigger": "new_sos",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "alert_key": latest_sos_key,
            "alert_timestamp": latest_sos.get("timestamp"),
            "vehicle_reading_key": latest_vehicle_key,
            "vehicle_timestamp": latest_vehicle.get("timestamp"),
            "health_reading_key": latest_health_key,
            "health_timestamp": latest_health.get("timestamp"),
        }

        emergency_event = {
            "active": True,
            "type": latest_sos.get("message", "SOS: Tilt/fall detected"),
            "priority": "CRITICAL",
            "device_name": latest_sos.get("device_id", settings.vehicle_node),
            "last_updated": latest_sos.get("timestamp"),
            "trigger_source": latest_sos.get("type", "sos"),
            "selection_preference": (
                str(
                    latest_sos.get("selection_preference")
                    or latest_sos.get("hospital_choice")
                    or "balanced"
                )
                .strip()
                .lower()
            ),
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "source": latest_location_source,
            },
            "health": health_payload,
            "refresh_metadata": refresh_metadata,
        }

        return self.recommender.build_react_native_payload(
            emergency_event=emergency_event,
            require_emergency=False,
            only_open=False,
        )

    def sync_user_document(self, user_id: str, collection: str | None = None) -> dict[str, Any]:
        if self.db is None:
            self.initialize()

        users_path = (collection or settings.users_path).strip("/")
        ref = self.db.reference(f"{users_path}/{user_id}")
        document = ref.get()
        if not document:
            raise ValueError(f"Realtime Database path not found or empty: {users_path}/{user_id}")

        payload = self.build_payload(document)
        ref.update(payload)
        return payload

    def sync_current_emergency(self) -> dict[str, Any]:
        if self.db is None:
            self.initialize()

        vehicle_root = self.db.reference(settings.vehicle_node).get() or {}
        alerts_root = self.db.reference(settings.alerts_node).get() or {}
        health_root = self.db.reference(settings.health_node).get() or {}

        payload = self.build_payload_from_nodes(vehicle_root, alerts_root, health_root)
        response_ref = self.db.reference(settings.response_path)
        response_ref.set(payload)
        self._update_latest_alert_assignment(payload)
        return payload

    def _update_latest_alert_assignment(self, payload: dict[str, Any]) -> None:
        if self.db is None:
            return

        refresh = payload.get("ai_results", {}).get("payload_refresh", {})
        sos_alert = payload.get("sos_alert", {})
        alert_key = str(refresh.get("alert_key", "")).strip()
        if not alert_key:
            return

        hospital = payload.get("ai_results", {}).get("hospital_ai", {})
        if not hospital:
            return

        assignment = {
            "hospital_name": hospital.get("hospital_name", "Nearest hospital"),
            "distance_km": hospital.get("distance_km"),
            "address": hospital.get("address", "Address not available"),
            "specialization": hospital.get("specialization", "General"),
            "hospital_phone": hospital.get("phone", "N/A"),
            "emergency_available": hospital.get("emergency_available"),
            "emergency_facility": hospital.get("emergency_facility"),
            "map_url": hospital.get("map_url"),
            "selected_at": hospital.get("selected_at"),
            "selection_method": hospital.get("selection_method"),
            "source": "ai_emergency_sync",
        }
        refresh_flat = {
            "refresh_trigger": refresh.get("refresh_trigger"),
            "refreshed_at": refresh.get("refreshed_at"),
            "vehicle_reading_key": refresh.get("vehicle_reading_key"),
            "vehicle_timestamp": refresh.get("vehicle_timestamp"),
            "health_reading_key": refresh.get("health_reading_key"),
            "health_timestamp": refresh.get("health_timestamp"),
            "alert_timestamp": refresh.get("alert_timestamp"),
        }
        sos_flat = {
            "active": sos_alert.get("active"),
            "type": sos_alert.get("type"),
            "trigger_source": sos_alert.get("trigger_source"),
            "priority": sos_alert.get("priority"),
            "device_name": sos_alert.get("device_name"),
            "last_updated": sos_alert.get("last_updated"),
            "location": sos_alert.get("location"),
            "health": sos_alert.get("health"),
        }

        alert_ref = self.db.reference(f"{settings.alerts_node.strip('/')}/{alert_key}")
        alert_ref.update(
            {
                "assigned_hospital": assignment,
                "refresh_metadata": refresh,
                **assignment,
                **refresh_flat,
                **sos_flat,
            }
        )

    def get_latest_sos_details(self) -> tuple[str, dict[str, Any]]:
        if self.db is None:
            self.initialize()

        alerts_root = self.db.reference(settings.alerts_node).get() or {}
        latest_key, latest_sos = self._latest_entry(
            alerts_root,
            predicate=lambda alert: str(alert.get("type", "")).lower() == "sos",
        )
        if not latest_sos:
            raise ValueError("No SOS alerts found under the alerts node.")
        return latest_key, latest_sos

    @staticmethod
    def _build_alert_dedupe_key(alert_key: str, alert: dict[str, Any]) -> str:
        return "|".join(
            [
                str(alert_key or ""),
                str(alert.get("timestamp", "")),
            ]
        )

    @staticmethod
    def _format_console_value(value: Any, fallback: str = "N/A") -> str:
        if value is None:
            return fallback

        text = str(value).strip()
        return text or fallback

    @classmethod
    def format_sos_console_output(
        cls,
        alert_key: str,
        latest_sos: dict[str, Any],
        event_path: str,
        payload: dict[str, Any],
    ) -> str:
        sos_alert = payload.get("sos_alert", {})
        hospital = payload.get("ai_results", {}).get("hospital_ai", {})
        selection = payload.get("ai_results", {}).get("hospital_selection", {})
        refresh = payload.get("ai_results", {}).get("payload_refresh", {})
        location = sos_alert.get("location", {})
        health = sos_alert.get("health", {})

        pretty_payload = json.dumps(payload, indent=2, sort_keys=True, default=str)
        lines = [
            "",
            "[SOS] ==================================================",
            "[SOS] New SOS detected",
            f"[SOS] Alert key: {cls._format_console_value(alert_key)}",
            f"[SOS] Event path: {cls._format_console_value(event_path)}",
            (
                "[SOS] Timestamp: "
                f"{cls._format_console_value(latest_sos.get('timestamp') or sos_alert.get('last_updated'), 'unknown')}"
            ),
            (
                "[SOS] Message: "
                f"{cls._format_console_value(latest_sos.get('message') or sos_alert.get('type'), 'SOS triggered')}"
            ),
            f"[SOS] Device: {cls._format_console_value(sos_alert.get('device_name'))}",
            (
                "[SOS] Location: "
                f"{cls._format_console_value(location.get('latitude'))}, "
                f"{cls._format_console_value(location.get('longitude'))} "
                f"({cls._format_console_value(location.get('source'))})"
            ),
            f"[SOS] Heart rate: {cls._format_console_value(health.get('heart_rate_bpm'))}",
            f"[SOS] SpO2: {cls._format_console_value(health.get('spo2'))}",
            f"[SOS] Refreshed at: {cls._format_console_value(refresh.get('refreshed_at'))}",
            (
                "[SOS] Recommended hospital: "
                f"{cls._format_console_value(hospital.get('hospital_name'), 'Unknown hospital')}"
            ),
            (
                "[SOS] Selection method: "
                f"{cls._format_console_value(selection.get('selection_method') or hospital.get('selection_method'))}"
            ),
            f"[SOS] Compared hospitals: {cls._format_console_value(selection.get('compared_hospitals'))}",
            f"[SOS] Distance (km): {cls._format_console_value(hospital.get('distance_km'))}",
            f"[SOS] Response time (sec): {cls._format_console_value(hospital.get('response_time_sec'))}",
            f"[SOS] Phone available: {cls._format_console_value(hospital.get('phone_available'))}",
            f"[SOS] Address: {cls._format_console_value(hospital.get('address'))}",
            f"[SOS] Phone: {cls._format_console_value(hospital.get('phone'))}",
            f"[SOS] Maps: {cls._format_console_value(hospital.get('map_url'))}",
            "[SOS] Full payload:",
        ]
        lines.extend(f"[SOS] {line}" if line else "[SOS]" for line in pretty_payload.splitlines())
        lines.append("[SOS] ==================================================")
        return "\n".join(lines)

    def _sync_and_log_sos(
        self,
        alert_key: str,
        latest_sos: dict[str, Any],
        event_path: str,
    ) -> None:
        payload = self.sync_current_emergency()
        print(
            self.format_sos_console_output(
                alert_key=alert_key,
                latest_sos=latest_sos,
                event_path=event_path,
                payload=payload,
            ),
            flush=True,
        )

    def _sync_latest_sos_if_needed(
        self,
        alert_key: str,
        latest_sos: dict[str, Any],
        event_path: str,
        last_synced_marker: str,
        ignore_marker: bool = False,
    ) -> str:
        current_marker = self._build_alert_dedupe_key(alert_key, latest_sos)
        if not ignore_marker and current_marker == last_synced_marker:
            return last_synced_marker

        self._sync_and_log_sos(
            alert_key=alert_key,
            latest_sos=latest_sos,
            event_path=event_path,
        )
        return current_marker

    def watch_sos(
        self,
        poll_interval_seconds: float = 2.0,
        sync_on_startup: bool = False,
    ) -> None:
        heartbeat_interval = max(float(poll_interval_seconds), 1.0)
        last_synced_marker = ""

        while True:
            listener = None
            try:
                if self.db is None:
                    self.initialize()

                alerts_ref = self.db.reference(settings.alerts_node)

                try:
                    alert_key, latest_sos = self.get_latest_sos_details()
                    print(
                        "[SOS] Listener connected. "
                        f"Current latest SOS key={alert_key}, "
                        f"timestamp={latest_sos.get('timestamp', 'unknown')}",
                        flush=True,
                    )
                    if sync_on_startup and self._build_alert_dedupe_key(alert_key, latest_sos) != last_synced_marker:
                        print("[SOS] Syncing the current latest SOS on startup.", flush=True)
                        try:
                            last_synced_marker = self._sync_latest_sos_if_needed(
                                alert_key=alert_key,
                                latest_sos=latest_sos,
                                event_path="/startup",
                                last_synced_marker=last_synced_marker,
                            )
                        except Exception as exc:
                            print(f"[SOS] Startup sync error: {exc}", flush=True)
                except ValueError:
                    print("[SOS] Listener connected. Waiting for the first SOS alert...", flush=True)

                def handle_event(event: Any) -> None:
                    nonlocal last_synced_marker
                    try:
                        event_path = getattr(event, "path", "/")
                        # Firebase listeners commonly emit an initial "/" snapshot on connect.
                        # Skip it so data refresh happens only for actual SOS updates.
                        if event_path in {"", "/"}:
                            return

                        alert_key, latest_sos = self.get_latest_sos_details()
                        last_synced_marker = self._sync_latest_sos_if_needed(
                            alert_key=alert_key,
                            latest_sos=latest_sos,
                            event_path=event_path,
                            last_synced_marker=last_synced_marker,
                            ignore_marker=True,
                        )
                    except ValueError:
                        return
                    except Exception as exc:
                        print(f"[SOS] Watch error: {exc}", flush=True)

                listener = alerts_ref.listen(handle_event)

                while True:
                    time.sleep(heartbeat_interval)
                    try:
                        alert_key, latest_sos = self.get_latest_sos_details()
                    except ValueError:
                        continue

                    current_marker = self._build_alert_dedupe_key(alert_key, latest_sos)
                    if current_marker == last_synced_marker:
                        continue

                    print("[SOS] Poll fallback detected a new SOS.", flush=True)
                    last_synced_marker = self._sync_latest_sos_if_needed(
                        alert_key=alert_key,
                        latest_sos=latest_sos,
                        event_path="/poll",
                        last_synced_marker=last_synced_marker,
                    )
            except KeyboardInterrupt:
                print("[SOS] Listener stopped.", flush=True)
                break
            except Exception as exc:
                print(f"[SOS] Listener connection lost: {exc}", flush=True)
                print(f"[SOS] Reconnecting in {heartbeat_interval:.1f} seconds...", flush=True)
                self.db = None
                time.sleep(heartbeat_interval)
            finally:
                if listener is not None:
                    try:
                        listener.close()
                    except Exception:
                        pass
