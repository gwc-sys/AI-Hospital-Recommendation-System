from src.firebase_bridge import FirebaseEmergencySync


def test_build_payload_from_nodes_uses_alert_location_and_health_metrics() -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")

    vehicle_root = {
        "readings": {
            "2060127": {
                "gps_lat": 18.675943,
                "gps_lon": 73.841827,
                "timestamp": 2059,
            }
        }
    }
    alerts_root = {
        "2056828": {
            "device_id": "Ai-based-smart-vehicle-health",
            "latitude": 18.675943,
            "longitude": 73.841827,
            "message": "SOS: Tilt/fall detected",
            "timestamp": 2056,
            "type": "sos",
        }
    }
    health_root = {
        "status": {
            "device_id": "mahesh_Raskar",
            "device_name": "mahesh Raskar",
        },
        "readings": {
            "24308": {
                "finger_detected": True,
                "heart_rate_bpm": 166,
                "heart_rate_valid": True,
                "oxygen_saturation_spo2": 100,
                "spo2_valid": True,
                "timestamp": 24,
            }
        },
    }

    payload = sync.build_payload_from_nodes(vehicle_root, alerts_root, health_root)

    assert payload["sos_alert"]["type"] == "SOS: Tilt/fall detected"
    assert payload["sos_alert"]["location"]["latitude"] == 18.675943
    assert payload["sos_alert"]["location"]["source"] == "alert"
    assert payload["sos_alert"]["health"]["heart_rate_bpm"] == 166
    assert payload["sos_alert"]["health"]["spo2"] == 100
    assert payload["ai_results"]["hospital_ai"]["hospital_name"] == "Emergency Hospital"
    assert payload["ai_results"]["payload_refresh"]["alert_key"] == "2056828"
    assert payload["ai_results"]["payload_refresh"]["refresh_trigger"] == "new_sos"
    assert "alert_marker" not in payload["sos_alert"]["refresh_metadata"]


def test_build_payload_from_nodes_falls_back_to_vehicle_gps() -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")

    vehicle_root = {
        "readings": {
            "2060127": {
                "gps_lat": 18.675943,
                "gps_lon": 73.841827,
                "timestamp": 2059,
            }
        }
    }
    alerts_root = {
        "2056828": {
            "device_id": "Ai-based-smart-vehicle-health",
            "message": "SOS: Tilt/fall detected",
            "timestamp": 2056,
            "type": "sos",
        }
    }
    health_root = {"readings": {}}

    payload = sync.build_payload_from_nodes(vehicle_root, alerts_root, health_root)

    assert payload["sos_alert"]["location"]["latitude"] == 18.675943
    assert payload["sos_alert"]["location"]["longitude"] == 73.841827
    assert payload["sos_alert"]["location"]["source"] == "vehicle_gps"


def test_build_payload_from_nodes_uses_latest_valid_health_metrics() -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")

    vehicle_root = {
        "readings": {
            "2060127": {
                "gps_lat": 18.675943,
                "gps_lon": 73.841827,
                "timestamp": 2059,
            }
        }
    }
    alerts_root = {
        "2056828": {
            "device_id": "Ai-based-smart-vehicle-health",
            "latitude": 18.675943,
            "longitude": 73.841827,
            "message": "SOS: Tilt/fall detected",
            "timestamp": 2056,
            "type": "sos",
        }
    }
    health_root = {
        "status": {
            "device_id": "mahesh_Raskar",
            "device_name": "mahesh Raskar",
        },
        "readings": {
            "24308": {
                "finger_detected": True,
                "heart_rate_bpm": 166,
                "heart_rate_valid": True,
                "oxygen_saturation_spo2": 100,
                "spo2_valid": True,
                "timestamp": 24,
            },
            "1588330": {
                "finger_detected": True,
                "heart_rate_bpm": -999,
                "heart_rate_valid": False,
                "oxygen_saturation_spo2": -999,
                "spo2_valid": False,
                "timestamp": 1588,
            },
        },
    }

    payload = sync.build_payload_from_nodes(vehicle_root, alerts_root, health_root)

    assert payload["sos_alert"]["health"]["heart_rate_bpm"] == 166
    assert payload["sos_alert"]["health"]["spo2"] == 100


def test_build_payload_from_nodes_prefers_alert_location_over_vehicle_location() -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")

    vehicle_root = {
        "readings": {
            "3313973": {
                "gps_lat": 18.675617,
                "gps_lon": 73.841995,
                "timestamp": 3313,
            }
        }
    }
    alerts_root = {
        "3310949": {
            "device_id": "Ai-based-smart-vehicle-health",
            "latitude": 18.675500,
            "longitude": 73.841900,
            "message": "SOS: Tilt/fall detected",
            "timestamp": 3310,
            "type": "sos",
        }
    }
    health_root = {"readings": {}}

    payload = sync.build_payload_from_nodes(vehicle_root, alerts_root, health_root)

    assert payload["sos_alert"]["location"]["latitude"] == 18.675500
    assert payload["sos_alert"]["location"]["longitude"] == 73.841900
    assert payload["sos_alert"]["location"]["source"] == "alert"
    assert "alert_marker" not in payload["ai_results"]["payload_refresh"]


def test_sync_and_log_sos_reports_updated_hospital(capsys) -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")
    sync.sync_current_emergency = lambda: {
        "sos_alert": {
            "device_name": "Ai-based-smart-vehicle-health",
            "location": {
                "latitude": 18.675943,
                "longitude": 73.841827,
                "source": "vehicle_gps",
            },
            "health": {
                "heart_rate_bpm": 166,
                "spo2": 100,
            },
        },
        "ai_results": {
            "payload_refresh": {
                "refreshed_at": "2026-04-05T10:00:37Z",
            },
            "hospital_ai": {
                "hospital_name": "Emergency Hospital",
                "distance_km": 0.42,
            }
        }
    }

    sync._sync_and_log_sos(
        "2056828",
        {
            "message": "SOS: Tilt/fall detected",
            "timestamp": 2056,
        },
        event_path="/poll",
    )

    output = capsys.readouterr().out
    assert "New SOS detected" in output
    assert "Recommended hospital: Emergency Hospital" in output
    assert "Refreshed at: 2026-04-05T10:00:37Z" in output
    assert '"distance_km": 0.42' in output


def test_format_sos_console_output_includes_payload_details() -> None:
    output = FirebaseEmergencySync.format_sos_console_output(
        "2056828",
        {
            "message": "SOS: Tilt/fall detected",
            "timestamp": 2056,
        },
        "/poll",
        {
            "sos_alert": {
                "device_name": "Ai-based-smart-vehicle-health",
                "location": {
                    "latitude": 18.675943,
                    "longitude": 73.841827,
                    "source": "vehicle_gps",
                },
                "health": {
                    "heart_rate_bpm": 166,
                    "spo2": 100,
                },
            },
            "ai_results": {
                "payload_refresh": {
                    "refreshed_at": "2026-04-05T10:00:37Z",
                },
                "hospital_ai": {
                    "hospital_name": "Emergency Hospital",
                    "distance_km": 0.42,
                    "address": "Pune",
                    "phone": "+91 9876543210",
                    "map_url": "https://maps.example/hospital",
                }
            },
        },
    )

    assert "New SOS detected" in output
    assert "Event path: /poll" in output
    assert "Heart rate: 166" in output
    assert "Refreshed at: 2026-04-05T10:00:37Z" in output
    assert "Alert marker" not in output
    assert "Recommended hospital: Emergency Hospital" in output
    assert '"hospital_name": "Emergency Hospital"' in output


def test_sync_latest_sos_if_needed_only_marks_alert_after_sync() -> None:
    sync = FirebaseEmergencySync(dataset_path="tests/fixtures/single_emergency_hospital.csv")
    calls: list[tuple[str, str]] = []

    def fake_sync(alert_key: str, latest_sos: dict[str, object], event_path: str) -> None:
        calls.append((alert_key, event_path))

    sync._sync_and_log_sos = fake_sync
    latest_sos = {
        "message": "SOS: Tilt/fall detected",
        "timestamp": 2056,
        "latitude": 18.675943,
        "longitude": 73.841827,
        "type": "sos",
    }

    first_marker = sync._sync_latest_sos_if_needed(
        "2056828",
        latest_sos,
        "/poll",
        last_synced_marker="",
    )
    second_marker = sync._sync_latest_sos_if_needed(
        "2056828",
        latest_sos,
        "/poll",
        last_synced_marker=first_marker,
    )

    assert first_marker == "2056828|2056"
    assert second_marker == first_marker
    assert calls == [("2056828", "/poll")]
