from src.emergency_response import EmergencyHospitalRecommender


def test_find_nearest_hospital_returns_closest_row() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/nearest_hospitals.csv"
    )
    hospital = recommender.find_nearest_hospital(18.5204, 73.8567)

    assert hospital["hospital_name"] == "Near Hospital"
    assert hospital["distance_km"] < 1


def test_build_react_native_payload_contains_sos_and_hospital_ai() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/single_emergency_hospital.csv"
    )
    payload = recommender.build_react_native_payload(
        {
            "active": True,
            "type": "SOS: Tilt/fall detected",
            "priority": "CRITICAL",
            "device_name": "Ai-based-smart-vehicle-health",
            "last_updated": "2026-04-04T22:30:10Z",
            "location": {"latitude": 18.5204, "longitude": 73.8567},
            "health": {"spo2": 95, "temperature": 37.1},
        }
    )

    assert payload["sos_alert"]["type"] == "SOS: Tilt/fall detected"
    assert payload["sos_alert"]["location"]["latitude"] == 18.5204
    assert payload["ai_results"]["hospital_ai"]["hospital_name"] == "Emergency Hospital"
    assert payload["ai_results"]["hospital_ai"]["ui_card_title"] == "Find Hospital"
    assert payload["ai_results"]["hospital_ai"]["response_time_sec"] >= 60
    assert "phone_available" in payload["ai_results"]["hospital_ai"]
    assert payload["ai_results"]["nearby_hospitals"][0]["hospital_name"] == "Emergency Hospital"
    assert payload["ai_results"]["hospital_selection"]["primary_algorithm"] == "Decision Tree / Simple ML model"


def test_find_nearest_hospital_ignores_drugstores_and_cleans_nan_website() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/mixed_emergency_places.csv"
    )
    hospital = recommender.find_nearest_hospital(18.5204, 73.8567)

    assert hospital["hospital_name"] == "Real Emergency Hospital"
    assert hospital["facility_type"] == "hospital"
    assert hospital["website"] == "N/A"


def test_find_nearest_hospital_excludes_clinic_and_medicare_names() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/mixed_emergency_places.csv"
    )
    hospital = recommender.find_nearest_hospital(18.5204, 73.8567)

    assert "clinic" not in hospital["hospital_name"].lower()
    assert "medicare" not in hospital["hospital_name"].lower()


def test_find_nearest_hospital_prefers_better_open_hospital_over_closed_doctor_row() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/emergency_priority_candidates.csv"
    )

    hospital = recommender.find_nearest_hospital(18.5204, 73.8567)

    assert hospital["hospital_name"] == "Open City Hospital"
    assert hospital["open_now"] is True
    assert hospital["facility_type"] == "establishment"
    assert hospital["selection_method"] in {"decision_tree_hybrid", "weighted_parameter_fallback"}


def test_build_react_native_payload_includes_ranked_nearby_hospitals() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/emergency_priority_candidates.csv"
    )

    payload = recommender.build_react_native_payload(
        {
            "active": True,
            "type": "Manual SOS button pressed",
            "priority": "CRITICAL",
            "device_name": "Ai-based-smart-vehicle-health",
            "last_updated": "2026-04-05T10:00:37Z",
            "location": {"latitude": 18.5204, "longitude": 73.8567},
            "health": {"spo2": 81, "heart_rate_bpm": 187},
        }
    )

    assert payload["ai_results"]["hospital_ai"]["hospital_name"] == "Open City Hospital"
    assert payload["ai_results"]["hospital_ai"]["reason"].startswith("Best nearby hospital")
    assert payload["ai_results"]["nearby_hospitals"][0]["hospital_name"] == "Open City Hospital"
    nearby_names = [item["hospital_name"] for item in payload["ai_results"]["nearby_hospitals"]]
    assert "Open Backup Hospital" in nearby_names
    assert "Closed Doctor Hospital" in nearby_names
    assert payload["ai_results"]["hospital_selection"]["compared_parameters"] == [
        "distance_km",
        "open_now",
        "emergency_available",
        "phone_available",
        "response_time_sec",
        "response_probability",
    ]
    compared = payload["ai_results"]["nearby_hospitals"][0]
    assert "response_time_sec" in compared
    assert "phone_available" in compared
    assert "parameter_score" in compared
    assert "ml_score" in compared
    assert "final_score" in compared


def test_build_react_native_payload_uses_parameter_fallback_when_models_disabled() -> None:
    recommender = EmergencyHospitalRecommender(
        dataset_path="tests/fixtures/emergency_priority_candidates.csv",
        use_models=False,
    )

    payload = recommender.build_react_native_payload(
        {
            "active": True,
            "type": "Manual SOS button pressed",
            "priority": "CRITICAL",
            "device_name": "Ai-based-smart-vehicle-health",
            "last_updated": "2026-04-05T10:00:37Z",
            "location": {"latitude": 18.5204, "longitude": 73.8567},
            "health": {"spo2": 81, "heart_rate_bpm": 187},
        }
    )

    assert payload["ai_results"]["hospital_selection"]["fallback_used"] is True
    assert payload["ai_results"]["hospital_selection"]["selection_method"] == "weighted_parameter_fallback"
    assert payload["ai_results"]["hospital_ai"]["selection_method"] == "weighted_parameter_fallback"
