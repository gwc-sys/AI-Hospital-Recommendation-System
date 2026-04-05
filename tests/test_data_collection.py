import httpx

from src.data_collection import GoogleMapsCollector


def test_calculate_response_probability_increases_with_contact_data() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    probability = collector._calculate_response_prob(
        {
            "currentOpeningHours": {"openNow": True},
            "nationalPhoneNumber": "+91-9999999999",
            "websiteUri": "https://example.com",
        }
    )
    assert probability == 0.85


def test_calculate_distance_returns_positive_value() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    collector.search_center = (12.9716, 77.5946)
    distance = collector._calculate_distance(
        {"location": {"latitude": 12.9816, "longitude": 77.6046}}
    )
    assert distance > 0


def test_fetch_places_pages_collects_paginated_results() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    calls = []
    text_calls = []

    def fake_search_nearby(location, radius, facility_type):
        calls.append(
            {
                "location": location,
                "radius": radius,
                "facility_type": facility_type,
            }
        )
        return {"places": [{"id": "1"}, {"id": "2"}]}

    def fake_search_text(query, location, radius):
        text_calls.append(
            {
                "query": query,
                "location": location,
                "radius": radius,
            }
        )
        return {"places": [{"id": "2"}, {"id": "3"}]}

    collector._search_nearby = fake_search_nearby
    collector._search_text = fake_search_text
    pages = collector._fetch_places_pages(
        location=(18.5204, 73.8567),
        radius=5000,
        facility_type="hospital",
        search_label="pune",
    )

    assert [place["id"] for place in pages] == ["1", "2", "3"]
    assert calls[0]["location"] == (18.5204, 73.8567)
    assert text_calls[0]["query"] == "hospital in pune"


def test_collector_defaults_to_hospital_only() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    collector.api_key = "fake-key"
    collector.facility_types = ["hospital"]

    assert collector.facility_types == ["hospital"]


def test_is_real_hospital_rejects_medical_store_noise() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)

    assert collector._is_real_hospital(
        {
            "primaryType": "hospital",
            "types": ["hospital", "health"],
            "displayName": {"text": "Talera Hospital"},
        }
    )


def test_build_text_queries_returns_backup_queries() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)

    queries = collector._build_text_queries("hospital", "pune")

    assert queries[0] == "hospital in pune"
    assert "government hospital in pune" in queries
    assert not collector._is_real_hospital(
        {
            "primaryType": "drugstore",
            "types": ["drugstore", "pharmacy", "health"],
            "displayName": {"text": "Omsainath Medical"},
        }
    )


def test_fetch_places_pages_falls_back_to_legacy_on_403() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    collector.use_places_new = True

    request = httpx.Request("POST", "https://places.googleapis.com/v1/places:searchNearby")
    response = httpx.Response(403, request=request)

    def raise_403(*args, **kwargs):
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    collector._search_nearby = raise_403
    collector._fetch_places_pages_legacy = lambda **kwargs: [{"id": "legacy-1"}]

    pages = collector._fetch_places_pages(
        location=(18.5204, 73.8567),
        radius=5000,
        facility_type="hospital",
        search_label="pune",
    )

    assert pages == [{"id": "legacy-1"}]
    assert collector.use_places_new is False
