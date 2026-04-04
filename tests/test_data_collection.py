from src.data_collection import GoogleMapsCollector


def test_calculate_response_probability_increases_with_contact_data() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    probability = collector._calculate_response_prob(
        {
            "opening_hours": {"open_now": True},
            "formatted_phone_number": "+91-9999999999",
            "website": "https://example.com",
        }
    )
    assert probability == 0.85


def test_calculate_distance_returns_positive_value() -> None:
    collector = GoogleMapsCollector.__new__(GoogleMapsCollector)
    collector.search_center = (12.9716, 77.5946)
    distance = collector._calculate_distance(
        {"geometry": {"location": {"lat": 12.9816, "lng": 77.6046}}}
    )
    assert distance > 0
