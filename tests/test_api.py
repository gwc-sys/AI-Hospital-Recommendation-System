from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_current_location_returns_ip_metadata() -> None:
    response = client.get("/location/current")

    assert response.status_code == 200
    payload = response.json()
    assert "ip_address" in payload
    assert payload["source"] in {"ip_only", "proxy_geo_headers", "client_coordinates"}


def test_current_location_uses_client_headers_when_available() -> None:
    response = client.get(
        "/location/current",
        headers={
            "x-forwarded-for": "203.0.113.10, 10.0.0.1",
            "x-user-latitude": "18.5204",
            "x-user-longitude": "73.8567",
            "x-user-city": "Pune",
            "x-user-region": "Maharashtra",
            "x-user-country": "IN",
            "x-user-postal-code": "411001",
            "x-user-timezone": "Asia/Kolkata",
            "x-user-location-accuracy-km": "0.1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ip_address"] == "203.0.113.10"
    assert payload["city"] == "Pune"
    assert payload["latitude"] == 18.5204
    assert payload["longitude"] == 73.8567
    assert payload["precision"] == "precise"

