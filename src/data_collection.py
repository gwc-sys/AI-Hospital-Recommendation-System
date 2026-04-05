"""
Google Maps API data collection module.
"""

from __future__ import annotations

from datetime import datetime
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
import googlemaps
import pandas as pd

from config.api_config import API_KEY
from src.utils import DATA_DIR, ensure_directories


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleMapsCollector:
    """Handles data collection from Google Maps Places API."""

    NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
    TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
    PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
    SEARCH_FIELD_MASK = ",".join(
        [
            "places.id",
            "places.displayName",
            "places.location",
            "places.formattedAddress",
            "places.primaryType",
            "places.types",
        ]
    )
    DETAILS_FIELD_MASK = ",".join(
        [
            "id",
            "displayName",
            "location",
            "formattedAddress",
            "primaryType",
            "types",
            "currentOpeningHours",
            "regularOpeningHours",
            "nationalPhoneNumber",
            "websiteUri",
            "rating",
            "userRatingCount",
        ]
    )

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or API_KEY
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "A valid GOOGLE_MAPS_API_KEY is required in the environment or .env file."
            )

        self.facility_types = ["hospital"]
        self.search_center: Optional[Tuple[float, float]] = None
        self.http = httpx.Client(timeout=30.0)
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.use_places_new = True

    def collect_hospitals(
        self,
        location: Tuple[float, float],
        radius: int = 10000,
        search_points: Optional[Iterable[Tuple[float, float]]] = None,
        search_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Collect facility data around one or more latitude/longitude points."""
        all_hospitals: list[Dict] = []
        seen_place_ids: set[str] = set()
        points = list(search_points or [location])

        for point in points:
            self.search_center = point

            for facility_type in self.facility_types:
                logger.info("Collecting %s data around %s...", facility_type, point)

                try:
                    for place in self._fetch_places_pages(
                        location=point,
                        radius=radius,
                        facility_type=facility_type,
                        search_label=search_label,
                    ):
                        place_id = place.get("id")
                        if not place_id or place_id in seen_place_ids:
                            continue
                        if not self._is_real_hospital(place):
                            continue

                        hospital_data = self._extract_hospital_data(place)
                        if not hospital_data:
                            continue
                        all_hospitals.append(hospital_data)
                        seen_place_ids.add(place_id)

                except Exception as exc:
                    logger.error("Error collecting %s around %s: %s", facility_type, point, exc)
                    continue

        return pd.DataFrame(all_hospitals)

    def fetch_hospitals(
        self,
        location: Tuple[float, float] = (12.9716, 77.5946),
        radius: int = 10000,
        search_points: Optional[Iterable[Tuple[float, float]]] = None,
        search_label: Optional[str] = None,
    ) -> List[Dict]:
        """Compatibility wrapper for the existing pipeline."""
        return self.collect_hospitals(
            location=location,
            radius=radius,
            search_points=search_points,
            search_label=search_label,
        ).to_dict(
            orient="records"
        )

    def _fetch_places_pages(
        self,
        location: Tuple[float, float],
        radius: int,
        facility_type: str,
        search_label: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch nearby places plus backup text search results and deduplicate by place ID."""
        merged_places: list[Dict] = []
        seen_place_ids: set[str] = set()

        if not getattr(self, "use_places_new", True):
            return self._fetch_places_pages_legacy(
                location=location,
                radius=radius,
                facility_type=facility_type,
                search_label=search_label,
            )

        try:
            response = self._search_nearby(location=location, radius=radius, facility_type=facility_type)
            for place in response.get("places", []):
                place_id = place.get("id")
                if place_id and place_id not in seen_place_ids:
                    merged_places.append(place)
                    seen_place_ids.add(place_id)

            for query in self._build_text_queries(facility_type=facility_type, search_label=search_label):
                response = self._search_text(query=query, location=location, radius=radius)
                for place in response.get("places", []):
                    place_id = place.get("id")
                    if place_id and place_id not in seen_place_ids:
                        merged_places.append(place)
                        seen_place_ids.add(place_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                logger.warning(
                    "Places API (New) returned 403. Falling back to legacy Google Maps Places client."
                )
                self.use_places_new = False
                return self._fetch_places_pages_legacy(
                    location=location,
                    radius=radius,
                    facility_type=facility_type,
                    search_label=search_label,
                )
            raise

        return merged_places

    def _fetch_places_pages_legacy(
        self,
        location: Tuple[float, float],
        radius: int,
        facility_type: str,
        search_label: Optional[str] = None,
    ) -> List[Dict]:
        merged_places: list[Dict] = []
        seen_place_ids: set[str] = set()

        response = self.gmaps.places_nearby(
            location=location,
            radius=radius,
            type=facility_type,
        )
        for place in response.get("results", []):
            normalized = self._normalize_legacy_place(place)
            place_id = normalized.get("id")
            if place_id and place_id not in seen_place_ids:
                merged_places.append(normalized)
                seen_place_ids.add(place_id)

        next_page_token = response.get("next_page_token")
        while next_page_token:
            time.sleep(2)
            response = self.gmaps.places_nearby(page_token=next_page_token)
            for place in response.get("results", []):
                normalized = self._normalize_legacy_place(place)
                place_id = normalized.get("id")
                if place_id and place_id not in seen_place_ids:
                    merged_places.append(normalized)
                    seen_place_ids.add(place_id)
            next_page_token = response.get("next_page_token")

        for query in self._build_text_queries(facility_type=facility_type, search_label=search_label):
            response = self.gmaps.places(query=query)
            for place in response.get("results", []):
                normalized = self._normalize_legacy_place(place)
                place_id = normalized.get("id")
                if place_id and place_id not in seen_place_ids:
                    merged_places.append(normalized)
                    seen_place_ids.add(place_id)

        return merged_places

    def _search_nearby(
        self,
        location: Tuple[float, float],
        radius: int,
        facility_type: str,
    ) -> Dict:
        payload = {
            "includedTypes": [facility_type],
            "includedPrimaryTypes": [facility_type],
            "maxResultCount": 20,
            "rankPreference": "DISTANCE",
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": location[0], "longitude": location[1]},
                    "radius": float(radius),
                }
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": self.SEARCH_FIELD_MASK,
        }
        response = self.http.post(self.NEARBY_SEARCH_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _build_text_queries(self, facility_type: str, search_label: Optional[str]) -> list[str]:
        normalized_label = (search_label or "").strip()
        if not normalized_label:
            return []

        return [
            f"{facility_type} in {normalized_label}",
            f"government {facility_type} in {normalized_label}",
            f"private {facility_type} in {normalized_label}",
            f"multispeciality {facility_type} in {normalized_label}",
            f"children {facility_type} in {normalized_label}",
            f"cardiac {facility_type} in {normalized_label}",
        ]

    def _search_text(
        self,
        query: str,
        location: Tuple[float, float],
        radius: int,
    ) -> Dict:
        payload = {
            "textQuery": query,
            "pageSize": 20,
            "includedType": "hospital",
            "locationBias": {
                "circle": {
                    "center": {"latitude": location[0], "longitude": location[1]},
                    "radius": float(radius),
                }
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": self.SEARCH_FIELD_MASK,
        }
        response = self.http.post(self.TEXT_SEARCH_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _fetch_place_details(self, place_id: str) -> Dict:
        if not self.use_places_new:
            details = self.gmaps.place(place_id).get("result", {})
            return self._normalize_legacy_place_details(details)

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": self.DETAILS_FIELD_MASK,
        }
        response = self.http.get(
            self.PLACE_DETAILS_URL.format(place_id=place_id),
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def _normalize_legacy_place(self, place: Dict) -> Dict:
        location = place.get("geometry", {}).get("location", {})
        return {
            "id": place.get("place_id"),
            "displayName": {"text": place.get("name", "")},
            "location": {
                "latitude": location.get("lat"),
                "longitude": location.get("lng"),
            },
            "formattedAddress": place.get("formatted_address", place.get("vicinity", "")),
            "primaryType": place.get("types", [""])[0] if place.get("types") else "",
            "types": place.get("types", []),
        }

    def _normalize_legacy_place_details(self, details: Dict) -> Dict:
        location = details.get("geometry", {}).get("location", {})
        opening_hours = details.get("opening_hours", {})
        return {
            "id": details.get("place_id"),
            "displayName": {"text": details.get("name", "")},
            "location": {
                "latitude": location.get("lat"),
                "longitude": location.get("lng"),
            },
            "formattedAddress": details.get("formatted_address", details.get("vicinity", "")),
            "primaryType": details.get("types", [""])[0] if details.get("types") else "",
            "types": details.get("types", []),
            "currentOpeningHours": {
                "openNow": opening_hours.get("open_now", False),
                "weekdayDescriptions": opening_hours.get("weekday_text", []),
            },
            "regularOpeningHours": {
                "weekdayDescriptions": opening_hours.get("weekday_text", []),
            },
            "nationalPhoneNumber": details.get("formatted_phone_number"),
            "websiteUri": details.get("website"),
            "rating": details.get("rating"),
            "userRatingCount": details.get("user_ratings_total"),
        }

    def _is_real_hospital(self, place: Dict) -> bool:
        primary_type = str(place.get("primaryType", "")).strip().lower()
        types = [str(value).strip().lower() for value in place.get("types", [])]
        display_name = place.get("displayName", {})
        name = str(display_name.get("text", place.get("name", ""))).strip().lower()

        if primary_type == "hospital" or "hospital" in types:
            return True

        reject_words = [
            "clinic",
            "diagnostic",
            "laboratory",
            "lab",
            "medical",
            "pharmacy",
            "medical store",
            "drugstore",
            "physiotherapy",
            "doctor",
            "nursing",
            "pathology",
            "chemist",
        ]
        if any(word in name for word in reject_words) and "hospital" not in name:
            return False

        return False

    def _extract_hospital_data(self, place: Dict) -> Dict:
        """Extract relevant data from a place payload."""
        place_id = place.get("id")
        if not place_id:
            return {}

        place_details = self._fetch_place_details(place_id)
        if not self._is_real_hospital(place_details):
            return {}

        opening_hours = (
            place_details.get("currentOpeningHours")
            or place_details.get("regularOpeningHours")
            or {}
        )
        display_name = place_details.get("displayName", {})
        location = place_details.get("location", place.get("location", {}))

        specialty = self._infer_specialty(place_details)
        rating = place_details.get("rating", 0.0)
        reviews = place_details.get("userRatingCount", 0)

        return {
            "name": display_name.get("text", ""),
            "location": place_details.get("formattedAddress", ""),
            "specialty": specialty,
            "facility_type": place_details.get("primaryType", "unknown"),
            "place_id": place_id,
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "vicinity": place_details.get("formattedAddress", ""),
            "rating": float(rating or 0.0),
            "reviews": int(reviews or 0),
            "is_open_now": opening_hours.get("openNow", False),
            "opening_hours": str(opening_hours.get("weekdayDescriptions", [])),
            "phone_number": place_details.get("nationalPhoneNumber", "N/A"),
            "website": place_details.get("websiteUri", "N/A"),
            "emergency_facility": self._check_emergency(place_details),
            "response_probability": self._calculate_response_prob(place_details),
            "distance_from_center": self._calculate_distance(place),
            "has_phone": 1 if place_details.get("nationalPhoneNumber") else 0,
            "has_website": 1 if place_details.get("websiteUri") else 0,
            "collected_at": datetime.now().isoformat(),
        }

    def _infer_specialty(self, place_details: Dict) -> str:
        types = " ".join(place_details.get("types", [])).lower()
        if "pharmacy" in types:
            return "Pharmacy"
        if "doctor" in types:
            return "General Practice"
        if "hospital" in types:
            return "General"
        if "clinic" in types:
            return "Clinic"
        return "Healthcare"

    def _check_emergency(self, place_details: Dict) -> bool:
        """Check if facility has emergency services."""
        types = " ".join(place_details.get("types", [])).lower()
        name = str(place_details.get("displayName", {}).get("text", "")).lower()
        emergency_keywords = ["emergency", "24/7", "icu", "trauma", "er"]
        return any(keyword in types or keyword in name for keyword in emergency_keywords)

    def _calculate_response_prob(self, place_details: Dict) -> float:
        """Estimate response probability without relying on ratings only."""
        probability = 0.5

        if (place_details.get("currentOpeningHours") or {}).get("openNow"):
            probability += 0.2
        if place_details.get("nationalPhoneNumber"):
            probability += 0.1
        if place_details.get("websiteUri"):
            probability += 0.05

        return min(1.0, probability)

    def _calculate_distance(self, place: Dict) -> float:
        """Calculate Haversine distance from the search center in meters."""
        if not self.search_center:
            return 0.0

        lat1, lon1 = self.search_center
        location = place.get("location", {})
        lat2 = location["latitude"]
        lon2 = location["longitude"]

        radius_earth = 6371000
        d_lat = radians(lat2 - lat1)
        d_lon = radians(lon2 - lon1)

        a = (
            sin(d_lat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
        )
        c = 2 * asin(sqrt(a))
        return radius_earth * c

    def save_data(self, df: pd.DataFrame, filepath: str | Path) -> Path:
        """Save collected data to CSV."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
        logger.info("Data saved to %s", target)
        return target

    def save_to_csv(
        self,
        records: List[Dict],
        output_path: Path | None = None,
    ) -> Path:
        """Compatibility wrapper for the existing pipeline."""
        ensure_directories(DATA_DIR / "raw")
        target = output_path or DATA_DIR / "raw" / "hospitals_raw_data.csv"
        return self.save_data(pd.DataFrame(records), target)


GoogleMapsHospitalCollector = GoogleMapsCollector

