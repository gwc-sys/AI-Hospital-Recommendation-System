"""
Google Maps API data collection module.
"""

from __future__ import annotations

from datetime import datetime
import logging
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import googlemaps
import pandas as pd

from config.api_config import API_KEY
from src.utils import DATA_DIR, ensure_directories


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleMapsCollector:
    """Handles data collection from Google Maps Places API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or API_KEY
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "A valid GOOGLE_MAPS_API_KEY is required in the environment or .env file."
            )

        self.gmaps = googlemaps.Client(key=self.api_key)
        self.facility_types = ["hospital", "doctor", "health", "clinic", "pharmacy"]
        self.search_center: Optional[Tuple[float, float]] = None

    def collect_hospitals(
        self,
        location: Tuple[float, float],
        radius: int = 10000,
    ) -> pd.DataFrame:
        """Collect facility data around a latitude/longitude point."""
        self.search_center = location
        all_hospitals: list[Dict] = []
        seen_place_ids: set[str] = set()

        for facility_type in self.facility_types:
            logger.info("Collecting %s data...", facility_type)

            try:
                places = self.gmaps.places_nearby(
                    location=location,
                    radius=radius,
                    type=facility_type,
                )

                for place in places.get("results", []):
                    place_id = place.get("place_id")
                    if not place_id or place_id in seen_place_ids:
                        continue

                    hospital_data = self._extract_hospital_data(place)
                    all_hospitals.append(hospital_data)
                    seen_place_ids.add(place_id)

                time.sleep(0.1)

            except Exception as exc:
                logger.error("Error collecting %s: %s", facility_type, exc)
                continue

        return pd.DataFrame(all_hospitals)

    def fetch_hospitals(
        self,
        location: Tuple[float, float] = (12.9716, 77.5946),
        radius: int = 10000,
    ) -> List[Dict]:
        """Compatibility wrapper for the existing pipeline."""
        return self.collect_hospitals(location=location, radius=radius).to_dict(
            orient="records"
        )

    def _extract_hospital_data(self, place: Dict) -> Dict:
        """Extract relevant data from a place payload."""
        place_details = self.gmaps.place(place["place_id"]).get("result", {})
        opening_hours = place_details.get("opening_hours", {})

        specialty = self._infer_specialty(place_details)
        rating = place_details.get("rating", place.get("rating", 0.0))
        reviews = place_details.get("user_ratings_total", place.get("user_ratings_total", 0))

        return {
            "name": place.get("name", ""),
            "location": place.get("vicinity", ""),
            "specialty": specialty,
            "facility_type": place.get("types", ["unknown"])[0],
            "place_id": place.get("place_id", ""),
            "latitude": place["geometry"]["location"]["lat"],
            "longitude": place["geometry"]["location"]["lng"],
            "vicinity": place.get("vicinity", ""),
            "rating": float(rating or 0.0),
            "reviews": int(reviews or 0),
            "is_open_now": opening_hours.get("open_now", False),
            "opening_hours": str(opening_hours.get("weekday_text", [])),
            "phone_number": place_details.get("formatted_phone_number", "N/A"),
            "website": place_details.get("website", "N/A"),
            "emergency_facility": self._check_emergency(place_details),
            "response_probability": self._calculate_response_prob(place_details),
            "distance_from_center": self._calculate_distance(place),
            "has_phone": 1 if place_details.get("formatted_phone_number") else 0,
            "has_website": 1 if place_details.get("website") else 0,
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
        emergency_keywords = ["emergency", "24/7", "icu", "trauma", "er"]
        return any(keyword in types for keyword in emergency_keywords)

    def _calculate_response_prob(self, place_details: Dict) -> float:
        """Estimate response probability without relying on ratings only."""
        probability = 0.5

        if place_details.get("opening_hours", {}).get("open_now"):
            probability += 0.2
        if place_details.get("formatted_phone_number"):
            probability += 0.1
        if place_details.get("website"):
            probability += 0.05

        return min(1.0, probability)

    def _calculate_distance(self, place: Dict) -> float:
        """Calculate Haversine distance from the search center in meters."""
        if not self.search_center:
            return 0.0

        lat1, lon1 = self.search_center
        lat2 = place["geometry"]["location"]["lat"]
        lon2 = place["geometry"]["location"]["lng"]

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

