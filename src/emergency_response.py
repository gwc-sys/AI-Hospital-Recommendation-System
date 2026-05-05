from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.utils import DATA_DIR, MODELS_DIR


FEATURE_NAMES = [
    "distance",
    "response_prob",
    "emergency_score",
    "has_phone",
    "has_website",
    "current_hour",
    "day_of_week",
    "distance_score",
]

COMPARED_PARAMETERS = [
    "distance_km",
    "specialization",
    "open_now",
    "emergency_available",
    "emergency_facility",
    "phone_available",
    "response_time_sec",
    "response_probability",
]

logger = logging.getLogger(__name__)
load_dotenv()


def calculate_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    from math import asin, cos, radians, sin, sqrt

    radius_earth_km = 6371.0
    delta_lat = radians(latitude_2 - latitude_1)
    delta_lon = radians(longitude_2 - longitude_1)
    a = (
        sin(delta_lat / 2) ** 2
        + cos(radians(latitude_1))
        * cos(radians(latitude_2))
        * sin(delta_lon / 2) ** 2
    )
    c = 2 * asin(sqrt(a))
    return radius_earth_km * c


class EmergencyHospitalRecommender:
    """Prepare Firebase-ready emergency payloads for the React Native app."""

    EXCLUDED_FACILITY_TERMS = {"pharmacy", "drugstore", "clinic", "medicare"}
    LIVE_EXCLUDED_NAME_TERMS = {
        "medical store",
        "pharmacy",
        "chemist",
        "clinic",
        "laboratory",
        "diagnostic",
        "engineers",
    }
    CANDIDATE_POOL_SIZE = 12
    LIVE_SEARCH_RADIUS_METERS = 5000
    LOCAL_DATASET_NEARBY_RADIUS_KM = 20.0
    SELECTION_PROFILES = {
        "balanced": {
            "distance_score": 0.22,
            "response_time_score": 0.18,
            "open_now": 0.18,
            "emergency_available": 0.18,
            "phone_available": 0.10,
            "response_probability": 0.08,
            "quality_score_norm": 0.04,
            "facility_priority": 0.02,
        },
        "best_of_best": {
            # Nearest + best quality: keep distance dominant while still preferring top hospitals.
            "distance_score": 0.36,
            "response_time_score": 0.14,
            "open_now": 0.10,
            "emergency_available": 0.14,
            "phone_available": 0.04,
            "response_probability": 0.10,
            "quality_score_norm": 0.10,
            "facility_priority": 0.02,
        },
    }

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        model_dir: Path | str | None = None,
        use_models: bool = True,
        live_lookup_enabled: bool | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else DATA_DIR / "processed" / "final_hospital_dataset.csv"
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.feature_names = FEATURE_NAMES
        self.classifier = None
        self.regressor = None
        self.live_lookup_enabled = (
            bool(live_lookup_enabled)
            if live_lookup_enabled is not None
            else dataset_path is None
        )
        self.model_status = "disabled"
        if use_models:
            self._load_models()

    def _load_models(self) -> None:
        self.classifier = self._safe_load_model(self.model_dir / "decision_tree_classifier.pkl")
        self.regressor = self._safe_load_model(self.model_dir / "decision_tree_regressor.pkl")
        if self.classifier is not None and self.regressor is not None:
            self.model_status = "ready"
        elif self.classifier is not None or self.regressor is not None:
            self.model_status = "partial"
        else:
            self.model_status = "unavailable"

    @staticmethod
    def _safe_load_model(path: Path):
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _load_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.dataset_path)
        defaults = {
            "facility_type": "hospital",
            "emergency_facility": False,
            "is_open_now": False,
            "phone_number": "N/A",
            "website": "N/A",
            "vicinity": "N/A",
            "response_probability": 0.5,
            "quality_score": 0.0,
            "has_phone": 0,
            "has_website": 0,
        }
        for column, default_value in defaults.items():
            if column not in data.columns:
                data[column] = default_value
        return data

    def _looks_like_real_hospital(self, hospital_name: Any, facility_type: Any) -> bool:
        normalized_name = str(hospital_name or "").strip().lower()
        normalized_type = str(facility_type or "").strip().lower()

        if any(term in normalized_name for term in self.LIVE_EXCLUDED_NAME_TERMS):
            return False
        if normalized_type in {"hospital", "health"}:
            return True
        return "hospital" in normalized_name

    def _enrich_hospital_defaults(self, data: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "facility_type": "hospital",
            "emergency_facility": False,
            "is_open_now": False,
            "phone_number": "N/A",
            "website": "N/A",
            "vicinity": "N/A",
            "response_probability": 0.5,
            "quality_score": 0.0,
            "has_phone": 0,
            "has_website": 0,
        }
        enriched = data.copy()
        for column, default_value in defaults.items():
            if column not in enriched.columns:
                enriched[column] = default_value

        quality_score_series = pd.to_numeric(
            enriched.get("quality_score", pd.Series(index=enriched.index, dtype="float64")),
            errors="coerce",
        )
        rating_series = pd.to_numeric(
            enriched.get("rating", pd.Series(index=enriched.index, dtype="float64")),
            errors="coerce",
        )
        enriched["quality_score"] = quality_score_series.fillna(rating_series).fillna(0.0).clip(lower=0.0)
        return enriched

    def _load_live_hospitals(self, latitude: float, longitude: float) -> pd.DataFrame:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        if not api_key:
            try:
                from config.api_config import API_KEY

                api_key = str(API_KEY or "").strip()
            except Exception:
                api_key = ""
        if not api_key:
            return pd.DataFrame()

        try:
            from src.data_collection import GoogleMapsCollector

            collector = GoogleMapsCollector(api_key=api_key)
            live_data = collector.collect_hospitals(
                location=(latitude, longitude),
                radius=self.LIVE_SEARCH_RADIUS_METERS,
                search_points=[(latitude, longitude)],
                search_label="live_location",
            )
        except Exception as exc:
            logger.warning("Live hospital lookup failed, falling back to dataset: %s", exc)
            return pd.DataFrame()

        if live_data.empty:
            return live_data

        filtered = live_data[
            live_data.apply(
                lambda row: self._looks_like_real_hospital(
                    hospital_name=row.get("name"),
                    facility_type=row.get("facility_type"),
                ),
                axis=1,
            )
        ].copy()
        return self._enrich_hospital_defaults(filtered)

    def _load_nearby_dataset_hospitals(self, latitude: float, longitude: float) -> pd.DataFrame:
        dataset_data = self._enrich_hospital_defaults(self._load_dataset())
        dataset_data = self._filter_hospital_rows(dataset_data)
        if dataset_data.empty:
            return dataset_data

        dataset_data = dataset_data.copy()
        dataset_data["distance_km"] = dataset_data.apply(
            lambda row: calculate_distance_km(
                latitude,
                longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )
        nearby = dataset_data[dataset_data["distance_km"] <= self.LOCAL_DATASET_NEARBY_RADIUS_KM].copy()
        return nearby.sort_values("distance_km", ascending=True).reset_index(drop=True)

    def _load_hospitals_for_location(self, latitude: float, longitude: float) -> tuple[pd.DataFrame, str]:
        nearby_dataset = self._load_nearby_dataset_hospitals(latitude, longitude)
        if not nearby_dataset.empty:
            return nearby_dataset, "dataset_csv_nearby"

        if self.live_lookup_enabled:
            live_data = self._load_live_hospitals(latitude, longitude)
            if not live_data.empty:
                return live_data, "live_google_maps"

        dataset_data = self._enrich_hospital_defaults(self._load_dataset())
        return dataset_data, "dataset_csv"

    @staticmethod
    def _clean_text(value: Any, fallback: str = "N/A") -> str:
        if pd.isna(value):
            return fallback
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return fallback
        return text

    def _filter_hospital_rows(self, hospitals: pd.DataFrame) -> pd.DataFrame:
        facility_types = hospitals["facility_type"].fillna("").astype(str).str.lower()
        name_values = hospitals["name"].fillna("").astype(str).str.lower()

        excluded_pattern = "|".join(sorted(self.EXCLUDED_FACILITY_TERMS))
        excluded = facility_types.str.contains(excluded_pattern, na=False) | name_values.str.contains(
            excluded_pattern,
            na=False,
        )

        hospital_like = facility_types.isin({"hospital", "health"}) | name_values.str.contains(
            "hospital",
            na=False,
        )
        return hospitals[hospital_like & ~excluded].copy()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clean_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _has_contact_value(value: Any) -> int:
        if pd.isna(value):
            return 0
        text = str(value).strip().lower()
        return int(bool(text) and text != "nan" and text != "n/a")

    @staticmethod
    def _normalize_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        minimum = numeric.min()
        maximum = numeric.max()

        if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
            return pd.Series(1.0, index=series.index)

        normalized = (numeric - minimum) / (maximum - minimum)
        if higher_is_better:
            return normalized.clip(0, 1)
        return (1 - normalized).clip(0, 1)

    @staticmethod
    def _facility_priority_score(facility_type: Any, hospital_name: Any) -> float:
        normalized_type = str(facility_type or "").strip().lower()
        normalized_name = str(hospital_name or "").strip().lower()

        if normalized_type == "hospital":
            return 1.0
        if normalized_type == "health":
            return 0.9
        if normalized_type == "establishment" and "hospital" in normalized_name:
            return 0.7
        if normalized_type == "doctor" and "hospital" in normalized_name:
            return 0.45
        if "hospital" in normalized_name:
            return 0.6
        return 0.25

    @staticmethod
    def _estimate_response_time_sec(
        distance_km: float,
        open_now: bool,
        emergency_available: bool,
        response_probability: float,
        phone_available: bool,
    ) -> int:
        travel_seconds = max(distance_km, 0.05) / 32 * 3600
        open_multiplier = 0.95 if open_now else 1.18
        emergency_multiplier = 0.92 if emergency_available else 1.10
        phone_multiplier = 0.96 if phone_available else 1.06
        readiness_multiplier = 1.18 - (max(min(response_probability, 1.0), 0.0) * 0.28)
        estimated = travel_seconds * open_multiplier * emergency_multiplier * phone_multiplier * readiness_multiplier
        return max(60, int(round(estimated)))

    def _prepare_candidate_pool(self, hospitals: pd.DataFrame) -> pd.DataFrame:
        candidate_pool_size = min(len(hospitals), self.CANDIDATE_POOL_SIZE)
        ranked = hospitals.nsmallest(candidate_pool_size, "distance_km").copy()

        ranked["distance_km"] = pd.to_numeric(ranked["distance_km"], errors="coerce").fillna(float("inf"))
        ranked["response_probability"] = (
            pd.to_numeric(ranked["response_probability"], errors="coerce").fillna(0.5).clip(0, 1)
        )
        ranked["quality_score"] = pd.to_numeric(ranked["quality_score"], errors="coerce").fillna(0).clip(lower=0)
        ranked["phone_available"] = ranked["phone_number"].apply(self._has_contact_value).astype(int)
        ranked["website_available"] = ranked["website"].apply(self._has_contact_value).astype(int)
        ranked["has_phone"] = ranked.get("has_phone", ranked["phone_available"]).fillna(ranked["phone_available"]).astype(int)
        ranked["has_website"] = (
            ranked.get("has_website", ranked["website_available"]).fillna(ranked["website_available"]).astype(int)
        )
        ranked["open_now"] = ranked["is_open_now"].apply(self._clean_bool)
        ranked["emergency_available"] = ranked["emergency_facility"].apply(self._clean_bool)
        ranked["distance_score"] = self._normalize_series(ranked["distance_km"], higher_is_better=False)
        ranked["response_time_sec"] = ranked.apply(
            lambda row: self._estimate_response_time_sec(
                distance_km=float(row["distance_km"]),
                open_now=bool(row["open_now"]),
                emergency_available=bool(row["emergency_available"]),
                response_probability=float(row["response_probability"]),
                phone_available=bool(row["phone_available"]),
            ),
            axis=1,
        )
        ranked["response_time_score"] = self._normalize_series(ranked["response_time_sec"], higher_is_better=False)
        ranked["quality_score_norm"] = self._normalize_series(ranked["quality_score"], higher_is_better=True)
        ranked["facility_priority"] = ranked.apply(
            lambda row: self._facility_priority_score(row.get("facility_type"), row.get("name")),
            axis=1,
        )
        now = datetime.now()
        ranked["current_hour"] = now.hour
        ranked["day_of_week"] = now.weekday()
        return ranked

    def _build_model_feature_frame(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "distance": candidates["distance_km"],
                "response_prob": candidates["response_probability"],
                "emergency_score": candidates["emergency_available"].astype(int),
                "has_phone": candidates["phone_available"].astype(int),
                "has_website": candidates["website_available"].astype(int),
                "current_hour": candidates["current_hour"].astype(int),
                "day_of_week": candidates["day_of_week"].astype(int),
                "distance_score": candidates["distance_score"],
            }
        )

    def _apply_model_scores(self, candidates: pd.DataFrame) -> pd.DataFrame:
        ranked = candidates.copy()
        ranked["ml_classifier_score"] = 0.0
        ranked["ml_regression_score_raw"] = 0.0
        ranked["ml_score"] = 0.0
        ranked["ml_confidence"] = 0.0
        ranked["ml_decision"] = "consider"

        if self.classifier is None and self.regressor is None:
            return ranked

        feature_frame = self._build_model_feature_frame(ranked)

        classifier_score = pd.Series(0.0, index=ranked.index)
        classifier_confidence = pd.Series(0.0, index=ranked.index)
        classifier_decision = pd.Series("consider", index=ranked.index, dtype="object")

        if self.classifier is not None:
            labels = self.classifier.predict(feature_frame[self.feature_names])
            classifier_decision = pd.Series(
                ["recommended" if int(label) == 1 else "consider" for label in labels],
                index=ranked.index,
                dtype="object",
            )

            if hasattr(self.classifier, "predict_proba"):
                probabilities = self.classifier.predict_proba(feature_frame[self.feature_names])
                classes = list(getattr(self.classifier, "classes_", []))
                if 1 in classes:
                    positive_index = classes.index(1)
                    classifier_score = pd.Series(probabilities[:, positive_index], index=ranked.index)
                else:
                    classifier_score = pd.Series(probabilities.max(axis=1), index=ranked.index)
                classifier_confidence = pd.Series(probabilities.max(axis=1), index=ranked.index)
            else:
                classifier_score = pd.Series([1.0 if decision == "recommended" else 0.4 for decision in classifier_decision], index=ranked.index)
                classifier_confidence = pd.Series(0.5, index=ranked.index)

        regression_score_raw = pd.Series(0.0, index=ranked.index)
        regression_score_norm = pd.Series(0.0, index=ranked.index)
        if self.regressor is not None:
            regression_score_raw = pd.Series(
                self.regressor.predict(feature_frame[self.feature_names]),
                index=ranked.index,
            )
            regression_score_norm = self._normalize_series(regression_score_raw, higher_is_better=True)

        if self.classifier is not None and self.regressor is not None:
            combined_model_score = regression_score_norm * 0.6 + classifier_score * 0.4
        elif self.regressor is not None:
            combined_model_score = regression_score_norm
        else:
            combined_model_score = classifier_score

        ranked["ml_classifier_score"] = classifier_score.clip(0, 1)
        ranked["ml_regression_score_raw"] = regression_score_raw
        ranked["ml_score"] = combined_model_score.clip(0, 1)
        ranked["ml_confidence"] = classifier_confidence.clip(0, 1)
        ranked["ml_decision"] = classifier_decision
        return ranked

    def _apply_parameter_scores(
        self,
        candidates: pd.DataFrame,
        selection_preference: str = "balanced",
    ) -> pd.DataFrame:
        ranked = candidates.copy()
        normalized = str(selection_preference or "balanced").strip().lower()
        profile = self.SELECTION_PROFILES.get(normalized, self.SELECTION_PROFILES["balanced"])
        ranked["parameter_score"] = (
            ranked["distance_score"] * profile["distance_score"]
            + ranked["response_time_score"] * profile["response_time_score"]
            + ranked["open_now"].astype(int) * profile["open_now"]
            + ranked["emergency_available"].astype(int) * profile["emergency_available"]
            + ranked["phone_available"].astype(int) * profile["phone_available"]
            + ranked["response_probability"] * profile["response_probability"]
            + ranked["quality_score_norm"] * profile["quality_score_norm"]
            + ranked["facility_priority"] * profile["facility_priority"]
        ).clip(0, 1)
        return ranked

    def _rank_emergency_candidates(
        self,
        hospitals: pd.DataFrame,
        selection_preference: str = "balanced",
    ) -> pd.DataFrame:
        ranked = self._prepare_candidate_pool(hospitals)
        ranked = self._apply_model_scores(ranked)
        ranked = self._apply_parameter_scores(
            ranked,
            selection_preference=selection_preference,
        )

        if self.classifier is not None or self.regressor is not None:
            ranked["selection_method"] = "decision_tree_hybrid"
            ranked["fallback_used"] = False
            ranked["final_score"] = (ranked["parameter_score"] * 0.55 + ranked["ml_score"] * 0.45).clip(0, 1)
        else:
            ranked["selection_method"] = "weighted_parameter_fallback"
            ranked["fallback_used"] = True
            ranked["final_score"] = ranked["parameter_score"].clip(0, 1)

        ranked["selection_score"] = ranked["final_score"]
        return ranked.sort_values(
            ["final_score", "distance_km", "quality_score"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    def _build_hospital_payload(
        self,
        hospital_row: pd.Series,
        reason: str,
        rank_position: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "hospital_name": self._clean_text(hospital_row["name"]),
            "distance_km": round(float(hospital_row["distance_km"]), 3),
            "address": self._clean_text(hospital_row.get("vicinity", "N/A")),
            "specialization": self._clean_text(hospital_row.get("specialty", "General"), fallback="General"),
            "phone": self._clean_text(hospital_row.get("phone_number", "N/A")),
            "website": self._clean_text(hospital_row.get("website", "N/A")),
            "open_now": bool(hospital_row.get("open_now", hospital_row.get("is_open_now", False))),
            "emergency_available": bool(
                hospital_row.get("emergency_available", hospital_row.get("emergency_facility", False))
            ),
            "emergency_facility": bool(
                hospital_row.get("emergency_facility", hospital_row.get("emergency_available", False))
            ),
            "phone_available": bool(hospital_row.get("phone_available", 0)),
            "website_available": bool(hospital_row.get("website_available", 0)),
            "facility_type": self._clean_text(
                hospital_row.get("facility_type", "hospital"),
                fallback="hospital",
            ),
            "latitude": self._safe_float(hospital_row.get("latitude", 0.0)),
            "longitude": self._safe_float(hospital_row.get("longitude", 0.0)),
            "reason": reason,
            "response_probability": round(float(hospital_row.get("response_probability", 0.0)), 3),
            "response_time_sec": int(hospital_row.get("response_time_sec", 0)),
            "quality_score": round(float(hospital_row.get("quality_score", 0.0)), 3),
            "parameter_score": round(float(hospital_row.get("parameter_score", 0.0)), 3),
            "ml_score": round(float(hospital_row.get("ml_score", 0.0)), 3),
            "ml_decision": self._clean_text(hospital_row.get("ml_decision", "consider"), fallback="consider"),
            "ml_confidence": round(float(hospital_row.get("ml_confidence", 0.0)), 3),
            "selection_method": self._clean_text(
                hospital_row.get("selection_method", "weighted_parameter_fallback"),
                fallback="weighted_parameter_fallback",
            ),
            "fallback_used": bool(hospital_row.get("fallback_used", False)),
            "final_score": round(float(hospital_row.get("final_score", 0.0)), 3),
            "selection_score": round(float(hospital_row.get("selection_score", 0.0)), 3),
            "comparison_parameters": COMPARED_PARAMETERS,
            "map_url": (
                "https://www.google.com/maps?q="
                f"{self._safe_float(hospital_row.get('latitude', 0.0))},"
                f"{self._safe_float(hospital_row.get('longitude', 0.0))}"
            ),
        }
        if rank_position is not None:
            payload["rank_position"] = rank_position
        return payload

    def find_best_hospital_options(
        self,
        latitude: float,
        longitude: float,
        require_emergency: bool = False,
        only_open: bool = False,
        selection_preference: str = "balanced",
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        hospitals, data_source = self._load_hospitals_for_location(latitude, longitude)
        hospitals = self._filter_hospital_rows(hospitals)
        hospitals["distance_km"] = hospitals.apply(
            lambda row: calculate_distance_km(
                latitude,
                longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )

        if require_emergency:
            hospitals = hospitals[hospitals["emergency_facility"].astype(bool)]
        if only_open:
            hospitals = hospitals[hospitals["is_open_now"].astype(bool)]

        if hospitals.empty:
            raise ValueError("No hospitals matched the emergency filters.")

        normalized_preference = str(selection_preference or "balanced").strip().lower()
        if normalized_preference not in self.SELECTION_PROFILES:
            normalized_preference = "balanced"

        ranked = self._rank_emergency_candidates(
            hospitals,
            selection_preference=normalized_preference,
        )
        best = ranked.iloc[0]
        best_reason = (
            "Best nearby hospital after comparing distance, open status, emergency availability, "
            "phone availability, response time, response probability, and the decision tree model."
        )
        nearby_options = [
            self._build_hospital_payload(
                row,
                reason="Nearby hospital evaluated for this SOS",
                rank_position=index + 1,
            )
            for index, (_, row) in enumerate(ranked.head(5).iterrows())
        ]
        selection_metadata = {
            "primary_algorithm": "Decision Tree / Simple ML model",
            "selection_method": self._clean_text(best.get("selection_method", "weighted_parameter_fallback")),
            "model_status": self.model_status,
            "data_source": data_source,
            "fallback_system": "Weighted parameter scoring fallback",
            "fallback_used": bool(best.get("fallback_used", False)),
            "compared_parameters": COMPARED_PARAMETERS,
            "response_time_source": "estimated_from_distance_and_readiness",
            "compared_hospitals": int(len(nearby_options)),
            "candidate_pool_size": int(len(ranked)),
            "selection_preference": normalized_preference,
        }
        return (
            self._build_hospital_payload(best, reason=best_reason, rank_position=1),
            nearby_options,
            selection_metadata,
        )

    def find_nearest_hospital(
        self,
        latitude: float,
        longitude: float,
        require_emergency: bool = False,
        only_open: bool = False,
        selection_preference: str = "balanced",
    ) -> dict[str, Any]:
        best_hospital, _, _ = self.find_best_hospital_options(
            latitude=latitude,
            longitude=longitude,
            require_emergency=require_emergency,
            only_open=only_open,
            selection_preference=selection_preference,
        )
        return best_hospital

    def build_react_native_payload(
        self,
        emergency_event: dict[str, Any],
        require_emergency: bool = False,
        only_open: bool = False,
    ) -> dict[str, Any]:
        location = emergency_event.get("location", {})
        refresh_metadata = dict(emergency_event.get("refresh_metadata", {}))
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
        location_payload = {
            "latitude": latitude,
            "longitude": longitude,
            "map_url": f"https://www.google.com/maps?q={latitude},{longitude}",
        }
        for extra_key, extra_value in location.items():
            if extra_key not in {"latitude", "longitude"}:
                location_payload[extra_key] = extra_value

        hospital, nearby_hospitals, selection_metadata = self.find_best_hospital_options(
            latitude=latitude,
            longitude=longitude,
            require_emergency=require_emergency,
            only_open=only_open,
            selection_preference=str(emergency_event.get("selection_preference", "balanced")),
        )

        timestamp = emergency_event.get("last_updated") or datetime.now(timezone.utc).isoformat()
        sos_type = emergency_event.get("type", "SOS Emergency")
        priority = emergency_event.get("priority", "CRITICAL")
        device_name = emergency_event.get("device_name", "Ai-based-smart-vehicle-health")
        selected_at = datetime.now(timezone.utc).isoformat()
        assigned_hospital = {
            "hospital_name": hospital["hospital_name"],
            "distance_km": hospital["distance_km"],
            "address": hospital["address"],
            "specialization": hospital["specialization"],
            "phone": hospital["phone"],
            "emergency_available": hospital["emergency_available"],
            "emergency_facility": hospital["emergency_facility"],
            "map_url": hospital["map_url"],
            "selected_at": selected_at,
        }

        return {
            "sos_alert": {
                "active": bool(emergency_event.get("active", True)),
                "type": sos_type,
                "priority": priority,
                "device_name": device_name,
                "last_updated": timestamp,
                "trigger_source": emergency_event.get("trigger_source", "vehicle_ai"),
                "location": location_payload,
                "health": emergency_event.get("health", {}),
                "refresh_metadata": refresh_metadata,
                "assigned_hospital": assigned_hospital,
                # Legacy flat keys for clients expecting hospital fields directly under sos_alert.
                "hospital_name": assigned_hospital["hospital_name"],
                "distance_km": assigned_hospital["distance_km"],
                "address": assigned_hospital["address"],
                "specialization": assigned_hospital["specialization"],
                "hospital_phone": assigned_hospital["phone"],
                "emergency_available": assigned_hospital["emergency_available"],
                "emergency_facility": assigned_hospital["emergency_facility"],
            },
            "ai_results": {
                "hospital_ai": {
                    **hospital,
                    "selected_at": selected_at,
                    "ui_card_title": "Find Hospital",
                    "ui_card_subtitle": "Best hospital after AI comparison",
                },
                "nearby_hospitals": nearby_hospitals,
                "hospital_selection": selection_metadata,
                "payload_refresh": refresh_metadata,
            },
            # Legacy top-level snapshot for clients that do not read nested ai_results.
            "assigned_hospital_from_firebase": assigned_hospital,
        }
