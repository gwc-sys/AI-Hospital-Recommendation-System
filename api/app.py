"""
FastAPI Application for Hospital Recommendation System.
"""

from __future__ import annotations

from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_hospital_dataset.csv"

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

EXCLUDED_FACILITY_TERMS = {"pharmacy", "drugstore", "clinic", "medicare"}


app = FastAPI(title="Hospital Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    latitude: float
    longitude: float
    max_distance_km: float = Field(default=5.0, gt=0)
    priority: str = "balanced"
    require_emergency: bool = False
    only_open: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class HospitalRecommendation(BaseModel):
    hospital_name: str
    type: str
    latitude: float
    longitude: float
    address: str
    open_now: bool
    phone: str
    website: str
    emergency_facility: bool
    response_probability: float
    distance: float
    recommendation_score: float
    ml_decision: str
    confidence: float


class UserLocationDetails(BaseModel):
    ip_address: str
    forwarded_for: List[str]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    timezone: Optional[str] = None
    accuracy_km: Optional[float] = None
    source: str
    precision: str
    note: str


class ModelManager:
    def __init__(self) -> None:
        self.feature_names = FEATURE_NAMES
        self.classifier = None
        self.regressor = None
        self.data = pd.DataFrame()
        self.model_status = "not_loaded"
        self.reload()

    def reload(self) -> None:
        self.data = self._load_data()
        self.classifier = self._safe_load_model(MODELS_DIR / "decision_tree_classifier.pkl")
        self.regressor = self._safe_load_model(MODELS_DIR / "decision_tree_regressor.pkl")
        self.model_status = (
            "ready" if self.classifier is not None and self.regressor is not None else "partial"
        )

    def _safe_load_model(self, path: Path):
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _load_data(self) -> pd.DataFrame:
        if not DATASET_PATH.exists():
            return pd.DataFrame()

        data = pd.read_csv(DATASET_PATH)
        defaults = {
            "facility_type": "hospital",
            "distance_from_center": 0.0,
            "response_probability": 0.5,
            "emergency_facility": False,
            "has_phone": 0,
            "has_website": 0,
            "is_open_now": False,
            "phone_number": "N/A",
            "website": "N/A",
            "collected_at": datetime.now().isoformat(),
        }
        for column, default_value in defaults.items():
            if column not in data.columns:
                data[column] = default_value
        return data


manager = ModelManager()


def calculate_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
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


def apply_priority_weights(data: pd.DataFrame, priority: str) -> pd.Series:
    normalized = priority.strip().lower()
    weights = {
        "distance": {"distance": 0.55, "response": 0.15, "emergency": 0.20, "availability": 0.10},
        "emergency": {"distance": 0.15, "response": 0.20, "emergency": 0.50, "availability": 0.15},
        "response": {"distance": 0.20, "response": 0.50, "emergency": 0.15, "availability": 0.15},
        "balanced": {"distance": 0.30, "response": 0.30, "emergency": 0.20, "availability": 0.20},
    }
    selected = weights.get(normalized, weights["balanced"])

    max_distance = data["distance_km"].max()
    if pd.isna(max_distance) or max_distance <= 0:
        distance_score = pd.Series(1.0, index=data.index)
    else:
        distance_score = 1 - (data["distance_km"] / max_distance)

    availability = (
        data["has_phone"].astype(int) * 0.5
        + data["has_website"].astype(int) * 0.3
        + data["is_open_now"].astype(int) * 0.2
    )

    return (
        distance_score * selected["distance"]
        + data["response_probability"].clip(0, 1) * selected["response"]
        + data["emergency_facility"].astype(int) * selected["emergency"]
        + availability.clip(0, 1) * selected["availability"]
    )


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")
    forwarded_for = [ip.strip() for ip in forwarded_for if ip.strip()]
    if forwarded_for:
        return forwarded_for[0]

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def get_forwarded_for_chain(request: Request) -> List[str]:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")
    return [ip.strip() for ip in forwarded_for if ip.strip()]


def _safe_float_header(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_location_from_headers(request: Request) -> UserLocationDetails:
    """
    Build a best-effort location response from proxy/CDN headers.

    Precise device GPS coordinates cannot be determined by the backend alone unless the
    client sends them explicitly after user consent.
    """
    headers = request.headers
    latitude = _safe_float_header(headers.get("x-user-latitude") or headers.get("cf-iplatitude"))
    longitude = _safe_float_header(headers.get("x-user-longitude") or headers.get("cf-iplongitude"))

    city = headers.get("x-user-city") or headers.get("x-vercel-ip-city") or headers.get("cf-ipcity")
    region = (
        headers.get("x-user-region")
        or headers.get("x-vercel-ip-country-region")
        or headers.get("cf-region")
    )
    country = (
        headers.get("x-user-country")
        or headers.get("x-vercel-ip-country")
        or headers.get("cf-ipcountry")
    )
    postal_code = headers.get("x-user-postal-code") or headers.get("x-vercel-ip-postal-code")
    timezone = headers.get("x-user-timezone") or headers.get("x-vercel-ip-timezone") or headers.get("cf-timezone")
    accuracy_km = _safe_float_header(headers.get("x-user-location-accuracy-km"))

    if latitude is not None and longitude is not None:
        source = "client_coordinates"
        precision = "precise"
        note = "Coordinates were supplied by the client request headers."
    elif any([city, region, country, postal_code, timezone]):
        source = "proxy_geo_headers"
        precision = "approximate"
        note = "Location was inferred from proxy or hosting provider geo headers and may be approximate."
    else:
        source = "ip_only"
        precision = "unknown"
        note = "Backend can only see IP/request metadata here. Exact GPS location requires coordinates from the client."

    return UserLocationDetails(
        ip_address=get_client_ip(request),
        forwarded_for=get_forwarded_for_chain(request),
        latitude=latitude,
        longitude=longitude,
        city=city,
        region=region,
        country=country,
        postal_code=postal_code,
        timezone=timezone,
        accuracy_km=accuracy_km,
        source=source,
        precision=precision,
        note=note,
    )


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    collected_at = pd.to_datetime(data["collected_at"], errors="coerce")
    max_distance = data["distance_km"].max()
    if pd.isna(max_distance) or max_distance <= 0:
        distance_score = pd.Series(1.0, index=data.index)
    else:
        distance_score = 1 - (data["distance_km"] / max_distance)

    return pd.DataFrame(
        {
            "distance": data["distance_km"],
            "response_prob": data["response_probability"].clip(0, 1),
            "emergency_score": data["emergency_facility"].astype(int),
            "has_phone": data["has_phone"].astype(int),
            "has_website": data["has_website"].astype(int),
            "current_hour": collected_at.dt.hour.fillna(datetime.now().hour).astype(int),
            "day_of_week": collected_at.dt.dayofweek.fillna(datetime.now().weekday()).astype(int),
            "distance_score": distance_score.clip(0, 1),
        }
    )


def filter_hospital_candidates(data: pd.DataFrame) -> pd.DataFrame:
    facility_types = data["facility_type"].fillna("").astype(str).str.lower()
    name_values = data["name"].fillna("").astype(str).str.lower()
    excluded_pattern = "|".join(sorted(EXCLUDED_FACILITY_TERMS))

    excluded = facility_types.str.contains(excluded_pattern, na=False) | name_values.str.contains(
        excluded_pattern,
        na=False,
    )
    hospital_like = facility_types.isin({"hospital", "health"}) | name_values.str.contains(
        "hospital",
        na=False,
    )
    return data[hospital_like & ~excluded].copy()


@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "message": "Hospital Recommendation System API",
        "version": "1.0.0",
        "endpoints": ["/recommend", "/health", "/stats", "/location/current"],
    }


@app.get("/location/current", response_model=UserLocationDetails)
async def get_current_user_location(request: Request) -> UserLocationDetails:
    """Return best-effort current user location details from request metadata."""
    return extract_location_from_headers(request)


@app.post("/recommend", response_model=List[HospitalRecommendation])
async def get_recommendations(request: RecommendationRequest) -> List[HospitalRecommendation]:
    """Get hospital recommendations based on location and preferences."""
    try:
        if manager.data.empty:
            raise HTTPException(status_code=503, detail="Hospital dataset is not available.")

        candidates = filter_hospital_candidates(manager.data.copy())
        candidates["distance_km"] = candidates.apply(
            lambda row: calculate_distance_km(
                request.latitude,
                request.longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )

        candidates = candidates[candidates["distance_km"] <= request.max_distance_km]
        if request.require_emergency:
            candidates = candidates[candidates["emergency_facility"].astype(bool)]
        if request.only_open:
            candidates = candidates[candidates["is_open_now"].astype(bool)]

        if candidates.empty:
            return []

        feature_frame = build_feature_frame(candidates)

        if manager.classifier is not None:
            ml_labels = manager.classifier.predict(feature_frame[manager.feature_names])
            if hasattr(manager.classifier, "predict_proba"):
                probabilities = manager.classifier.predict_proba(feature_frame[manager.feature_names])
                confidences = probabilities.max(axis=1)
            else:
                confidences = [0.5] * len(candidates)
        else:
            ml_labels = (candidates["response_probability"] >= 0.6).astype(int)
            confidences = [0.5] * len(candidates)

        if manager.regressor is not None:
            recommendation_scores = manager.regressor.predict(feature_frame[manager.feature_names])
        else:
            recommendation_scores = apply_priority_weights(candidates, request.priority).to_numpy()

        candidates["recommendation_score"] = recommendation_scores
        candidates["ml_decision"] = ["recommended" if label == 1 else "consider" for label in ml_labels]
        candidates["confidence"] = confidences
        candidates["priority_score"] = apply_priority_weights(candidates, request.priority)

        ranked = candidates.sort_values(
            ["recommendation_score", "priority_score", "distance_km"],
            ascending=[False, False, True],
        ).head(request.limit)

        return [
            HospitalRecommendation(
                hospital_name=str(row["name"]),
                type=str(row.get("facility_type", "hospital")),
                latitude=float(row.get("latitude", 0.0)),
                longitude=float(row.get("longitude", 0.0)),
                address=str(row.get("vicinity", "N/A")),
                open_now=bool(row.get("is_open_now", False)),
                phone=str(row.get("phone_number", "N/A")),
                website=str(row.get("website", "N/A")),
                emergency_facility=bool(row.get("emergency_facility", False)),
                response_probability=round(float(row.get("response_probability", 0.0)), 3),
                distance=round(float(row["distance_km"]), 3),
                recommendation_score=round(float(row["recommendation_score"]), 3),
                ml_decision=str(row["ml_decision"]),
                confidence=round(float(row["confidence"]), 3),
            )
            for _, row in ranked.iterrows()
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": manager.model_status,
        "dataset_loaded": not manager.data.empty,
    }


@app.get("/stats")
async def get_stats() -> dict:
    """Get dataset statistics."""
    if manager.data.empty:
        raise HTTPException(status_code=503, detail="Hospital dataset is not available.")

    collected_at = pd.to_datetime(manager.data["collected_at"], errors="coerce")
    last_updated = collected_at.max()
    return {
        "total_hospitals": int(len(manager.data)),
        "emergency_available": int(manager.data["emergency_facility"].astype(int).sum()),
        "avg_response_probability": float(manager.data["response_probability"].mean()),
        "open_now": int(manager.data["is_open_now"].astype(int).sum()),
        "last_updated": last_updated.isoformat() if pd.notna(last_updated) else None,
    }
