"""
Main pipeline execution script.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection import GoogleMapsCollector
from src.data_processing import HospitalDataProcessor
from src.model_training import HospitalModelTrainer


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def normalize_location_name(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def resolve_runtime_location(
    config: dict,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    label: str | None = None,
) -> dict:
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise ValueError("Both --latitude and --longitude are required for coordinate-based runs.")

        location_name = normalize_location_name(label) or "live_location"
        return {
            "name": location_name,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "search_points": [
                {
                    "name": location_name,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                }
            ],
            "source": "coordinates",
        }

    return resolve_city_config(config, city)


def resolve_city_config(config: dict, city: str | None = None) -> dict:
    """Resolve the active city configuration."""
    cities = config["data_collection"].get("cities", {})
    city_name = normalize_location_name(city or config["data_collection"].get("default_city", "pune"))
    city_config = cities.get(city_name)
    if city_config:
        return {"name": city_name, **city_config}

    for parent_city_name, parent_city_config in cities.items():
        for point in parent_city_config.get("search_points", []):
            point_name = normalize_location_name(point.get("name"))
            if point_name == city_name:
                return {
                    "name": point_name,
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                    "search_points": [
                        {
                            "name": point_name,
                            "latitude": point["latitude"],
                            "longitude": point["longitude"],
                        }
                    ],
                    "parent_city": parent_city_name,
                }

    available_cities = sorted(cities.keys())
    available_points = sorted(
        normalize_location_name(point.get("name"))
        for city_config in cities.values()
        for point in city_config.get("search_points", [])
        if point.get("name")
    )
    available = ", ".join(available_cities + available_points)
    raise ValueError(f"Unknown city '{city_name}'. Available cities/search points: {available}")


def run_full_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    label: str | None = None,
):
    """Run complete data collection, processing, and model training pipeline."""
    config = load_config(config_path)
    city_config = resolve_runtime_location(
        config,
        city=city,
        latitude=latitude,
        longitude=longitude,
        label=label,
    )

    logger.info("Step 1: Collecting data from Google Maps API for %s", city_config["name"])
    collector = GoogleMapsCollector(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))

    location = (
        city_config["latitude"],
        city_config["longitude"],
    )
    search_points = [
        (point["latitude"], point["longitude"])
        for point in city_config.get("search_points", [])
    ] or [location]
    radius = config["google_maps"]["search_radius"]

    df_raw = collector.collect_hospitals(
        location=location,
        radius=radius,
        search_points=search_points,
        search_label=city_config["name"].replace("_", " "),
    )
    if df_raw.empty:
        raise ValueError("No hospital data was collected from Google Maps API.")

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_path, index=False)
    logger.info("Collected %s facilities", len(df_raw))

    logger.info("Step 2: Processing data")
    processor = HospitalDataProcessor()
    enhanced_path = PROJECT_ROOT / config["data"]["enhanced_path"]
    final_path = PROJECT_ROOT / config["data"]["processed_path"]
    processor.process(
        input_path=raw_path,
        enhanced_path=enhanced_path,
        final_path=final_path,
    )
    df_processed = None
    if final_path.exists():
        df_processed = pd.read_csv(final_path)

    logger.info("Step 3: Training decision tree models")
    trainer = HospitalModelTrainer(config)
    metrics = trainer.train(final_path)

    logger.info("Pipeline completed successfully")
    return {
        "raw_path": str(raw_path),
        "enhanced_path": str(enhanced_path),
        "final_path": str(final_path),
        "metrics": metrics,
        "rows_collected": len(df_raw),
        "rows_processed": 0 if df_processed is None else len(df_processed),
        "city": city_config["name"],
        "source": city_config.get("source", "config"),
    }


def update_data(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    label: str | None = None,
):
    """Refresh the raw and processed datasets using the configured location."""
    result = run_full_pipeline(
        config_path,
        city=city,
        latitude=latitude,
        longitude=longitude,
        label=label,
    )
    logger.info("Update completed with %s collected rows", result["rows_collected"])
    return result


def train_only(config_path: str | Path = DEFAULT_CONFIG_PATH):
    """Train models using the existing processed dataset."""
    config = load_config(config_path)
    final_path = PROJECT_ROOT / config["data"]["processed_path"]
    if not final_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {final_path}")

    logger.info("Training models using %s", final_path)
    trainer = HospitalModelTrainer(config)
    metrics = trainer.train(final_path)
    logger.info("Training completed successfully")
    return {"final_path": str(final_path), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "update", "train"], default="full")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--city", nargs="+", default=None)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--label", nargs="+", default=None)
    args = parser.parse_args()
    city_value = " ".join(args.city) if args.city else None
    label_value = " ".join(args.label) if args.label else None

    if args.mode == "full":
        print(
            run_full_pipeline(
                args.config,
                city=city_value,
                latitude=args.latitude,
                longitude=args.longitude,
                label=label_value,
            )
        )
    elif args.mode == "update":
        print(
            update_data(
                args.config,
                city=city_value,
                latitude=args.latitude,
                longitude=args.longitude,
                label=label_value,
            )
        )
    elif args.mode == "train":
        print(train_only(args.config))
