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


def run_full_pipeline(config_path: str | Path = DEFAULT_CONFIG_PATH):
    """Run complete data collection, processing, and model training pipeline."""
    config = load_config(config_path)

    logger.info("Step 1: Collecting data from Google Maps API")
    collector = GoogleMapsCollector(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))

    location = (
        config["data_collection"]["default_location"]["latitude"],
        config["data_collection"]["default_location"]["longitude"],
    )
    radius = config["google_maps"]["search_radius"]

    df_raw = collector.collect_hospitals(location=location, radius=radius)
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
    }


def update_data(config_path: str | Path = DEFAULT_CONFIG_PATH):
    """Refresh the raw and processed datasets using the configured location."""
    result = run_full_pipeline(config_path)
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
    args = parser.parse_args()

    if args.mode == "full":
        print(run_full_pipeline(args.config))
    elif args.mode == "update":
        print(update_data(args.config))
    elif args.mode == "train":
        print(train_only(args.config))
