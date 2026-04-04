import pandas as pd

from src.data_processing import HospitalDataProcessor
from src.model_training import HospitalModelTrainer


def test_clean_dataset_adds_quality_score() -> None:
    df = pd.DataFrame(
        [
            {"name": "A", "rating": "4.5", "reviews": "100", "specialty": "General"},
            {"name": "B", "rating": "4.2", "reviews": "80", "specialty": "General"},
        ]
    )
    processed = HospitalDataProcessor().clean_dataset(df)
    assert "quality_score" in processed.columns
    assert processed["quality_score"].min() > 0


def test_prepare_features_adds_model_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "distance_from_center": 1200.0,
                "response_probability": 0.75,
                "emergency_facility": True,
                "has_phone": 1,
                "has_website": 1,
                "collected_at": "2026-04-04T12:30:00",
            },
            {
                "distance_from_center": 3500.0,
                "response_probability": 0.55,
                "emergency_facility": False,
                "has_phone": 1,
                "has_website": 0,
                "collected_at": "2026-04-04T16:00:00",
            },
        ]
    )
    trainer = HospitalModelTrainer()
    features = trainer.prepare_features(df)
    assert "recommendation_score" in features.columns
    assert "target_class" in features.columns
    assert set(trainer.feature_names).issubset(features.columns)
