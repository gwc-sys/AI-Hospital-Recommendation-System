"""
Decision Tree Model Training Module.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DATA_DIR, MODELS_DIR, ensure_directories


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class HospitalModelTrainer:
    """Train and save decision tree models."""

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or self._load_config()
        self.classifier: DecisionTreeClassifier | None = None
        self.regressor: DecisionTreeRegressor | None = None
        self.label_encoders: dict = {}
        self.feature_names: list[str] | None = None
        ensure_directories(MODELS_DIR)

    def _load_config(self) -> Dict:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        required_columns = {
            "distance_from_center",
            "response_probability",
            "emergency_facility",
            "has_phone",
            "has_website",
            "collected_at",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns for training: {sorted(missing_columns)}")

        features = pd.DataFrame(index=df.index)
        features["distance"] = pd.to_numeric(df["distance_from_center"], errors="coerce").fillna(0.0)
        features["response_prob"] = pd.to_numeric(
            df["response_probability"], errors="coerce"
        ).fillna(0.0)
        features["emergency_score"] = df["emergency_facility"].astype(int)
        features["has_phone"] = pd.to_numeric(df["has_phone"], errors="coerce").fillna(0).astype(int)
        features["has_website"] = pd.to_numeric(
            df["has_website"], errors="coerce"
        ).fillna(0).astype(int)

        collected_at = pd.to_datetime(df["collected_at"], errors="coerce")
        features["current_hour"] = collected_at.dt.hour.fillna(0).astype(int)
        features["day_of_week"] = collected_at.dt.dayofweek.fillna(0).astype(int)

        max_distance = features["distance"].max()
        if pd.isna(max_distance) or max_distance <= 0:
            features["distance_score"] = 1.0
        else:
            features["distance_score"] = 1 - (features["distance"] / max_distance)

        features["recommendation_score"] = self._calculate_recommendation_score(df)
        threshold = self.config["recommendation"].get("classification_threshold", 0.6)
        features["target_class"] = (features["recommendation_score"] > threshold).astype(int)

        self.feature_names = [
            "distance",
            "response_prob",
            "emergency_score",
            "has_phone",
            "has_website",
            "current_hour",
            "day_of_week",
            "distance_score",
        ]
        return features

    def _calculate_recommendation_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate recommendation score without ratings."""
        weights = self.config["recommendation"]["weights"]

        distance = pd.to_numeric(df["distance_from_center"], errors="coerce").fillna(0.0)
        response_probability = pd.to_numeric(
            df["response_probability"], errors="coerce"
        ).fillna(0.0)
        emergency = df["emergency_facility"].astype(int)
        has_phone = pd.to_numeric(df["has_phone"], errors="coerce").fillna(0).astype(int)
        has_website = pd.to_numeric(df["has_website"], errors="coerce").fillna(0).astype(int)

        max_distance = distance.max()
        if pd.isna(max_distance) or max_distance <= 0:
            distance_component = np.ones(len(df))
        else:
            distance_component = 1 - (distance / max_distance)

        score = (
            response_probability * weights["response_probability"]
            + emergency * weights["emergency_score"]
            + distance_component * weights["distance"]
            + has_phone * weights["has_phone"]
            + has_website * weights["has_website"]
        )

        score_min = score.min()
        score_max = score.max()
        if pd.isna(score_min) or pd.isna(score_max) or score_max == score_min:
            return np.full(len(df), 0.5, dtype=float)

        return ((score - score_min) / (score_max - score_min)).to_numpy(dtype=float)

    def train(self, data: pd.DataFrame | Path | str | None = None) -> dict[str, float]:
        """Train both classifier and regressor models."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            dataset_path = Path(data) if data else DATA_DIR / "processed" / "final_hospital_dataset.csv"
            df = pd.read_csv(dataset_path)

        features_df = self.prepare_features(df)
        x = features_df[self.feature_names]
        y_class = features_df["target_class"]
        y_reg = features_df["recommendation_score"]

        test_size = self.config["model"].get("test_size", 0.2)
        random_state = self.config["model"].get("random_state", 42)
        should_stratify = y_class.nunique() > 1 and y_class.value_counts().min() >= 2
        stratify_target = y_class if should_stratify else None

        x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            x,
            y_class,
            y_reg,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )

        self.classifier = DecisionTreeClassifier(**self.config["model"]["classifier"])
        self.classifier.fit(x_train, y_train_class)

        self.regressor = DecisionTreeRegressor(**self.config["model"]["regressor"])
        self.regressor.fit(x_train, y_train_reg)

        metrics = self._evaluate_models(x_test, y_test_class, y_test_reg)
        self.save_models()
        return metrics

    def _evaluate_models(
        self,
        x_test: pd.DataFrame,
        y_test_class: pd.Series,
        y_test_reg: pd.Series,
    ) -> dict[str, float]:
        """Evaluate trained models."""
        if self.classifier is None or self.regressor is None:
            raise ValueError("Models must be trained before evaluation.")

        y_pred_class = self.classifier.predict(x_test)
        y_pred_reg = self.regressor.predict(x_test)

        logger.info("Classifier Performance:\n%s", classification_report(y_test_class, y_pred_class))

        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg) if len(y_test_reg) > 1 else float("nan")
        importance = dict(zip(self.feature_names or [], self.classifier.feature_importances_))

        logger.info("Regressor MSE: %.3f", mse)
        logger.info("Regressor R2: %.3f", r2)
        logger.info("Feature Importance: %s", importance)

        return {
            "classifier_accuracy": float(self.classifier.score(x_test, y_test_class)),
            "regressor_mse": float(mse),
            "regressor_r2": float(r2),
        }

    def save_models(self, model_dir: str | Path = MODELS_DIR) -> None:
        """Save trained models to disk."""
        target_dir = Path(model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, target_dir / "decision_tree_classifier.pkl")
        joblib.dump(self.regressor, target_dir / "decision_tree_regressor.pkl")
        joblib.dump(self.label_encoders, target_dir / "label_encoders.pkl")
        logger.info("Models saved to %s", target_dir)

    def load_models(self, model_dir: str | Path = MODELS_DIR) -> None:
        """Load trained models from disk."""
        target_dir = Path(model_dir)
        self.classifier = joblib.load(target_dir / "decision_tree_classifier.pkl")
        self.regressor = joblib.load(target_dir / "decision_tree_regressor.pkl")
        self.label_encoders = joblib.load(target_dir / "label_encoders.pkl")
        logger.info("Models loaded from %s", target_dir)
 

if __name__ == "__main__":
    metrics = HospitalModelTrainer().train()
    print(metrics)
