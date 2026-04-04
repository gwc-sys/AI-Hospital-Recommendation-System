from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR


class HospitalRecommendationEngine:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or DATA_DIR / "processed" / "final_hospital_dataset.csv"

    def recommend(
        self,
        specialty: str | None = None,
        min_rating: float = 4.0,
        top_k: int = 5,
    ) -> list[dict]:
        df = pd.read_csv(self.dataset_path)
        filtered = df[df["rating"] >= min_rating]

        if specialty and "specialty" in filtered.columns:
            filtered = filtered[
                filtered["specialty"].str.lower() == specialty.strip().lower()
            ]

        ranked = filtered.sort_values(
            ["quality_score", "reviews", "rating"],
            ascending=False,
        )
        return ranked.head(top_k).to_dict(orient="records")

