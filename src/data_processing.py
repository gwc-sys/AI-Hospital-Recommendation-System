from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DATA_DIR, ensure_directories, safe_float, setup_logger


logger = setup_logger("data_processing", Path("logs/app.log"))


class HospitalDataProcessor:
    def __init__(self) -> None:
        ensure_directories(DATA_DIR / "processed")

    def filter_hospital_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df.copy()
        facility_type = filtered.get("facility_type", pd.Series("", index=filtered.index)).fillna("").astype(str).str.lower()
        name = filtered.get("name", pd.Series("", index=filtered.index)).fillna("").astype(str).str.lower()
        specialty = filtered.get("specialty", pd.Series("", index=filtered.index)).fillna("").astype(str).str.lower()
        vicinity = filtered.get("vicinity", pd.Series("", index=filtered.index)).fillna("").astype(str).str.lower()

        blocked_facility_types = {"drugstore", "pharmacy", "store", "laboratory"}
        blocked_keywords = {
            "medical",
            "medical store",
            "general store",
            "drug store",
            "drugstore",
            "laboratory",
            "pathology",
            "diagnostic",
            "chemist",
            "pharmacy",
        }
        hospital_keywords = {
            "hospital",
            "clinic",
            "multispeciality",
            "multi speciality",
            "trauma",
            "institute",
            "care centre",
            "care center",
        }

        has_hospital_signal = (
            facility_type.eq("hospital")
            | name.apply(lambda value: any(keyword in value for keyword in hospital_keywords))
            | vicinity.apply(lambda value: any(keyword in value for keyword in hospital_keywords))
        )
        has_blocked_signal = (
            facility_type.isin(blocked_facility_types)
            | specialty.eq("pharmacy")
            | name.apply(lambda value: any(keyword in value for keyword in blocked_keywords))
            | vicinity.apply(lambda value: any(keyword in value for keyword in blocked_keywords))
        )

        return filtered[has_hospital_signal & ~has_blocked_signal].copy()

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.filter_hospital_rows(df)
        cleaned["rating"] = cleaned["rating"].apply(safe_float)
        cleaned["reviews"] = cleaned["reviews"].apply(lambda value: int(safe_float(value)))
        cleaned["quality_score"] = cleaned["rating"] * 20 + cleaned["reviews"].clip(upper=500) / 10
        cleaned["is_highly_rated"] = (cleaned["rating"] >= 4.5).astype(int)
        return cleaned

    def process(
        self,
        input_path: Path | None = None,
        enhanced_path: Path | None = None,
        final_path: Path | None = None,
    ) -> tuple[Path, Path]:
        source = input_path or DATA_DIR / "raw" / "hospitals_raw_data.csv"
        enhanced = enhanced_path or DATA_DIR / "processed" / "enhanced_hospitals_data.csv"
        final = final_path or DATA_DIR / "processed" / "final_hospital_dataset.csv"

        df = pd.read_csv(source)
        processed = self.clean_dataset(df)
        processed.to_csv(enhanced, index=False)
        processed.sort_values(["quality_score", "rating"], ascending=False).to_csv(
            final,
            index=False,
        )
        logger.info("Processed dataset saved to %s and %s", enhanced, final)
        return enhanced, final


if __name__ == "__main__":
    HospitalDataProcessor().process()

