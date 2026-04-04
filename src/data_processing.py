from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DATA_DIR, ensure_directories, safe_float, setup_logger


logger = setup_logger("data_processing", Path("logs/app.log"))


class HospitalDataProcessor:
    def __init__(self) -> None:
        ensure_directories(DATA_DIR / "processed")

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
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

