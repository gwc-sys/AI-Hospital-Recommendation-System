from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection import GoogleMapsHospitalCollector


def main() -> None:
    collector = GoogleMapsHospitalCollector()
    records = collector.fetch_hospitals()
    collector.save_to_csv(records)
    print("Data update complete.")


if __name__ == "__main__":
    main()
