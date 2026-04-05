from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection import GoogleMapsHospitalCollector
from scripts.run_pipeline import load_config, resolve_city_config


def main() -> None:
    config = load_config()
    city_config = resolve_city_config(config, city="pune")
    collector = GoogleMapsHospitalCollector()
    location = (city_config["latitude"], city_config["longitude"])
    search_points = [
        (point["latitude"], point["longitude"])
        for point in city_config.get("search_points", [])
    ] or [location]
    radius = config["google_maps"]["search_radius"]
    records = collector.fetch_hospitals(
        location=location,
        radius=radius,
        search_points=search_points,
    )
    collector.save_to_csv(records)
    print("Data update complete for Pune.")


if __name__ == "__main__":
    main()
