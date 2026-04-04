import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class APISettings:
    google_maps_api_key: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
    fastapi_host: str = os.getenv("API_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("API_PORT", "8000"))


settings = APISettings()
API_KEY = settings.google_maps_api_key
