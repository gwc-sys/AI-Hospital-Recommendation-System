from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class FirebaseSettings:
    service_account_path: str = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    database_url: str = os.getenv("FIREBASE_DATABASE_URL", "")
    users_path: str = os.getenv("FIREBASE_USERS_PATH", "users")
    vehicle_node: str = os.getenv("FIREBASE_VEHICLE_NODE", "Ai-based-smart-vehicle-health")
    alerts_node: str = os.getenv("FIREBASE_ALERTS_NODE", "alerts")
    health_node: str = os.getenv("FIREBASE_HEALTH_NODE", "mahesh_Raskar")
    response_path: str = os.getenv(
        "FIREBASE_RESPONSE_PATH",
        "Ai-based-smart-vehicle-health/emergency_response/current",
    )


settings = FirebaseSettings()
