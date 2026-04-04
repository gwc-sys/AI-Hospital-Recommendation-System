from fastapi import APIRouter, Query

from src.recommendation_engine import HospitalRecommendationEngine


router = APIRouter(prefix="/api", tags=["recommendations"])
engine = HospitalRecommendationEngine()


@router.get("/recommend")
def recommend_hospitals(
    specialty: str | None = Query(default=None),
    min_rating: float = Query(default=4.0, ge=0.0, le=5.0),
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    recommendations = engine.recommend(
        specialty=specialty,
        min_rating=min_rating,
        top_k=top_k,
    )
    return {"count": len(recommendations), "results": recommendations}

