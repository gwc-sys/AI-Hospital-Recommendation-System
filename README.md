<<<<<<< HEAD
# AI-Hospital-Recommendation-System
The AI-Based Emergency Hospital Recommendation System is an intelligent solution designed to automatically identify and recommend the most suitable hospital during emergency situations. The system integrates real-time location data, Google Maps APIs, and machine learning models to provide fast and reliable healthcare assistance.
=======
# AI Hospital Recommendation System

Starter project for collecting hospital data, processing features, training decision-tree models, and exposing recommendation results for later integration into your real application.

## Structure

- `src/`: data collection, processing, training, and recommendation logic
- `api/`: optional FastAPI layer for testing or later app integration
- `scripts/`: runnable helpers for pipeline, updates, and deployment checks
- `data/`: raw and processed datasets
- `models/`: generated model artifacts

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
uvicorn api.app:app --reload
pytest
```

## Current Focus

This repository is intentionally UI-free so you can validate the AI/ML workflow first:

- collect hospital data
- clean and engineer features
- train the recommendation model
- expose predictions through Python modules or API endpoints

Once the model behaves the way you want, it can be connected to your React Native project through the API layer.
>>>>>>> a95f7bb (Initial commit - AI Hospital Recommendation System)
