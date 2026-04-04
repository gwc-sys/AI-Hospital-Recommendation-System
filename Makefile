.PHONY: help install run-collect run-train run-api test clean update-data

help:
	@echo Available commands:
	@echo   make install      - Install project dependencies
	@echo   make run-collect  - Run full data collection and training pipeline
	@echo   make run-train    - Train models using existing processed data
	@echo   make run-api      - Start FastAPI server
	@echo   make update-data  - Refresh data using the configured location
	@echo   make test         - Run test suite
	@echo   make clean        - Remove caches and generated logs

install:
	pip install -r requirements.txt
	pip install -r api/requirements.txt

run-collect:
	python scripts/run_pipeline.py --mode full

run-train:
	python scripts/run_pipeline.py --mode train

run-api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

update-data:
	python scripts/run_pipeline.py --mode update

test:
	pytest tests -v

clean:
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue"
	powershell -Command "if (Test-Path logs\\*.log) { Remove-Item logs\\*.log -Force -ErrorAction SilentlyContinue }"
