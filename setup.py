from setuptools import find_packages, setup


setup(
    name="ai-hospital-recommendation-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
        "fastapi",
        "uvicorn",
        "pyyaml",
    ],
)
