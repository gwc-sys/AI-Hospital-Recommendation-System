from pathlib import Path


def main() -> None:
    model_path = Path("models/decision_tree_classifier.pkl")
    if model_path.exists():
        print(f"Model ready for deployment: {model_path}")
    else:
        print("Model artifact not found. Run the training pipeline first.")


if __name__ == "__main__":
    main()

