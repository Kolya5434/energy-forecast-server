from huggingface_hub import hf_hub_download
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPO_ID = "Mykola121/energy-forecast-models"

# List of all model files
MODEL_FILES = [
    "arima_model.pkl",
    "sarima_baseline_model.pkl",
    "prophet_baseline_model.json",
    "lstm_model.keras",
    "gru_model.keras",
    "transformer_model.keras",
    "random_forest_model.pkl",
    "xgboost_model.pkl",
    "xgboost_tuned_model.pkl",
    "light_gbm_model.pkl",
    "voting_model.pkl",
    "stacking_model.pkl",
    "standard_scaler.pkl",
    "minmax_scaler.pkl",
]


def download_models_from_hf():
    """Downloads models from Hugging Face Hub"""

    # Checking if there are enough models
    if MODELS_DIR.exists():
        model_files = (
                list(MODELS_DIR.glob("*.pkl")) +
                list(MODELS_DIR.glob("*.keras")) +
                list(MODELS_DIR.glob("*.json"))
        )
        if len(model_files) >= 12:
            print(f"✅ Моделі вже є ({len(model_files)} файлів)")
            print(f"📁 Шлях: {MODELS_DIR}")
            return

    print("📥 Завантаження моделей з Hugging Face Hub...")
    print(f"📦 Репозиторій: {REPO_ID}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for filename in MODEL_FILES:
            model_path = MODELS_DIR / filename
            if model_path.exists():
                print(f"  ✓ {filename} вже є")
                continue

            print(f"  ⬇️  Завантажую {filename}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False
            )

        print("✅ Всі моделі успішно завантажені з Hugging Face!")

        files = list(MODELS_DIR.glob("*"))
        print(f"📦 Завантажено файлів: {len(files)}")
        for f in sorted(files):
            print(f"  - {f.name}")

    except Exception as e:
        print(f"❌ Помилка завантаження моделей з Hugging Face: {e}")
        raise