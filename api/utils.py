import gdown
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FOLDER_ID = "1gNyV2xouP_78_dv3kr1cOFheQepbZ3nI"


def download_models_from_gdrive():
    """Завантажує всі моделі з Google Drive папки"""

    if MODELS_DIR.exists():
        model_files = (
                list(MODELS_DIR.glob("*.pkl")) +
                list(MODELS_DIR.glob("*.keras")) +
                list(MODELS_DIR.glob("*.json"))
        )
        if len(model_files) >= 12:
            print(f"✅ Моделі вже завантажені ({len(model_files)} файлів)")
            print(f"📁 Шлях: {MODELS_DIR}")
            return

    print("📥 Завантаження моделей з Google Drive...")
    print(f"📁 Цільова папка: {MODELS_DIR}")

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        gdown.download_folder(url, output=str(MODELS_DIR), quiet=False, use_cookies=False)

        print("✅ Моделі успішно завантажені!")

        files = list(MODELS_DIR.glob("*"))
        print(f"📦 Завантажено файлів: {len(files)}")
        for f in files:
            print(f"  - {f.name}")

    except Exception as e:
        print(f"❌ Помилка завантаження моделей: {e}")
        raise