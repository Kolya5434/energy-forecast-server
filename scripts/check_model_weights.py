"""
Script for checking DL models' weights for NaN/Inf
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

models_to_check = {
    "LSTM": BASE_DIR / "models/lstm_model.keras",
    "GRU": BASE_DIR / "models/gru_model.keras",
    "Transformer": BASE_DIR / "models/transformer_model.keras"
}

for model_name, model_path in models_to_check.items():
    print(f"\n{model_name}:")

    if not model_path.exists():
        print(f"  ✗ Файл не знайдено!")
        continue

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"  ✓ Модель завантажено")

        # Перевіряємо всі шари
        total_weights = 0
        nan_weights = 0
        inf_weights = 0

        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                total_weights += w.size
                nan_count = np.isnan(w).sum()
                inf_count = np.isinf(w).sum()
                nan_weights += nan_count
                inf_weights += inf_count

                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠ Layer '{layer.name}': NaN={nan_count}, Inf={inf_count}")

        print(f"  ✓ Всього ваг: {total_weights:,}")
        print(f"  ✓ NaN ваг: {nan_weights:,}")
        print(f"  ✓ Inf ваг: {inf_weights:,}")

        if nan_weights > 0 or inf_weights > 0:
            print(f"  → Потрібно перетренувати модель")
        else:
            print(f"  ✓ Всі ваги валідні")

            input_shape = model.input_shape
            random_input = np.random.randn(1, input_shape[1], input_shape[2]).astype(np.float32)

            try:
                pred = model.predict(random_input, verbose=0)[0][0]
                print(f"  ✓ Тестовий прогноз на випадкових даних: {pred:.6f}")

                if np.isnan(pred):
                    print(f"  ✗ ПРОБЛЕМА: Модель повертає NaN навіть на валідних даних!")
                    print(f"  → Можлива проблема з архітектурою або активаціями")
                elif np.isinf(pred):
                    print(f"  ✗ ПРОБЛЕМА: Модель повертає Inf!")
                else:
                    print(f"  ✓ Модель працює коректно")
            except Exception as e:
                print(f"  ✗ Помилка при тестовому прогнозі: {e}")

    except Exception as e:
        print(f"  ✗ ПОМИЛКА: {e}")
        import traceback

        traceback.print_exc()