"""
Script for training DL models from scratch
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/dataset_for_modeling.csv"
SCALER_PATH = BASE_DIR / "models/standard_scaler.pkl"
MODELS_DIR = BASE_DIR / "models"

SEQUENCE_LENGTH = 24
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

print("\n1. Завантаження даних...")
data = pd.read_csv(DATA_PATH, index_col='DateTime', parse_dates=True)
print(f"   ✓ Завантажено {len(data)} рядків")

print("\n2. Завантаження scaler...")
scaler = joblib.load(SCALER_PATH)
scaler_features = list(scaler.get_feature_names_out())
print(f"   ✓ Ознаки: {scaler_features}")

print("\n3. Підготовка даних...")
data_subset = data[scaler_features].copy()

if data_subset.isnull().values.any():
    print("   ⚠ Знайдено NaN, заповнюємо...")
    data_subset = data_subset.ffill().bfill().fillna(0)

scaled_data = scaler.transform(data_subset)
print(f"   ✓ Дані масштабовано: {scaled_data.shape}")

target_idx = scaler_features.index('Global_active_power')
print(f"   ✓ Target index: {target_idx}")

def create_sequences(data, seq_length, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X_seq = np.delete(data[i:i+seq_length], target_idx, axis=1)
        y_val = data[i+seq_length, target_idx]
        X.append(X_seq)
        y.append(y_val)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, target_idx)
print(f"   ✓ Створено послідовності: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# ===== LSTM MODEL =====
print("\n4. Тренування LSTM...")
lstm_model = keras.Sequential([
    layers.LSTM(50, activation='tanh', return_sequences=True, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(50, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("   Архітектура:")
lstm_model.summary()

history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Перевірка на NaN
weights_check = []
for layer in lstm_model.layers:
    for w in layer.get_weights():
        if np.isnan(w).any():
            print(f"   ✗ WARNING: NaN in {layer.name}")
            weights_check.append(False)
        else:
            weights_check.append(True)

if all(weights_check):
    print("   ✓ Модель натренована успішно, NaN відсутні")
    lstm_model.save(MODELS_DIR / "lstm_model.keras")
    print("   ✓ Модель збережена")
else:
    print("   ✗ ПОМИЛКА: Модель містить NaN!")

# ===== GRU MODEL =====
print("\n5. Тренування GRU...")
gru_model = keras.Sequential([
    layers.GRU(50, activation='tanh', input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    layers.Dropout(0.2),
    layers.Dense(1)
])

gru_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("   Архітектура:")
gru_model.summary()

history_gru = gru_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Перевірка на NaN
weights_check = []
for layer in gru_model.layers:
    for w in layer.get_weights():
        if np.isnan(w).any():
            print(f"   ✗ WARNING: NaN in {layer.name}")
            weights_check.append(False)
        else:
            weights_check.append(True)

if all(weights_check):
    print("   ✓ Модель натренована успішно, NaN відсутні")
    gru_model.save(MODELS_DIR / "gru_model.keras")
    print("   ✓ Модель збережена")
else:
    print("   ✗ ПОМИЛКА: Модель містить NaN!")

# ===== TRANSFORMER MODEL =====
print("\n6. Тренування Transformer...")

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed forward
    x = layers.Dense(ff_dim, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

inputs = keras.Input(shape=(SEQUENCE_LENGTH, X.shape[2]))
x = inputs
# Stack 4 transformer blocks
for _ in range(4):
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)

x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1)(x)

transformer_model = keras.Model(inputs, outputs)

transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("   Архітектура:")
transformer_model.summary()

history_transformer = transformer_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

weights_check = []
for layer in transformer_model.layers:
    for w in layer.get_weights():
        if np.isnan(w).any():
            print(f"   ✗ WARNING: NaN in {layer.name}")
            weights_check.append(False)
        else:
            weights_check.append(True)

if all(weights_check):
    print("   ✓ Модель натренована успішно, NaN відсутні")
    transformer_model.save(MODELS_DIR / "transformer_model.keras")
    print("   ✓ Модель збережена")
else:
    print("   ✗ ПОМИЛКА: Модель містить NaN!")

print("\n" + "="*60)
print("ТРЕНУВАННЯ ЗАВЕРШЕНО")
print("="*60)
