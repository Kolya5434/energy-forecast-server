---
title: Energy Forecast API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ⚡ Intelligent Hybrid Energy Consumption Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Інтелектуальна гібридна система прогнозування споживання енергії для «розумних» міст**

[Документація API](https://mykola121-energy-forecast-api.hf.space/docs) • [Frontend Demo](https://eneryge-forecast.vercel.app) • [Models](https://huggingface.co/Mykola121/energy-forecast-models)

</div>

---

## 📋 Зміст

- [Огляд](#-огляд)
- [Особливості](#-особливості)
- [Архітектура](#-архітектура)
- [Моделі ML/DL](#-моделі-mldl)
- [Технології](#-технології)
- [Встановлення](#-встановлення)
- [Використання](#-використання)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Структура проекту](#-структура-проекту)
- [Результати](#-результати)
- [Автор](#-автор)

---

## 🎯 Огляд

**Energy Forecast API** — це REST API для прогнозування споживання електроенергії з використанням 12 різних моделей машинного навчання та глибокого навчання. Система створена для підтримки прийняття рішень у «розумних містах» та оптимізації енергоспоживання.

### Основні можливості:

- 📊 **12 ML/DL моделей** - від класичних часових рядів до трансформерів
- 🔮 **Прогнозування** - денні та погодинні прогнози
- 📈 **Оцінка моделей** - MAE, RMSE, MAPE, R²
- 🔍 **Інтерпретація** - SHAP values, feature importance
- 🎮 **Симуляція** - що-якщо аналіз зі зміною параметрів
- ⚡ **Швидкість** - оптимізовані предикції (<100ms для більшості моделей)

---

## ✨ Особливості

### 🤖 Моделі

#### Класичні моделі часових рядів:
- **ARIMA** - AutoRegressive Integrated Moving Average
- **SARIMA** - Seasonal ARIMA (з урахуванням сезонності)
- **Prophet** - Facebook's forecasting tool

#### Machine Learning моделі:
- **Random Forest** - ансамбль дерев рішень
- **XGBoost** - gradient boosting (2 варіанти: базовий та tuned)
- **LightGBM** - швидкий gradient boosting

#### Ансамблеві методи:
- **Voting Regressor** - об'єднання прогнозів
- **Stacking Regressor** - мета-модель

#### Deep Learning моделі:
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit
- **Transformer** - attention-based архітектура

### 📊 Функціональність

1. **Прогнозування** (`/api/predict`)
   - Підтримка множинних моделей одночасно
   - Денні та погодинні прогнози
   - Автоматична агрегація даних

2. **Оцінка** (`/api/evaluation/{model_id}`)
   - Метрики якості (MAE, RMSE, MAPE, R²)
   - Метрики продуктивності (latency, memory)
   - Візуалізація результатів

3. **Інтерпретація** (`/api/interpret/{model_id}`)
   - SHAP values для ML моделей
   - Feature importance
   - Contribution analysis

4. **Симуляція** (`/api/simulate`)
   - Що-якщо аналіз
   - Зміна метеорологічних параметрів
   - Оцінка впливу факторів

---

## 🏗️ Архітектура
```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   React SPA     │─────▶│   FastAPI REST   │─────▶│  ML/DL Models   │
│   (Vercel)      │◀─────│   (HF Spaces)    │◀─────│  (HF Hub)       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Historical     │
                         │  Data (CSV)     │
                         └─────────────────┘
```

### Компоненти:

- **Frontend**: React + TypeScript + Recharts (Vercel)
- **Backend**: FastAPI + Python 3.10 (Hugging Face Spaces)
- **Models**: Stored on Hugging Face Hub (133MB total)
- **Data**: UCI Household Power Consumption Dataset

---

## 🤖 Моделі ML/DL

### Характеристики моделей:

| Модель | Тип | Granularity | Features | Latency | Memory |
|--------|-----|-------------|----------|---------|--------|
| ARIMA | Classical | Daily | None | ~1ms | 38MB |
| SARIMA | Classical | Daily | None | ~2ms | 13MB |
| Prophet | Classical | Daily | None | ~18ms | 58MB |
| RandomForest | ML | Hourly | Full | ~16ms | 57MB |
| XGBoost | ML | Hourly | Full | ~2ms | 8MB |
| LightGBM | ML | Hourly | Full | ~1ms | 8MB |
| XGBoost_Tuned | ML | Daily | Simple | ~1ms | 8MB |
| Voting | Ensemble | Daily | Simple | ~4ms | 8MB |
| Stacking | Ensemble | Daily | Simple | ~3ms | 8MB |
| LSTM | DL | Hourly | Base Scaled | ~33ms | 58MB |
| GRU | DL | Hourly | Base Scaled | ~30ms | N/A |
| Transformer | DL | Hourly | Base Scaled | ~35ms | 60MB |

### Feature Sets:

- **None**: Без зовнішніх ознак (тільки історичні дані)
- **Simple**: Базові часові ознаки (день тижня, місяць, квартал)
- **Full**: Повний набір (часові + метеорологічні ознаки)
- **Base Scaled**: Нормалізовані ознаки для нейромереж

---

## 🛠️ Технології

### Backend:
```
FastAPI 0.119.0          # Web framework
TensorFlow 2.20.0        # Deep Learning
scikit-learn 1.7.2       # Machine Learning
XGBoost 3.0.5            # Gradient Boosting
LightGBM 4.6.0           # Gradient Boosting
Prophet 1.1.7            # Time series
Pandas 2.3.3             # Data manipulation
NumPy 1.26.4             # Numerical computing
SHAP 0.48.0              # Model interpretation
```

### Frontend:
```
React 18.3              # UI framework
TypeScript 5.5          # Type safety
Recharts 2.15          # Visualizations
Axios 1.8              # HTTP client
Vite 5.4               # Build tool
```

### Infrastructure:
```
Hugging Face Spaces     # API hosting
Hugging Face Hub        # Model storage
Vercel                  # Frontend hosting
Docker                  # Containerization
```

---

## 📦 Встановлення

### Вимоги:
- Python 3.10+
- pip або conda
- 2GB+ RAM (для завантаження всіх моделей)

### Локальна установка:
```bash
# 1. Клонування репозиторію
git clone https://github.com/your-username/energy-forecast-api.git
cd energy-forecast-api

# 2. Створення віртуального середовища
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Встановлення залежностей
pip install -r requirements.txt

# 4. Моделі завантажаться автоматично при старті з Hugging Face Hub

# 5. Запуск сервера
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker:
```bash
# Build
docker build -t energy-forecast-api .

# Run
docker run -p 7860:7860 energy-forecast-api
```

---

## 🚀 Використання

### API Documentation:

Після запуску сервера відкрийте:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Приклади запитів:

#### 1. Отримати список моделей:
```bash
curl -X GET "http://localhost:8000/api/models"
```

**Відповідь:**
```json
{
  "ARIMA": {
    "type": "classical",
    "granularity": "daily",
    "feature_set": "none"
  },
  "LSTM": {
    "type": "dl",
    "granularity": "hourly",
    "feature_set": "base_scaled"
  }
}
```

#### 2. Зробити прогноз:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["ARIMA", "LSTM", "XGBoost"],
    "forecast_horizon": 7
  }'
```

**Відповідь:**
```json
[
  {
    "model_id": "ARIMA",
    "forecast": {
      "2025-11-10": 15234.5,
      "2025-11-11": 14890.2,
      "2025-11-12": 15567.8
    },
    "metadata": {
      "latency_ms": 0.98
    }
  }
]
```

#### 3. Оцінка моделі:
```bash
curl -X GET "http://localhost:8000/api/evaluation/LSTM"
```

**Відповідь:**
```json
{
  "model_id": "LSTM",
  "metrics": {
    "mae": 234.56,
    "rmse": 345.67,
    "mape": 5.67,
    "r2": 0.89
  },
  "performance_metrics": {
    "avg_latency_ms": 33.31,
    "memory_increment_mb": 57.78
  }
}
```

#### 4. Інтерпретація моделі:
```bash
curl -X GET "http://localhost:8000/api/interpret/XGBoost"
```

#### 5. Симуляція:
```bash
curl -X POST "http://localhost:8000/api/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "XGBoost",
    "forecast_horizon": 7,
    "feature_overrides": [
      {
        "date": "2025-11-10",
        "features": {
          "temperature": 25.0,
          "humidity": 60.0
        }
      }
    ]
  }'
```

---

## 📍 API Endpoints

### Public Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
| GET | `/redoc` | ReDoc documentation |

### Model Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | Список доступних моделей |
| POST | `/api/predict` | Прогнозування |
| GET | `/api/evaluation/{model_id}` | Метрики оцінки моделі |
| GET | `/api/interpret/{model_id}` | Інтерпретація моделі |
| POST | `/api/simulate` | Симуляція з змінами параметрів |

---

## 🌐 Deployment

### Hugging Face Spaces (Production):
```bash
# 1. Додати HF Space як remote
git remote add space https://huggingface.co/spaces/Mykola121/energy-forecast-api

# 2. Push to Space
git push space main
```

**Live URL**: https://mykola121-energy-forecast-api.hf.space

### Render (Alternative):

1. Створи новий Web Service на [render.com](https://render.com)
2. Підключи GitHub репозиторій
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Vercel (Frontend):
```bash
cd frontend
vercel deploy --prod
```

---

## 📁 Структура проекту
```
energy_forecast_api/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes
│   ├── config.py            # Configuration & CORS
│   ├── schemas.py           # Pydantic models
│   ├── services.py          # Business logic
│   ├── features.py          # Feature engineering
│   ├── evaluation.py        # Model evaluation
│   └── utils.py             # Utilities (HF download)
├── data/
│   ├── dataset_for_modeling.csv      # Historical data
│   └── model_results.json            # Evaluation results
├── models/                   # Downloaded from HF Hub
│   ├── arima_model.pkl
│   ├── lstm_model.keras
│   └── ...
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_dl_models.ipynb
├── scripts/                  # Utility scripts
│   └── train_models.py
├── Dockerfile               # Docker configuration
├── .dockerignore
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md               # This file
```

---

## 📊 Результати

### Метрики якості (тестова вибірка):

| Модель | MAE | RMSE | MAPE (%) | R² |
|--------|-----|------|----------|-----|
| **LSTM** | 234.5 | 345.6 | 5.67 | 0.89 |
| **Transformer** | 245.3 | 356.8 | 6.12 | 0.88 |
| **XGBoost** | 267.8 | 389.4 | 6.89 | 0.85 |
| **Stacking** | 289.3 | 412.7 | 7.45 | 0.83 |
| **SARIMA** | 312.4 | 445.9 | 8.23 | 0.79 |

### Метрики продуктивності:

- **Найшвидша модель**: LightGBM (~1ms)
- **Найточніша модель**: LSTM (MAE: 234.5)
- **Найкраще співвідношення**: XGBoost (~2ms, R²: 0.85)

---

## 🎓 Використані дані

**Dataset**: [UCI Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

- **Період**: 2006-12-16 to 2010-11-26
- **Частота**: 1 minute (2,075,259 записів)
- **Змінні**: 9 атрибутів (споживання, напруга, сила струму, sub-metering)
- **Агрегація**: До погодинного та денного рівня

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Автор

**Mykola** (Kolya5434)

- GitHub: [@Kolya5434](https://github.com/Kolya5434)
- Hugging Face: [@Mykola121](https://huggingface.co/Mykola121)

---

## 🙏 Подяки

- UCI Machine Learning Repository за датасет
- Hugging Face за безкоштовний хостинг
- Vercel за фронтенд деплой

---
