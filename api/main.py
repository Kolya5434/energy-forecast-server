from fastapi import HTTPException, Request
from typing import List

from .config import app
from .schemas import PredictionRequest, PredictionResponse, SimulationRequest, CompareRequest
from . import services
from starlette.concurrency import run_in_threadpool

from fastapi.responses import JSONResponse

# Import scientific routes
from .scientific_routes import router as scientific_router

# Include scientific routes
app.include_router(scientific_router)

# --- Exception Handler ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global handler for unexpected errors."""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "message": "An unexpected internal server error occurred.",
            # "detail": str(exc) # Might expose sensitive info in production
        },
    )

# --- API Routes ---
@app.get("/api/models", summary="Get list of available models")
def get_models():
    """Returns a list and description of all available forecasting models."""
    return services.get_models_service()

@app.post("/api/predict", response_model=List[PredictionResponse], summary="Generate forecasts using selected models")
async def predict(request: PredictionRequest):
    """Accepts a list of model IDs and a forecast horizon, returns forecasts."""
    try:
        response = await run_in_threadpool(services.predict_service, request)
        if not response:
            raise HTTPException(status_code=404, detail="No valid models found for prediction.")
        return response
    except FileNotFoundError as e:
         raise HTTPException(status_code=500, detail=f"Server configuration error: {e}")
    except ConnectionError as e:
         raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    except Exception as e:
        raise e


@app.get("/api/evaluation/{model_id}", summary="Get evaluation metrics for a model")
async def get_evaluation(model_id: str):
    """Returns performance and accuracy metrics from the results file."""
    try:
        result = await run_in_threadpool(services.get_evaluation_service, model_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result.get("error", "Data not found."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка під час оцінки: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/interpret/{model_id}", summary="Get interpretation data for a model")
async def get_interpretation(model_id: str):
    """Returns SHAP explanation or feature importance for the selected model."""
    try:
        result = await run_in_threadpool(services.get_interpretation_service, model_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка під час інтерпретації: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate", response_model=PredictionResponse, summary="Запустити симуляцію прогнозу зі зміненими ознаками")
async def simulate_prediction(request: SimulationRequest):
    """
    Accepts the ID of a single model, the horizon, and a list of changed attributes.
    Returns a single simulated forecast.
    """
    try:
        response = await run_in_threadpool(services.simulate_service, request)
        return response
    except (KeyError, NotImplementedError, ValueError, FileNotFoundError) as e:
        print(f"Помилка під час симуляції: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Неочікувана помилка під час симуляції: {e}")
        raise HTTPException(status_code=500, detail="Внутрішня помилка сервера під час симуляції.")

@app.get("/api/historical", summary="Отримати історичні дані споживання")
async def get_historical(
    days: int = 30,
    granularity: str = "daily",
    include_stats: bool = False
):
    """
    Повертає історичні дані споживання енергії.

    - **days**: Кількість днів історії (1-365)
    - **granularity**: 'daily' або 'hourly'
    - **include_stats**: Чи включати статистику (min, max, mean, std, median)
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Параметр 'days' має бути від 1 до 365.")
        if granularity not in ["daily", "hourly"]:
            raise HTTPException(status_code=400, detail="Параметр 'granularity' має бути 'daily' або 'hourly'.")

        result = await run_in_threadpool(
            services.get_historical_service,
            days,
            granularity,
            include_stats
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при отриманні історичних даних: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features/{model_id}", summary="Отримати інформацію про ознаки моделі")
async def get_features(model_id: str):
    """
    Повертає детальну інформацію про ознаки, які використовує модель.

    - Список всіх ознак моделі
    - Категорії доступних умов (weather, calendar, time, energy, zone_consumption)
    - Чи підтримує модель умови для прогнозування
    """
    try:
        result = await run_in_threadpool(services.get_features_service, model_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при отриманні ознак моделі: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== NEW ANALYTICS ENDPOINTS ==============

@app.get("/api/patterns", summary="Отримати сезонні патерни споживання")
async def get_patterns(period: str = "daily"):
    """
    Повертає сезонні патерни споживання енергії.

    - **period**: 'hourly', 'daily', 'weekly', 'monthly', 'yearly'
    """
    try:
        valid_periods = ["hourly", "daily", "weekly", "monthly", "yearly"]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Параметр 'period' має бути одним з: {', '.join(valid_periods)}"
            )
        result = await run_in_threadpool(services.get_patterns_service, period)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при отриманні патернів: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies", summary="Отримати історію аномалій споживання")
async def get_anomalies(
    threshold: float = 2.0,
    days: int = 30,
    include_details: bool = True
):
    """
    Повертає аналіз аномалій споживання енергії.

    - **threshold**: Поріг стандартних відхилень (1.5-3.0 рекомендовано)
    - **days**: Кількість днів для аналізу (1-365)
    - **include_details**: Чи включати деталізовану інформацію
    """
    try:
        if threshold < 0.5 or threshold > 5.0:
            raise HTTPException(status_code=400, detail="Параметр 'threshold' має бути від 0.5 до 5.0")
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Параметр 'days' має бути від 1 до 365")

        result = await run_in_threadpool(
            services.get_anomalies_service,
            threshold,
            days,
            include_details
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при отриманні аномалій: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/peaks", summary="Отримати пікові періоди споживання")
async def get_peaks(
    top_n: int = 10,
    granularity: str = "hourly"
):
    """
    Повертає топ пікових та мінімальних періодів споживання.

    - **top_n**: Кількість топ записів (1-100)
    - **granularity**: 'hourly' або 'daily'
    """
    try:
        if top_n < 1 or top_n > 100:
            raise HTTPException(status_code=400, detail="Параметр 'top_n' має бути від 1 до 100")
        if granularity not in ["hourly", "daily"]:
            raise HTTPException(status_code=400, detail="Параметр 'granularity' має бути 'hourly' або 'daily'")

        result = await run_in_threadpool(services.get_peaks_service, top_n, granularity)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при отриманні піків: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decomposition", summary="Отримати сезонну декомпозицію")
async def get_decomposition(period: int = 24):
    """
    Повертає сезонну декомпозицію часового ряду (trend, seasonal, residual).

    - **period**: Період сезонності в годинах (24=добова, 168=тижнева)
    """
    try:
        valid_periods = [24, 168, 12, 48]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Параметр 'period' має бути одним з: {', '.join(map(str, valid_periods))}"
            )
        result = await run_in_threadpool(services.get_decomposition_service, period)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при декомпозиції: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", summary="Порівняти сценарії прогнозування")
async def compare_scenarios(request: CompareRequest):
    """
    Порівнює baseline прогноз з кількома сценаріями.

    Приклад запиту:
    ```json
    {
        "model_id": "XGBoost_Tuned",
        "forecast_horizon": 7,
        "scenarios": [
            {"name": "cold_weather", "weather": {"temperature": -5}},
            {"name": "hot_weather", "weather": {"temperature": 35}},
            {"name": "holiday", "calendar": {"is_holiday": true}}
        ]
    }
    ```
    """
    try:
        result = await run_in_threadpool(services.compare_scenarios_service, request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Помилка при порівнянні сценаріїв: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint - швидкий відгук без моделей"""
    return {"message": "Energy Forecast API is running"}

@app.get("/health")
def health():
    """Health check для моніторингу"""
    return {"status": "healthy"}