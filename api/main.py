from fastapi import HTTPException, Request
from typing import List

from .config import app
from .schemas import PredictionRequest, PredictionResponse, SimulationRequest
from . import services
from starlette.concurrency import run_in_threadpool

from fastapi.responses import JSONResponse

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