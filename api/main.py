from fastapi import HTTPException, Request
from typing import List

from .config import app
from .schemas import PredictionRequest, PredictionResponse
from . import services

from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Глобальний обробник, який ловить усі непередбачувані помилки.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "message": "An unexpected internal server error occurred.",
            "detail": str(exc)
        },
    )

@app.get("/api/models", summary="Отримати список доступних моделей")
def get_models():
    """Повертає список та опис усіх доступних для прогнозування моделей."""
    return services.get_models_service()

@app.post("/api/predict", response_model=List[PredictionResponse], summary="Зробити прогноз за допомогою обраних моделей")
def predict(request: PredictionRequest):
    """Приймає список ID моделей та горизонт прогнозування, повертає прогнози."""
    response = services.predict_service(request)
    if not response:
        raise HTTPException(status_code=404, detail="None of the requested models were found.")
    return response

@app.get("/api/evaluation/{model_id}", summary="Отримати метрики якості для моделі")
def get_evaluation(model_id: str):
    """Повертає звіт про продуктивність та точність моделі з файлу результатів."""
    result = services.get_evaluation_service(model_id)
    if "error" in result or not result:
        raise HTTPException(status_code=404, detail=result.get("error", "Data not found."))
    return result

@app.get("/api/interpret/{model_id}", summary="Отримати SHAP-інтерпретацію для моделі")
def get_interpretation(model_id: str):
    """Повертає SHAP-пояснення для одного прогнозу найкращої ML-моделі."""
    result = services.get_interpretation_service(model_id)
    if "error" in result or "message" in result:
        raise HTTPException(status_code=400, detail=result.get("error") or result.get("message"))
    return result