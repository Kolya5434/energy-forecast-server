from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    model_ids: List[str] = Field(..., description="Список ID моделей для прогнозування", example=["XGBoost_Tuned", "SARIMA"])
    forecast_horizon: int = Field(..., description="Горизонт прогнозування в днях", example=7)

class PredictionResponse(BaseModel):
    model_id: str
    forecast: Dict[str, float]
    metadata: Dict[str, Any]

class FeatureOverride(BaseModel):
    date: str = Field(..., description="Дата для зміни (у форматі YYYY-MM-DD)", example="2010-11-29")
    features: Dict[str, Any] = Field(..., description="Словник зі зміненими ознаками та їхніми значеннями", example={"day_of_week": 6})

class SimulationRequest(BaseModel):
    model_id: str = Field(..., description="ID моделі для симуляції (лише ML/Ensemble)", example="XGBoost_Tuned")
    forecast_horizon: int = Field(..., description="Горизонт прогнозування в днях", example=7)
    feature_overrides: List[FeatureOverride] = Field(..., description="Список змін в ознаках")