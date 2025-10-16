from pydantic import BaseModel, Field
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    model_ids: List[str] = Field(..., description="Список ID моделей для прогнозування", example=["XGBoost_Tuned", "SARIMA"])
    forecast_horizon: int = Field(..., description="Горизонт прогнозування в днях", example=7)

class PredictionResponse(BaseModel):
    model_id: str
    forecast: Dict[str, float]
    metadata: Dict[str, Any]
