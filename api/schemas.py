from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class WeatherConditions(BaseModel):
    """Погодні умови для прогнозування."""
    temperature: Optional[float] = Field(None, description="Температура (°C)", example=15.0)
    humidity: Optional[float] = Field(None, description="Вологість (%)", ge=0, le=100, example=65.0)
    wind_speed: Optional[float] = Field(None, description="Швидкість вітру (м/с)", ge=0, example=5.5)


class CalendarConditions(BaseModel):
    """Календарні умови для прогнозування."""
    is_holiday: Optional[bool] = Field(None, description="Чи є цей день святом", example=True)
    is_weekend: Optional[bool] = Field(None, description="Чи є цей день вихідним", example=False)


class TimeScenario(BaseModel):
    """Часові сценарії для моделювання пікових годин та сезонності."""
    hour: Optional[int] = Field(None, description="Година доби (0-23)", ge=0, le=23, example=19)
    day_of_week: Optional[int] = Field(None, description="День тижня (0=Пн, 6=Нд)", ge=0, le=6, example=0)
    day_of_month: Optional[int] = Field(None, description="День місяця (1-31)", ge=1, le=31, example=15)
    day_of_year: Optional[int] = Field(None, description="День року (1-366)", ge=1, le=366, example=180)
    week_of_year: Optional[int] = Field(None, description="Тиждень року (1-53)", ge=1, le=53, example=26)
    month: Optional[int] = Field(None, description="Місяць (1-12)", ge=1, le=12, example=7)
    year: Optional[int] = Field(None, description="Рік", ge=2000, example=2024)
    quarter: Optional[int] = Field(None, description="Квартал (1-4)", ge=1, le=4, example=3)


class EnergyConditions(BaseModel):
    """Енергетичні параметри мережі для what-if аналізу."""
    voltage: Optional[float] = Field(None, description="Напруга в мережі (V)", ge=0, example=240.0)
    global_reactive_power: Optional[float] = Field(None, description="Реактивна потужність", ge=0, example=0.5)
    global_intensity: Optional[float] = Field(None, description="Сила струму (A)", ge=0, example=4.5)


class ZoneConsumption(BaseModel):
    """Зонове споживання електроенергії по категоріях приладів."""
    sub_metering_1: Optional[float] = Field(None, description="Кухня: посудомийка, духовка, мікрохвильовка (Wh)", ge=0, example=30.0)
    sub_metering_2: Optional[float] = Field(None, description="Пральня: пральна машина, сушарка, холодильник, освітлення (Wh)", ge=0, example=20.0)
    sub_metering_3: Optional[float] = Field(None, description="Клімат: бойлер, кондиціонер, електроопалення (Wh)", ge=0, example=15.0)


class PredictionRequest(BaseModel):
    model_ids: List[str] = Field(..., description="Список ID моделей для прогнозування", example=["XGBoost_Tuned", "SARIMA"])
    forecast_horizon: int = Field(..., description="Горизонт прогнозування в днях", example=7)
    weather: Optional[WeatherConditions] = Field(None, description="Опціональні погодні умови для всього горизонту прогнозування")
    calendar: Optional[CalendarConditions] = Field(None, description="Опціональні календарні умови")
    time_scenario: Optional[TimeScenario] = Field(None, description="Часові сценарії (пікові години, сезонність)")
    energy: Optional[EnergyConditions] = Field(None, description="Енергетичні параметри мережі")
    zone_consumption: Optional[ZoneConsumption] = Field(None, description="Зонове споживання по категоріях приладів")
    is_anomaly: Optional[bool] = Field(None, description="Примусово позначити період як аномальний")

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
    feature_overrides: List[FeatureOverride] = Field(default=[], description="Список змін в ознаках")
    weather: Optional[WeatherConditions] = Field(None, description="Опціональні погодні умови для всього горизонту симуляції")
    calendar: Optional[CalendarConditions] = Field(None, description="Опціональні календарні умови для всього горизонту симуляції")
    time_scenario: Optional[TimeScenario] = Field(None, description="Часові сценарії (пікові години, сезонність)")
    energy: Optional[EnergyConditions] = Field(None, description="Енергетичні параметри мережі")
    zone_consumption: Optional[ZoneConsumption] = Field(None, description="Зонове споживання по категоріях приладів")
    is_anomaly: Optional[bool] = Field(None, description="Примусово позначити період як аномальний")