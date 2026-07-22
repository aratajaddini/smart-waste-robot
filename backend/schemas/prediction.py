from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    id: int
    filename: str
    top_class: str        # ✅ keep as top_class for /predict
    confidence: float
    scores: Dict[str, float]

class HistoryItem(BaseModel):
    id: int
    filename: str
    predicted_class: str  # ✅ changed from top_class to predicted_class
    confidence: float
    created_at: str