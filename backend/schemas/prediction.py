from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    id: int
    filename: str
    top_class: str
    confidence: float
    scores: Dict[str, float]


class HistoryItem(BaseModel):
    id: int
    filename: str
    top_class: str
    confidence: float
    created_at: str
