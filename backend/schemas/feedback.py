from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct_class: str


class FeedbackResponse(BaseModel):
    id: int
    prediction_id: int
    correct_class: str
