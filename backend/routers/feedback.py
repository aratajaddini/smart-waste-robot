from fastapi import APIRouter, HTTPException
from backend.models.database import get_conn
from backend.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest):
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM predictions WHERE id = ?", (payload.prediction_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="prediction not found")

        cur = conn.execute(
            "INSERT INTO feedback (prediction_id, correct_class) VALUES (?, ?)",
            (payload.prediction_id, payload.correct_class),
        )
        fb_id = cur.lastrowid

    return FeedbackResponse(
        id=fb_id,
        prediction_id=payload.prediction_id,
        correct_class=payload.correct_class,
    )
