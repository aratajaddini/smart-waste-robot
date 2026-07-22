from typing import List
from fastapi import APIRouter
from backend.models.database import get_conn
from backend.schemas.prediction import HistoryItem

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=List[HistoryItem])
async def history():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, top_class, confidence, created_at "
            "FROM predictions ORDER BY id DESC"
        ).fetchall()
    return [HistoryItem(**dict(r)) for r in rows]
