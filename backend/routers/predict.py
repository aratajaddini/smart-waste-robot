import shutil
from fastapi import APIRouter, UploadFile, File
from backend.config import UPLOAD_DIR
from backend.inference import run_inference
from backend.models.database import get_conn
from backend.schemas.prediction import PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = run_inference(str(dest))

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (filename, top_class, confidence) VALUES (?, ?, ?)",
            (file.filename, result["top_class"], result["confidence"]),
        )
        pred_id = cur.lastrowid

    return PredictionResponse(
        id=pred_id,
        filename=file.filename,
        top_class=result["top_class"],
        confidence=result["confidence"],
        scores=result["scores"],
    )
