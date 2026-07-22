import io
from functools import lru_cache
from PIL import Image
import numpy as np
from backend.config import MODEL_PATH   # ✅ import the path from config


@lru_cache(maxsize=1)
def _get_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH), task="classify")


def run_inference(image_bytes: bytes) -> dict:
    model = _get_model()
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img_pil)

    results = model(img_np, verbose=False)[0]
    probs = results.probs
    names = model.names

    scores = {
        names[i]: round(float(probs.data[i]), 4)
        for i in names
    }

    return {
        "top_class": names[int(probs.top1)],
        "confidence": round(float(probs.top1conf), 4),
        "scores": scores,
    }