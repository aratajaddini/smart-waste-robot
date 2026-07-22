from functools import lru_cache
from backend.config import MODEL_PATH, CLASSES


@lru_cache(maxsize=1)
def get_model():
    """Lazy load. Runs only on the first real call."""
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def run_inference(image_path: str) -> dict:
    model = get_model()
    results = model(image_path)
    probs = results[0].probs
    scores = {CLASSES[i]: float(probs.data[i]) for i in range(len(CLASSES))}
    top_class = max(scores, key=scores.get)
    return {
        "top_class": top_class,
        "confidence": scores[top_class],
        "scores": scores,
    }
