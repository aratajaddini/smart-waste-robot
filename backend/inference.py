from functools import lru_cache
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent / "weights" / "best.pt"


@lru_cache(maxsize=1)
def _load_model() -> YOLO:
    """Load YOLO model once and cache it."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def run_inference(image_path: str) -> dict:
    """
    Run YOLOv8 inference on the given image path.
    Returns top class name and per-class confidence scores.
    """
    model = _load_model()
    results = model(image_path)

    probs = results[0].probs  # Classification probabilities
    names = results[0].names  # Class index -> name mapping

    scores = {names[i]: round(float(probs.data[i]), 4) for i in range(len(names))}
    top_class = names[int(probs.top1)]

    return {
        "top_class": top_class,
        "scores": scores,
    }
