from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weights" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "waste.db"

CLASSES = ["Glass", "Metal", "Paper", "Plastic", "Waste"]

UPLOAD_DIR.mkdir(exist_ok=True)
