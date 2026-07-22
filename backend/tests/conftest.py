import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect DB and uploads to a temp path
    import backend.config as config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)

    import backend.models.database as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")

    # Mock inference so the real model is never loaded
    def mock_inference(image_path: str) -> dict:
        scores = {"Glass": 0.7, "Metal": 0.1, "Paper": 0.1,
                  "Plastic": 0.05, "Waste": 0.05}
        return {"top_class": "Glass", "confidence": 0.7, "scores": scores}

    import backend.routers.predict as predict_router
    monkeypatch.setattr(predict_router, "run_inference", mock_inference)

    from backend.main import app
    with TestClient(app) as c:
        yield c
