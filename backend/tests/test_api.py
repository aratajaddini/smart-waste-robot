import io


def _fake_image():
    return io.BytesIO(b"fakeimagebytes")


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict(client):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["top_class"] == "Glass"
    assert set(data["scores"].keys()) == {"Glass", "Metal", "Paper", "Plastic", "Waste"}
    assert data["id"] >= 1


def test_history(client):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    client.post("/predict", files=files)
    r = client.get("/history")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_feedback(client):
    files = {"file": ("test.jpg", _fake_image(), "image/jpeg")}
    pred = client.post("/predict", files=files).json()
    r = client.post("/feedback", json={"prediction_id": pred["id"], "correct_class": "Metal"})
    assert r.status_code == 200
    assert r.json()["correct_class"] == "Metal"


def test_feedback_not_found(client):
    r = client.post("/feedback", json={"prediction_id": 9999, "correct_class": "Metal"})
    assert r.status_code == 404
