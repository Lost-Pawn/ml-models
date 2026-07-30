from fastapi.testclient import TestClient
from api.main import app

def test_predict_endpoint():
    client = TestClient(app)
    with open("data/test/cats/cat.65.jpg", "rb") as f:
        response = client.post("/predict/", files={"file": ("cat.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    assert "prediction" in response.json()