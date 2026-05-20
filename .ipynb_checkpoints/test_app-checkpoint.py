from app import app
import json

# Create test client
client = app.test_client() 

def test_home():

    response = client.get("/")

    assert response.status_code == 200

def test_prediction():

    data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = client.post(
        "/predict",
        data=json.dumps(data),
        content_type="application/json"
    )

    assert response.status_code == 200