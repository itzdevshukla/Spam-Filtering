"""Integration tests for Django views and REST API endpoints."""

import json
import os
import sys
from pathlib import Path
import pytest
from django.test import Client

# Configure Django settings for test runner
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "spamshield_web"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "spamshield_project.settings")
import django
django.setup()

from detector.models import PredictionLog


@pytest.fixture
def client():
    return Client()


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SpamShield" in response.content
    assert b"Real-Time Message Classifier" in response.content


def test_index_form_classification(client):
    response = client.post("/", {"message_text": "URGENT! You won £1000 cash. Call 0800123456 now!"})
    assert response.status_code == 200
    assert b"SPAM" in response.content
    # Should create a database log
    assert PredictionLog.objects.filter(predicted_label="SPAM").exists()


def test_batch_page_loads(client):
    response = client.get("/batch/")
    assert response.status_code == 200
    assert b"Batch Message Classification" in response.content


def test_dashboard_page_loads(client):
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert b"Model Evaluation & Academic Viva Dashboard" in response.content


def test_api_predict_endpoint(client):
    payload = {"text": "Hey Mike, are we studying at the library tonight?"}
    response = client.post(
        "/api/predict/",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["label"] == "HAM"
    assert data["spam_probability"] < 0.35


def test_api_batch_predict_endpoint(client):
    payload = {
        "texts": [
            "WINNER!! Claim your £5000 prize now!",
            "Can you pick up groceries on the way home?",
        ]
    }
    response = client.post(
        "/api/batch-predict/",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 2
