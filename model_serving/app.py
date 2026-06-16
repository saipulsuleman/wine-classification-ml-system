from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import os
import time
import logging
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="ML Model Serving API for Breast Cancer Classification using MLflow",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_request_count_total",
    "Total number of API requests",
    ["method", "endpoint", "status"]
)
PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Total number of predictions made",
    ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing predictions",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_ERROR_COUNT = Counter(
    "prediction_error_count_total",
    "Total number of prediction errors"
)
MODEL_INFO = Gauge(
    "model_info",
    "Current model information",
    ["model_name", "model_version"]
)

# Global model holder
model = None
model_name = "unknown"

CLASS_NAMES = {0: "malignant", 1: "benign"}
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="List of 30 breast cancer features",
        example=[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    )


class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_class_name: str
    probabilities: List[float]
    model_name: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


@app.on_event("startup")
async def startup_event():
    global model, model_name
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Try to load model from MLflow model registry or local path
    model_uri = os.getenv("MODEL_URI", None)
    if model_uri is None:
        # Try to find latest run and load from there
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("breast-cancer-classification-experiment")
            if experiment:
                runs = client.search_runs(
                    experiment.experiment_id,
                    order_by=["metrics.accuracy DESC"],
                    max_results=1
                )
                if runs:
                    best_run = runs[0]
                    model_uri = f"runs:/{best_run.info.run_id}/model"
                    model_name = best_run.info.run_name or "best_model"
                    logger.info(f"Loading model from run: {best_run.info.run_id}")
        except Exception as e:
            logger.warning(f"Could not find model from experiment: {e}")

    if model_uri:
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully: {model_name}")
            MODEL_INFO.labels(model_name=model_name, model_version="1.0").set(1)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning("No model URI found. Predictions will fail until model is loaded.")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/metrics", "/docs"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_name
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    global model
    if model is None:
        PREDICTION_ERROR_COUNT.inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure MLflow experiment has been run.")

    start_time = time.time()
    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        inference_time_ms = (time.time() - start_time) * 1000

        PREDICTION_COUNT.labels(predicted_class=CLASS_NAMES.get(int(prediction), str(prediction))).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()

        logger.info(f"Prediction: class={prediction}, time={inference_time_ms:.2f}ms")

        return PredictionOutput(
            predicted_class=int(prediction),
            predicted_class_name=CLASS_NAMES.get(int(prediction), str(prediction)),
            probabilities=probabilities,
            model_name=model_name,
            inference_time_ms=round(inference_time_ms, 3)
        )
    except Exception as e:
        PREDICTION_ERROR_COUNT.inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
