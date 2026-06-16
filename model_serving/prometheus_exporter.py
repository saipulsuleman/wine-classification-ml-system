"""
Prometheus metrics definitions for Breast Cancer Classification API.
These counters and histograms are exposed via the /metrics endpoint.
"""
from prometheus_client import Counter, Histogram, Gauge

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