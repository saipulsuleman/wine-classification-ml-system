@echo off
echo ===== Breast Cancer Classification ML System =====
echo.

echo [1] Starting MLflow UI di http://localhost:5000 ...
start "MLflow UI" cmd /k "cd /d "E:\Proyek Akhir" && python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000"
timeout /t 2 > nul

echo [2] Starting Model Serving API di http://localhost:8000 ...
start "Model API" cmd /k "cd /d "E:\Proyek Akhir" && python -m uvicorn model_serving.app:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 > nul

echo [3] Starting Prometheus di http://localhost:9090 ...
cd /d "E:\Proyek Akhir\monitoring\prometheus-windows\prometheus-2.47.0.windows-amd64"
start "Prometheus" prometheus.exe --config.file=prometheus.yml --storage.tsdb.path=data --web.listen-address=0.0.0.0:9090
timeout /t 2 > nul

echo [4] Grafana berjalan sebagai Windows Service di http://localhost:3000
echo     Jika belum jalan, jalankan: net start Grafana

echo.
echo === Service URLs ===
echo   MLflow UI:     http://localhost:5000
echo   Model API:     http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Metrics:       http://localhost:8000/metrics
echo   Prometheus:    http://localhost:9090
echo   Grafana:       http://localhost:3000
echo   Grafana login: admin / admin123
echo   Dashboard:     http://localhost:3000/d/saipulkarimsuleman-bc-ml
echo.
pause
