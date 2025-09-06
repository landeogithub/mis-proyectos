@echo off
setlocal
set "BASEDIR=D:\Archivos\models"

:: ---- JupyterLab (sin token/clave) ----
start "JupyterLab" /D "%BASEDIR%" cmd /k ^
  jupyter lab --ServerApp.token= --ServerApp.password= ^
  --ServerApp.ip=127.0.0.1 --ServerApp.port=8888 ^
  --ServerApp.root_dir="%BASEDIR%" --ServerApp.open_browser=False

:: ---- MLflow ----
start "MLflow" /D "%BASEDIR%" cmd /k ^
  mlflow server --host 127.0.0.1 --port 8080 ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root ./mlruns

:: ---- Ray (head) ----
start "Ray Head" /D "%BASEDIR%" cmd /k ^
  ray start --head --ray-client-server-port=10001 --dashboard-port=8265 --port=6379

:: Abrir UIs
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8080/#/
start "" http://127.0.0.1:8265/#/overview
start "" http://127.0.0.1:8888/

endlocal
