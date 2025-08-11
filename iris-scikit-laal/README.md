# ⚡ Quick Start — JupyterLab, MLflow y Ray (Windows)

![Windows](https://img.shields.io/badge/Windows-10%2B-informational) ![Localhost](https://img.shields.io/badge/localhost-127.0.0.1-blue)

> **Nota:** Ejecuta estos comandos en **CMD** o **PowerShell**.

---

### 🧪 JupyterLab
```powershell
jupyter lab
```

### 📊 MLflow
```powershell
mlflow server --host 127.0.0.1 --port 8080
```

### 🧠 Ray
```powershell
ray start --head --ray-client-server-port=10001 --port=6379 --dashboard-port=8265
```

— listo 🚀