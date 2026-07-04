# MINDAEYE

**MIND** + **EYE** — AI medical image analysis in one Flask app.

| Name | Domain | Input | Model |
|------|--------|-------|-------|
| **MIND** | Brain tumor | MRI (JPG/PNG) | EfficientNet-B0, 4 classes |
| **EYE** | Diabetic retinopathy | Fundus photo (JPG/PNG) | EfficientNet-B0, 5 stages |

> Decision-support tool for portfolio and research. Not a licensed medical device.

---

## Features

- Dual PyTorch models with lazy loading
- Full softmax probability bars (all classes)
- Prediction history per user (SQLite)
- Grad-CAM heatmaps locally (`ENABLE_GRADCAM=true`)
- Simple PDF report download
- User signup / login

---

## Quick start (local)

```bash
cd MIND-A-EYE
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-local.txt   # includes Grad-CAM

# Place dr_model.pth and brain_tumor.pth in models/
python app.py
```

Open http://localhost:5000

---

## Deploy on Render

### Current issue (build failed)

If you see:
```text
ERROR: No matching distribution found for torch==2.5.1
Using Python version 3.14.3
```

**Cause:** Render defaulted to Python 3.14, but old torch pins only exist for Python 3.11. This repo now uses `torch==2.12.1` (CPU) which works on 3.14.

**Also pin Python 3.11** (recommended, smaller/faster builds):

1. Render Dashboard → your service → **Environment**
2. Add: `PYTHON_VERSION` = `3.11.11`
3. Redeploy

Or ensure `runtime.txt` (contains `python-3.11.11`) is committed and pushed to GitHub.

### Render checklist

Render free tier has **512 MB RAM**. The default `pip install torch` pulled **CUDA PyTorch + NVIDIA drivers (~2 GB)** and Grad-CAM pulled **matplotlib/scipy**, which exceeded memory at startup.

### Fixes applied

| Fix | Why |
|-----|-----|
| CPU-only PyTorch (`--extra-index-url` in `requirements.txt`) | No CUDA packages |
| Python 3.11 (`runtime.txt`) | Avoids Python 3.14 pulling latest huge torch |
| Grad-CAM removed from production deps | Saves ~200 MB at import |
| `ENABLE_GRADCAM=false` on Render | No matplotlib at startup |
| Lazy torch imports | Faster gunicorn bind |
| `--workers 1 --threads 1` | One process only |

### Render checklist

1. Push this code + model weights (`models/*.pth`) — use [Git LFS](https://git-lfs.com) for large files
2. In Render dashboard → **Environment**:
   - `ENABLE_GRADCAM` = `false`
   - `SECRET_KEY` = (auto-generated or set manually)
3. Redeploy — build should install ~200 MB torch, not ~2 GB CUDA stack
4. If still OOM on **first prediction**, upgrade to Render **Starter** plan (512 MB → 2 GB)

### Local vs Render

| | Local | Render |
|---|-------|--------|
| Install | `pip install -r requirements-local.txt` | `pip install -r requirements.txt` |
| Grad-CAM | Yes | Disabled (`ENABLE_GRADCAM=false`) |
| Python | 3.11+ recommended | 3.11.11 (pinned) |

---

## API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/predict/dr` | Yes | EYE — diabetic retinopathy |
| POST | `/predict/brain_tumor` | Yes | MIND — brain tumor |
| GET | `/history` | Yes | Past predictions |
| GET | `/download_report` | Yes | PDF report |

---

## Author

Mohit Sharma — Data Science / ML portfolio project
