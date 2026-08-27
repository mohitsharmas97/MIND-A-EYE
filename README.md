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

## Quick Start (Local)

```bash
cd MIND-A-EYE
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements-local.txt

# Place model weights in models/
#   models/best_dr_transfer_model.pth
#   models/best_brain_tumor_model.pth

python app.py
```

Open http://localhost:5000

---

## Training Notebooks

| Notebook | Model saved |
|----------|-------------|
| `diabetic_retinopathy_tl.ipynb` | `models/best_dr_transfer_model.pth` |
| `brain_tumor_tl.ipynb` | `models/best_brain_tumor_model.pth` |

Both use EfficientNet-B0 with transfer learning (two-stage fine-tuning).

---

## Deploy on Render

1. Push code + model weights (`models/*.pth`) — use [Git LFS](https://git-lfs.com) for large files
2. In Render dashboard → **Environment**:
   - `ENABLE_GRADCAM` = `false`
   - `SECRET_KEY` = (set a secure value)
3. Ensure `runtime.txt` (`python-3.11.11`) is committed

| | Local | Render |
|---|-------|--------|
| Install | `pip install -r requirements-local.txt` | `pip install -r requirements.txt` |
| Grad-CAM | Yes | Disabled (`ENABLE_GRADCAM=false`) |
| Python | 3.11+ | 3.11.11 (pinned) |

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
