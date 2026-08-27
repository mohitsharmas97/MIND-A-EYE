"""
MINDAEYE — Medical imaging AI
  MIND → Brain tumor classification (MRI)
  EYE  → Diabetic retinopathy staging (fundus)
"""

import base64
import io
import json
import os
import uuid
from datetime import datetime, timedelta
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

# Keep matplotlib from building font cache at import (Grad-CAM pulls it in)
os.environ.setdefault("MPLBACKEND", "Agg")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "mindaye-dev-secret-change-in-production")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///mindaye.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ── Model config ──────────────────────────────────────────────────────────────

ENABLE_GRADCAM = os.getenv("ENABLE_GRADCAM", "true").lower() == "true"

_device = None


def get_device():
    global _device
    if _device is None:
        import torch
        torch.set_num_threads(1)
        _device = torch.device("cpu")
    return _device

MODEL_CONFIG = {
    "dr": {
        "path": "models/best_dr_transfer_model.pth",
        "classes": ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"],
        "num_classes": 5,
        "label": "Diabetic Retinopathy (Eye)",
        "specialist": "Ophthalmologist",
    },
    "brain_tumor": {
        "path": "models/best_brain_tumor_model.pth",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "num_classes": 4,
        "label": "Brain Tumor (Mind)",
        "specialist": "Neurologist",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_model_cache = {}
_latest_result = {}


def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Database ──────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship("Prediction", backref="user", lazy="dynamic")


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)
    model_label = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def probability_dict(self):
        return json.loads(self.probabilities)

    def sorted_probabilities(self):
        return sorted(self.probability_dict().items(), key=lambda x: x[1], reverse=True)


with app.app_context():
    db.create_all()


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_user():
    user = None
    if "user_id" in session:
        row = db.session.get(User, session["user_id"])
        if row:
            user = {"id": row.id, "name": row.name, "email": row.email}
    return {"current_user": user}


# ── ML helpers ────────────────────────────────────────────────────────────────

def load_model(model_type):
    if model_type in _model_cache:
        return _model_cache[model_type]

    import torch
    import torch.nn as nn
    from torchvision import models

    device = get_device()
    cfg = MODEL_CONFIG[model_type]
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, cfg["num_classes"], bias=True),
    )
    model.load_state_dict(torch.load(cfg["path"], map_location=device, weights_only=False))
    model.to(device).eval()
    _model_cache[model_type] = model
    return model


def read_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    data = list(image.getdata())
    clean = Image.new(image.mode, image.size)
    clean.putdata(data)
    return clean


def predict(model, image, classes):
    import torch
    import torch.nn.functional as F

    device = get_device()
    tensor = get_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)

    prob_values = probs[0].cpu().tolist()
    probabilities = {cls: round(p * 100, 2) for cls, p in zip(classes, prob_values)}
    result_class = classes[idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    return result_class, confidence_pct, tensor, probabilities


def generate_heatmap(model, input_tensor, original_image):
    if not ENABLE_GRADCAM:
        return None
    try:
        import numpy as np
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    img_resized = original_image.resize((224, 224))
    img_norm = np.array(img_resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_prediction(model_type, file_bytes):
    cfg = MODEL_CONFIG[model_type]
    model = load_model(model_type)

    image = read_image(file_bytes)
    result_class, confidence, tensor, probabilities = predict(model, image, cfg["classes"])
    heatmap_b64 = generate_heatmap(model, tensor, image)

    result = {
        "result": result_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_type": model_type,
        "model_label": cfg["label"],
        "recommended_specialist": cfg["specialist"],
    }
    if heatmap_b64:
        result["heatmap_data"] = f"data:image/jpeg;base64,{heatmap_b64}"
    return result


def save_prediction(user_id, result):
    record = Prediction(
        user_id=user_id,
        model_type=result["model_type"],
        model_label=result["model_label"],
        result=result["result"],
        confidence=result["confidence"],
        probabilities=json.dumps(result["probabilities"]),
    )
    db.session.add(record)
    db.session.commit()
    return record.id


def make_pdf_report(result, patient_name="Patient"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MINDAEYE Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Patient: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Analysis: {result.get('model_label', 'N/A')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    diagnosis = result["result"].replace("_", " ")
    pdf.cell(0, 10, f"Diagnosis: {diagnosis}", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Confidence: {result.get('confidence', 'N/A')}%", ln=True)
    pdf.cell(0, 8, f"Recommended specialist: {result.get('recommended_specialist', 'N/A')}", ln=True)
    pdf.ln(3)

    probs = result.get("probabilities")
    if probs:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Class probabilities:", ln=True)
        pdf.set_font("Arial", size=10)
        for label, pct in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            pdf.cell(0, 6, f"  {label.replace('_', ' ')}: {pct}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0, 5,
        "DISCLAIMER: AI-generated preliminary analysis only. "
        "Not a substitute for professional medical diagnosis. "
        "Consult a qualified healthcare provider.",
    )
    pdf.ln(3)
    pdf.cell(0, 8, "Generated by MINDAEYE", ln=True, align="C")

    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", f"{patient_name}_report.pdf")
    pdf.output(path)
    return path


# ── Routes: pages ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/upload")
@login_required
def upload_page():
    return render_template("upload.html")


@app.route("/history")
@login_required
def history_page():
    rows = (
        Prediction.query.filter_by(user_id=session["user_id"])
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return render_template("history.html", predictions=rows)


@app.route("/login")
def login_page():
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/signup")
def signup_page():
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("signup.html")


# ── Routes: auth API ──────────────────────────────────────────────────────────

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered."}), 400

    user = User(
        name=name,
        email=email,
        password=generate_password_hash(password, method="pbkdf2:sha256"),
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"success": True, "message": "Account created. Please log in."}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    session["user_id"] = user.id
    session["user_name"] = user.name
    session["user_email"] = user.email
    if data.get("remember"):
        session.permanent = True

    return jsonify({"success": True, "message": "Login successful.", "redirect": "/upload"})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ── Routes: ML API ────────────────────────────────────────────────────────────

@app.route("/predict/<model_type>", methods=["POST"])
@login_required
def predict_endpoint(model_type):
    global _latest_result

    if model_type not in MODEL_CONFIG:
        return jsonify({"error": "Invalid model type."}), 400

    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image provided."}), 400

    try:
        result = run_prediction(model_type, file.read())
        _latest_result = result
        prediction_id = save_prediction(session["user_id"], result)
        session["latest_result"] = {
            k: v for k, v in result.items() if k != "heatmap_data"
        }
        session["latest_prediction_id"] = prediction_id
        result["prediction_id"] = prediction_id
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/download_report")
@login_required
def download_report():
    result = session.get("latest_result") or _latest_result
    if not result:
        return jsonify({"error": "No diagnosis available. Run an analysis first."}), 400

    name = request.args.get("name") or session.get("user_name", "Patient")
    path = make_pdf_report(result, name)
    return send_file(path, as_attachment=True, download_name=f"{name}_mindaye_report.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
