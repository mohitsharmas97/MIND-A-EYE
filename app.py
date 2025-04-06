from flask import Flask, render_template, request, jsonify, send_file
from fpdf import FPDF
import os
import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from chatbot import chatbot_response, PROJECT_INFO, MEDICAL_INFO, FAQ_DATABASE, GENERAL_KNOWLEDGE

# === Initialize Flask App ===
app = Flask(__name__, static_folder='static', template_folder='templates')


# === Globals ===
latest_result = {}  # Stores the latest diagnosis result
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATHS = {
    'dr': 'models/dr_model.pth',
    'brain_tumor': 'models/brain_tumor.pth',
}
model_cache = {}

# === Lazy Model Loading ===
def load_model(model_type):
    if model_type in model_cache:
        return model_cache[model_type]

    if model_type == 'dr':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=5, bias=True)
        )
    elif model_type == 'brain_tumor':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=4, bias=True)
        )
    else:
        return None

    try:
        model.load_state_dict(torch.load(MODEL_PATHS[model_type], map_location=device))
        model.to(device)
        model.eval()
        model_cache[model_type] = model
        return model
    except FileNotFoundError:
        print(f"Model file for {model_type} not found.")
        return None

# === Prediction Function ===
def predict_image(model, image_path, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# === Routes ===
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict/dr', methods=['POST'])
def predict_dr():
    global latest_result
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    os.makedirs('uploads', exist_ok=True)
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)

    try:
        model = load_model('dr')
        if model is None:
            return jsonify({'error': 'DR model not found'}), 500

        classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
        result = predict_image(model, temp_path, classes)
        latest_result = {'result': result}
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/predict/brain_tumor', methods=['POST'])
def predict_brain_tumor():
    global latest_result
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    os.makedirs('uploads', exist_ok=True)
    temp_path = os.path.join('uploads', file.filename)
    file.save(temp_path)

    try:
        model = load_model('brain_tumor')
        if model is None:
            return jsonify({'error': 'Brain tumor model not found'}), 500

        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        result = predict_image(model, temp_path, classes)
        latest_result = {'result': result}
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/get_latest_diagnosis_result')
def get_latest_diagnosis_result():
    """API endpoint to fetch the latest diagnosis result for the chatbot"""
    if latest_result.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        latest_result['condition_type'] = 'diabetic_retinopathy'
        latest_result['recommended_specialist'] = 'Ophthalmologist'
    elif latest_result.get('result') in ['glioma', 'meningioma', 'pituitary']:
        latest_result['condition_type'] = 'brain_tumor'
        latest_result['recommended_specialist'] = 'Neurologist'
    
    # Return empty dict if no diagnosis yet
    return latest_result or {}

def get_latest_diagnosis_result():
    """Function that returns the latest diagnosis result for the chatbot"""
    global latest_result
    
    if not latest_result:
        return {}
        
    result_copy = latest_result.copy()
    
    # Add additional information based on diagnosis type
    if result_copy.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        result_copy['condition_type'] = 'diabetic_retinopathy'
        result_copy['recommended_specialist'] = 'Ophthalmologist'
    elif result_copy.get('result') in ['glioma', 'meningioma', 'pituitary']:
        result_copy['condition_type'] = 'brain_tumor'
        result_copy['recommended_specialist'] = 'Neurologist'
    elif result_copy.get('result') == 'No_DR':
        result_copy['condition_type'] = 'normal'
        result_copy['notes'] = 'No signs of diabetic retinopathy detected'
    elif result_copy.get('result') == 'notumor':
        result_copy['condition_type'] = 'normal'
        result_copy['notes'] = 'No brain tumor detected'
        
    return result_copy

@app.route('/download_report')
def download_report():
    patient_name = request.args.get('name', 'Anonymous')
    if not latest_result:
        return jsonify({'error': 'No diagnosis available'}), 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, txt=f"Diagnosis: {latest_result['result']}", ln=True, align='L')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="This report is AI-generated. Please consult a medical professional.", align='L')
    pdf.set_font("Arial", style='I', size=10)
    pdf.cell(0, 10, txt="Generated by AI Diagnosis System", ln=True, align='C')

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"{patient_name}_diagnosis_report.pdf")
    pdf.output(report_path)

    return send_file(report_path, as_attachment=True, download_name=f"{patient_name}_diagnosis_report.pdf")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    # Use the chatbot_response function imported from chatbot.py
    response = chatbot_response(user_message)
    return jsonify({'response': response})


# Other UI routes (optional)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/consult')
def consult():
    return render_template('consult.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html') 

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    return jsonify({'success': True, 'message': 'User registered successfully'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
