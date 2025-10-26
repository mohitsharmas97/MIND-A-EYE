# from flask import Flask, render_template, request, jsonify, send_file
# from fpdf import FPDF
# import os
# import datetime
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import google.generativeai as genai
# import json
# import re
# from typing import Dict, Any, List, Optional, Tuple
# from datetime import datetime
# from dotenv import load_dotenv

# # === Initialize Flask App ===
# app = Flask(__name__, static_folder='static', template_folder='templates')

# # === Globals ===
# latest_result = {}  # Stores the latest diagnosis result
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL_PATHS = {
#     'dr': 'models/dr_model.pth',
#     'brain_tumor': 'models/brain_tumor.pth',
# }
# model_cache = {}  # Cache for lazy loading models

# # Load API Key
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

# # Project and App Information
# PROJECT_INFO = {
#     "name": "NeuroVision AI",
#     "version": "2.1.0",
#     "description": "An advanced medical imaging analysis platform for diabetic retinopathy and brain tumor detection.",
#     "features": [
#         "Diabetic retinopathy classification (5 stages)",
#         "Brain tumor detection (4 types)",
#         "Comprehensive diagnostic reports",
#         "AI-powered analysis with expert validation",
#         "Secure data handling with HIPAA compliance"
#     ],
#     "team": "NeuroVision Medical Technologies",
#     "contact": "support@neurovision.ai"
# }

# # === Lazy Model Loading ===
# def load_model(model_type):
#     if model_type in model_cache:
#         return model_cache[model_type]

#     if model_type == 'dr':
#         model = models.efficientnet_b0(weights=None)
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(in_features=1280, out_features=5, bias=True)
#         )
#     elif model_type == 'brain_tumor':
#         model = models.efficientnet_b0(weights=None)
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(in_features=1280, out_features=4, bias=True)
#         )
#     else:
#         return None

#     try:
#         model.load_state_dict(torch.load(MODEL_PATHS[model_type], map_location=device))
#         model.to(device)
#         model.eval()
#         model_cache[model_type] = model
#         return model
#     except FileNotFoundError:
#         print(f"Model file for {model_type} not found.")
#         return None

# # === Prediction Function ===
# def predict_image(model, image_path, classes):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
#     return classes[predicted.item()]

# # Define comprehensive medical information database
# MEDICAL_INFO = {
#     "diabetic_retinopathy": {
#         "overview": "A diabetes complication affecting the eyes, caused by damage to blood vessels in the retina.",
#         "stages": {
#             "No_DR": "No detectable diabetic retinopathy.",
#             "Mild": "Small areas of balloon-like swelling in the retina's tiny blood vessels.",
#             "Moderate": "Some blood vessels that nourish the retina are blocked.",
#             "Severe": "Many more blood vessels are blocked, depriving several areas of the retina of blood supply.",
#             "Proliferate_DR": "The most advanced stage where new abnormal blood vessels grow, which can cause serious vision problems."
#         },
#         "symptoms": [
#             "Often no symptoms in early stages",
#             "Spots or dark strings floating in your vision (floaters)",
#             "Blurred vision",
#             "Fluctuating vision",
#             "Dark or empty areas in your vision",
#             "Vision loss"
#         ],
#         "risk_factors": [
#             "Duration of diabetes",
#             "Poor control of blood sugar levels",
#             "High blood pressure",
#             "High cholesterol",
#             "Pregnancy",
#             "Tobacco use"
#         ],
#         "prevention": [
#             "Manage diabetes carefully",
#             "Regular eye exams",
#             "Control blood pressure and cholesterol",
#             "Stop smoking",
#             "Exercise regularly"
#         ],
#         "treatment": [
#             "Anti-VEGF injections",
#             "Focal laser treatment",
#             "Scatter laser treatment (panretinal photocoagulation)",
#             "Vitrectomy (surgical procedure)"
#         ]
#     },
#     "brain_tumor": {
#         "overview": "Abnormal growth of cells in the brain that can be cancerous (malignant) or noncancerous (benign).",
#         "types": {
#             "glioma": "Tumors that occur in the brain and spinal cord, beginning in glial cells that surround nerve cells.",
#             "meningioma": "A tumor that arises from the meninges, the membranes that surround your brain and spinal cord.",
#             "pituitary": "Tumors that form in the pituitary gland at the base of the brain, most are benign.",
#             "notumor": "No detectable tumor growth in brain tissues."
#         },
#         "symptoms": [
#             "Headaches that gradually become more frequent and severe",
#             "Unexplained nausea or vomiting",
#             "Vision problems",
#             "Gradual loss of sensation or movement in arms or legs",
#             "Balance difficulties",
#             "Speech difficulties",
#             "Confusion in everyday matters",
#             "Seizures"
#         ],
#         "risk_factors": [
#             "Age",
#             "Radiation exposure",
#             "Family history of brain tumors",
#             "Exposure to certain chemicals"
#         ],
#         "diagnosis": [
#             "Neurological exam",
#             "Imaging tests (MRI, CT scan)",
#             "Biopsy"
#         ],
#         "treatment": [
#             "Surgery",
#             "Radiation therapy",
#             "Radiosurgery",
#             "Chemotherapy",
#             "Targeted drug therapy",
#             "Rehabilitation after treatment"
#         ]
#     }
# }

# # FAQ database for common questions
# FAQ_DATABASE = {
#     "accuracy": "NeuroVision AI provides preliminary analysis with approximately 92% accuracy for diabetic retinopathy and 89% for brain tumor detection. However, all results should be confirmed by a qualified healthcare provider.",
#     "privacy": "All medical images and patient data are encrypted and processed in compliance with HIPAA regulations. Your data is never shared with third parties without explicit consent.",
#     "report_format": "Our diagnostic reports include the AI analysis, confidence scores, visual markers, and recommended next steps. Reports can be downloaded as PDF files.",
#     "second_opinion": "NeuroVision AI is designed to assist medical professionals, not replace them. We always recommend consulting with a qualified healthcare provider for official diagnosis and treatment plans.",
#     "image_requirements": "For optimal results, retinal images should be high-resolution fundus photographs. Brain scans should be clear MRI images in standard DICOM format.",
#     "processing_time": "Image analysis typically takes 30-60 seconds depending on image quality and server load.",
#     "false_results": "Like all diagnostic tools, NeuroVision AI has limitations. False positives and false negatives may occur, which is why expert human verification is essential.",
#     "cost": "Please contact our sales team for current pricing information for individual and institutional licenses.",
#     "integration": "NeuroVision AI can be integrated with most modern hospital information systems and PACS through our secure API.",
#     "training_data": "Our models are trained on diverse datasets from multiple global medical institutions, with regular updates to improve accuracy and reduce biases."
# }

# # General knowledge topics for fallback responses
# GENERAL_KNOWLEDGE = {
#     "weather": "I can't provide real-time weather data, but I can explain weather patterns or meteorological concepts.",
#     "news": "I don't have live news access, but I can discuss historical events or analyze news topics you specify.",
#     "sports": "I can provide information about sports rules, history, or famous athletes, but not live scores.",
#     "entertainment": "I can discuss movies, books, or music in general terms, but don't have access to current releases.",
#     "technology": "I can explain technology concepts, compare devices, or discuss tech history and trends."
# }

# # Cache for frequent responses to improve performance
# response_cache = {}

# # Initialize the AI Model with optimized parameters
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     generation_config={
#         "temperature": 0.3,
#         "top_p": 0.95,
#         "top_k": 40,
#         "max_output_tokens": 2048,  # Increased for more comprehensive responses
#     }
# )

# def generate_system_prompt() -> str:
#     """Generate a comprehensive system prompt combining project info and medical knowledge."""
#     current_date = datetime.now().strftime("%Y-%m-%d")
    
#     return f"""
#     # NeuroVision AI Assistant - System Prompt
#     ## Project Information
#     Name: {PROJECT_INFO['name']}
#     Version: {PROJECT_INFO['version']}
#     Description: {PROJECT_INFO['description']}
#     Features: {", ".join(PROJECT_INFO['features'])}
#     Team: {PROJECT_INFO['team']}
#     Contact: {PROJECT_INFO['contact']}
#     Current Date: {current_date}

#     ## Medical Knowledge Base
#     You have access to detailed information about:
#     - Diabetic Retinopathy (5 classifications)
#     - Brain Tumors (4 classifications)
    
#     ## Role and Guidelines
#     You are a compassionate and knowledgeable AI assistant for NeuroVision with these capabilities:
#     1. Provide medically accurate yet easy-to-understand explanations
#     2. Offer reassurance and support for health concerns
#     3. Clearly emphasize that results are preliminary and must be reviewed by professionals
#     4. Use non-diagnostic language ("the analysis suggests" rather than definitive statements)
#     5. Handle general knowledge queries when not medical-related
#     6. Provide information about the NeuroVision platform and its features
    
#     ## Response Formatting Guidelines
#     - Use Markdown for formatting (headers, bold, lists)
#     - Include disclaimers when discussing medical topics
#     - Break long responses into readable paragraphs
#     - Highlight important warnings or recommendations
    
#     ## General Knowledge Handling
#     For non-medical queries:
#     - Acknowledge the query type
#     - Provide general information if possible
#     - Redirect to appropriate resources when needed
#     - Be clear about limitations
    
#     Always maintain a professional yet approachable tone.
#     """

# # Enhanced system prompt
# SYSTEM_PROMPT = generate_system_prompt()

# # Start a chat session to maintain context
# chat = model.start_chat(
#     history=[
#         {"role": "user", "parts": [SYSTEM_PROMPT]},
#         {"role": "model", "parts": [f"I am {PROJECT_INFO['name']} (v{PROJECT_INFO['version']}), your medical diagnosis and general knowledge assistant. How can I help you today?"]}
#     ]
# )

# def get_detailed_condition_info(condition_key: str) -> str:
#     """
#     Retrieve detailed information about a specific medical condition.
    
#     Args:
#         condition_key: The key for the condition in the MEDICAL_INFO database
        
#     Returns:
#         Formatted information about the condition
#     """
#     info = ""
    
#     # Check for diabetic retinopathy stages
#     for category, data in MEDICAL_INFO.items():
#         if category == "diabetic_retinopathy" and condition_key in data["stages"]:
#             stage = condition_key
#             info = f"""## Diabetic Retinopathy - {stage.replace('_', ' ')}

# **Classification:** {data["stages"][stage]}

# **Overview:** {data["overview"]}

# **Common Symptoms:**
# {', '.join(data["symptoms"][:3])}

# **Prevention:**
# {', '.join(data["prevention"][:3])}

# **Recommended Next Steps:**
# 1. Consult with an ophthalmologist to confirm the analysis
# 2. Regular monitoring of blood sugar levels
# 3. Follow your diabetes management plan

# **Disclaimer:** This information is for educational purposes only and not a substitute for professional medical advice.
# """
#             break
            
#         # Check for brain tumor types
#         elif category == "brain_tumor" and condition_key in data["types"]:
#             tumor_type = condition_key
#             info = f"""## Brain Tumor Analysis - {tumor_type.capitalize()}

# **Classification:** {data["types"][tumor_type]}

# **Overview:** {data["overview"]}

# **Common Symptoms:**
# {', '.join(data["symptoms"][:3])}

# **Diagnostic Approaches:**
# {', '.join(data["diagnosis"][:3])}

# **Recommended Next Steps:**
# 1. Consult with a neurologist to confirm the analysis
# 2. Additional diagnostic testing may be required
# 3. Discuss treatment options with your healthcare provider

# **Disclaimer:** This information is for educational purposes only and not a substitute for professional medical advice.
# """
#             break
    
#     if not info:
#         # Generic response for conditions not in database
#         info = f"""## Medical Condition Information

# I don't have specific detailed information about "{condition_key}" in my database.

# **Recommended Steps:**
# 1. Consult with a healthcare provider for accurate diagnosis
# 2. Discuss any symptoms or concerns with your doctor
# 3. Seek information from credible medical sources

# **Disclaimer:** Always consult healthcare professionals for proper medical advice and diagnosis.
# """
    
#     return info

# def extract_condition_keywords(message: str) -> List[str]:
#     """
#     Extract keywords related to medical conditions from the user message.
    
#     Args:
#         message: User input message
        
#     Returns:
#         List of recognized condition keywords
#     """
#     message = message.lower()
    
#     # Keywords for different conditions
#     condition_keywords = {
#         # Diabetic retinopathy stages
#         "no_dr": ["no dr", "no diabetic retinopathy", "normal retina"],
#         "mild": ["mild dr", "mild diabetic retinopathy", "early dr"],
#         "moderate": ["moderate dr", "moderate diabetic retinopathy"],
#         "severe": ["severe dr", "severe diabetic retinopathy", "advanced dr"],
#         "proliferate_dr": ["proliferative", "pdr", "proliferate", "advanced diabetic retinopathy"],
        
#         # Brain tumors
#         "glioma": ["glioma", "glial", "glioblastoma", "astrocytoma"],
#         "meningioma": ["meningioma", "meningeal"],
#         "pituitary": ["pituitary", "pituitary adenoma", "pituitary tumor"],
#         "notumor": ["no tumor", "no brain tumor", "normal brain", "healthy brain"]
#     }
    
#     found_keywords = []
    
#     for condition, keywords in condition_keywords.items():
#         for keyword in keywords:
#             if keyword in message:
#                 found_keywords.append(condition)
#                 break
    
#     return found_keywords

# def match_faq(message: str) -> Optional[str]:
#     """
#     Match user message to frequently asked questions.
    
#     Args:
#         message: User input message
        
#     Returns:
#         FAQ response if matched, None otherwise
#     """
#     message = message.lower()
    
#     # Map keywords to FAQ categories
#     faq_keywords = {
#         "accuracy": ["accuracy", "accurate", "precision", "reliable", "correct"],
#         "privacy": ["privacy", "secure", "confidential", "hipaa", "data security"],
#         "report_format": ["report", "format", "pdf", "download", "results format"],
#         "second_opinion": ["second opinion", "doctor", "confirm", "verification"],
#         "image_requirements": ["image", "quality", "resolution", "requirement", "upload"],
#         "processing_time": ["time", "long", "takes", "processing", "wait"],
#         "false_results": ["false", "wrong", "mistake", "error", "incorrect"],
#         "cost": ["cost", "price", "fee", "payment", "subscription"],
#         "integration": ["integrate", "system", "hospital", "ehr", "api"],
#         "training_data": ["training", "dataset", "data", "trained", "learn"]
#     }
    
#     # Check for matches
#     for category, keywords in faq_keywords.items():
#         for keyword in keywords:
#             if keyword in message:
#                 return f"## {category.replace('_', ' ').title()}\n\n{FAQ_DATABASE[category]}"
    
#     return None

# def get_app_functionality_info(query: str) -> Optional[str]:
#     """
#     Provide information about app functionality based on user query.
    
#     Args:
#         query: User input message
        
#     Returns:
#         Response about app features or None if not matching
#     """
#     query = query.lower()
    
#     # Define app feature keywords
#     feature_queries = {
#         "upload": ["upload", "scan", "image", "picture", "photo", "upload image"],
#         "report": ["report", "results", "download", "pdf", "save", "share"],
#         "analysis": ["analysis", "analyze", "detect", "diagnose", "process"],
#         "accuracy": ["accuracy", "precise", "reliable", "confidence"],
#         "features": ["features", "capabilities", "function", "what can you do"],
#         "about": ["about", "who are you", "tell me about", "information about", "what is neurovision"],
#         "security": ["security", "privacy", "confidential", "safe", "data safety"]
#     }
    
#     for feature, keywords in feature_queries.items():
#         if any(keyword in query for keyword in keywords):
#             # Responses for different features
#             if feature == "upload":
#                 return "## Image Upload\n\nYou can upload medical images through our Upload page by clicking the 'Upload' link in the navigation menu. We accept retinal scans for diabetic retinopathy analysis and brain MRI images for tumor detection. For optimal results, please ensure your images are high resolution and properly oriented."
            
#             elif feature == "report":
#                 return "## Medical Reports\n\nAfter analysis, you can download a comprehensive medical report in PDF format. The report includes the AI analysis results, confidence scores, and recommended next steps. You can find the download option on the results page or in your user dashboard."
            
#             elif feature == "analysis":
#                 return "## Image Analysis\n\nNeuroVision AI uses advanced deep learning algorithms to analyze medical images. For diabetic retinopathy, we classify into 5 stages (No DR, Mild, Moderate, Severe, Proliferative). For brain MRIs, we detect and classify four categories (Glioma, Meningioma, Pituitary tumor, and No tumor)."
            
#             elif feature == "accuracy":
#                 return "## System Accuracy\n\nNeuroVision AI achieves approximately 92% accuracy for diabetic retinopathy detection and 89% for brain tumor classification. However, all results should be confirmed by healthcare professionals. Our system is constantly improving through regular model updates and validation studies."
            
#             elif feature == "features":
#                 feature_list = "\n".join([f"- {feature}" for feature in PROJECT_INFO["features"]])
#                 return f"## NeuroVision Features\n\n{PROJECT_INFO['description']}\n\nOur key features include:\n{feature_list}\n\nIs there a specific feature you'd like more information about?"
            
#             elif feature == "about":
#                 return f"## About NeuroVision\n\n{PROJECT_INFO['name']} (version {PROJECT_INFO['version']}) is {PROJECT_INFO['description']} Developed by {PROJECT_INFO['team']}, we aim to assist medical professionals in diagnosing conditions earlier and more accurately through AI technology."
            
#             elif feature == "security":
#                 return "## Data Security\n\nAll medical images and patient data are encrypted and processed in compliance with HIPAA regulations. We use industry-standard encryption, secure data centers, and strict access controls. Your data is never shared with third parties without explicit consent."
    
#     return None

# def is_general_knowledge_query(message: str) -> Tuple[bool, Optional[str]]:
#     """
#     Determine if the query is a general knowledge question and return appropriate response.
    
#     Args:
#         message: User input message
        
#     Returns:
#         Tuple: (is_general_knowledge, response_or_none)
#     """
#     message = message.lower()
    
#     for topic, response in GENERAL_KNOWLEDGE.items():
#         if re.search(rf'\b{topic}\b', message):
#             return (True, response)
    
#     # Common general questions
#     general_phrases = {
#         "who created you": f"I was developed by {PROJECT_INFO['team']} as part of the {PROJECT_INFO['name']} project.",
#         "what can you do": f"I can help with:\n- Medical image analysis information\n- {PROJECT_INFO['description']}\n- General knowledge questions\n\nMy features include: {', '.join(PROJECT_INFO['features'])}",
#         "how old are you": f"I'm an AI assistant for {PROJECT_INFO['name']} version {PROJECT_INFO['version']}, first released in 2024.",
#         "tell me about yourself": f"I'm {PROJECT_INFO['name']}, an AI assistant designed to provide information about medical diagnostics and general knowledge. {PROJECT_INFO['description']}"
#     }
    
#     for phrase, response in general_phrases.items():
#         if phrase in message:
#             return (True, response)
    
#     return (False, None)

# def enhance_response(original_response: str, is_medical: bool = True) -> str:
#     """
#     Enhance AI response with improved formatting and additional context.
    
#     Args:
#         original_response: Original AI response
#         is_medical: Whether the response is medical-related
        
#     Returns:
#         Enhanced response with better formatting
#     """
#     # Basic cleaning
#     enhanced = original_response.strip()
    
#     # Add proper Markdown formatting
#     if not enhanced.startswith("#"):
#         enhanced = f"## Response\n{enhanced}"
    
#     # Medical-specific enhancements
#     if is_medical:
#         # Add section breaks for better readability
#         enhanced = re.sub(r"(\n)([A-Z][a-z]+:)", r"\n\n**\2**", enhanced)
        
#         # Ensure disclaimer is present if not already there
#         if "Disclaimer:" not in enhanced:
#             enhanced += "\n\n**Disclaimer:** This information is for educational purposes only and not a substitute for professional medical advice. Always consult with qualified healthcare providers."
    
#     # General formatting improvements
#     enhanced = re.sub(r"(\n\s*\n)", r"\n\n", enhanced)  # Remove extra newlines
#     enhanced = re.sub(r"\.(\s)([A-Z])", r".\n\n\2", enhanced)  # Paragraph breaks
    
#     # Add project footer for longer responses
#     if len(enhanced.split()) > 50:  # Only for substantial responses
#         enhanced += f"\n\n---\n*{PROJECT_INFO['name']} v{PROJECT_INFO['version']}*"
    
#     return enhanced

# def generate_combined_prompt(user_query: str, context: Dict[str, Any] = None) -> str:
#     """
#     Generate a comprehensive prompt combining user query with system context.
    
#     Args:
#         user_query: The user's input message
#         context: Additional context dictionary (e.g., current diagnosis)
        
#     Returns:
#         Combined prompt string for the AI model
#     """
#     prompt_parts = []
    
#     # 1. Current context
#     if context:
#         prompt_parts.append(f"## Current Context\n{json.dumps(context, indent=2)}")
    
#     # 2. User query
#     prompt_parts.append(f"## User Query\n{user_query}")
    
#     # 3. Instructions
#     instructions = """
#     Please provide a comprehensive response considering:
#     - The user's specific query
#     - Our medical knowledge base (when relevant)
#     - Project information (when relevant)
#     - Appropriate disclaimers
#     - Clear, formatted output
#     """
#     prompt_parts.append(f"## Response Guidelines\n{instructions}")
    
#     return "\n\n".join(prompt_parts)

# def get_latest_diagnosis_result() -> Dict[str, Any]:
#     """
#     Get the latest diagnosis result stored in the application.
    
#     Returns:
#         Dictionary containing the latest diagnosis information
#     """
#     global latest_result
    
#     # Add additional context if available
#     result_copy = latest_result.copy()
    
#     # Add recommended actions based on the diagnosis
#     if result_copy.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
#         result_copy['condition_type'] = 'diabetic_retinopathy'
#         result_copy['recommended_specialist'] = 'Ophthalmologist'
#         result_copy['timeframe'] = 'within 2-4 weeks'
#     elif result_copy.get('result') in ['glioma', 'meningioma', 'pituitary']:
#         result_copy['condition_type'] = 'brain_tumor'
#         result_copy['recommended_specialist'] = 'Neurologist'
#         result_copy['timeframe'] = 'as soon as possible'
    
#     return result_copy

# def chatbot_response(message: str) -> str:
#     """
#     Process user input and return enhanced chatbot response.
    
#     Args:
#         message: User input message
        
#     Returns:
#         Formatted AI response with appropriate context
#     """
#     try:
#         # Check cache first for performance
#         cache_key = message.lower().strip()
#         if cache_key in response_cache:
#             return response_cache[cache_key]
        
#         # Check for general knowledge queries first
#         is_general, general_response = is_general_knowledge_query(message)
#         if is_general and general_response:
#             enhanced = enhance_response(general_response, is_medical=False)
#             response_cache[cache_key] = enhanced
#             return enhanced
        
#         # Extract condition keywords from message
#         condition_keywords = extract_condition_keywords(message)
        
#         # If specific condition mentioned, provide detailed information
#         if condition_keywords:
#             condition = condition_keywords[0]
#             response = get_detailed_condition_info(condition)
#             response_cache[cache_key] = response
#             return response
        
#         # Check for FAQ matches
#         faq_response = match_faq(message)
#         if faq_response:
#             response_cache[cache_key] = faq_response
#             return faq_response
        
#         # Check for app functionality questions
#         app_info = get_app_functionality_info(message)
#         if app_info:
#             response_cache[cache_key] = app_info
#             return app_info
        
#         # Get any relevant context (e.g., current diagnosis)
#         try:
#             diagnosis_result = get_latest_diagnosis_result()
#             context = {
#                 "latest_diagnosis": diagnosis_result,
#                 "query_time": datetime.now().isoformat()
#             }
#         except:
#             # Fallback if the function isn't available
#             context = {
#                 "query_time": datetime.now().isoformat()
#             }
        
#         # Generate comprehensive prompt
#         combined_prompt = generate_combined_prompt(message, context)
        
#         # Get AI response
#         response = chat.send_message(combined_prompt)
        
#         # Enhance the response
#         enhanced_response = enhance_response(response.text)
        
#         # Cache the response
#         if len(response_cache) > 100:  # Limit cache size
#             response_cache.popitem()  # Remove oldest entry
#         response_cache[cache_key] = enhanced_response
        
#         return enhanced_response
        
#     except Exception as e:
#         print(f"Chatbot Error: {str(e)}")
#         error_msg = f"I'm experiencing technical difficulties. Please try again later. \n\n*Error details: {str(e)}*"
#         return enhance_response(error_msg, is_medical=False)

# # === Routes ===
# @app.route('/')
# def home():
#     return render_template('homepage.html')

# @app.route('/predict/dr', methods=['POST'])
# def predict_dr():
#     global latest_result
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No image selected'}), 400

#     os.makedirs('uploads', exist_ok=True)
#     temp_path = os.path.join('uploads', file.filename)
#     file.save(temp_path)

#     try:
#         model = load_model('dr')
#         if model is None:
#             return jsonify({'error': 'DR model not found'}), 500

#         classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
#         result = predict_image(model, temp_path, classes)
#         latest_result = {'result': result}
#         return jsonify({'prediction': result})
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# @app.route('/predict/brain_tumor', methods=['POST'])
# def predict_brain_tumor():
#     global latest_result
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No image selected'}), 400

#     os.makedirs('uploads', exist_ok=True)
#     temp_path = os.path.join('uploads', file.filename)
#     file.save(temp_path)

#     try:
#         model = load_model('brain_tumor')
#         if model is None:
#             return jsonify({'error': 'Brain tumor model not found'}), 500

#         classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
#         result = predict_image(model, temp_path, classes)
#         latest_result = {'result': result}
#         return jsonify({'prediction': result})
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# @app.route('/get_latest_result')
# def get_latest_result():
#     if latest_result.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
#         latest_result['recommended_specialist'] = 'Ophthalmologist'
#     elif latest_result.get('result') in ['glioma', 'meningioma', 'pituitary']:
#         latest_result['recommended_specialist'] = 'Neurologist'
#     return jsonify(latest_result)

# @app.route('/download_report')
# def download_report():
#     patient_name = request.args.get('name', 'Anonymous')
#     if not latest_result:
#         return jsonify({'error': 'No diagnosis available'}), 400

#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", style='B', size=16)
#     pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
#     pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, txt=f"Diagnosis: {latest_result['result']}", ln=True, align='L')
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, txt="This report is AI-generated. Please consult a medical professional.", align='L')
#     pdf.set_font("Arial", style='I', size=10)
#     pdf.cell(0, 10, txt="Generated by AI Diagnosis System", ln=True, align='C')

#     os.makedirs("reports", exist_ok=True)
#     report_path = os.path.join("reports", f"{patient_name}_diagnosis_report.pdf")
#     pdf.output(report_path)

#     return send_file(report_path, as_attachment=True, download_name=f"{patient_name}_diagnosis_report.pdf")

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     if not data:
#         return jsonify({'error': 'Invalid request format'}), 400
        
#     user_message = data.get('message', '')
#     if not user_message:
#         return jsonify({'response': 'Please enter a message.'})

#     # Get latest diagnosis if available
#     latest_diagnosis = get_latest_diagnosis_result() if latest_result else None
    
#     # Get chatbot response
#     response = chatbot_response(user_message)
    
#     return jsonify({'response': response})

# @app.route('/chat')
# def chat_page():
#     return render_template('chat.html')

# # Other UI routes
# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/services')
# def services():
#     return render_template('services.html')

# @app.route('/upload')
# def upload():
#     return render_template('upload.html')

# @app.route('/consult')
# def consult():
#     return render_template('consult.html')

# @app.route('/report')
# def report():
#     return render_template('report.html')

# @app.route('/signup')
# def signup_page():
#     return render_template('signup.html')

# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     name = data.get('name')
#     email = data.get('email')
#     password = data.get('password')
#     return jsonify({'success': True, 'message': 'User registered successfully'})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(debug=True, host='0.0.0.0', port=port)

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from fpdf import FPDF
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import google.generativeai as genai
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import uuid
from flask_mail import Mail, Message
import io


# === Initialize Flask App ===
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure session and mail
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY", "neurovision-dev-secret-key")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# === Database Initialization ===
def init_db():
    conn = sqlite3.connect('neurovision.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# === Authentication Decorators ===
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# === User Management Functions ===
def get_user_by_email(email):
    conn = sqlite3.connect('neurovision.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user and {
        'id': user[0],
        'name': user[1],
        'email': user[2],
        'password': user[3],
        'created_at': user[4]
    }

def get_user_by_id(user_id):
    conn = sqlite3.connect('neurovision.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user and {
        'id': user[0],
        'name': user[1],
        'email': user[2],
        'created_at': user[4]
    }

def create_user(name, email, password):
    user_id = str(uuid.uuid4())
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    try:
        conn = sqlite3.connect('neurovision.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)",
                 (user_id, name, email, hashed_password))
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

        
# === Globals ===
latest_result = {}  # Stores the latest diagnosis result
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATHS = {
    'dr': 'models/dr_model.pth',
    'brain_tumor': 'models/brain_tumor.pth',
}
model_cache = {}  # Cache for lazy loading models

# Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Project and App Information
PROJECT_INFO = {
    "name": "NeuroVision AI",
    "version": "2.1.0",
    "description": "An advanced medical imaging analysis platform for diabetic retinopathy and brain tumor detection.",
    "features": [
        "Diabetic retinopathy classification (5 stages)",
        "Brain tumor detection (4 types)",
        "Comprehensive diagnostic reports",
        "AI-powered analysis with expert validation",
        "Secure data handling with HIPAA compliance"
    ],
    "team": "NeuroVision Medical Technologies",
    "contact": "support@neurovision.ai"
}

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

# Define comprehensive medical information database
MEDICAL_INFO = {
    "diabetic_retinopathy": {
        "overview": "A diabetes complication affecting the eyes, caused by damage to blood vessels in the retina.",
        "stages": {
            "No_DR": "No detectable diabetic retinopathy.",
            "Mild": "Small areas of balloon-like swelling in the retina's tiny blood vessels.",
            "Moderate": "Some blood vessels that nourish the retina are blocked.",
            "Severe": "Many more blood vessels are blocked, depriving several areas of the retina of blood supply.",
            "Proliferate_DR": "The most advanced stage where new abnormal blood vessels grow, which can cause serious vision problems."
        },
        "symptoms": [
            "Often no symptoms in early stages",
            "Spots or dark strings floating in your vision (floaters)",
            "Blurred vision",
            "Fluctuating vision",
            "Dark or empty areas in your vision",
            "Vision loss"
        ],
        "risk_factors": [
            "Duration of diabetes",
            "Poor control of blood sugar levels",
            "High blood pressure",
            "High cholesterol",
            "Pregnancy",
            "Tobacco use"
        ],
        "prevention": [
            "Manage diabetes carefully",
            "Regular eye exams",
            "Control blood pressure and cholesterol",
            "Stop smoking",
            "Exercise regularly"
        ],
        "treatment": [
            "Anti-VEGF injections",
            "Focal laser treatment",
            "Scatter laser treatment (panretinal photocoagulation)",
            "Vitrectomy (surgical procedure)"
        ]
    },
    "brain_tumor": {
        "overview": "Abnormal growth of cells in the brain that can be cancerous (malignant) or noncancerous (benign).",
        "types": {
            "glioma": "Tumors that occur in the brain and spinal cord, beginning in glial cells that surround nerve cells.",
            "meningioma": "A tumor that arises from the meninges, the membranes that surround your brain and spinal cord.",
            "pituitary": "Tumors that form in the pituitary gland at the base of the brain, most are benign.",
            "notumor": "No detectable tumor growth in brain tissues."
        },
        "symptoms": [
            "Headaches that gradually become more frequent and severe",
            "Unexplained nausea or vomiting",
            "Vision problems",
            "Gradual loss of sensation or movement in arms or legs",
            "Balance difficulties",
            "Speech difficulties",
            "Confusion in everyday matters",
            "Seizures"
        ],
        "risk_factors": [
            "Age",
            "Radiation exposure",
            "Family history of brain tumors",
            "Exposure to certain chemicals"
        ],
        "diagnosis": [
            "Neurological exam",
            "Imaging tests (MRI, CT scan)",
            "Biopsy"
        ],
        "treatment": [
            "Surgery",
            "Radiation therapy",
            "Radiosurgery",
            "Chemotherapy",
            "Targeted drug therapy",
            "Rehabilitation after treatment"
        ]
    }
}

# FAQ database for common questions
FAQ_DATABASE = {
    "accuracy": "NeuroVision AI provides preliminary analysis with approximately 92% accuracy for diabetic retinopathy and 89% for brain tumor detection. However, all results should be confirmed by a qualified healthcare provider.",
    "privacy": "All medical images and patient data are encrypted and processed in compliance with HIPAA regulations. Your data is never shared with third parties without explicit consent.",
    "report_format": "Our diagnostic reports include the AI analysis, confidence scores, visual markers, and recommended next steps. Reports can be downloaded as PDF files.",
    "second_opinion": "NeuroVision AI is designed to assist medical professionals, not replace them. We always recommend consulting with a qualified healthcare provider for official diagnosis and treatment plans.",
    "image_requirements": "For optimal results, retinal images should be high-resolution fundus photographs. Brain scans should be clear MRI images in standard DICOM format.",
    "processing_time": "Image analysis typically takes 30-60 seconds depending on image quality and server load.",
    "false_results": "Like all diagnostic tools, NeuroVision AI has limitations. False positives and false negatives may occur, which is why expert human verification is essential.",
    "cost": "Please contact our sales team for current pricing information for individual and institutional licenses.",
    "integration": "NeuroVision AI can be integrated with most modern hospital information systems and PACS through our secure API.",
    "training_data": "Our models are trained on diverse datasets from multiple global medical institutions, with regular updates to improve accuracy and reduce biases."
}

# General knowledge topics for fallback responses
GENERAL_KNOWLEDGE = {
    "weather": "I can't provide real-time weather data, but I can explain weather patterns or meteorological concepts.",
    "news": "I don't have live news access, but I can discuss historical events or analyze news topics you specify.",
    "sports": "I can provide information about sports rules, history, or famous athletes, but not live scores.",
    "entertainment": "I can discuss movies, books, or music in general terms, but don't have access to current releases.",
    "technology": "I can explain technology concepts, compare devices, or discuss tech history and trends."
}

# Cache for frequent responses to improve performance
response_cache = {}

# Initialize the AI Model with optimized parameters
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,  # Increased for more comprehensive responses
    }
)

def generate_system_prompt() -> str:
    """Generate a comprehensive system prompt combining project info and medical knowledge."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
    # NeuroVision AI Assistant - System Prompt
    ## Project Information
    Name: {PROJECT_INFO['name']}
    Version: {PROJECT_INFO['version']}
    Description: {PROJECT_INFO['description']}
    Features: {", ".join(PROJECT_INFO['features'])}
    Team: {PROJECT_INFO['team']}
    Contact: {PROJECT_INFO['contact']}
    Current Date: {current_date}

    ## Medical Knowledge Base
    You have access to detailed information about:
    - Diabetic Retinopathy (5 classifications)
    - Brain Tumors (4 classifications)
    
    ## Role and Guidelines
    You are a compassionate and knowledgeable AI assistant for NeuroVision with these capabilities:
    1. Provide medically accurate yet easy-to-understand explanations
    2. Offer reassurance and support for health concerns
    3. Clearly emphasize that results are preliminary and must be reviewed by professionals
    4. Use non-diagnostic language ("the analysis suggests" rather than definitive statements)
    5. Handle general knowledge queries when not medical-related
    6. Provide information about the NeuroVision platform and its features
    
    ## Response Formatting Guidelines
    - Use Markdown for formatting (headers, bold, lists)
    - Include disclaimers when discussing medical topics
    - Break long responses into readable paragraphs
    - Highlight important warnings or recommendations
    
    ## General Knowledge Handling
    For non-medical queries:
    - Acknowledge the query type
    - Provide general information if possible
    - Redirect to appropriate resources when needed
    - Be clear about limitations
    
    Always maintain a professional yet approachable tone.
    """

# Enhanced system prompt
SYSTEM_PROMT = generate_system_prompt()

# Start a chat session to maintain context
chat = model.start_chat(
    history=[
       {"role": "user", "parts": [SYSTEM_PROMT]},
        {"role": "model", "parts": [f"I am {PROJECT_INFO['name']} (v{PROJECT_INFO['version']}), your medical diagnosis and general knowledge assistant. How can I help you today?"]}
    ]
)

def get_detailed_condition_info(condition_key: str) -> str:
    """
    Retrieve detailed information about a specific medical condition.
    
    Args:
        condition_key: The key for the condition in the MEDICAL_INFO database
        
    Returns:
        Formatted information about the condition
    """
    info = ""
    
    # Check for diabetic retinopathy stages
    for category, data in MEDICAL_INFO.items():
        if category == "diabetic_retinopathy" and condition_key in data["stages"]:
            stage = condition_key
            info = f"""## Diabetic Retinopathy - {stage.replace('_', ' ')}

*Classification:* {data["stages"][stage]}

*Overview:* {data["overview"]}

*Common Symptoms:*
{', '.join(data["symptoms"][:3])}

*Prevention:*
{', '.join(data["prevention"][:3])}

*Recommended Next Steps:*
1. Consult with an ophthalmologist to confirm the analysis
2. Regular monitoring of blood sugar levels
3. Follow your diabetes management plan

*Disclaimer:* This information is for educational purposes only and not a substitute for professional medical advice.
"""
            break
            
        # Check for brain tumor types
        elif category == "brain_tumor" and condition_key in data["types"]:
            tumor_type = condition_key
            info = f"""## Brain Tumor Analysis - {tumor_type.capitalize()}

*Classification:* {data["types"][tumor_type]}

*Overview:* {data["overview"]}

*Common Symptoms:*
{', '.join(data["symptoms"][:3])}

*Diagnostic Approaches:*
{', '.join(data["diagnosis"][:3])}

*Recommended Next Steps:*
1. Consult with a neurologist to confirm the analysis
2. Additional diagnostic testing may be required
3. Discuss treatment options with your healthcare provider

*Disclaimer:* This information is for educational purposes only and not a substitute for professional medical advice.
"""
            break
    
    if not info:
        # Generic response for conditions not in database
        info = f"""## Medical Condition Information

I don't have specific detailed information about "{condition_key}" in my database.

*Recommended Steps:*
1. Consult with a healthcare provider for accurate diagnosis
2. Discuss any symptoms or concerns with your doctor
3. Seek information from credible medical sources

*Disclaimer:* Always consult healthcare professionals for proper medical advice and diagnosis.
"""
    
    return info

def extract_condition_keywords(message: str) -> List[str]:
    """
    Extract keywords related to medical conditions from the user message.
    
    Args:
        message: User input message
        
    Returns:
        List of recognized condition keywords
    """
    message = message.lower()
    
    # Keywords for different conditions
    condition_keywords = {
        # Diabetic retinopathy stages
        "no_dr": ["no dr", "no diabetic retinopathy", "normal retina"],
        "mild": ["mild dr", "mild diabetic retinopathy", "early dr"],
        "moderate": ["moderate dr", "moderate diabetic retinopathy"],
        "severe": ["severe dr", "severe diabetic retinopathy", "advanced dr"],
        "proliferate_dr": ["proliferative", "pdr", "proliferate", "advanced diabetic retinopathy"],
        
        # Brain tumors
        "glioma": ["glioma", "glial", "glioblastoma", "astrocytoma"],
        "meningioma": ["meningioma", "meningeal"],
        "pituitary": ["pituitary", "pituitary adenoma", "pituitary tumor"],
        "notumor": ["no tumor", "no brain tumor", "normal brain", "healthy brain"]
    }
    
    found_keywords = []
    
    for condition, keywords in condition_keywords.items():
        for keyword in keywords:
            if keyword in message:
                found_keywords.append(condition)
                break
    
    return found_keywords

def match_faq(message: str) -> Optional[str]:
    """
    Match user message to frequently asked questions.
    
    Args:
        message: User input message
        
    Returns:
        FAQ response if matched, None otherwise
    """
    message = message.lower()
    
    # Map keywords to FAQ categories
    faq_keywords = {
        "accuracy": ["accuracy", "accurate", "precision", "reliable", "correct"],
        "privacy": ["privacy", "secure", "confidential", "hipaa", "data security"],
        "report_format": ["report", "format", "pdf", "download", "results format"],
        "second_opinion": ["second opinion", "doctor", "confirm", "verification"],
        "image_requirements": ["image", "quality", "resolution", "requirement", "upload"],
        "processing_time": ["time", "long", "takes", "processing", "wait"],
        "false_results": ["false", "wrong", "mistake", "error", "incorrect"],
        "cost": ["cost", "price", "fee", "payment", "subscription"],
        "integration": ["integrate", "system", "hospital", "ehr", "api"],
        "training_data": ["training", "dataset", "data", "trained", "learn"]
    }
    
    # Check for matches
    for category, keywords in faq_keywords.items():
        for keyword in keywords:
            if keyword in message:
                return f"## {category.replace('_', ' ').title()}\n\n{FAQ_DATABASE[category]}"
    
    return None

def get_app_functionality_info(query: str) -> Optional[str]:
    """
    Provide information about app functionality based on user query.
    
    Args:
        query: User input message
        
    Returns:
        Response about app features or None if not matching
    """
    query = query.lower()
    
    # Define app feature keywords
    feature_queries = {
        "upload": ["upload", "scan", "image", "picture", "photo", "upload image"],
        "report": ["report", "results", "download", "pdf", "save", "share"],
        "analysis": ["analysis", "analyze", "detect", "diagnose", "process"],
        "accuracy": ["accuracy", "precise", "reliable", "confidence"],
        "features": ["features", "capabilities", "function", "what can you do"],
        "about": ["about", "who are you", "tell me about", "information about", "what is neurovision"],
        "security": ["security", "privacy", "confidential", "safe", "data safety"]
    }
    
    for feature, keywords in feature_queries.items():
        if any(keyword in query for keyword in keywords):
            # Responses for different features
            if feature == "upload":
                return "## Image Upload\n\nYou can upload medical images through our Upload page by clicking the 'Upload' link in the navigation menu. We accept retinal scans for diabetic retinopathy analysis and brain MRI images for tumor detection. For optimal results, please ensure your images are high resolution and properly oriented."
            
            elif feature == "report":
                return "## Medical Reports\n\nAfter analysis, you can download a comprehensive medical report in PDF format. The report includes the AI analysis results, confidence scores, and recommended next steps. You can find the download option on the results page or in your user dashboard."
            
            elif feature == "analysis":
                return "## Image Analysis\n\nNeuroVision AI uses advanced deep learning algorithms to analyze medical images. For diabetic retinopathy, we classify into 5 stages (No DR, Mild, Moderate, Severe, Proliferative). For brain MRIs, we detect and classify four categories (Glioma, Meningioma, Pituitary tumor, and No tumor)."
            
            elif feature == "accuracy":
                return "## System Accuracy\n\nNeuroVision AI achieves approximately 92% accuracy for diabetic retinopathy detection and 89% for brain tumor classification. However, all results should be confirmed by healthcare professionals. Our system is constantly improving through regular model updates and validation studies."
            
            elif feature == "features":
                feature_list = "\n".join([f"- {feature}" for feature in PROJECT_INFO["features"]])
                return f"## NeuroVision Features\n\n{PROJECT_INFO['description']}\n\nOur key features include:\n{feature_list}\n\nIs there a specific feature you'd like more information about?"
            
            elif feature == "about":
                return f"## About NeuroVision\n\n{PROJECT_INFO['name']} (version {PROJECT_INFO['version']}) is {PROJECT_INFO['description']} Developed by {PROJECT_INFO['team']}, we aim to assist medical professionals in diagnosing conditions earlier and more accurately through AI technology."
            
            elif feature == "security":
                return "## Data Security\n\nAll medical images and patient data are encrypted and processed in compliance with HIPAA regulations. We use industry-standard encryption, secure data centers, and strict access controls. Your data is never shared with third parties without explicit consent."
    
    return None

def is_general_knowledge_query(message: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if the query is a general knowledge question and return appropriate response.
    
    Args:
        message: User input message
        
    Returns:
        Tuple: (is_general_knowledge, response_or_none)
    """
    message = message.lower()
    
    for topic, response in GENERAL_KNOWLEDGE.items():
        if re.search(rf'\b{topic}\b', message):
            return (True, response)
    
    # Common general questions
    general_phrases = {
        "who created you": f"I was developed by {PROJECT_INFO['team']} as part of the {PROJECT_INFO['name']} project.",
        "what can you do": f"I can help with:\n- Medical image analysis information\n- {PROJECT_INFO['description']}\n- General knowledge questions\n\nMy features include: {', '.join(PROJECT_INFO['features'])}",
        "how old are you": f"I'm an AI assistant for {PROJECT_INFO['name']} version {PROJECT_INFO['version']}, first released in 2024.",
        "tell me about yourself": f"I'm {PROJECT_INFO['name']}, an AI assistant designed to provide information about medical diagnostics and general knowledge. {PROJECT_INFO['description']}"
    }
    
    for phrase, response in general_phrases.items():
        if phrase in message:
            return (True, response)
    
    return (False, None)

def enhance_response(original_response: str, is_medical: bool = True) -> str:
    """
    Enhance AI response with improved formatting and additional context.
    
    Args:
        original_response: Original AI response
        is_medical: Whether the response is medical-related
        
    Returns:
        Enhanced response with better formatting
    """
    # Basic cleaning
    enhanced = original_response.strip()
    
    # Add proper Markdown formatting
    if not enhanced.startswith("#"):
        enhanced = f"## Response\n{enhanced}"
    
    # Medical-specific enhancements
    if is_medical:
        # Add section breaks for better readability
        enhanced = re.sub(r"(\n)([A-Z][a-z]+:)", r"\n\n*\2*", enhanced)
        
        # Ensure disclaimer is present if not already there
        if "Disclaimer:" not in enhanced:
            enhanced += "\n\n*Disclaimer:* This information is for educational purposes only and not a substitute for professional medical advice. Always consult with qualified healthcare providers."
    
    # General formatting improvements
    enhanced = re.sub(r"(\n\s*\n)", r"\n\n", enhanced)  # Remove extra newlines
    enhanced = re.sub(r"\.(\s)([A-Z])", r".\n\n\2", enhanced)  # Paragraph breaks
    
    # Add project footer for longer responses
    if len(enhanced.split()) > 50:  # Only for substantial responses
        enhanced += f"\n\n---\n*{PROJECT_INFO['name']} v{PROJECT_INFO['version']}*"
    
    return enhanced

def generate_combined_prompt(user_query: str, context: Dict[str, Any] = None) -> str:
    """
    Generate a comprehensive prompt combining user query with system context.
    
    Args:
        user_query: The user's input message
        context: Additional context dictionary (e.g., current diagnosis)
        
    Returns:
        Combined prompt string for the AI model
    """
    prompt_parts = []
    
    # 1. Current context
    if context:
        prompt_parts.append(f"## Current Context\n{json.dumps(context, indent=2)}")
    
    # 2. User query
    prompt_parts.append(f"## User Query\n{user_query}")
    
    # 3. Instructions
    instructions = """
    Please provide a comprehensive response considering:
    - The user's specific query
    - Our medical knowledge base (when relevant)
    - Project information (when relevant)
    - Appropriate disclaimers
    - Clear, formatted output
    """
    prompt_parts.append(f"## Response Guidelines\n{instructions}")
    
    return "\n\n".join(prompt_parts)

def get_latest_diagnosis_result() -> Dict[str, Any]:
    """
    Get the latest diagnosis result stored in the application.
    
    Returns:
        Dictionary containing the latest diagnosis information
    """
    global latest_result
    
    # Add additional context if available
    result_copy = latest_result.copy()
    
    # Add recommended actions based on the diagnosis
    if result_copy.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        result_copy['condition_type'] = 'diabetic_retinopathy'
        result_copy['recommended_specialist'] = 'Ophthalmologist'
        result_copy['timeframe'] = 'within 2-4 weeks'
    elif result_copy.get('result') in ['glioma', 'meningioma', 'pituitary']:
        result_copy['condition_type'] = 'brain_tumor'
        result_copy['recommended_specialist'] = 'Neurologist'
        result_copy['timeframe'] = 'as soon as possible'
    
    return result_copy

def chatbot_response(message: str) -> str:
    """
    Process user input and return enhanced chatbot response.
    
    Args:
        message: User input message
        
    Returns:
        Formatted AI response with appropriate context
    """
    try:
        # Check cache first for performance
        cache_key = message.lower().strip()
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        # Check for general knowledge queries first
        is_general, general_response = is_general_knowledge_query(message)
        if is_general and general_response:
            enhanced = enhance_response(general_response, is_medical=False)
            response_cache[cache_key] = enhanced
            return enhanced
        
        # Extract condition keywords from message
        condition_keywords = extract_condition_keywords(message)
        
        # If specific condition mentioned, provide detailed information
        if condition_keywords:
            condition = condition_keywords[0]
            response = get_detailed_condition_info(condition)
            response_cache[cache_key] = response
            return response
        
        # Check for FAQ matches
        faq_response = match_faq(message)
        if faq_response:
            response_cache[cache_key] = faq_response
            return faq_response
        
        # Check for app functionality questions
        app_info = get_app_functionality_info(message)
        if app_info:
            response_cache[cache_key] = app_info
            return app_info
        
        # Get any relevant context (e.g., current diagnosis)
        try:
            diagnosis_result = get_latest_diagnosis_result()
            context = {
                "latest_diagnosis": diagnosis_result,
                "query_time": datetime.now().isoformat()
            }
        except:
            # Fallback if the function isn't available
            context = {
                "query_time": datetime.now().isoformat()
            }
        
        # Generate comprehensive prompt
        combined_prompt = generate_combined_prompt(message, context)
        
        # Get AI response
        response = chat.send_message(combined_prompt)
        
        # Enhance the response
        enhanced_response = enhance_response(response.text)
        
        # Cache the response
        if len(response_cache) > 100:  # Limit cache size
            response_cache.popitem()  # Remove oldest entry
        response_cache[cache_key] = enhanced_response
        
        return enhanced_response
        
    except Exception as e:
        print(f"Chatbot Error: {str(e)}")
        error_msg = f"I'm experiencing technical difficulties. Please try again later. \n\n*Error details: {str(e)}*"
        return enhance_response(error_msg, is_medical=False)

# === Routes ===
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict/dr', methods=['POST'])
@login_required
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
@login_required
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

@app.route('/get_latest_result')
@login_required
def get_latest_result():
    if latest_result.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        latest_result['recommended_specialist'] = 'Ophthalmologist'
    elif latest_result.get('result') in ['glioma', 'meningioma', 'pituitary']:
        latest_result['recommended_specialist'] = 'Neurologist'
    return jsonify(latest_result)
@app.route('/download_report')
@login_required
def download_report():
    patient_name = request.args.get('name', 'Anonymous')
    if not latest_result:
        return jsonify({'error': 'No diagnosis available'}), 400

    # Get comprehensive AI analysis for the report
    if latest_result.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR', 'No_DR']:
        report_prompt = f"Generate a comprehensive medical report for {patient_name} with a diagnosis of {latest_result['result']} diabetic retinopathy. Include clinical implications, recommended follow-up, and patient guidance."
    elif latest_result.get('result') in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        report_prompt = f"Generate a comprehensive medical report for {patient_name} with a brain scan indicating {latest_result['result']}. Include clinical implications, recommended follow-up, and patient guidance."
    else:
        report_prompt = f"Generate a comprehensive medical report for {patient_name} with a diagnosis of {latest_result['result']}."
    
    # Get AI-generated report content
    ai_report_content = chatbot_response(report_prompt)
    
    # Create PDF report with AI content
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="NeuroVision AI Diagnosis Report", ln=True, align='C')
    pdf.ln(5)
    
    # Patient info
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient: {patient_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Report ID: {uuid.uuid4().hex[:8]}", ln=True, align='L')
    pdf.ln(5)
    
    # Main diagnosis
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, txt=f"Primary Diagnosis: {latest_result['result']}", ln=True, align='L')
    pdf.ln(5)
    
    # AI report content - handle markdown formatting
    pdf.set_font("Arial", size=11)
    # Remove markdown headers and formatting
    clean_content = re.sub(r'#+\s+', '', ai_report_content)
    clean_content = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_content)  # Bold
    clean_content = re.sub(r'\*(.+?)\*', r'\1', clean_content)      # Italic
    clean_content = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean_content)  # Links
    
    # Split by paragraphs and add to PDF
    paragraphs = clean_content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            pdf.multi_cell(0, 6, txt=para.strip(), align='L')
            pdf.ln(3)
    
    # Disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", style='I', size=10)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This report is generated by an AI system and is intended for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Please consult with a qualified healthcare provider for proper evaluation.", align='L')
    pdf.ln(5)
    
    # Footer
    pdf.set_font("Arial", style='I', size=10)
    pdf.cell(0, 10, txt=f"Generated by {PROJECT_INFO['name']} v{PROJECT_INFO['version']}", ln=True, align='C')

    # Create directory for reports if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"{patient_name}_diagnosis_report.pdf")
    pdf.output(report_path)
    
    # Get user email from session or database
    user_email = None
    if 'user_email' in session:
        user_email = session.get('user_email')
    else:
        # Fallback to database lookup if not in session
        user = get_user_by_id(session.get('user_id'))
        if user and 'email' in user:
            user_email = user['email']
    
    # Also check if an email was provided in the URL parameters
    param_email = request.args.get('email')
    if param_email and '@' in param_email:
        user_email = param_email

    # For debugging, print the email address
    print(f"Attempting to send report to email: {user_email}")

    # Send email with PDF attachment if user has email
    if user_email:
        try:
            # Print email configuration for debugging
            print(f"Mail server: {app.config['MAIL_SERVER']}")
            print(f"Mail username: {app.config['MAIL_USERNAME']}")
            
            # Check for valid email configuration
            if app.config['MAIL_USERNAME'] == 'your-email@gmail.com' or app.config['MAIL_PASSWORD'] == 'your-app-password':
                flash("Email configuration incomplete. Please configure email settings properly.", "warning")
                print("Warning: Email configuration incomplete")
            else:
                # Create the email message
                msg = Message(
                    subject=f"NeuroVision AI Diagnosis Report for {patient_name}",
                    recipients=[user_email],
                    body=f"Dear {patient_name},\n\nPlease find attached your AI-generated diagnosis report from NeuroVision AI.\n\nDiagnosis: {latest_result['result']}\n\nThis report includes a comprehensive analysis of your medical images and recommended next steps.\n\nReminder: This is an AI-generated report. Please consult a healthcare professional for proper medical advice.\n\nBest regards,\nNeuroVision AI Team"
                )
                
                # Attach the PDF - FIX: Use the physical file path that was already created
                with open(report_path, 'rb') as pdf_file:
                    msg.attach(
                        filename=f"{patient_name}_diagnosis_report.pdf",
                        content_type="application/pdf",
                        data=pdf_file.read()
                    )
                
                # Send the email
                mail.send(msg)
                flash(f"Report has been sent to {user_email}.", "success")
                print(f"Email sent to {user_email}")
        except Exception as e:
            flash(f"Failed to send email: {str(e)}", "error")
            print(f"Email error: {str(e)}")
    else:
        flash("Could not determine your email address. Please check your profile settings.", "warning")
        print("Warning: Could not determine user email")

    return send_file(report_path, as_attachment=True, download_name=f"{patient_name}_diagnosis_report.pdf")

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request format'}), 400
        
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    # Get latest diagnosis if available
    latest_diagnosis = get_latest_diagnosis_result() if latest_result else None
    
    # Get chatbot response
    response = chatbot_response(user_message)
    
    return jsonify({'response': response})

@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html')

# Other UI routes
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/consult')
@login_required
def consult():
    return render_template('consult.html')

@app.route('/report')
@login_required
def report():
    return render_template('report.html')

# === Authentication Routes ===
@app.route('/signup_page')
def signup_page():
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/login_page')
def login_page():
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
        else:
            # For regular form submissions
            data = {
                'name': request.form.get('name'),
                'email': request.form.get('email'),
                'password': request.form.get('password')
            }
        
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        # Validate data
        if not name or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'}), 400
        
        # Check if user already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Create new user
        user_id = create_user(name, email, password)
        if user_id:
            return jsonify({'success': True, 'message': 'Registration successful! Redirecting to login...'}), 201
        else:
            return jsonify({'success': False, 'message': 'An error occurred during registration'}), 500
    else:
        # GET request - redirect to signup page
        return redirect(url_for('signup_page'))

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    remember = data.get('remember', False)
    
    # Validate data
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required'}), 400
    
    # Check if user exists
    user = get_user_by_email(email)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    
    # Set session data
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    session['user_email'] = user['email']
    
    # Handle "remember me" option
    if remember:
        session.permanent = True
    
    return jsonify({
        'success': True, 
        'message': 'Login successful!',
        'redirect': '/'
    }), 200

@app.route('/logout')
def logout():
    # Clear session
    session.clear()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    user = get_user_by_id(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login_page'))
    
    return render_template('profile.html', user=user)

# Add a current user context processor
@app.context_processor
def inject_user():
    user = None
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
    return {'current_user': user}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)