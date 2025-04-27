# MINDAEYE-Healthcare
MindAye helps in early detection of diabetic retinopathy and brain tumors using AI analysis of medical images.
It allows users to upload eye or brain scans and instantly receive a preliminary diagnosis.
The system offers specialist recommendations based on results to guide next medical steps.
A built-in chatbot answers health-related questions using real-time responses powered by the Gemini API.
Users can also save consultation records and download personalized diagnosis reports for further care.

![Screenshot 2025-04-06 010243](https://github.com/user-attachments/assets/514b5b81-44c3-4064-9535-e46759f10b6d)
![Screenshot 2025-04-06 011300](https://github.com/user-attachments/assets/02a08920-74cb-4473-9df5-dff51c69c262)
![Screenshot 2025-04-06 010905](https://github.com/user-attachments/assets/7355d648-1cef-4391-896b-9d68977d771f)
![Screenshot 2025-04-06 011104](https://github.com/user-attachments/assets/3d71e36f-9d70-46d9-808a-3a817a800951)
![Screenshot 2025-04-06 011739](https://github.com/user-attachments/assets/5b49bd2b-c5bd-43fb-b360-9ff38208d273)
![Screenshot 2025-04-06 010932](https://github.com/user-attachments/assets/82cdbe87-ed5e-4f76-abdf-8422e53d855a)
![Screenshot 2025-04-06 011018](https://github.com/user-attachments/assets/2add493f-ab30-4c49-beb5-e2c0dc67eada)

# MINDAEYE Features
AI-powered diagnosis of diabetic retinopathy from retina scans.

AI-based detection of brain tumors from MRI images.

Instant specialist recommendation based on diagnosis results.

Gemini API-powered chatbot for health and project-related queries.

Chat interface for interactive medical assistance.

Generates downloadable PDF diagnosis reports.

Saves consultation records with patient and doctor info.

Simple signup and login system for user access.

Placeholder endpoint for future video consultation integration.

# Supported Disease Classifications

Brain Tumor Classification:
Users upload brain MRI images, which are analyzed by an AI model to detect and classify tumors.
The system identifies one of four classes: Glioma, Meningioma, Pituitary, or No Tumor.

Diabetic Retinopathy Classification:
Users submit retina fundus images for AI-based analysis of retinal damage.
The model classifies the image into one of five stages: No_DR, Mild, Moderate, Severe, or Proliferate_DR.

# Model Accuracies
1. Brain Tumor:98%
2. Blind Retnopathy:84%

Brain Tumor:

![Screenshot 2025-04-06 014659](https://github.com/user-attachments/assets/fd6b25d3-c5cd-488b-985a-b42c75b22bc4)

#  Technologies Used

Python – Core programming language

Flask – Web framework for building the application

PyTorch – Deep learning framework for model development and inference

Torchvision – Pretrained models and image transformation utilities

OpenCV & PIL – Image processing and manipulation

FPDF – PDF report generation

HTML/CSS/JavaScript – Frontend for user interface

Gemini API – Powering the AI chatbot for user queries

Google Colab / Jupyter Notebooks – Model training and experimentation

Git & GitHub – Version control and project hosting

# Team Contributions

Frontend Development:Mohit 

Backend Development:Mohit

Chatbot Integration:Mohit

AI Model Development:Pushkar(developed both models from scratch) 

# Future Enhancements

Add real-time video consultation using Twilio or Agora integration.

Implement secure user authentication and patient history tracking.

Expand disease classification to include skin cancer, pneumonia, etc.

Deploy the app on cloud platforms like AWS, Azure, or Heroku.

Enhance chatbot intelligence for more accurate and broader medical support.

Integrate electronic health records (EHR) for complete patient profiles.

Enable multilingual support for wider accessibility.

# Usage

Upload Medical Images – Upload retina fundus or brain MRI images through the web interface.

Get Instant Diagnosis – The AI model analyzes the image and provides a classification result.

Chat with AI Assistant – Ask medical-related queries through the integrated chatbot powered by Gemini API.

Download Report – Generate and download a PDF report of the diagnosis.

Record Consultations – Save consultation notes for future reference.

# Disclaimer

This platform is intended for research and educational purposes only. It does not provide a certified medical diagnosis. Always consult a licensed healthcare professional for medical concerns.



