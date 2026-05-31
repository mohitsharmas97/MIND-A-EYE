# MINDAEYE-Healthcare
MindAye helps in early detection of diabetic retinopathy and brain tumors using AI analysis of medical images.
It allows users to upload eye or brain scans and instantly receive a preliminary diagnosis.
The system offers specialist recommendations based on results to guide next medical steps.
A built-in chatbot answers health-related questions using real-time responses powered by the Gemini API.
Users can also save consultation records and download personalized diagnosis reports for further care.

![Screenshot 2025-04-06 010243](https://github.com/user-attachments/assets/bd42ad45-211f-48e4-97fc-d3a1436d0cba)
![Screenshot 2025-04-06 010905](https://github.com/user-attachments/assets/66b9ad0f-4b61-418d-a290-e8a05093b14d)
![Screenshot 2025-04-06 010932](https://github.com/user-attachments/assets/9d351879-93df-447f-8053-c19da731e3cf)
![Screenshot 2025-04-06 011018](https://github.com/user-attachments/assets/9101f815-8f8f-46bb-b68f-3a4e18847e10)
![Screenshot 2025-04-06 011104](https://github.com/user-attachments/assets/9bdbac5a-94ff-4b4b-ba74-47a81a3fa009)
![image](https://github.com/user-attachments/assets/eb66cbfb-10e9-41d2-a832-6d0c87ea6145)
![Screenshot 2025-04-06 011300](https://github.com/user-attachments/assets/d472c218-6c6f-4ebb-8696-20130a03d9e0)




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

![Screenshot 2025-04-06 014659](https://github.com/user-attachments/assets/66e26014-63f0-4e7e-a40e-c6d451d9b9e4)


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

Chatbot Integration:Pushkar

AI Model Development:Mohit(developed both models from scratch) 

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

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Client [Client Side]
        U[User Browser]
        UI[Tailwind UI]
        U <--> UI
    end

    subgraph Server [Flask Backend]
        API[Routing / Controllers]
        SEC[EXIF Stripper & Privacy]
        DB[(SQLAlchemy DB <br> PostgreSQL/SQLite)]
        PDF[PDF Engine - FPDF]
        CHAT[Chatbot Logic]
        MAIL[Email Service]
    end

    subgraph ML [AI Inference Engine]
        MEM[In-Memory BytesIO]
        DC[DICOM Parser]
        PT[PyTorch EfficientNet Models]
        XAI[Grad-CAM Generator]
    end

    subgraph External [External APIs]
        GEM[Google Gemini API]
    end

    %% Flow
    UI -- "Upload Scan (JPG/PNG/DICOM)" --> API
    API -- "Auth & Session" <--> DB
    API -- "Raw File" --> MEM
    MEM -- "Extract Pixels" --> DC
    DC -- "Strip EXIF" --> SEC
    SEC -- "Clean Tensor" --> PT
    PT -- "Prediction & Confidence Score" --> API
    PT -- "Feature Maps" --> XAI
    XAI -- "Base64 Heatmap" --> API
    
    API -- "JSON Results" --> UI
    
    UI -- "Chat Query" --> CHAT
    CHAT <--> GEM
    CHAT -- "Response" --> UI
    
    UI -- "Request Report" --> API
    API -- "Query Context" --> GEM
    GEM -- "Clinical Text" --> API
    API -- "Generate" --> PDF
    PDF -- "Email Report" --> MAIL
    MAIL --> U
