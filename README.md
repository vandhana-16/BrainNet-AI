# BrainNet AI

A dual-model deep learning system for brain tumor MRI classification with explainability and trust scoring.

## Features
- Dual Model: EfficientNetB0 (86.4%) + ResNet50V2 (94.6%)
- GradCAM++: Visual explanation of model decisions
- SCC (Spatial Consensus Confidence): Novel trust metric using IoU of GradCAM++ heatmaps
- FastAPI Backend: REST API with JWT authentication
- SQLite Database: Persistent scan history per doctor
- PDF Reports: Downloadable clinical report per scan

## Project Structure
```
BrainNet_AI/
├── main.py
├── requirements.txt
├── app/
│   ├── database.py
│   ├── core/
│   │   ├── auth.py
│   │   ├── ml_engine.py
│   │   └── pdf_generator.py
│   └── routers/
│       ├── auth.py
│       └── scan.py
├── templates/
│   ├── index.html
│   └── dashboard.html
└── notebooks/
    └── README.md
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/register | Register a doctor |
| POST | /auth/login | Login, get JWT token |
| POST | /scan/predict | Upload MRI, get prediction + SCC |
| GET | /scan/history | Get all scans for logged-in doctor |
| GET | /scan/download/{id} | Download PDF report |
| GET | /health | Server health check |

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset
- Classes: Glioma, Meningioma, Pituitary, No Tumor
- Training: 5,600 images | Testing: 1,600 images

## Novel Contribution — SCC
SCC measures the Intersection over Union (IoU) of GradCAM++ heatmaps from two architecturally different models.
- SCC >= 0.35 → HIGH trust
- SCC 0.15–0.35 → MEDIUM trust
- SCC < 0.15 → LOW trust — refer to radiologist

## Results
- High SCC (>0.3) predictions: 92.7% accuracy
- Low SCC (<0.1) predictions: 77.2% accuracy
- 15.5 percentage point gap proves SCC outperforms softmax confidence

> For research use only. Not a clinical diagnostic tool.

## Live Demo
Deployed on Google Colab + ngrok.
Run the notebook to start the server.

## How to Run
1. Open Google Colab
2. Mount Google Drive
3. Run all cells in order
4. Open the ngrok URL printed at the end
