
from pathlib import Path
import io

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="Indian Currency Detection API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the denomination detection model
denomination_model = YOLO(str(MODEL_DIR / "best.pt"))

# Load the REAL / FAKE classification model
authenticity_model = resnet18(weights=None)
authenticity_model.fc = nn.Linear(authenticity_model.fc.in_features, 2)

authenticity_model.load_state_dict(
    torch.load(
        MODEL_DIR / "real_fake_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)
authenticity_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

@app.get("/")
def home():
    return {"message": "Indian Currency Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Detect currency denominations
    with torch.inference_mode():
        results = denomination_model.predict(
        image,
        conf=0.25,
        verbose=False,
        imgsz=320,
        device="cpu",
    )

    detections = []

    if not any(len(result.boxes) > 0 for result in results):
       return {
        "denomination_detections": [],
        "authenticity": "UNCERTAIN",
        "authenticity_confidence": 0.0,
        "warning": "No currency note was detected. Please upload a clear image of an Indian currency note."
    }

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "denomination": str(result.names[class_id]),
                "confidence": round(float(box.conf.item()) * 100, 2),
                "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })

    # Classify the image as REAL or FAKE
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(authenticity_model(tensor), dim=1)[0]

    classes = ["fake", "real"]
    predicted_id = int(torch.argmax(probabilities).item())

    return {
        "denomination_detections": detections,
        "authenticity": classes[predicted_id].upper(),
        "authenticity_confidence": round(
            float(probabilities[predicted_id]) * 100, 2
        ),
        "warning": "AI assessment only; this is not proof that a note is genuine or counterfeit."
    }
