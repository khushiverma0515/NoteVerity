
from pathlib import Path
import io
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

IMAGE_SIZE = 320
YOLO_CONFIDENCE = 0.20
MAX_DETECTIONS = 5

app = FastAPI(title="Indian Currency Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading YOLO model...")
denomination_model = YOLO(str(MODEL_DIR / "best.pt"))

print("Loading authenticity model...")
authenticity_model = resnet18(weights=None)
authenticity_model.fc = nn.Linear(
    authenticity_model.fc.in_features, 2
)

authenticity_model.load_state_dict(
    torch.load(
        MODEL_DIR / "real_fake_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)

authenticity_model.eval()


# --------------------------------------------------
# IMAGE TRANSFORMATION
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# MODEL WARMUP
# --------------------------------------------------

@app.on_event("startup")
def warmup_models():
    print("Warming up YOLO...")

    dummy_image = Image.new("RGB", (320, 320))

    with torch.inference_mode():
        denomination_model.predict(
            dummy_image,
            conf=YOLO_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

    print("YOLO warmup complete.")
    print("NoteVerity API ready.")


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Indian Currency Detection API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# CURRENCY PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload an image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    # ----------------------------------------------
    # YOLO DENOMINATION DETECTION
    # ----------------------------------------------

    yolo_start = time.perf_counter()

    with torch.inference_mode():
        results = denomination_model.predict(
            image,
            conf=YOLO_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

    yolo_time = time.perf_counter() - yolo_start

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())

            detections.append({
                "denomination": str(result.names[class_id]),
                "confidence": round(
                    float(box.conf.item()) * 100, 2
                ),
                "box": [
                    round(float(v), 1)
                    for v in box.xyxy[0].tolist()
                ],
            })

    print(f"YOLO time: {yolo_time:.2f} seconds")

    # ----------------------------------------------
    # NO NOTE DETECTED
    # ----------------------------------------------

    if not detections:
        return {
            "denomination_detections": [],
            "authenticity": "UNCERTAIN",
            "authenticity_confidence": 0.0,
            "warning": (
                "No currency note detected. "
                "Hold the note clearly in front of the camera."
            ),
            "processing_time": round(yolo_time, 2),
        }

    # ----------------------------------------------
    # SELECT BEST DETECTION
    # ----------------------------------------------

    best = max(
        detections,
        key=lambda item: item["confidence"],
    )

    # ----------------------------------------------
    # AUTHENTICITY CLASSIFICATION
    # ----------------------------------------------

    auth_start = time.perf_counter()

    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(
            authenticity_model(tensor),
            dim=1,
        )[0]

    classes = ["fake", "real"]
    predicted_id = int(torch.argmax(probabilities).item())

    authenticity = classes[predicted_id].upper()

    authenticity_confidence = round(
        float(probabilities[predicted_id]) * 100,
        2,
    )

    auth_time = time.perf_counter() - auth_start

    total_time = yolo_time + auth_time

    print(f"Authenticity time: {auth_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")

    # ----------------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------------

    return {
        "denomination_detections": detections,
        "authenticity": authenticity,
        "authenticity_confidence": authenticity_confidence,
        "warning": (
            "AI assessment only; this is not proof "
            "that a note is genuine or counterfeit."
        ),
        "processing_time": round(total_time, 2),
    }
