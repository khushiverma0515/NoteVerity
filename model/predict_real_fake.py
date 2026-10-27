import os
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image


# -----------------------------
# 1. Model path
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "real_fake_model.pth"
)


# -----------------------------
# 2. Load model
# -----------------------------

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()


# -----------------------------
# 3. Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 4. Prediction function
# -----------------------------

def predict_real_fake(image):

    image = Image.fromarray(image).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    classes = ["fake", "real"]

    prediction = classes[predicted.item()]

    confidence = confidence.item() * 100
    return prediction, confidence


if __name__ == "__main__":
    import cv2

    image_path = input("Enter image path: ")

    image = cv2.imread(image_path)

    prediction, confidence = predict_real_fake(image)

    print("Prediction :", prediction.upper())
    print("Confidence :", f"{confidence:.2f}%")