from ultralytics import YOLO

model = YOLO("../model/best.pt")


def detect_currency(image):
    results = model(image)

    result = results[0]

    if len(result.boxes) == 0:
        return None, 0, result

    confidence = float(result.boxes.conf[0])
    class_id = int(result.boxes.cls[0])

    names = model.names
    denomination = names[class_id]

    return denomination, confidence, result