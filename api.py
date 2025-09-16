import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware   # ✅ added
from torchvision import transforms
from PIL import Image
import io

from CNN.model import get_model  # make sure this points to your CNN model file

# ---------------------
# Init FastAPI app
# ---------------------
app = FastAPI(
    title="Satellite Image Classifier",
    description="Classify satellite images into cloudy, desert, green_area, water",
    version="1.0"
)

# ---------------------
# CORS setup
# ---------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Device config
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Classes (must match training)
# ---------------------
classes = ['cloudy', 'desert', 'green_area', 'water']

# ---------------------
# Load trained model
# ---------------------
model = get_model("resnet18", num_classes=len(classes), pretrained=True, fine_tune=True)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ---------------------
# Preprocessing (same as training)
# ---------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ---------------------
# API Routes
# ---------------------

@app.get("/")
def read_root():
    return {"message": "Satellite CNN API is running. Use /predict to classify images."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image (jpg, png, etc.) and get back predictions."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(torch.argmax(torch.tensor(probs)))
    pred_label = classes[pred_idx]
    pred_confidence = float(probs[pred_idx]) * 100

    return {
        "predicted_class": pred_label,
        "confidence": f"{pred_confidence:.2f}%",
        "all_probabilities": {
            classes[i]: f"{float(p) * 100:.2f}%" for i, p in enumerate(probs)
        }
    }
