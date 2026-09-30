import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.model import MnistClassifier
from torchvision import transforms



app = FastAPI()

# Load the trained model
model = MnistClassifier()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "MNIST Classifier",
        "version": "1.0"
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = file.file.read()
        # Preprocess the image (assuming it's a grayscale image)
        image_tensor = torch.tensor(list(image_bytes), dtype=torch.float32).view(1, 1, 28, 28) / 255.0

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        return {"predicted_digit": predicted.item(), "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
