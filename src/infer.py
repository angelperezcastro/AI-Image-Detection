# src/infer.py
import torch
from PIL import Image
from torchvision import transforms
from model import build_model, load_checkpoint
import os

def predict(image_path, model_path="models/saved_models/epoch_5.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, pretrained=False).to(device)
    model = load_checkpoint(model, path=model_path, device=device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    classes = ["Real", "Generada"]
    print(f"Predicción: {classes[pred_class]} ({probs[0][pred_class]*100:.2f}%)")

if __name__ == "__main__":
    predict("data/samples/example.jpg")
