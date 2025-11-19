# src/infer.py
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import build_model, load_checkpoint
import os
import sys

def predict(image_path, model_path):
    # -------------------------
    # 1) Validación de rutas
    # -------------------------
    if not os.path.isfile(image_path):
        print(f"❌ ERROR: No se encontró la imagen: {image_path}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"❌ ERROR: No se encontró el modelo: {model_path}")
        sys.exit(1)

    # -------------------------
    # 2) Preparar modelo
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando dispositivo: {device}")

    model = build_model(num_classes=2, pretrained=False).to(device)
    model = load_checkpoint(model, path=model_path, device=device)
    model.eval()

    # -------------------------
    # 3) Preprocesamiento
    # -------------------------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # -------------------------
    # 4) Inferencia
    # -------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    classes = ["Generada", "Real"]

    print("\n==============================")
    print(f"📌 Imagen: {image_path}")
    print(f"📦 Modelo: {model_path}")
    print("------------------------------")
    print(f"🔍 Predicción: {classes[pred_class]}")
    print(f"📊 Confianza: {probs[0][pred_class] * 100:.2f}%")
    print("==============================\n")

    return classes[pred_class], float(probs[0][pred_class])


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia del modelo IA-Image-Detection")
    
    parser.add_argument(
        "--image_path",
        required=True,
        type=str,
        help="Ruta a la imagen a clasificar"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Ruta al archivo .pth del modelo"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.image_path, args.model_path)
