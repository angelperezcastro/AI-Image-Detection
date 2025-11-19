# src/gradcam_infer.py
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import build_model, load_checkpoint
import os
import sys


# ========= GRAD-CAM CORE ========= #

def generate_gradcam(model, img_tensor, target_layer):
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    # detach() es importante para convertir a numpy
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam, pred_class


def overlay_heatmap(original, cam, save_path):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = 0.4 * heatmap + 0.6 * original
    cv2.imwrite(save_path, overlay)
    print(f"🔥 Grad-CAM guardado en: {save_path}")


# ========= PIPELINE PRINCIPAL ========= #

def gradcam_pipeline(image_path, model_path, target_layer_name):
    if not os.path.isfile(image_path):
        print(f"❌ No existe la imagen: {image_path}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"❌ No existe el modelo: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    model = build_model(num_classes=2, pretrained=False).to(device)
    model = load_checkpoint(model, path=model_path, device=device)
    model.eval()

    named_layers = dict(model.named_modules())

    if target_layer_name not in named_layers:
        print(f"❌ La capa '{target_layer_name}' no existe.")
        print("Capas disponibles:")
        for name in named_layers:
            print("  -", name)
        sys.exit(1)

    target_layer = named_layers[target_layer_name]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    cam, pred_class = generate_gradcam(model, img_tensor, target_layer)

    original_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)

    # ========== NUEVO: generar nombre automático ========== #
    base_name = os.path.basename(image_path)                     # "0_adm_7.PNG"
    name_no_ext = os.path.splitext(base_name)[0]                 # "0_adm_7"
    save_name = f"{name_no_ext}_gradcam.jpg"                     # "0_adm_7_gradcam.jpg"

    output_dir = "results/gradcams"
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = os.path.join(output_dir, save_name)

    overlay_heatmap(original_cv, cam, final_output_path)

    classes = ["Generada", "Real"]
    print("\n========= RESULTADO =========")
    print(f"📌 Imagen: {image_path}")
    print(f"📦 Modelo: {model_path}")
    print(f"🎯 Capa usada: {target_layer_name}")
    print(f"🔍 Predicción: {classes[pred_class]}")
    print(f"💾 Guardado en: {final_output_path}")
    print("=============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generación de Grad-CAM")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument(
        "--target_layer",
        type=str,
        default="stages.3.blocks.2.norm",
        help="Capa objetivo de ConvNeXt Tiny"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gradcam_pipeline(
        image_path=args.image_path,
        model_path=args.model_path,
        target_layer_name=args.target_layer
    )
