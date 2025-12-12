# src/evaluate.py
import argparse
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dataset import load_dataloaders
from model import build_model, load_checkpoint


def evaluate_model(model_path, data_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    _, val_loader, class_names = load_dataloaders(
        base_dir=data_dir,
        batch_size=batch_size,
    )

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model = load_checkpoint(model, path=model_path, device=device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ====== MÉTRICAS ======
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 === CLASIFICATION REPORT ===\n")
    print(report_txt)
    print("\n📉 === MATRIZ DE CONFUSIÓN ===\n")
    print(cm)

    # Guardar reporte (TXT + JSON)
    with open("results/metrics/classification_report.txt", "w") as f:
        f.write(report_txt)
        f.write("\n")

    with open("results/metrics/classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Guardar matriz de confusión como CSV
    np.savetxt("results/metrics/confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # ====== FIGURA: matriz de confusión ======
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Escribir los números dentro (útil para el informe)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png")
    plt.close()

    print("\n🧾 Métricas guardadas en: results/metrics/")
    print("📈 Figura guardada en: results/figures/confusion_matrix.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación del modelo ConvNeXt IA Detector")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al archivo .pth del modelo")
    parser.add_argument("--data_dir", type=str, required=True, help="Directorio base con train/ y val/")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
