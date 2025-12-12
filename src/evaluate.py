
import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import load_dataloaders
from model import build_model, load_checkpoint


def evaluate_model(model_path, data_dir, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    # SOLO cargamos el val_loader (no necesitamos train)
    _, val_loader, class_names = load_dataloaders(
        base_dir=data_dir,
        batch_size=batch_size,
    )

    # Crear modelo y cargar pesos
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

    print("\n📊 === CLASIFICATION REPORT ===\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\n📉 === MATRIZ DE CONFUSIÓN ===\n")
    print(confusion_matrix(y_true, y_pred))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación del modelo ConvNeXt IA Detector")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Ruta al archivo .pth del modelo")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directorio base con train/ y val/")
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
