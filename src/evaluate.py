# src/evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import load_dataloaders
from model import build_model, load_checkpoint

def evaluate_model(model_path="models/saved_models/epoch_5.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, class_names = load_dataloaders()
    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model = load_checkpoint(model, path=model_path, device=device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("📊 Reporte de clasificación:\n", classification_report(y_true, y_pred, target_names=class_names))
    print("📉 Matriz de confusión:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
