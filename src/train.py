# src/train.py
import argparse
import os
import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import load_dataloaders
from model import build_model, save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc="Entrenando"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validando"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return val_loss / len(dataloader), accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento ConvNeXt IA Detector")
    parser.add_argument("--data_dir", type=str, default="data/raw/ADM", help="Directorio base con train/ y val/")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size para entrenamiento")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Usando dataset: {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    train_loader, val_loader, class_names = load_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Guardar historial de métricas
    train_losses, val_losses, val_accs = [], [], []

    # Crear carpetas de resultados
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Época {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            path=f"models/saved_models/epoch_{epoch}.pth",
        )

    # ====== FIGURAS (results/figures) ======
    epochs_range = list(range(1, args.epochs + 1))

    # Loss
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolución del Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/loss_curve.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Evolución de la Accuracy en Validación")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/val_accuracy_curve.png")
    plt.close()

    # ====== MÉTRICAS (results/metrics) ======
    # 1) CSV con el historial
    history_csv = "results/metrics/train_history.csv"
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for e, tr, vl, va in zip(epochs_range, train_losses, val_losses, val_accs):
            writer.writerow([e, tr, vl, va])

    # 2) Resumen en JSON (útil para el informe)
    best_epoch_idx = max(range(len(val_accs)), key=lambda i: val_accs[i])
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_dir": args.data_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "class_names": class_names,
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        "final_val_acc": float(val_accs[-1]) if val_accs else None,
        "best_val_acc": float(val_accs[best_epoch_idx]) if val_accs else None,
        "best_epoch_by_val_acc": int(epochs_range[best_epoch_idx]) if val_accs else None,
    }

    with open("results/metrics/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n📈 Figuras guardadas en: results/figures/")
    print("🧾 Métricas guardadas en: results/metrics/")


if __name__ == "__main__":
    main()
