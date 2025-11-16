# src/train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/ADM",
        help="Directorio base con train/ y val/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size para entrenamiento",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Número de épocas",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
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

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Época {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            path=f"models/saved_models/epoch_{epoch}.pth",
        )


if __name__ == "__main__":
    main()
