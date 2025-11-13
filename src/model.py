# src/model.py
import torch
import timm

def build_model(num_classes=2, pretrained=True):
    model = timm.create_model('convnext_tiny', pretrained=pretrained)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
    return model

def save_checkpoint(model, optimizer, epoch, path="models/saved_models/checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint guardado en {path}")

def load_checkpoint(model, optimizer=None, path="models/saved_models/checkpoint.pth", device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Checkpoint cargado desde {path}")
    return model
