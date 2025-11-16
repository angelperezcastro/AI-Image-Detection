# src/dataset.py
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(train: bool = True):
    """Devuelve las transformaciones para train o val."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def load_dataloaders(
    base_dir: str = "data/raw/ADM",
    batch_size: int = 16,
    num_workers: int = 2,
):
    """
    Carga los DataLoaders de train y val usando una estructura tipo:
    base_dir/
      train/
        ai/
        nature/
      val/
        ai/
        nature/
    """
    base_dir = Path(base_dir)

    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    print("Cargando datos desde:")
    print(f" - Train: {train_dir}")
    print(f" - Val:   {val_dir}")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_transforms(train=False),
    )

    class_names = train_dataset.classes
    print("Clases detectadas:", class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, class_names
