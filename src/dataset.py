# src/dataset.py

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
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


def _detect_generator_roots(base_dir: Path):
    """
    Devuelve una lista de carpetas que actúan como 'root' de dataset:
    cada una debe tener train/ y val/ dentro.
    """
    roots = []

    # Caso 1: base_dir YA es un dataset (tiene train/ y val/)
    if (base_dir / "train").is_dir() and (base_dir / "val").is_dir():
        roots.append(base_dir)
        return roots

    # Caso 2: base_dir es una carpeta contenedora (p.ej. data/raw)
    # buscamos subcarpetas que tengan train/ y val/
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue

        train_dir = sub / "train"
        val_dir = sub / "val"
        if train_dir.is_dir() and val_dir.is_dir():
            roots.append(sub)

    return roots


def load_dataloaders(
    base_dir: str = "data/raw/generador1_ADM_descomprimido",
    batch_size: int = 16,
    num_workers: int = 2,
):
    """
    Carga los DataLoaders de train y val.

    Soporta dos modos:

    1) base_dir = carpeta que contiene directamente:
         base_dir/train/{ai,nature}
         base_dir/val/{ai,nature}

    2) base_dir = carpeta contenedora (p.ej. data/raw) con varias subcarpetas
       que sí tienen esa estructura (tus 3 generadores). En ese caso,
       junta todos los generadores en un único dataset usando ConcatDataset.
    """
    base_dir = Path(base_dir)

    generator_roots = _detect_generator_roots(base_dir)

    if not generator_roots:
        raise ValueError(
            f"No se han encontrado datasets válidos en {base_dir}. "
            f"Asegúrate de que hay carpetas con train/ y val/ dentro."
        )

    print("Cargando datos desde los siguientes roots:")
    for root in generator_roots:
        print(f" - {root}")

    train_datasets = []
    val_datasets = []

    for root in generator_roots:
        train_dir = root / "train"
        val_dir = root / "val"

        train_ds = datasets.ImageFolder(
            root=train_dir,
            transform=get_transforms(train=True),
        )
        val_ds = datasets.ImageFolder(
            root=val_dir,
            transform=get_transforms(train=False),
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    # Comprobamos que todas las clases coinciden (ai, nature, etc.)
    class_names = train_datasets[0].classes
    for ds in train_datasets[1:] + val_datasets:
        if ds.classes != class_names:
            raise ValueError(
                "Las clases no coinciden entre generadores. "
                f"Esperado {class_names}, pero encontrado {ds.classes}"
            )

    # Unimos todos los datasets en uno solo
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

    print("Clases detectadas:", class_names)
    print(f"Nº de datasets combinados: {len(train_datasets)}")

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
