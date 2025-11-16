import os
from glob import glob
from pprint import pprint

# ================================
#  UTILIDADES GENERALES
# ================================

def count_images(path):
    """Cuenta imágenes en una carpeta (recursivo)."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(path, "**", e), recursive=True))
    return files


def show_samples(path, limit=5):
    """Muestra algunos ejemplos del directorio."""
    files = count_images(path)
    print(f"Samples {path}:")
    for f in files[:limit]:
        print("  -", f)
    if len(files) == 0:
        print("  (vacío)")
    print()


# ================================
#  RUTINA PRINCIPAL
# ================================
if __name__ == "__main__":
    print("\n==============================")
    print(" TEST DATASET: ADM + BigGAN")
    print("==============================\n")

    # Rutas principales
    adm_root = "data/raw/ADM"
    biggan_root = "data/raw/BigGAN"

    datasets = {
        "ADM": adm_root,
        "BigGAN": biggan_root
    }

    for name, root in datasets.items():
        print(f"### Dataset: {name}")
        if not os.path.isdir(root):
            print(f"❌ No se encontró la carpeta {root}\n")
            continue

        train_path = os.path.join(root, "train")
        val_path = os.path.join(root, "val")

        print("Subcarpetas en TRAIN:", os.listdir(train_path) if os.path.isdir(train_path) else "no existe")
        print("Subcarpetas en VAL:", os.listdir(val_path) if os.path.isdir(val_path) else "no existe")

        print("\nConteo de imágenes:")

        for split in ["train", "val"]:
            split_path = os.path.join(root, split)
            if not os.path.isdir(split_path):
                print(f"  {split}: carpeta no encontrada")
                continue

            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                files = count_images(cls_path)
                print(f"  {split}/{cls}: {len(files)} imágenes")

        print("\nEjemplos:")
        for split in ["train", "val"]:
            split_path = os.path.join(root, split)
            if os.path.isdir(split_path):
                for cls in os.listdir(split_path):
                    cls_path = os.path.join(split_path, cls)
                    if os.path.isdir(cls_path):
                        show_samples(cls_path, limit=3)

        print("\n-----------------------------------\n")
