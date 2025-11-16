# scripts/create_samples.py

import os, random, shutil
from pathlib import Path

RAW_ROOT = Path("data/raw")
DEST_ROOT = Path("data/samples/ALL")
N_PER_CLASS = 50

def get_all_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return files

def process_generator(gen_name):
    print(f"Procesando generador: {gen_name}")
    gen_path = RAW_ROOT / gen_name

    for split in ["train", "val"]:
        for cls in ["ai", "nature"]:
            src = gen_path / split / cls
            dst = DEST_ROOT / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            imgs = get_all_images(src)
            if len(imgs) == 0:
                print(f"[AVISO] No hay imágenes en {src}")
                continue

            samples = random.sample(imgs, min(N_PER_CLASS, len(imgs)))
            for img in samples:
                shutil.copy(img, dst / f"{gen_name}_{img.name}")

if __name__ == "__main__":
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Detecta generadores automáticamente
    generators = [d.name for d in RAW_ROOT.iterdir() if d.is_dir()]

    for gen in generators:
        process_generator(gen)

    print("\nSamples combinados creados en", DEST_ROOT)
