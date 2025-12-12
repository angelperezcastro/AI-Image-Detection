# scripts/create_full_all.py

import shutil
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

RAW_ROOT = Path("data/raw")                  # Root folder containing generators (ADM, BigGAN, ...)
DEST_ROOT = Path("data/raw/ALL_full")        # Destination folder for the combined dataset

EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


# ==============================
# HELPER FUNCTIONS
# ==============================

def get_all_images(folder: Path):
    """Return all image files inside a folder."""
    files = []
    for ext in EXTS:
        files.extend(folder.glob(ext))
    return files


def process_generator(generator_name: str):
    """Copy all images from a generator into the combined dataset."""
    gen_path = RAW_ROOT / generator_name
    print(f"\n=== Processing generator: {generator_name} ===")

    for split in ["train", "val"]:
        for cls in ["ai", "nature"]:
            src = gen_path / split / cls

            if not src.is_dir():
                print(f"[WARNING] Folder not found: {src}")
                continue

            dst = DEST_ROOT / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            images = get_all_images(src)
            if len(images) == 0:
                print(f"[WARNING] No images found in {src}")
                continue

            print(f"📂 Copying {len(images)} images from {src} → {dst}")

            for img in images:
                # Prefix filename with generator name to avoid collisions
                dst_path = dst / f"{generator_name}_{img.name}"
                shutil.copy2(img, dst_path)

    print(f"=== Finished {generator_name} ===")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Automatically detect generators (all folders in RAW_ROOT except ALL_full)
    generators = [
        d.name for d in RAW_ROOT.iterdir()
        if d.is_dir() and d.name not in ["ALL_full"]
    ]

    print("Detected generators:", generators)

    for gen in generators:
        process_generator(gen)

    print("\n✅ FULL combined dataset created at:", DEST_ROOT)
    print("   Structure: ALL_full/train/{ai,nature}, ALL_full/val/{ai,nature}")
