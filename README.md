# IA Image Generated Detection 

This project provides a **full pipeline** for detecting whether an image is AI-generated or real, using a **ConvNeXt-Tiny model** trained on a custom dataset of image generators.
It includes:

- Training pipeline

- Evaluation utilities

- Standard inference

- Grad-CAM explainability

- Tools for building combined datasets and sampling subsets

The system is implemented in PyTorch + timm and is designed to run both locally and in Google Colab.

## 1. Model Architecture

The classifier is built using ConvNeXt-Tiny, with its classification head replaced by a 2-class fully connected layer.

**Implementation:**
```
model = timm.create_model('convnext_tiny', pretrained=True)
model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
```

The two classes are:

- ai → AI-generated images

- nature → real images

<!-- IMAGE: Diagram of ConvNeXt-Tiny with replaced 2-class head -->
![ConvNext-Tiny Diagram with replaced 2-class head](.\assets\auxImage1.png) 
## 2. Dataset Format

All training and validation loaders expect the following structure (ImageFolder-compatible):
```
.
└── dataset_root/
    ├── train/
    │   ├── ai/
    │   └── nature/
    └── val/
        ├── ai/
        └── nature/
```

The data loader automatically applies augmentation for training and normalization compatible with ConvNeXt.
**Implementation:**

- Train transforms: resize, random horizontal flip, normalization

- Val transforms: resize, normalization

## 3. Training

Training script: src/train.py


##### 3.1. Run training (local)
```
python src/train.py \
    --data_dir data/raw/ADM \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4
```
##### 3.2. Arguments

| Argument        | Default        | Description                                      |
|-----------------|----------------|--------------------------------------------------|
| `--data_dir`    | `data/raw/ADM` | Dataset root (must contain `/train` and `/val`) |
| `--epochs`      | `5`            | Number of training epochs                        |
| `--batch_size`  | `16`           | Batch size                                       |
| `--lr`          | `1e-4`         | Learning rate                                    |


##### 3.3. Output

At each epoch, the script prints:

- Training loss

- Validation loss

- Validation accuracy

And saves a checkpoint to:

```
models/saved_models/epoch_{N}.pth
```



![Training loss curve](.\results\figures\loss_curve.png) ![Training loss curve](.\results\figures\val_accuracy_curve.png)


## 4. Evaluation on Validation Set

Evaluation script: `src/evaluate.py`


##### 4.1. Run evaluation
```
python src/evaluate.py --model_path models/saved_models/epoch_5.pth
```

The script outputs:

- Precision, recall, F1-score for each class

- Confusion matrix


![evaluation example](.\assets\screenshot1.png) ![confusion matrix](.\results\figures\confusion_matrix.png)

## 5. Standard Inference (Single Image)

Inference script: src/infer.py


##### 5.1. Run inference
```
python src/infer.py \
    --image_path example.jpg \
    --model_path models/saved_models/epoch_5.pth
```

The script will:

- Validate the image path and model path

- Load the ConvNeXt-Tiny model

- Preprocess the image (resize, normalize)

- Run a forward pass

- Print prediction and confidence

Classes used internally:

- ["Generada", "Real"]


**Output example:**

🔍 Predicción: Real
📊 Confianza: 97.32%

![infer example](.\assets\screenshot2.png)
## 6. Grad-CAM Explainability

Grad-CAM inference script: src/gradcam_infer.py

This script generates a **Grad-CAM heatmap** overlay showing which regions of the image most influenced the model’s decision.

##### 6.1. Run Grad-CAM
```
python src/gradcam_infer.py \
    --image_path example.jpg \
    --model_path models/saved_models/epoch_5.pth \
    --target_layer "stages.2.blocks.4.conv_dw"
```

--target_layer is the name of the ConvNeXt layer used to compute Grad-CAM 
>(stages.2.blocks.4.conv_dw recommended).

The script lists available layer names if the provided one is invalid.

##### 6.2. Output

A file is saved as:
```
results/gradcams/<original_name>_gradcam.jpg
```

The script also prints:
- Prediction (Generada / Real)

- Used layer name

- Path to the Grad-CAM output

![gradcam_infer example](.\assets\screenshot3.png)
![original image](.\data\samples\ALL\train\ai\ADM_19_adm_63.png)  ![heatmap](.\results\gradcams\ADM_19_adm_63_gradcam.jpg)
## 7. Dataset Tools
##### 7.1. Create Balanced Sample Dataset

Script: scripts/create_samples.py
This script:

- Detects all generator folders under data/raw/

- Randomly samples N_PER_CLASS images for each class (ai, nature) and each split (train, val)

- Copies them into a combined samples dataset:

- python scripts/create_samples.py


Output location (default):

- data/samples/ALL/


Structure:

```
.
└── data/samples/ALL
    ├── train/
    │   ├── ai/
    │   └── nature/
    └── val/
        ├── ai/
        └── nature/
```
##### 7.2. Create Combined Full Dataset

Script: scripts/create_full_all.py

This script merges all generator-specific datasets into a single unified dataset (ALL_full). It was only used in colab due to the high volume of images:
```
python scripts/create_full_all.py
```

Output location:
```
data/raw/ALL_full/
```

Structure:
```
.
└── data/raw/ALL_full/
    ├── train/
    │   ├── ai/
    │   └── nature/
    └── val/
        ├── ai/
        └── nature/
```

Each copied file is prefixed with the generator name (e.g. ADM_, BigGAN_) to avoid name collisions.

## 8. Requirements

- Python 3.9+

- PyTorch

- timm

- torchvision

- scikit-learn

- OpenCV

- tqdm

Example manual installation:
```
 pip install torch torchvision
pip install timm scikit-learn opencv-python tqdm
```
## 9. Summary

This project provides a complete AI-image detection framework with:

- ConvNeXt-Tiny backbone

- Training, evaluation, and inference pipelines

- Grad-CAM explainability

- Dataset construction and sampling tools

- Support for local and Google Colab training

