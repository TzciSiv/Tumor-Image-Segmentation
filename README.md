# Tumor Image Segmentation: A PyTorch Deep Learning Project

## Overview

This project implements a **U-Net–style segmentation model** using a **pretrained ResNet-50 encoder** to identify tumor regions in medical images. It provides a complete end-to-end workflow including preprocessing, augmentation, K-fold cross-validation, and detailed per-pixel metric evaluation.

Across validation folds, the model achieved an average segmentation accuracy of **~95%**.

## Architecture

The application is organized into several core components:

1. **Model Definition (`model.py`)**
   - Implements a U-Net decoder paired with a pretrained ResNet-50 backbone for strong feature extraction.
   - Produces per-pixel probability maps indicating tumor vs. non-tumor regions.

2. **Data Pipeline (`dataset.py`)**
   - Loads image–mask pairs from a CSV index, resizing and normalizing them for training.
   - Includes label-preserving augmentations such as flips and rotations.

3. **Training Engine (`train.py`)**
   - Performs forward passes, computes segmentation losses (Dice, Combo Loss), and updates model weights.
   - Evaluates performance each epoch using accuracy, Dice, IoU, FPR, and FNR.

4. **Experiment Controller (`main.py`)**
   - Manages **K-fold cross-validation** to ensure reliable segmentation performance.
   - Saves prediction masks for visual inspection and aggregates fold-level metrics.

5. **Configuration Layer (`config.py`)**
   - Stores all hyperparameters: image size, learning rate, batch size, number of folds, output paths, and device settings.

6. **Metrics Module (`metrics.py`)**
   - Calculates segmentation metrics including:
     - Accuracy  
     - Dice Score  
     - Intersection-over-Union (IoU)  
     - False-Positive Rate (FPR)  
     - False-Negative Rate (FNR)

## Dataset

Download and unzip the dataset, then name the folder **`Dataset`** and place it next to the project directory (outside the Python files):

🔗 **Dataset Link:**  
https://drive.google.com/file/d/1K4ReQ5Cy2f2RwGFn1LU29kwSbRDMLmMq/view

Your directory structure should look like:
```bash
├─ Dataset/
├─ project/
│  ├─ main.py
│  ├─ model.py
│  ├─ train.py
│  ├─ metrics.py
│  ├─ config.py
│  ├─ loss.py
│  └─ dataset.py
```

## Sample Run

Before running the script, install any missing packages based on error messages

Then run:

```bash
python main.py

```
