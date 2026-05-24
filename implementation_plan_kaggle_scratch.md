# Implementation Plan - Custom CNN Coin Counting (Kaggle/Colab + Local Inference)

This plan details the implementation of a custom Convolutional Neural Network (CNN) trained from scratch to count coins in images. The training is designed to run on a cloud platform (Kaggle or Google Colab) with GPU acceleration, and the trained model weights will be exported for local CPU inference in the `compter_pieces` pipeline.

---

## User Review Required

> [!IMPORTANT]
> **1. Training on Kaggle/Colab & Exporting Weights**
> - We will provide a clean script/notebook code (`entrainer_nn.py`) that can be executed on Kaggle/Colab.
> - The training process will output `meilleur_modele_nn.pth` (a serialized PyTorch state dictionary) which you can download and place in your local workspace.
>
> **2. Custom CNN Architecture from Scratch**
> - Instead of a pre-trained backbone, we will design a custom CNN architecture from scratch matching your lecture concepts:
>   - **Input**: 3-channel preprocessed image ($128 \times 128$).
>   - **Convolutional Layers**: 4 layers of 2D Convolution (extracting feature maps using local filters and learning biases) alternating with Max Pooling (reducing spatial dimensions and computational load) and ReLU activations.
>   - **Fully Connected Layers**: A hidden dense layer with ReLU activation, followed by a linear output layer producing the continuous count scalar.
>   - **Loss**: Mean Squared Error (MSE) loss, optimized using Adam.
>
> **3. Local Inference (PyTorch CPU)**
> - Since PyTorch is installed locally, `traitement_nn.py` will rebuild the same custom CNN structure, load the downloaded `meilleur_modele_nn.pth` weights file, and run fast CPU-based inference for single images.

---

## Proposed Changes

### Component 1: Local Dataset Preparation

We will split the current validation set (140 images) into a training set (120 images) and a validation set (20 images) to prepare the data for cloud uploading.

#### [NEW] [preparer_train_set.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/preparer_train_set.py)
A script that:
- Reads `data/validation.json`.
- Creates `data/train` and moves the first 120 images (`img_001.jpg` to `img_120.jpg`) there.
- Creates `data/train.json` containing only the labels for those 120 images.
- Updates `data/validation.json` to keep only the remaining 20 images (`img_121.jpg` to `img_140.jpg`).

---

### Component 2: Cloud Training Pipeline

We will write the training code designed to run on Kaggle/Colab with GPU support.

#### [NEW] [entrainer_nn.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/entrainer_nn.py)
A training script containing:
1. **Preprocessing Pipeline**:
   - Luminance conversion ($Y = 0.299R + 0.587G + 0.114B$) for intensity.
   - HSL Saturation extraction for color richness.
   - Sobel gradient magnitude for coin boundaries.
   - Resize to $128 \times 128$ and stack these three channels.
2. **Custom CNN Model**:
   - Alternates Conv2d (with biases) -> ReLU -> MaxPool2d.
   - Flattens features and passes them through Fully Connected (Linear) layers to output a single count scalar.
3. **Optimizations**:
   - Data augmentation (flips, rotations).
   - Training loop using MSE Loss, tracking train/val MAE and MSE over 100+ epochs.
   - Saves model weights to `meilleur_modele_nn.pth`.

---

### Component 3: Local Inference Integration

#### [NEW] [traitement_nn.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/traitement_nn.py)
The local interface module that:
- Defines the exact same `CustomCNN` architecture.
- Implements the 3-channel preprocessing pipeline.
- Loads the weights from `meilleur_modele_nn.pth` on the CPU.
- Exposes `compter_pieces(chemin_image, **kwargs)` which runs inference on the image, rounds the result to the nearest integer, and returns it.

#### [MODIFY] [evaluation.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/evaluation.py)
- Change line 3 to import `compter_pieces` from `traitement_nn` instead of `traitement`.

---

## Verification Plan

### Automated Tests
1. **Locally Split Dataset**:
   ```bash
   python preparer_train_set.py
   ```
2. **Run Cloud Training**:
   - Zip `data/train`, `data/validation`, `data/train.json`, `data/validation.json` and upload them to Kaggle/Colab.
   - Run `entrainer_nn.py` on Kaggle/Colab.
   - Download the trained `meilleur_modele_nn.pth` weights and place it in the local `CoineX` workspace.
3. **Local Evaluation**:
   ```bash
   python evaluation.py
   ```
