# Implementation Plan - CNN-based Coin Counting

This plan details the implementation of a Convolutional Neural Network (CNN) pipeline to count coins in images. The pipeline leverages concepts from the image processing lectures, including color space conversion (Luminance grayscale and HSL saturation), edge detection (Sobel gradient), regression models, and deep learning.

---

## User Review Required

> [!IMPORTANT]
> **1. Regression vs. Classification**
> - The dataset only contains the **total count of coins** per image (e.g., `{"img_001.jpg": 2}`). It does not have coordinates or bounding boxes for individual coins.
> - Therefore, we will implement **count regression** (predicting a continuous scalar representing the count, trained with Mean Squared Error loss and rounded to the nearest integer) rather than object detection (bounding boxes). This aligns with Semaine 12 (Slide 39) for regression tasks.
>
> **2. Fine-tuning Pre-trained ResNet18 on CPU**
> - Since CUDA is not available on your system, we will use a pre-trained ResNet18 backbone and fine-tune it.
> - With only 120 training images, training a CNN from scratch would lead to severe overfitting and poor generalization. Fine-tuning a pre-trained model is the standard, best-practice approach in computer vision for small datasets.
> - To make CPU training fast, we will train for a small number of epochs (e.g., 20-30 epochs), which should complete in under 2 minutes.

---

## Open Questions

> [!NOTE]
> There are no major open questions, as the dataset annotations clearly dictate a regression model. We will proceed with splitting the data, implementing the preprocessing pipeline, training the CNN, and providing the `compter_pieces` interface in `traitement_nn.py`.

---

## Proposed Changes

### Component 1: Dataset Preparation

We will create a helper script to split the current validation set (140 images) into a training set (120 images) and a validation set (20 images).

#### [NEW] [preparer_train_set.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/preparer_train_set.py)
A script that:
- Reads `data/validation.json`.
- Copies the first 120 images (`img_001.jpg` to `img_120.jpg`) to `data/train/`.
- Writes `data/train.json` with the corresponding 120 labels.
- Updates `data/validation.json` to keep only the remaining 20 images (`img_121.jpg` to `img_140.jpg`), ensuring independent training/validation splits.

---

### Component 2: Neural Network Pipeline

We will implement the CNN architecture, preprocessing, and training code.

#### [NEW] [entrainer_nn.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/entrainer_nn.py)
A script to train the model, containing:
1. **Preprocessing Pipeline**:
   - For each input image:
     - Compute standard Luminance grayscale: $Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$ (Semaine 4).
     - Extract HSL Saturation channel (Semaine 4).
     - Compute Sobel gradient magnitude map (Semaine 9) to capture coin boundaries.
     - Stack these three single-channel maps into a single 3-channel tensor (Grayscale, Saturation, Sobel). This provides the network with highly structured geometric and color features rather than raw RGB colors.
     - Resize to $224 \times 224$ pixels and normalize to $[0, 1]$.
2. **Model Definition**:
   - Use `torchvision.models.resnet18` with pre-trained weights.
   - Replace the final fully connected layer (`fc`) with a regression layer outputting a single continuous count scalar.
3. **Training Loop**:
   - Train using MSE Loss and Adam optimizer.
   - Apply data augmentation (random rotation, flips) on the training set.
   - Save the best weights to `meilleur_modele_nn.pth`.

#### [NEW] [traitement_nn.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/traitement_nn.py)
A new interface file that:
- Loads the trained weights from `meilleur_modele_nn.pth`.
- Defines `compter_pieces(chemin_image, **kwargs)` to load, preprocess (same 3-channel Grayscale/Saturation/Sobel pipeline), resize, and forward-pass the image.
- Rounds the continuous output to the nearest integer, clips it to $\ge 0$, and returns it.

#### [MODIFY] [evaluation.py](file:///c:/Users/xps/Desktop/GitHub/CoineX/evaluation.py)
- Update line 3 to import `compter_pieces` from `traitement_nn` instead of `traitement`.

---

## Verification Plan

### Automated Tests
1. **Prepare Dataset**:
   ```bash
   python preparer_train_set.py
   ```
2. **Train Model**:
   ```bash
   python entrainer_nn.py
   ```
3. **Evaluate Pipeline**:
   ```bash
   python evaluation.py
   ```

### Manual Verification
- We will monitor the training and validation MSE/MAE in the console during the training run.
- We will run the evaluation on the validation set (the remaining 20 images) and test set (60 images) to print the MAE, MSE, and accuracy metrics.
