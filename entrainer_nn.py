"""
Entraînement du CNN de comptage de pièces.

Améliorations clés (par rapport à la version précédente) :
  - Pipeline 4 canaux partagé via pipeline_traitement.py
  - Normalisation par-canal (mean/std calculés sur le train) - Semaine 7
  - Architecture VGG-like (double conv par bloc) plus profonde
  - Augmentation forte : photométrique + géométrique fin
  - Loss Huber (SmoothL1) au lieu de MSE pur - mieux aligné avec la MAE
  - Mixed precision training (AMP) - ~1.7x plus rapide sur GPU
  - Cosine annealing avec warm restarts
  - Early stopping (patience configurable)
  - Gradient clipping pour stabilité
  - Statistiques de normalisation sauvegardées dans le checkpoint
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from pipeline_traitement import pretraiter_image_brut


# =============================================================================
# 1. DATASET avec normalisation par-canal et augmentation forte
# =============================================================================
def calculer_statistiques_dataset(preloaded_data):
    """
    Calcule la moyenne et l'écart-type par canal sur les données pré-chargées
    (non augmentées, non normalisées). Renvoie deux tenseurs de taille (4,).

    Référence cours Semaine 7 : « la normalisation permet aux algorithmes de
    se concentrer sur la distribution plutôt que sur l'illumination globale ».
    """
    sum_ = torch.zeros(4, dtype=torch.float64)
    sum_sq = torch.zeros(4, dtype=torch.float64)
    n = 0
    for x, _ in preloaded_data:
        flat = x.view(4, -1).to(torch.float64)
        sum_ += flat.sum(dim=1)
        sum_sq += (flat * flat).sum(dim=1)
        n += flat.shape[1]
    mean = sum_ / n
    var = sum_sq / n - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.to(torch.float32), std.to(torch.float32)


class CoinDataset(Dataset):
    def __init__(self, dossier_images, fichier_json, augment=False):
        self.augment = augment
        self.mean = None
        self.std = None

        with open(fichier_json, 'r') as f:
            self.labels_dict = json.load(f)
        self.liste_images = sorted(list(self.labels_dict.keys()))

        # Pré-chargement RAM : élimine le goulot CPU disque + prétraitement
        # à chaque itération. Les augmentations s'appliquent à la volée sur
        # les tenseurs déjà prêts.
        self.preloaded_data = []
        print(f"  Pré-chargement {len(self.liste_images)} images depuis '{dossier_images}'...")
        for nom_image in self.liste_images:
            chemin = os.path.join(dossier_images, nom_image)
            x = pretraiter_image_brut(chemin)            # 4 x 256 x 256 dans [0, 1]
            y = float(self.labels_dict[nom_image])
            self.preloaded_data.append((x, y))

    def set_statistics(self, mean, std):
        """Fixe les statistiques de normalisation par-canal."""
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __len__(self):
        return len(self.liste_images)

    def __getitem__(self, idx):
        x_raw, y = self.preloaded_data[idx]
        x = x_raw.clone()

        if self.augment:
            x = self._augmenter(x)

        # Normalisation par-canal (Semaine 7 : I_norm = (I - mean) / std)
        if self.mean is not None:
            x = (x - self.mean) / self.std

        return x, torch.tensor([y], dtype=torch.float32)

    def _augmenter(self, x):
        """
        Augmentation forte sur tenseur (4 x H x W, valeurs dans [0, 1]).

        - Géométrique (tous canaux ensemble) : flips, rotations 90°, rotation libre
        - Photométrique (canal par canal) :
            * Luminosité : luminance uniquement (Semaine 7 - décalage histogramme)
            * Contraste : luminance uniquement
            * Bruit Gaussien : canaux continus (Sem. 9 - simule le grain capteur)
            * Flou Gaussien : luminance + Sobel (simule défocus)
        Le masque Otsu (canal 3) reste protégé des changements photométriques
        car il est binaire par construction.
        """
        # --- Géométrique ---
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])                 # flip horizontal
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])                 # flip vertical
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, dims=[1, 2])          # rotation 90/180/270

        if random.random() < 0.7:
            # Rotation à angle libre (faible amplitude pour limiter les bords noirs)
            angle = random.uniform(-20.0, 20.0)
            x = TF.rotate(
                x.unsqueeze(0), angle, fill=0.0,
                interpolation=TF.InterpolationMode.BILINEAR,
            ).squeeze(0)

        # --- Photométrique ---
        # Décalage de luminosité sur la luminance (canal 0)
        if random.random() < 0.5:
            shift = random.uniform(-0.15, 0.15)
            x[0] = (x[0] + shift).clamp(0.0, 1.0)

        # Étirement de contraste sur la luminance
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            m = x[0].mean()
            x[0] = ((x[0] - m) * scale + m).clamp(0.0, 1.0)

        # Bruit Gaussien (canaux continus 0, 1, 2)
        if random.random() < 0.5:
            noise = torch.randn(3, x.shape[1], x.shape[2]) * 0.02
            x[:3] = (x[:3] + noise).clamp(0.0, 1.0)

        # Flou Gaussien (canaux continus, simule un défocus léger)
        if random.random() < 0.3:
            sigma = random.uniform(0.5, 1.5)
            x_blur = TF.gaussian_blur(
                x[:3].unsqueeze(0),
                kernel_size=[5, 5],
                sigma=[sigma, sigma],
            ).squeeze(0)
            x[:3] = x_blur

        return x


# =============================================================================
# 2. ARCHITECTURE CNN - VGG-like (double conv par bloc) avec 4 canaux d'entrée
# =============================================================================
class CustomCNN(nn.Module):
    """
    CNN from scratch (Semaine 12) :
      - 4 blocs convolutifs (Conv-BN-ReLU x2 + MaxPool)
      - Global Average Pooling (robuste à la translation)
      - 2 couches denses avec Dropout
    Total : ~785k paramètres
    """

    def __init__(self, in_channels=4):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.b1 = block(in_channels, 32)   # 256 -> 128
        self.b2 = block(32, 64)            # 128 -> 64
        self.b3 = block(64, 128)           # 64 -> 32
        self.b4 = block(128, 256)          # 32 -> 16

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)         # B x 256
        x = self.dropout(x)
        x = F.relu(self.fc1(x))            # B x 64
        x = self.fc2(x)                    # B x 1
        return x


# =============================================================================
# 3. ENTRAÎNEMENT
# =============================================================================
def entrainer_modele(epochs=200, batch_size=16, lr=1e-3, patience=30,
                     seed=42, chemin_sauvegarde="meilleur_modele_nn.pth"):
    # Reproductibilité
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Périphérique ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"SUCCÈS : GPU détecté → {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("ATTENTION : Aucun GPU - entraînement sur CPU.")

    # --- [1/5] Chargement datasets ---
    print("\n[1/5] Chargement des datasets...")
    train_dataset = CoinDataset("data/train", "data/train.json", augment=True)
    val_dataset = CoinDataset("data/validation", "data/validation.json", augment=False)

    # --- [2/5] Statistiques de normalisation sur le train ---
    print("\n[2/5] Calcul des statistiques de normalisation par canal...")
    mean, std = calculer_statistiques_dataset(train_dataset.preloaded_data)
    print(f"  Mean (par canal) : {[f'{m:.4f}' for m in mean.tolist()]}")
    print(f"  Std  (par canal) : {[f'{s:.4f}' for s in std.tolist()]}")
    train_dataset.set_statistics(mean, std)
    val_dataset.set_statistics(mean, std)

    # --- DataLoaders ---
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=pin_mem, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem,
    )

    # --- [3/5] Modèle, loss, optimiseur ---
    print("\n[3/5] Initialisation du modèle...")
    model = CustomCNN(in_channels=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {n_params:,}")

    # Huber / SmoothL1 : robuste aux outliers ET dérivable en 0
    # (Semaine 3 : MSE pénalise lourdement les outliers, MAE n'est pas dérivable)
    criterion = nn.SmoothL1Loss(beta=1.0)

    # AdamW : meilleure régularisation L2 que Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    # Cosine annealing with warm restarts : aide à sortir des minima locaux
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    # Mixed precision : ~1.7x plus rapide sur T4 (Kaggle)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

    # --- [4/5] Boucle d'entraînement ---
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0

    print(f"\n[4/5] Entraînement (max {epochs} epochs, early stop patience={patience})...")
    print("-" * 78)

    verif_hw_affichee = False

    for epoch in range(1, epochs + 1):
        # ===== PHASE TRAIN =====
        model.train()
        train_loss_acc = 0.0
        train_mae_acc = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if not verif_hw_affichee:
                print(f"  [VÉRIF HW] Entrée: {inputs.device} | Cible: {targets.device} "
                      f"| Modèle: {next(model.parameters()).device}\n")
                verif_hw_affichee = True

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item() * inputs.size(0)
            train_mae_acc += torch.sum(torch.abs(predictions.detach() - targets)).item()

        scheduler.step()
        train_loss_acc /= len(train_dataset)
        train_mae_acc /= len(train_dataset)

        # ===== PHASE VAL =====
        model.eval()
        val_loss_acc = 0.0
        val_mae_acc = 0.0
        val_exact = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)

                val_loss_acc += loss.item() * inputs.size(0)
                # Métrique alignée avec evaluation.py : prédictions arrondies + clamp >= 0
                pred_rounded = torch.round(predictions.float()).clamp(min=0.0)
                val_mae_acc += torch.sum(torch.abs(pred_rounded - targets)).item()
                val_exact += (pred_rounded == targets).sum().item()

        val_loss_acc /= len(val_dataset)
        val_mae_acc /= len(val_dataset)
        val_exact_pct = 100.0 * val_exact / len(val_dataset)

        improved = val_mae_acc < best_val_mae - 1e-4
        if improved:
            best_val_mae = val_mae_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean.tolist(),
                'std': std.tolist(),
                'in_channels': 4,
                'val_mae': val_mae_acc,
                'val_exact_pct': val_exact_pct,
                'epoch': epoch,
            }, chemin_sauvegarde)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or improved or patience_counter >= patience:
            lr_now = optimizer.param_groups[0]['lr']
            tag = "  <-- BEST" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss_acc:.4f} | Train MAE: {train_mae_acc:.2f} | "
                f"Val MAE: {val_mae_acc:.2f} | Val Exact: {val_exact_pct:4.1f}% | "
                f"LR: {lr_now:.2e}{tag}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"\n>>> Early stopping après {patience} epochs sans amélioration.")
            break

    # --- [5/5] Résumé final ---
    print("-" * 78)
    print(f"[5/5] Entraînement terminé.")
    print(f"  Meilleure Val MAE (arrondie) : {best_val_mae:.2f} pièces (epoch {best_epoch})")
    print(f"  Modèle + stats sauvegardés dans : {chemin_sauvegarde}")


if __name__ == "__main__":
    # Détection automatique de l'environnement (Kaggle/Colab vs local)
    is_cloud = (
        "COLAB_GPU" in os.environ
        or "KAGGLE_URL" in os.environ
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    )
    if is_cloud:
        entrainer_modele(epochs=200, batch_size=16, lr=1e-3, patience=30)
    else:
        # Test local rapide : 3 epochs pour vérifier que le code tourne sans erreur
        entrainer_modele(epochs=3, batch_size=8, lr=1e-3, patience=30)
