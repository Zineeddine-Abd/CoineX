import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. PIPELINE DE PRÉTRAITEMENT DE L'IMAGE
# =============================================================================
def pretraiter_image(chemin_image):
    """
    Applique le pipeline de prétraitement sur l'image d'origine :
    1) Conversion en niveaux de gris (Luminance).
    2) Extraction du canal Saturation HSL.
    3) Détection de contours avec le gradient de Sobel.
    4) Assemblage en 3 canaux et redimensionnement à 256x256.
    """
    img_bgr = cv2.imread(chemin_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image {chemin_image}")
        
    # Optimisation de performance : Redimensionner en 256x256 d'abord
    # pour accélérer les calculs (Luminance, Saturation HSL, Sobel) d'un facteur 110x
    img_bgr_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    
    # --- Canal 1 : Niveaux de gris normalisés (Luminance) ---
    # Formule du cours Semaine 4 : Y = 0.299*R + 0.587*G + 0.114*B
    gray = ((0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0).astype(np.float32)
    
    # --- Canal 2 : Saturation HSL normalisée ---
    # Formule du cours Semaine 4
    img_normalized = img_rgb.astype(np.float32) / 255.0
    max_c = np.max(img_normalized, axis=2)
    min_c = np.min(img_normalized, axis=2)
    delta = max_c - min_c
    L = (max_c + min_c) / 2.0
    
    S = np.zeros_like(L)
    mask = delta > 1e-6
    denom = 1.0 - np.abs(2.0 * L - 1.0)
    S[mask] = delta[mask] / (denom[mask] + 1e-6)
    S = np.clip(S, 0.0, 1.0)
    
    # --- Canal 3 : Magnitude du gradient de Sobel normalisée ---
    # Détection de contours Semaine 9
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    val_max = sobel_mag.max()
    if val_max > 0:
        sobel_mag = sobel_mag / val_max
        
    # --- Assemblage ---
    # Empilement des 3 caractéristiques (Gris, Saturation, Sobel)
    stacked = np.stack([gray, S, sobel_mag], axis=2) # 256 x 256 x 3
    
    # Transposer pour avoir le format PyTorch : C x H x W (3 x 256 x 256)
    tensor = stacked.transpose(2, 0, 1)
    return torch.from_numpy(tensor).float()

# =============================================================================
# 2. DATASET PYTORCH (COIN DATASET)
# =============================================================================
class CoinDataset(Dataset):
    def __init__(self, dossier_images, fichier_json, augment=False):
        self.augment = augment
        
        with open(fichier_json, 'r') as f:
            self.labels_dict = json.load(f)
            
        self.liste_images = sorted(list(self.labels_dict.keys()))
        
        # Optimisation : Pré-charger toutes les images pré-traitées en RAM.
        # Cela élimine le goulot d'étranglement CPU (chargement disque + prétraitement à la volée),
        # et permet au GPU de tourner en continu (GPU usage maximal).
        self.preloaded_data = []
        print(f"Pré-chargement de {len(self.liste_images)} images de '{dossier_images}' en RAM...")
        for nom_image in self.liste_images:
            chemin_image = os.path.join(dossier_images, nom_image)
            x = pretraiter_image(chemin_image)
            y = float(self.labels_dict[nom_image])
            self.preloaded_data.append((x, y))
            
    def __len__(self):
        return len(self.liste_images)
        
    def __getitem__(self, idx):
        x, y = self.preloaded_data[idx]
        
        # Cloner le tenseur pour éviter de modifier les données en RAM lors de l'augmentation
        x = x.clone()
        
        # Augmentation de données sur tenseur
        if self.augment:
            # Flips horizontaux / verticaux
            if random.random() > 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[1])
            # Rotation aléatoire de multiples de 90°
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                x = torch.rot90(x, k, [1, 2])
                
        return x, torch.tensor([y], dtype=torch.float32)

# =============================================================================
# 3. ARCHITECTURE DU MODÈLE (CUSTOM CNN FROM SCRATCH)
# =============================================================================
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Étape 1 : Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Étape 2 : Conv -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Étape 3 : Conv -> BN -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Étape 4 : Conv -> BN -> ReLU -> MaxPool
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        
        # Couches de régression
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1) # 1 sortie continue (Régression Semaine 12)
        
    def forward(self, x):
        # x : B x 3 x 256 x 256
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # B x 32 x 128 x 128
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # B x 64 x 64 x 64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # B x 128 x 32 x 32
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # B x 256 x 16 x 16
        
        # Global Average Pooling (GAP)
        # Moyenne globale spatiale pour conserver la robustesse et réduire les paramètres
        x = F.adaptive_avg_pool2d(x, (1, 1))             # B x 256 x 1 x 1
        x = torch.flatten(x, 1)                          # B x 256
        
        # Classification/Régression finale
        x = self.dropout(x)
        x = F.relu(self.fc1(x))                          # B x 128
        x = self.fc2(x)                                  # B x 1
        return x

# =============================================================================
# 4. SCRIPT D'ENTRAÎNEMENT PRINCIPAL
# =============================================================================
def entrainer_modele(epochs=120, batch_size=16, lr=1e-3):
    # Paramètres de périphériques (Semaine 12)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nom_gpu = torch.cuda.get_device_name(0)
        print(f"SUCCÈS : GPU détecté ! Modèle et données seront chargés sur : {nom_gpu}")
    else:
        device = torch.device("cpu")
        print("ATTENTION : Aucun GPU trouvé par PyTorch. Entraînement sur CPU.")
        
    # Chargement des bases distinctes (Protocole de Gestion des Données)
    train_dataset = CoinDataset("data/train", "data/train.json", augment=True)
    val_dataset = CoinDataset("data/validation", "data/validation.json", augment=False)
    
    # pin_memory=True accélère le transfert des tenseurs du CPU (RAM) vers le GPU (VRAM)
    utiliser_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=utiliser_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=utiliser_pin_memory)
    
    # Initialisation du modèle
    model = CustomCNN().to(device)
    
    # Fonction de perte : MSE (Erreur Quadratique Moyenne - Régression Semaine 12)
    criterion = nn.MSELoss()
    
    # Optimiseur : Adam (Descente de gradient optimisée)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Planificateur de taux d'apprentissage
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)
    
    meilleure_mae_val = float('inf')
    best_epoch = 0
    
    print("\nLancement de l'apprentissage du CNN...")
    print("-" * 50)
    
    for epoch in range(1, epochs + 1):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Message de débogage pour prouver l'utilisation du GPU sur l'interface
            if epoch == 1 and step == 0:
                print(f"--> [VÉRIFICATION MATÉRIELLE] Tenseur d'entrée : {inputs.device} | Tenseur cible : {targets.device} | Poids du modèle : {next(model.parameters()).device}\n")
            
            # Passe avant (Forward Pass)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
            # Passe arrière (Backward Pass) et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.sum(torch.abs(predictions - targets)).item()
            
        train_loss /= len(train_dataset)
        train_mae /= len(train_dataset)
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item() * inputs.size(0)
                # Métriques sur la validation arrondie (comme à l'évaluation finale)
                pred_rounded = torch.round(predictions)
                val_mae += torch.sum(torch.abs(pred_rounded - targets)).item()
                
        val_loss /= len(val_dataset)
        val_mae /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        # Affichage toutes les 5 époques ou à la première
        if epoch == 1 or epoch % 5 == 0 or val_mae < meilleure_mae_val:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.4f} | Train MAE: {train_mae:.2f} | Val MSE: {val_loss:.4f} | Val MAE (arrondie): {val_mae:.2f} | LR: {current_lr:.6f}")
            
        # Sauvegarde du meilleur modèle (Règle d'Or de validation)
        if val_mae < meilleure_mae_val:
            meilleure_mae_val = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), "meilleur_modele_nn.pth")
            
    print("-" * 50)
    print(f"Apprentissage terminé ! Meilleur modèle obtenu à l'époque {best_epoch} avec une MAE de validation de {meilleure_mae_val:.2f} pièces.")
    print("Fichier de poids sauvegardé sous : 'meilleur_modele_nn.pth'")

if __name__ == "__main__":
    # Si exécuté localement, on fait un petit entraînement de test rapide.
    # Sur Kaggle/Colab, augmentez le nombre d'époques (ex: 150)
    is_colab = "COLAB_GPU" in os.environ or "KAGGLE_URL" in os.environ
    epochs = 150 if is_colab else 3 # 3 epochs localement pour vérifier que le code tourne sans erreur
    entrainer_modele(epochs=epochs, batch_size=16, lr=1e-3)
