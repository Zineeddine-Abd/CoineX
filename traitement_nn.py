import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fallback sur la méthode classique si les poids ne sont pas encore entraînés
try:
    from traitement import compter_pieces as compter_pieces_classique
except ImportError:
    def compter_pieces_classique(chemin_image, *args, **kwargs):
        print("[ERREUR] Méthode classique non trouvable.")
        return 0

# =============================================================================
# 1. PRÉTRAITEMENT DE L'IMAGE
# =============================================================================
def pretraiter_image_inference(chemin_image):
    """
    Applique le même pipeline de prétraitement qu'à l'entraînement :
    Luminance + Saturation HSL + Sobel -> 3 canaux -> 256x256
    """
    img_bgr = cv2.imread(chemin_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {chemin_image}")
        
    # Optimisation de performance : Redimensionner en 256x256 d'abord
    # pour accélérer les calculs d'inférence CPU d'un facteur 110x
    img_bgr_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    
    # 1) Gris Luminance
    gray = ((0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]) / 255.0).astype(np.float32)
    
    # 2) Saturation HSL
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
    
    # 3) Magnitude de Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    val_max = sobel_mag.max()
    if val_max > 0:
        sobel_mag = sobel_mag / val_max
        
    # Assemblage
    stacked = np.stack([gray, S, sobel_mag], axis=2) # 256 x 256 x 3
    
    # Transposition vers format PyTorch (C x H x W)
    tensor = stacked.transpose(2, 0, 1)
    
    # Ajouter la dimension de batch (1 x C x H x W)
    tensor = np.expand_dims(tensor, axis=0)
    return torch.from_numpy(tensor).float()

# =============================================================================
# 2. ARCHITECTURE DU MODÈLE (CUSTOM CNN)
# =============================================================================
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Conv + BN + ReLU + MaxPool blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        
        # Régression
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instance globale du modèle (chargé paresseusement lors du premier appel)
_model_instance = None
_model_failed = False

def get_model(chemin_poids="meilleur_modele_nn.pth"):
    global _model_instance, _model_failed
    
    if _model_failed:
        return None
        
    if _model_instance is not None:
        return _model_instance
        
    if not os.path.exists(chemin_poids):
        print(f"\n[ATTENTION] Fichier de poids '{chemin_poids}' introuvable.")
        print("-> Veuillez d'abord lancer l'entraînement sur Colab/Kaggle en utilisant 'entrainer_nn.py'")
        print("-> Téléchargez ensuite le fichier 'meilleur_modele_nn.pth' et placez-le dans ce dossier.")
        print("-> Fallback automatique vers le pipeline classique de traitement d'images...\n")
        _model_failed = True
        return None
        
    try:
        model = CustomCNN()
        # Charger les poids sur le CPU (Inférence locale)
        model.load_state_dict(torch.load(chemin_poids, map_location=torch.device('cpu')))
        model.eval()
        _model_instance = model
        return _model_instance
    except Exception as e:
        print(f"[ERREUR] Échec du chargement du modèle CNN : {e}")
        _model_failed = True
        return None

# =============================================================================
# 3. INTERFACE DE COMPTAGE DES PIÈCES
# =============================================================================
def compter_pieces(chemin_image, *args, **kwargs):
    """
    Interface compatible avec evaluation.py.
    Utilise le modèle CNN si disponible, sinon bascule sur l'approche classique.
    """
    model = get_model()
    
    # Fallback si le modèle n'est pas disponible / pas encore entraîné
    if model is None:
        # On passe la taille du flou gaussien si elle est fournie
        taille_flou = kwargs.get("taille_flou", (7, 7))
        if len(args) > 0:
            taille_flou = args[0]
        return compter_pieces_classique(chemin_image, taille_flou)
        
    try:
        # Inférence avec le CNN
        with torch.no_grad():
            x = pretraiter_image_inference(chemin_image)
            prediction = model(x)
            
            # Récupérer la valeur scalaire
            valeur_predite = prediction.item()
            
            # Régression : arrondir à l'entier le plus proche et forcer >= 0 (Semaine 12)
            nombre_pieces = max(0, int(round(valeur_predite)))
            return nombre_pieces
    except Exception as e:
        print(f"[ERREUR] Inférence CNN échouée pour {chemin_image} : {e}. Fallback classique...")
        return compter_pieces_classique(chemin_image)
