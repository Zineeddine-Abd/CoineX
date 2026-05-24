"""
Inférence du CNN de comptage de pièces.

Améliorations clés par rapport à la version précédente :
  - Pipeline 4 canaux partagé via pipeline_traitement.py
  - Chargement automatique des statistiques de normalisation depuis le checkpoint
  - Test-Time Augmentation (TTA) : moyenne sur 8 transformations dihédrales
  - Fallback gracieux vers le pipeline classique si le modèle est absent
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline_traitement import pretraiter_image_brut

# Fallback sur le pipeline classique (traitement.py) si le modèle est indisponible
try:
    from traitement import compter_pieces as compter_pieces_classique
except ImportError:
    def compter_pieces_classique(chemin_image, *args, **kwargs):
        print("[ERREUR] Méthode classique non importable.")
        return 0


# =============================================================================
# ARCHITECTURE - doit correspondre exactement à entrainer_nn.py
# =============================================================================
class CustomCNN(nn.Module):
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

        self.b1 = block(in_channels, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.b4 = block(128, 256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Chargement paresseux du modèle + statistiques de normalisation
# =============================================================================
_model_instance = None
_norm_stats = None          # tuple (mean tensor 4x1x1, std tensor 4x1x1)
_model_failed = False


def get_model(chemin_poids="meilleur_modele_nn.pth"):
    global _model_instance, _norm_stats, _model_failed

    if _model_failed:
        return None
    if _model_instance is not None:
        return _model_instance

    if not os.path.exists(chemin_poids):
        print(f"\n[ATTENTION] Fichier de poids '{chemin_poids}' introuvable.")
        print("-> Lancez d'abord entrainer_nn.py sur Kaggle/Colab.")
        print("-> Téléchargez ensuite 'meilleur_modele_nn.pth' dans ce dossier.")
        print("-> Fallback automatique vers le pipeline classique.\n")
        _model_failed = True
        return None

    try:
        ckpt = torch.load(chemin_poids, map_location=torch.device('cpu'))

        # Nouveau format : dict avec 'model_state_dict', 'mean', 'std'
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            in_ch = ckpt.get('in_channels', 4)
            model = CustomCNN(in_channels=in_ch)
            model.load_state_dict(ckpt['model_state_dict'])

            mean = torch.tensor(ckpt['mean'], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(ckpt['std'], dtype=torch.float32).view(-1, 1, 1)
            _norm_stats = (mean, std)

            val_mae = ckpt.get('val_mae', None)
            if val_mae is not None:
                print(f"[INFO] Modèle chargé (Val MAE = {val_mae:.2f}, epoch {ckpt.get('epoch', '?')}).")
        else:
            # Ancien format détecté → l'architecture est incompatible
            print("[ATTENTION] Checkpoint en ancien format (incompatible avec le nouveau pipeline 4 canaux).")
            print("-> Réentraînez avec entrainer_nn.py.\n")
            _model_failed = True
            return None

        model.eval()
        _model_instance = model
        return _model_instance

    except Exception as e:
        print(f"[ERREUR] Échec du chargement du modèle : {e}")
        _model_failed = True
        return None


def _normaliser(x):
    """Applique la normalisation par-canal stockée dans le checkpoint."""
    mean, std = _norm_stats
    return (x - mean) / std


def _predire_avec_tta(model, x_norm):
    """
    Test-Time Augmentation : moyenne des prédictions sur les 8 transformations
    du groupe diédral D4 (4 rotations x 2 flips = 8 variantes).
    Robuste aux symétries qui pourraient être absentes du jeu d'entraînement.
    """
    predictions = []
    with torch.no_grad():
        for k in range(4):
            for flip in (False, True):
                x_aug = x_norm
                if k > 0:
                    x_aug = torch.rot90(x_aug, k, dims=[2, 3])
                if flip:
                    x_aug = torch.flip(x_aug, dims=[3])
                pred = model(x_aug).item()
                predictions.append(pred)
    return sum(predictions) / len(predictions)


# =============================================================================
# INTERFACE PUBLIQUE - compatible avec evaluation.py
# =============================================================================
def compter_pieces(chemin_image, *args, **kwargs):
    """
    Compte le nombre de pièces dans l'image.
    Utilise le CNN si disponible, sinon bascule sur le pipeline classique.

    Signature compatible avec evaluation.py :
        compter_pieces(chemin, taille_flou)
    """
    model = get_model()

    if model is None:
        # Fallback : on transmet la taille du flou au pipeline classique
        taille_flou = args[0] if args else kwargs.get("taille_flou", (7, 7))
        return compter_pieces_classique(chemin_image, taille_flou)

    try:
        # 1) Prétraitement (4 canaux, dans [0, 1])
        x = pretraiter_image_brut(chemin_image)         # 4 x 256 x 256
        x = x.unsqueeze(0)                              # 1 x 4 x 256 x 256

        # 2) Normalisation par-canal
        x = _normaliser(x)

        # 3) Inférence + TTA (8 variantes)
        prediction = _predire_avec_tta(model, x)

        # 4) Régression → entier positif (Semaine 12)
        return max(0, int(round(prediction)))

    except Exception as e:
        print(f"[ERREUR] Inférence CNN échouée pour {chemin_image} : {e}. Fallback classique.")
        return compter_pieces_classique(chemin_image)
