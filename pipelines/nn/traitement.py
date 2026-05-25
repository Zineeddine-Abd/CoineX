"""
Pipeline NN - entrée d'inférence (interface compatible avec evaluation.py).

Expose `compter_pieces(chemin_image, **kwargs)` qui :
  1. Charge le checkpoint paresseusement (1 fois par session)
  2. Prétraite l'image en 5 canaux
  3. Normalise avec les stats stockées dans le checkpoint
  4. Applique Test-Time Augmentation (TTA) sur 8 transformations D4
  5. Moyenne les prédictions et arrondit

Si le checkpoint est absent ou incompatible, bascule vers la méthode
morphologie pour ne pas casser l'évaluation.
"""
import os

import torch
import torch.nn.functional as F

from pipelines.nn.pretraitement import pretraiter_image_brut
from pipelines.nn.modele import CustomCNN


# Chemins possibles pour le checkpoint
# 1. à la racine du projet (où l'utilisateur dépose le .pth téléchargé de Kaggle)
# 2. dans pipelines/nn/ (alternative locale)
CHEMIN_POIDS_CANDIDATS = [
    "meilleur_modele_nn.pth",
    os.path.join("pipelines", "nn", "meilleur_modele_nn.pth"),
]


# Fallback : pipeline morphologie si le modèle NN n'est pas disponible
try:
    from pipelines.morphologie.traitement import compter_pieces as compter_pieces_fallback
except ImportError:
    def compter_pieces_fallback(chemin_image, **kwargs):
        print("[ERREUR] Pipeline de fallback (morphologie) introuvable.")
        return 0


# =============================================================================
# Chargement paresseux du modèle + statistiques de normalisation
# =============================================================================
_model_instance = None
_norm_stats = None          # tuple (mean tensor 5x1x1, std tensor 5x1x1)
_model_failed = False


def _trouver_checkpoint():
    """Cherche le .pth dans les emplacements connus, renvoie le chemin trouvé ou None."""
    for c in CHEMIN_POIDS_CANDIDATS:
        if os.path.exists(c):
            return c
    return None


def get_model():
    """Charge le checkpoint (poids + stats de normalisation). Retourne None si KO."""
    global _model_instance, _norm_stats, _model_failed

    if _model_failed:
        return None
    if _model_instance is not None:
        return _model_instance

    chemin = _trouver_checkpoint()
    if chemin is None:
        print(f"\n[ATTENTION] Fichier de poids 'meilleur_modele_nn.pth' introuvable.")
        print("-> Entraînez d'abord avec pipelines/nn/coinex-nn-pipeline.ipynb sur Kaggle.")
        print("-> Téléchargez ensuite 'meilleur_modele_nn.pth' à la racine du projet.")
        print("-> Fallback automatique vers le pipeline morphologie.\n")
        _model_failed = True
        return None

    try:
        ckpt = torch.load(chemin, map_location=torch.device('cpu'))

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            in_ch = ckpt.get('in_channels', 5)
            model = CustomCNN(in_channels=in_ch)
            model.load_state_dict(ckpt['model_state_dict'])

            mean = torch.tensor(ckpt['mean'], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(ckpt['std'], dtype=torch.float32).view(-1, 1, 1)
            _norm_stats = (mean, std)

            val_mae = ckpt.get('val_mae', None)
            if val_mae is not None:
                print(f"[INFO] Modèle NN chargé (Val MAE = {val_mae:.2f}, epoch {ckpt.get('epoch', '?')}).")
        else:
            print("[ATTENTION] Checkpoint en ancien format (incompatible avec le pipeline 5 canaux).")
            print("-> Réentraînez avec coinex-nn-pipeline.ipynb.\n")
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
    """Normalisation par-canal avec les stats stockées dans le checkpoint."""
    mean, std = _norm_stats
    return (x - mean) / std


def _predire_avec_tta(model, x_norm):
    """
    Test-Time Augmentation : moyenne sur les 8 transformations D4 du groupe diédral
    (4 rotations × 2 flips horizontaux = 8 variantes lossless).
    Réduit la variance par √8 ≈ 2.83x.
    """
    predictions = []
    with torch.no_grad():
        for k in range(4):                     # rotations 0°, 90°, 180°, 270°
            for flip in (False, True):         # avec/sans flip horizontal
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
def compter_pieces(chemin_image, **kwargs):
    """
    Compte le nombre de pièces dans l'image.
    Utilise le CNN si disponible, sinon bascule sur la morphologie.

    Signature compatible avec evaluation.py (accepte **kwargs comme les autres pipelines).
    """
    model = get_model()

    if model is None:
        # Fallback gracieux vers la morphologie
        return compter_pieces_fallback(chemin_image, **kwargs)

    try:
        # 1) Prétraitement (5 canaux, valeurs [0, 1])
        x = pretraiter_image_brut(chemin_image)         # (5, 384, 384)
        x = x.unsqueeze(0)                              # (1, 5, 384, 384)

        # 2) Normalisation par-canal (stats embarquées dans le checkpoint)
        x = _normaliser(x)

        # 3) Inférence + TTA (8 variantes lossless)
        prediction = _predire_avec_tta(model, x)

        # 4) Régression → entier positif
        return max(0, int(round(prediction)))

    except Exception as e:
        print(f"[ERREUR] Inférence CNN échouée pour {chemin_image} : {e}. Fallback morphologie.")
        return compter_pieces_fallback(chemin_image, **kwargs)
