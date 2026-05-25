"""
Pipeline de prétraitement partagé entre entraînement et inférence.
Toutes les étapes sont alignées sur les concepts du cours (Cours_Image.pdf).

STRATÉGIE : Prétraitement à la RÉSOLUTION ORIGINALE de chaque image,
            réduction unique à TARGET_SIZE (384x384) à la toute fin.

Le pipeline produit un tenseur 5 canaux (5 x 384 x 384) NON normalisé :
    Canal 0 : Luminance Y (Semaine 4)                  [continu]
    Canal 1 : Saturation HLS (cv2 - stable)             [continu]
    Canal 2 : Magnitude du gradient Sobel               [continu]
    Canal 3 : Masque Otsu + Ouverture + Fermeture       [binaire]
    Canal 4 : Détection de bords Canny (4 étapes)       [binaire]

Les canaux 0-2 sont continus (sensibles aux augmentations photométriques).
Les canaux 3-4 sont binaires (protégés des augmentations photométriques).

La normalisation par-canal (mean/std) est appliquée par le Dataset, pas ici.
"""

import cv2
import numpy as np
import torch

# Résolution finale envoyée au CNN
TARGET_SIZE = 384

# Nombre de canaux produits par le pipeline (à utiliser dans Dataset et CNN)
NUM_CHANNELS = 5


def pretraiter_image_brut(chemin_image):
    """
    Pipeline complet, sans normalisation (le Dataset s'en charge).

    Étapes :
      1) Lecture à la résolution originale
      2) Pré-débruitage Gaussien (Semaine 9 - étape 1 de Canny)
      3) Égalisation d'histogramme sur V de HSV (Semaine 7)
      4) Calcul des 5 canaux :
         - Luminance, Saturation, Sobel : à pleine résolution
         - Otsu + Canny : à pleine résolution
      5) Réduction à 384x384 (INTER_AREA = anti-aliasing)
      6) Morphologie Open+Close à 384x384 sur le masque Otsu (Semaine 10)

    Retourne : torch.Tensor (5, 384, 384), valeurs dans [0, 1].
    """
    img_bgr = cv2.imread(chemin_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {chemin_image}")

    # ----- 1) Pré-débruitage Gaussien (Semaine 9) -----
    # Cible le bruit pixel-par-pixel (capteur, JPEG) à l'échelle absolue.
    img_bgr = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)

    # On garde une version "brute" (juste débruitée) pour Canny.
    # L'égalisation amplifie les micro-gradients en faux bords → Canny détecte
    # 80%+ de pixels comme "bords". Mieux vaut détecter les bords naturels.
    gray_raw_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ----- 2) Égalisation V de HSV (Semaine 7) -----
    # Robustesse aux changements d'éclairage entre photos.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img_bgr_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)

    # ----- Canal 0 : Luminance Y (sur image égalisée, Semaine 4) -----
    rgb_f = img_rgb.astype(np.float32) / 255.0
    gray = (
        0.299 * rgb_f[:, :, 0]
        + 0.587 * rgb_f[:, :, 1]
        + 0.114 * rgb_f[:, :, 2]
    )
    gray_u8 = (gray * 255.0).astype(np.uint8)

    # ----- Canal 1 : Saturation HLS (cv2 - stable) -----
    hls = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2HLS)
    sat = hls[:, :, 2].astype(np.float32) / 255.0

    # ----- Canal 2 : Magnitude Sobel à PLEINE résolution (Semaine 9) -----
    # ksize=3 reste optimal (opérateur de dérivée scale-invariant).
    # La résolution de l'IMAGE donne la finesse, pas le noyau.
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    sm_max = float(sobel_mag.max())
    if sm_max > 1e-6:
        sobel_mag = sobel_mag / sm_max

    # ----- Canal 3 : Otsu à pleine résolution (Semaines 5-6) -----
    # Threshold optimal calculé sur l'histogramme plein res (image égalisée).
    _, otsu_hi = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Forcer 'pièces = 1' (foreground minoritaire en général)
    if (otsu_hi > 0).mean() > 0.5:
        otsu_hi = 255 - otsu_hi

    # ----- Réduction à 384x384 -----
    tgt = (TARGET_SIZE, TARGET_SIZE)
    gray = cv2.resize(gray, tgt, interpolation=cv2.INTER_AREA)
    sat = cv2.resize(sat, tgt, interpolation=cv2.INTER_AREA)
    sobel_mag = cv2.resize(sobel_mag, tgt, interpolation=cv2.INTER_AREA)
    # Image grise non-égalisée downsamplée (pour Canny)
    gray_raw_384 = cv2.resize(gray_raw_u8, tgt, interpolation=cv2.INTER_AREA)

    # Otsu : downsample puis re-binarise pour récupérer un vrai masque {0, 1}
    otsu_small = cv2.resize(otsu_hi, tgt, interpolation=cv2.INTER_AREA)
    _, otsu_bin = cv2.threshold(otsu_small, 127, 255, cv2.THRESH_BINARY)

    # ----- Canal 4 : Canny à 384x384 (Semaine 9 - algo complet) -----
    # Canny = 4 étapes : 1) flou Gaussien (déjà fait), 2) gradient,
    # 3) suppression des non-maxima (affine à 1 px), 4) seuillage hystérésis.
    # Donne des bords binaires ULTRA-NETS (contrairement à Sobel grayscale).
    #
    # On calcule Canny DIRECTEMENT à 384x384 (et non au plein res + dilatation) :
    # - évite l'épaississement artificiel par dilatation
    # - les bords sont déjà à la bonne échelle (1 px = 1 px d'entrée du CNN)
    # - cv2.Canny renvoie déjà du binaire {0, 255}
    #
    # Seuils fixes 100/200 (standard Canny) appliqués sur l'image NON-égalisée.
    # L'égalisation amplifie les micro-gradients en faux bords ; sur image brute,
    # on détecte les bords NATURELS (pourtours des pièces, principalement).
    canny_bin = cv2.Canny(gray_raw_384, 100, 200)

    # ----- Morphologie sur Otsu à 384x384 (Semaine 10) -----
    # Noyaux 5 et 11 = ~1.3% et 2.9% de la largeur, cohérent avec coins ~30-50px
    # Ouverture (supprime bruit isolé) puis fermeture (bouche trous gravures)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, k_close)
    otsu_mask = otsu_bin.astype(np.float32) / 255.0

    # Canny : pas de morphologie - on veut garder les bords FINS (qualité Canny)
    canny_mask = canny_bin.astype(np.float32) / 255.0

    # ----- Empilement final : 5 canaux -----
    stacked = np.stack(
        [gray, sat, sobel_mag, otsu_mask, canny_mask], axis=0
    ).astype(np.float32)
    return torch.from_numpy(stacked)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        t = pretraiter_image_brut(sys.argv[1])
        print(f"Shape : {tuple(t.shape)}")
        print(f"Min/Max par canal : {t.amin(dim=(1,2)).tolist()} / {t.amax(dim=(1,2)).tolist()}")
        print(f"Mean par canal : {t.mean(dim=(1,2)).tolist()}")
    else:
        print("Usage : python pipeline_traitement.py <chemin_image>")
