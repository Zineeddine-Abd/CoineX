"""
Pipeline de prétraitement partagé entre entraînement et inférence.
Toutes les étapes sont alignées sur les concepts du cours (Cours_Image.pdf).

STRATÉGIE : Prétraitement à la RÉSOLUTION ORIGINALE de chaque image,
            avec noyaux morphologiques scalés proportionnellement.
            Réduction unique à TARGET_SIZE (384x384) à la toute fin.

Avantages vs ancien pipeline (intermédiaire 512x512) :
  - Sobel/Otsu calculés sur l'image pleine résolution = bords ultra-nets
  - Le downsampling final (avec INTER_AREA = anti-aliasing) préserve la finesse
  - Morphologie scalée selon la taille = action consistante quelle que soit l'image

Le pipeline produit un tenseur 4 canaux (4 x 384 x 384) NON normalisé :
    Canal 0 : Luminance Y (formule du cours Semaine 4)
    Canal 1 : Saturation HLS (via cv2 - numériquement stable)
    Canal 2 : Magnitude du gradient de Sobel
    Canal 3 : Masque Otsu + ouverture + fermeture (binaire)

La normalisation par-canal (mean/std) est appliquée par le Dataset, pas ici.
"""

import cv2
import numpy as np
import torch

# Résolution finale envoyée au CNN
TARGET_SIZE = 384


def pretraiter_image_brut(chemin_image):
    """
    Pipeline complet, sans normalisation (le Dataset s'en charge).

    Étapes :
      1) Lecture à la résolution originale
      2) Pré-débruitage Gaussien (Semaine 9 - étape 1 de Canny)
      3) Égalisation d'histogramme sur V de HSV (Semaine 7)
      4) Calcul des 4 canaux à PLEINE résolution
      5) Morphologie avec noyaux scalés selon la taille d'image (Semaine 10)
      6) Réduction finale à 384x384 avec anti-aliasing (INTER_AREA)

    Retourne : torch.Tensor de forme (4, 384, 384), valeurs dans [0, 1].
    """
    img_bgr = cv2.imread(chemin_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {chemin_image}")

    # ----- 1) Pré-débruitage Gaussien (Semaine 9) -----
    # Petit noyau 5x5 : on cible le bruit pixel-par-pixel (capteur, JPEG)
    # qui existe à la même échelle absolue quelle que soit la résolution.
    img_bgr = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)

    # ----- 2) Égalisation V de HSV (Semaine 7) -----
    # Robustesse aux changements d'éclairage entre photos.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img_bgr_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)

    # ----- Canal 0 : Luminance Y (Semaine 4) -----
    rgb_f = img_rgb.astype(np.float32) / 255.0
    gray = (
        0.299 * rgb_f[:, :, 0]
        + 0.587 * rgb_f[:, :, 1]
        + 0.114 * rgb_f[:, :, 2]
    )

    # ----- Canal 1 : Saturation HLS (cv2, stable) -----
    hls = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2HLS)
    sat = hls[:, :, 2].astype(np.float32) / 255.0

    # ----- Canal 2 : Magnitude du gradient Sobel à PLEINE résolution -----
    # ksize=3 reste optimal : c'est un opérateur de dérivée, scale-invariant
    # dans son effet. C'est la résolution de l'IMAGE qui donne la finesse.
    gray_u8 = (gray * 255.0).astype(np.uint8)
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    sm_max = float(sobel_mag.max())
    if sm_max > 1e-6:
        sobel_mag = sobel_mag / sm_max

    # ----- Canal 3 : Otsu à pleine résolution, puis morphologie à 384 -----
    # On garde Otsu sur la haute résolution (calcul d'histogramme plus précis,
    # threshold optimal) MAIS on déplace la morphologie après le downsampling.
    # Pourquoi : un noyau morpho de 117x117 sur une image 4Mpx prend ~8s.
    # Comme on downsample à 384 à la fin de toute façon, les détails morpho
    # plus fins que 1/384e de l'image sont perdus → autant les faire à 384.
    _, otsu_hi = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Forcer 'pièces = 1' (foreground minoritaire en général)
    if (otsu_hi > 0).mean() > 0.5:
        otsu_hi = 255 - otsu_hi

    # ----- Réduction finale à 384x384 (avant morphologie pour Otsu) -----
    # INTER_AREA = anti-aliasing correct pour downsampling.
    target = (TARGET_SIZE, TARGET_SIZE)
    gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
    sat = cv2.resize(sat, target, interpolation=cv2.INTER_AREA)
    sobel_mag = cv2.resize(sobel_mag, target, interpolation=cv2.INTER_AREA)
    # Otsu : downsample puis re-binarise pour récupérer un vrai masque binaire
    otsu_small = cv2.resize(otsu_hi, target, interpolation=cv2.INTER_AREA)
    _, otsu_bin = cv2.threshold(otsu_small, 127, 255, cv2.THRESH_BINARY)

    # Morphologie à 384x384 (rapide, et c'est l'échelle de l'entrée du CNN)
    # Noyaux 5 et 11 = ~1.3% et 2.9% de la largeur → cohérent avec coins ~30-50px
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, k_close)
    otsu_mask = otsu_bin.astype(np.float32) / 255.0

    # Empilement C x H x W (format PyTorch)
    stacked = np.stack([gray, sat, sobel_mag, otsu_mask], axis=0).astype(np.float32)
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
