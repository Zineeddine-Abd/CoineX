"""
Prétraitement à 5 canaux pour le pipeline NN.

Chaîne complète :
  1. Lecture image à résolution originale
  2. Flou Gaussien 5x5 (débruitage pixel, étape 1 de Canny)
  3. Égalisation V de HSV (robustesse éclairage)
  4. Calcul des 5 canaux :
       0. Luminance Y                
       1. Saturation HLS via cv2     
       2. Magnitude Sobel            
       3. Otsu + Morphologie         (binaire)
       4. Canny (gray non-équalisé)  (binaire)
  5. Réduction unique à 384x384 (INTER_AREA = anti-aliasing)
  6. Empilement en tenseur (5, 384, 384) dans [0, 1]

Le tenseur retourné n'est PAS normalisé : la normalisation par-canal
(mean/std du train set) est faite plus tard par le Dataset / l'inférence.
"""
import cv2
import numpy as np
import torch


# Résolution finale envoyée au CNN
TARGET_SIZE = 384

# Nombre de canaux produits (à utiliser dans Dataset / CNN / checkpoint)
NUM_CHANNELS = 5


def pretraiter_image_brut(chemin_image):
    """
    Pipeline complet, sans normalisation.

    Retourne : torch.Tensor (5, 384, 384), valeurs dans [0, 1].
    """
    img_bgr = cv2.imread(chemin_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {chemin_image}")

    h, w = img_bgr.shape[:2]
    # Facteur d'échelle : dimensionne la dilatation Canny pour que les bords
    # fins survivent au downsampling vers 384.
    scale = max(h, w) / TARGET_SIZE

    # ----- 1) Pré-débruitage Gaussien (étape 1 de Canny) -----
    # Cible le bruit pixel-par-pixel (capteur, JPEG) à l'échelle absolue.
    img_bgr = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)

    # On garde une version "brute" (juste débruitée) pour Canny.
    # L'égalisation amplifie les micro-gradients en faux bords → Canny détecte
    # 80%+ de pixels comme "bords". Mieux vaut détecter les bords naturels.
    gray_raw_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ----- 2) Égalisation V de HSV -----
    # Robustesse aux changements d'éclairage entre photos.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img_bgr_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)

    # ----- Canal 0 : Luminance Y -----
    rgb_f = img_rgb.astype(np.float32) / 255.0
    gray = (
        0.299 * rgb_f[:, :, 0]
        + 0.587 * rgb_f[:, :, 1]
        + 0.114 * rgb_f[:, :, 2]
    )
    gray_u8 = (gray * 255.0).astype(np.uint8)

    # ----- Canal 1 : Saturation HLS (cv2 - stable) -----
    # Remplace la formule manuelle qui pouvait diviser par zéro
    # quand L approche 0 ou 1.
    hls = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2HLS)
    sat = hls[:, :, 2].astype(np.float32) / 255.0

    # ----- Canal 2 : Magnitude Sobel à PLEINE résolution -----
    # ksize=3 reste optimal (opérateur de dérivée scale-invariant).
    # La résolution de l'IMAGE donne la finesse, pas le noyau.
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    sm_max = float(sobel_mag.max())
    if sm_max > 1e-6:
        sobel_mag = sobel_mag / sm_max

    # ----- Canal 3 : Otsu à pleine résolution -----
    # Threshold optimal calculé sur l'histogramme plein res (image égalisée).
    _, otsu_hi = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Forcer 'pièces = 1' (foreground minoritaire en général)
    if (otsu_hi > 0).mean() > 0.5:
        otsu_hi = 255 - otsu_hi

    # ----- Canal 4 : Canny à pleine résolution (algo complet) -----
    # Canny = 4 étapes : 1) flou (déjà fait), 2) gradient, 3) suppression
    # des non-maxima (affine à 1 px), 4) seuillage par hystérésis (Sem. 5).
    # Donne des bords binaires ULTRA-NETS (contrairement à Sobel grayscale).
    # Important : calculé sur l'image NON-égalisée pour éviter que les
    # micro-gradients amplifiés ne soient classés comme bords.
    canny_hi = cv2.Canny(gray_raw_u8, 100, 200)

    # Dilatation légère pour préserver les bords (1-2 px) à travers le downsampling.
    dilate_radius = max(1, int(round(scale / 4.0)))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
    )
    canny_hi = cv2.dilate(canny_hi, dilate_kernel, iterations=1)

    # ----- Réduction à 384x384 (INTER_AREA = anti-aliasing) -----
    tgt = (TARGET_SIZE, TARGET_SIZE)
    gray = cv2.resize(gray, tgt, interpolation=cv2.INTER_AREA)
    sat = cv2.resize(sat, tgt, interpolation=cv2.INTER_AREA)
    sobel_mag = cv2.resize(sobel_mag, tgt, interpolation=cv2.INTER_AREA)

    # Otsu et Canny : downsample puis re-binarise pour récupérer un vrai masque {0, 1}
    otsu_small = cv2.resize(otsu_hi, tgt, interpolation=cv2.INTER_AREA)
    _, otsu_bin = cv2.threshold(otsu_small, 127, 255, cv2.THRESH_BINARY)
    canny_small = cv2.resize(canny_hi, tgt, interpolation=cv2.INTER_AREA)
    _, canny_bin = cv2.threshold(canny_small, 127, 255, cv2.THRESH_BINARY)

    # ----- Morphologie sur Otsu à 384x384 -----
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
        print(f"Min par canal : {t.amin(dim=(1,2)).tolist()}")
        print(f"Max par canal : {t.amax(dim=(1,2)).tolist()}")
        print(f"Mean par canal: {t.mean(dim=(1,2)).tolist()}")
    else:
        print("Usage : python -m pipelines.nn.pretraitement <chemin_image>")
