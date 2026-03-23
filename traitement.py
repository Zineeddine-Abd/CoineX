import math
from collections import deque

import matplotlib.image as mpimg
import numpy as np


MAX_DIMENSION = 520


def lire_image_rgb(chemin_image):
    try:
        image = mpimg.imread(chemin_image)
    except FileNotFoundError:
        return None

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)

    if image.shape[2] > 3:
        image = image[..., :3]

    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    hauteur, largeur = image.shape[:2]
    echelle = min(1.0, MAX_DIMENSION / max(largeur, hauteur))
    if echelle < 1.0:
        image = redimensionner_bilineaire(
            image,
            max(1, int(round(hauteur * echelle))),
            max(1, int(round(largeur * echelle))),
        )

    return image


def redimensionner_bilineaire(image, nouvelle_hauteur, nouvelle_largeur):
    hauteur, largeur = image.shape[:2]
    if nouvelle_hauteur == hauteur and nouvelle_largeur == largeur:
        return image.copy()

    y = np.linspace(0, hauteur - 1, nouvelle_hauteur, dtype=np.float32)
    x = np.linspace(0, largeur - 1, nouvelle_largeur, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, largeur - 1)
    y1 = np.clip(y0 + 1, 0, hauteur - 1)

    wx = xx - x0
    wy = yy - y0

    image = image.astype(np.float32)
    haut_gauche = image[y0, x0]
    haut_droite = image[y0, x1]
    bas_gauche = image[y1, x0]
    bas_droite = image[y1, x1]

    haut = haut_gauche * (1.0 - wx)[..., None] + haut_droite * wx[..., None]
    bas = bas_gauche * (1.0 - wx)[..., None] + bas_droite * wx[..., None]
    resultat = haut * (1.0 - wy)[..., None] + bas * wy[..., None]
    return np.clip(np.rint(resultat), 0, 255).astype(np.uint8)


def rgb_vers_hsl(image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    maximum = np.max(image, axis=2)
    minimum = np.min(image, axis=2)
    delta = maximum - minimum

    luminosite = (maximum + minimum) / 2.0
    saturation = np.zeros_like(luminosite)

    masque = delta > 1e-6
    denominateur = 1.0 - np.abs(2.0 * luminosite - 1.0)
    saturation[masque] = delta[masque] / (denominateur[masque] + 1e-6)

    return luminosite, np.clip(saturation, 0.0, 1.0)


def rgb_vers_gris(image_rgb):
    image = image_rgb.astype(np.float32)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ) / 255.0


def noyau_gaussien_1d(taille, sigma):
    rayon = taille // 2
    axe = np.arange(-rayon, rayon + 1, dtype=np.float32)
    noyau = np.exp(-(axe * axe) / (2.0 * sigma * sigma))
    somme = np.sum(noyau)
    if somme > 0:
        noyau /= somme
    return noyau


def convolution_1d_lignes(image, noyau):
    pad = len(noyau) // 2
    image_pad = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=1)
    return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)


def convolution_1d_colonnes(image, noyau):
    pad = len(noyau) // 2
    image_pad = np.pad(image, ((pad, pad), (0, 0)), mode="edge")
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, len(noyau), axis=0)
    return np.einsum("ijk,k->ij", fenetres, noyau[::-1], optimize=True)


def flou_gaussien(image, taille):
    taille = max(3, int(taille))
    if taille % 2 == 0:
        taille += 1
    sigma = max(1.0, taille / 3.0)
    noyau = noyau_gaussien_1d(taille, sigma)
    image = image.astype(np.float32)
    image = convolution_1d_lignes(image, noyau)
    return convolution_1d_colonnes(image, noyau)


def histogramme_u8(image):
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return np.bincount(image_u8.ravel(), minlength=256).astype(np.float64)


def seuil_otsu(image):
    hist = histogramme_u8(image)
    total = hist.sum()
    if total == 0:
        return 0.5

    probabilites = hist / total
    cumul_prob = np.cumsum(probabilites)
    cumul_moy = np.cumsum(probabilites * np.arange(256))
    moyenne_totale = cumul_moy[-1]

    variance_inter = (moyenne_totale * cumul_prob - cumul_moy) ** 2
    variance_inter /= cumul_prob * (1.0 - cumul_prob) + 1e-12

    seuil = int(np.argmax(variance_inter))
    return seuil / 255.0


def erosion_binaire(image_binaire, taille):
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool), ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))
    return np.all(fenetres, axis=(2, 3))


def dilatation_binaire(image_binaire, taille):
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    image_pad = np.pad(
        image_binaire.astype(bool), ((pad, pad), (pad, pad)), mode="constant", constant_values=False
    )
    fenetres = np.lib.stride_tricks.sliding_window_view(image_pad, (taille, taille))
    return np.any(fenetres, axis=(2, 3))


def ouverture_binaire(image_binaire, taille):
    return dilatation_binaire(erosion_binaire(image_binaire, taille), taille)


def composantes_connexes(image_binaire):
    masque = image_binaire.astype(bool)
    hauteur, largeur = masque.shape
    etiquettes = np.zeros((hauteur, largeur), dtype=np.int32)
    composantes = []
    etiquette = 0
    voisins = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for y in range(hauteur):
        for x in range(largeur):
            if not masque[y, x] or etiquettes[y, x] != 0:
                continue

            etiquette += 1
            file_pixels = deque([(y, x)])
            etiquettes[y, x] = etiquette
            pixels_y = []
            pixels_x = []
            touche_bord = False

            while file_pixels:
                cy, cx = file_pixels.popleft()
                pixels_y.append(cy)
                pixels_x.append(cx)

                if cy == 0 or cx == 0 or cy == hauteur - 1 or cx == largeur - 1:
                    touche_bord = True

                for dy, dx in voisins:
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < hauteur
                        and 0 <= nx < largeur
                        and masque[ny, nx]
                        and etiquettes[ny, nx] == 0
                    ):
                        etiquettes[ny, nx] = etiquette
                        file_pixels.append((ny, nx))

            pixels_y = np.asarray(pixels_y, dtype=np.int32)
            pixels_x = np.asarray(pixels_x, dtype=np.int32)
            area = int(pixels_y.size)
            y_min = int(pixels_y.min())
            y_max = int(pixels_y.max())
            x_min = int(pixels_x.min())
            x_max = int(pixels_x.max())
            hauteur_bbox = y_max - y_min + 1
            largeur_bbox = x_max - x_min + 1

            sous_masque = etiquettes[y_min : y_max + 1, x_min : x_max + 1] == etiquette
            contour = sous_masque & ~erosion_binaire(sous_masque, 3)
            perimetre = int(np.count_nonzero(contour))
            remplissage = area / float(hauteur_bbox * largeur_bbox)
            circularite = 4.0 * math.pi * area / max(1.0, perimetre * perimetre)

            composantes.append(
                {
                    "label": etiquette,
                    "area": area,
                    "bbox": (y_min, x_min, y_max, x_max),
                    "hauteur_bbox": hauteur_bbox,
                    "largeur_bbox": largeur_bbox,
                    "remplissage": remplissage,
                    "circularite": circularite,
                    "touche_bord": touche_bord,
                }
            )

    return composantes


def extraire_composantes_utiles(masque, aire_min, aire_max):
    composantes = composantes_connexes(masque)
    return [
        comp
        for comp in composantes
        if aire_min <= comp["area"] <= aire_max and not comp["touche_bord"]
    ]


def estimer_nombre_depuis_composantes(composantes):
    if not composantes:
        return 0

    aires_reference = [
        comp["area"]
        for comp in composantes
        if comp["circularite"] >= 0.45
        and comp["remplissage"] >= 0.45
        and 0.65
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= 1.55
    ]

    if not aires_reference:
        return len(composantes)

    aire_reference = float(np.median(aires_reference))
    compteur = 0

    for comp in composantes:
        ratio = comp["area"] / max(1.0, aire_reference)
        if ratio >= 1.8 and comp["remplissage"] >= 0.35:
            compteur += max(1, int(round(ratio)))
        else:
            compteur += 1

    return compteur


def detection_principale(image_rgb, taille_flou):
    _, saturation = rgb_vers_hsl(image_rgb)
    taille_locale = max(
        int(round((taille_flou[0] + taille_flou[1]) / 2.0)),
        int(round(min(image_rgb.shape[:2]) / 90)),
        5,
    )
    if taille_locale % 2 == 0:
        taille_locale += 1

    saturation_floue = flou_gaussien(saturation, taille_locale)
    seuil = seuil_otsu(saturation_floue)
    masque = saturation_floue > seuil

    taille_morpho = max(3, int(round(min(image_rgb.shape[:2]) / 110)))
    if taille_morpho % 2 == 0:
        taille_morpho += 1
    masque = ouverture_binaire(masque, taille_morpho)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(500, int(aire_image * 0.0010))
    aire_max = int(aire_image * 0.18)

    composantes = extraire_composantes_utiles(masque, aire_min, aire_max)
    return estimer_nombre_depuis_composantes(composantes), composantes


def detection_piece_unique(image_rgb):
    gris = rgb_vers_gris(image_rgb)

    marge = max(8, int(round(min(gris.shape) * 0.03)))
    bord = np.concatenate(
        [
            gris[:marge, :].ravel(),
            gris[-marge:, :].ravel(),
            gris[:, :marge].ravel(),
            gris[:, -marge:].ravel(),
        ]
    )
    fond = float(np.median(bord))

    difference = np.abs(gris - fond)
    difference_floue = flou_gaussien(difference, max(5, marge | 1))
    seuil = max(0.08, seuil_otsu(difference_floue))
    masque = ouverture_binaire(difference_floue > seuil, 5)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(600, int(aire_image * 0.003))
    aire_max = int(aire_image * 0.30)
    composantes = extraire_composantes_utiles(masque, aire_min, aire_max)

    candidates = [
        comp
        for comp in composantes
        if comp["remplissage"] >= 0.62
        and comp["circularite"] >= 0.40
        and 0.78
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= 1.28
    ]

    return len(candidates) == 1


def compter_pieces(chemin_image, taille_flou=(7, 7)):
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        return 0

    prediction_principale, composantes = detection_principale(image_rgb, taille_flou)
    piece_unique = detection_piece_unique(image_rgb)

    if composantes:
        aires = sorted(comp["area"] for comp in composantes)
        aire_mediane = float(np.median(aires))
        grandes_pieces_circulaires = [
            comp
            for comp in composantes
            if comp["circularite"] >= 0.48
            and comp["remplissage"] >= 0.68
            and comp["area"] >= 8.0 * max(1.0, aire_mediane)
        ]

        if prediction_principale >= 4 and len(grandes_pieces_circulaires) == 1:
            return 1

        if (
            prediction_principale >= 4
            and len(composantes) <= 2
            and any(comp["circularite"] >= 0.55 and comp["remplissage"] >= 0.72 for comp in composantes)
        ):
            return 1

    if piece_unique:
        if prediction_principale == 0:
            return 1
        if prediction_principale >= 3 * max(1, len(composantes)):
            return 1

    return prediction_principale
