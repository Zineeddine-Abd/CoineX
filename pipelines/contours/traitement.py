"""
Pipeline contours de détection de pièces.

Chaîne complète :
  0. Sous-échantillonnage                        
  1. Conversion en niveaux de gris par luminance 
  2. Pré-lissage gaussien (tue la texture)       
  3. Égalisation d'histogramme                   
  4. Second lissage doux avant dérivée           
  5. Détection de contours par Sobel (X, Y, mag) 
  6. Seuillage automatique d'Otsu                
  7. Fermeture morphologique (petit noyau)       (post-traitement, )
  8. Remplissage des silhouettes par masque      
  9a. Ouverture (élimination des speckles)       (post-traitement)
  9b. Érosion avec noyau circulaire              (sépare les pièces tangentes)
 10. Filtrage par propriétés géométriques        (extraction de primitives, S2) :
        - aire (S5/V1)
        - circularité 4π·A_hull / P_hull²        (calculée sur l'enveloppe convexe)
        - solidité  A / A_hull
        - ratio largeur/hauteur de la bbox
"""
import cv2
import numpy as np


def _impair(n):
    return n if n % 2 == 1 else n + 1


def pipeline_contours(
    chemin_image,
    largeur_cible=800,
    taille_pre_flou=(8, 8),
    taille_flou=(2, 2),
    taille_fermeture=5,
    taille_ouverture=9,
    taille_erosion=7,
    aire_min=300,
    aire_max=200000,
    circularite_min=0.55,
    ratio_max=1.8,
    solidite_min=0.85,
    # --- Stratégies anti-trous d'anneaux (OPT-IN : voir docstring) ---
    # Désactivées par défaut : elles aident sur les pièces ISOLÉES avec un
    # gap dans leur anneau, mais fusionnent les pièces qui se touchent. Sur
    # ce dataset (beaucoup d'amas) elles dégradent la MAE. Activer cas par cas.
    n_fermetures=1,            # >1 = scelle des gaps plus larges (mais fusionne)
    remplir_par_hull=False,    # True = remplit l'enveloppe convexe (pièces isolées)
    seuil_bas_hysteresis=None, # 0-255 = union avec un seuil bas (hystérésis S5)
    # rétro-compatibilité : ancien paramètre unique
    taille_noyau_morpho=None,
):
    """Exécute le pipeline complet et renvoie un dictionnaire avec :
        - toutes les étapes intermédiaires (pour le visualiseur)
        - les contours retenus et rejetés
        - le compteur final
    """

    if taille_noyau_morpho is not None:
        taille_fermeture = taille_noyau_morpho
        taille_ouverture = taille_noyau_morpho

    img = cv2.imread(chemin_image)
    if img is None:
        return {"erreur": f"Image introuvable: {chemin_image}", "compteur": 0}

    # 0. Sous-échantillonnage : réduire la résolution pour calmer le Sobel
    #    et accélérer le traitement .
    h, w = img.shape[:2]
    if w > largeur_cible:
        r = largeur_cible / w
        img = cv2.resize(img, (largeur_cible, int(h * r)), interpolation=cv2.INTER_AREA)

    # 1. Niveaux de gris par formule de luminance .
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Pré-lissage fort : tue la texture/grain du support avant l'égalisation,
    #    sinon equalizeHist amplifie ce bruit .
    p_x, p_y = _impair(taille_pre_flou[0]), _impair(taille_pre_flou[1])
    pre_flou = cv2.GaussianBlur(gris, (p_x, p_y), 0)
    # pre_flou = gris

    # 3. Égalisation d'histogramme : étaler la dynamique .
    egalisee = cv2.equalizeHist(pre_flou)

    # 4. Second lissage doux juste avant Sobel pour stabiliser la dérivée.
    t_x, t_y = _impair(taille_flou[0]), _impair(taille_flou[1])
    floute = cv2.GaussianBlur(egalisee, (t_x, t_y), 0)

    # 5. Sobel : convolution avec les noyaux dérivateurs .
    sobel_x = cv2.Sobel(floute, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(floute, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    m_max = magnitude.max() if magnitude.max() > 0 else 1.0
    magnitude_u8 = np.uint8(255.0 * magnitude / m_max)

    # 6. Seuillage automatique d'Otsu sur la magnitude .
    _, contours_bin_otsu = cv2.threshold(
        magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # 6bis. Union avec un seuil bas (équivalent du seuillage par hystérésis
    #       §1.3) : capture les bords faibles d'anneaux fragmentés.
    if seuil_bas_hysteresis is not None:
        _, contours_bin_bas = cv2.threshold(
            magnitude_u8, int(seuil_bas_hysteresis), 255, cv2.THRESH_BINARY
        )
        contours_bin = cv2.bitwise_or(contours_bin_otsu, contours_bin_bas)
    else:
        contours_bin = contours_bin_otsu

    # 7. Fermeture morphologique itérée (petit noyau, plusieurs passes) :
    #    chaque itération scelle des trous un peu plus larges sans souder les
    #    pièces voisines, contrairement à un gros noyau unique.
    noyau_close = np.ones((_impair(taille_fermeture),) * 2, np.uint8)
    fermee = cv2.morphologyEx(
        contours_bin, cv2.MORPH_CLOSE, noyau_close, iterations=max(1, n_fermetures)
    )

    # 8. Remplissage : on transforme les contours fermés en silhouettes pleines
    #    grâce à un masque binaire .
    #    Quand `remplir_par_hull=True`, on remplit l'enveloppe convexe de chaque
    #    contour : un anneau troué reste convexe-fermé donc se remplit comme un
    #    disque. Sinon on remplit le contour tel quel.
    contours_externes, _ = cv2.findContours(
        fermee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    masque_plein = np.zeros_like(fermee)
    if remplir_par_hull:
        hulls = [cv2.convexHull(c) for c in contours_externes]
        cv2.drawContours(masque_plein, hulls, -1, 255, thickness=cv2.FILLED)
    else:
        cv2.drawContours(
            masque_plein, contours_externes, -1, 255, thickness=cv2.FILLED
        )

    # 9a. Ouverture (élimine d'abord les speckles, noyau modéré).
    noyau_open = np.ones((_impair(taille_ouverture),) * 2, np.uint8)
    propre = cv2.morphologyEx(masque_plein, cv2.MORPH_OPEN, noyau_open)

    # 9b. Érosion avec un noyau CIRCULAIRE : casse les ponts fins entre pièces
    #     tangentes. On utilise un disque, car des pièces se touchent en un
    #     point (tangence) et un élément structurant rond érode ce point
    #     beaucoup plus vite que l'intérieur des pièces.
    k_erode = max(3, _impair(taille_erosion))
    noyau_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
    erodee = cv2.erode(propre, noyau_erode)

    # 10. Extraction des contours finaux et filtrage par propriétés mathématiques.
    contours_finaux, _ = cv2.findContours(
        erodee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    retenus, rejetes = [], []
    for c in contours_finaux:
        aire = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        aire_hull = cv2.contourArea(hull)
        perim_hull = cv2.arcLength(hull, True)
        # Circularité robuste : calculée sur l'enveloppe convexe (lisse),
        # une pièce ronde donne ~1.0 même si le contour est dentelé.
        circularite = (
            (4 * np.pi * aire_hull / (perim_hull ** 2)) if perim_hull > 0 else 0.0
        )
        # Solidité = aire / aire(enveloppe). Une pièce pleine ≈ 1.0,
        # un nuage de bruit irrégulier << 1.
        solidite = (aire / aire_hull) if aire_hull > 0 else 0.0
        ratio = max(bw, bh) / max(min(bw, bh), 1)

        info = {
            "contour": c,
            "aire": float(aire),
            "circularite": float(circularite),
            "solidite": float(solidite),
            "ratio": float(ratio),
            "bbox": (int(x), int(y), int(bw), int(bh)),
        }

        if not (aire_min <= aire <= aire_max):
            info["raison"] = f"aire={aire:.0f} hors [{aire_min},{aire_max}]"
            rejetes.append(info)
            continue
        if circularite < circularite_min:
            info["raison"] = f"circ={circularite:.2f} < {circularite_min}"
            rejetes.append(info)
            continue
        if solidite < solidite_min:
            info["raison"] = f"solidité={solidite:.2f} < {solidite_min}"
            rejetes.append(info)
            continue
        if ratio > ratio_max:
            info["raison"] = f"ratio={ratio:.2f} > {ratio_max}"
            rejetes.append(info)
            continue

        info["raison"] = "OK"
        retenus.append(info)

    return {
        "originale": img,
        "gris": gris,
        "pre_floutee": pre_flou,
        "egalisee": egalisee,
        "floutee": floute,
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "magnitude": magnitude_u8,
        "binaire": contours_bin,
        "fermee": fermee,
        "masque_plein": masque_plein,
        "propre": propre,
        "erodee": erodee,
        "retenus": retenus,
        "rejetes": rejetes,
        "compteur": len(retenus),
        "params": {
            "largeur_cible": largeur_cible,
            "taille_pre_flou": (p_x, p_y),
            "taille_flou": (t_x, t_y),
            "taille_fermeture": _impair(taille_fermeture),
            "n_fermetures": max(1, n_fermetures),
            "remplir_par_hull": remplir_par_hull,
            "seuil_bas_hysteresis": seuil_bas_hysteresis,
            "taille_ouverture": _impair(taille_ouverture),
            "taille_erosion": k_erode,
            "aire_min": aire_min,
            "aire_max": aire_max,
            "circularite_min": circularite_min,
            "solidite_min": solidite_min,
            "ratio_max": ratio_max,
        },
    }


def compter_pieces_contours(chemin_image, **kwargs):
    """Renvoie uniquement le nombre de pièces (interface compatible évaluation)."""
    res = pipeline_contours(chemin_image, **kwargs)
    return res.get("compteur", 0)

def compter_pieces(chemin_image, **kwargs):
    """Renvoie uniquement le nombre de pièces (interface compatible évaluation)."""
    res = pipeline_contours(chemin_image, **kwargs)
    return res.get("compteur", 0)
