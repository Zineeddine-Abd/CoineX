"""
traitement_new.py — Pipeline de comptage par contours (Canny) + remplissage BFS
=================================================================================

Différence fondamentale avec traitement.py :
─────────────────────────────────────────────
  traitement.py  → segmente des RÉGIONS colorées (canal saturation HSL)
                   → un blob = zone colorée, peut être une pièce OU une ombre

  traitement_new → détecte les CONTOURS des pièces (Canny),
                   les ferme morphologiquement, puis REMPLIT les formes fermées
                   → un blob rempli = forcément à l'intérieur d'un contour fermé fort
                   → beaucoup moins de fausses détections

Pipeline complet :
─────────────────────────────────────────────────
  S1-S2  lire_image_rgb           lecture et normalisation de l'image
  S4     rgb_vers_gris            conversion en canal unique (luminance)
  S7     egaliser_histogramme     améliore le contraste AVANT la détection
  S8     flou_gaussien            réduit le bruit haute fréquence
  S9     gradient_sobel_xy        calcule Gx, Gy séparément (direction du gradient)
  S9     suppression_non_max      amincit les bords à 1 pixel (Canny étape 2)
  S9     hysteresis_seuillage     double seuil + propagation BFS (Canny étape 3)
  S10    fermeture_binaire        rebouche les petits manques dans les cercles
  BFS    remplir_interieurs       inondation depuis le bord → remplit les cercles
  S10    ouverture_binaire        supprime les petits artefacts après remplissage
  S7     composantes_connexes     étiquetage BFS + descripteurs de forme
  S7     estimer_nombre           ratio d'aire pour les blobs de pièces fusionnées
"""

import math
from collections import deque
import numpy as np
import matplotlib.image as mpimg

# Fonctions importées de traitement.py (utilitaires partagés non re-implémentés)
from traitement import (
    lire_image_rgb,           # lecture + normalisation + redimensionnement
    egaliser_histogramme,     # S7  — égalisation d'histogramme
    seuil_otsu,               # S5  — seuillage automatique Otsu
    composantes_connexes,     # S7  — BFS + descripteurs de forme
    estimer_nombre_depuis_composantes,  # S7 — ratio d'aire pour le comptage
)


# =============================================================================
# HYPERPARAMÈTRES
# =============================================================================

# ── Canny : double seuillage ──────────────────────────────────────────────────
# Le seuil_haut est calculé automatiquement par Otsu sur la carte NMS.
# Le seuil_bas = CANNY_SEUIL_BAS_RATIO × seuil_haut.
# Convention standard : ratio entre 0.33 et 0.5
CANNY_SEUIL_BAS_RATIO  = 0.40

# ── Morphologie après Canny ──────────────────────────────────────────────────
# Fermeture : referme les petites interruptions dans les arcs de cercle.
# Si trop grande, fusionne des pièces proches. Si trop petite, les cercles restent ouverts.
CANNY_FERMETURE_TAILLE = 7

# Ouverture : supprime les petits artefacts qui se retrouveraient remplis.
# Ex : petites zones fermées par du bruit de gradient.
CANNY_OUVERTURE_TAILLE = 5


# =============================================================================
# S4 — CONVERSION EN NIVEAUX DE GRIS
# =============================================================================
def rgb_vers_gris(image_rgb):
    """
    Convertit une image RGB en niveaux de gris normalisés dans [0, 1].

    Formule de luminance standard :
        Gray = 0.299 R + 0.587 G + 0.114 B

    Ces coefficients reflètent la sensibilité différente de l'œil humain
    aux trois couleurs (plus sensible au vert, moins au bleu).
    Ce n'est pas une simple moyenne (R+G+B)/3.
    """
    img = image_rgb.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0


# =============================================================================
# S8 — CONVOLUTIONS 1D SÉPARABLES
# =============================================================================
def _conv1d_lignes(image, noyau):
    """
    Convolution 1D horizontale, appliquée ligne par ligne.

    Séparabilité et convolution discrète :
        (f * g)[n] = Σ f[m] · g[n−m]

    On utilise np.convolve sur chaque ligne individuellement.
    Mode 'same' : la sortie a la même taille que l'entrée (zero-padding aux bords).

    Pourquoi ligne par ligne :
    Le noyau est 1D (horizontal), donc on l'applique indépendamment sur chaque
    ligne de pixels. C'est l'étape "horizontale" du filtre séparable 2D.
    """
    H, W = image.shape
    resultat = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        resultat[y, :] = np.convolve(image[y, :].astype(np.float32), noyau, mode='same')
    return resultat


def _conv1d_colonnes(image, noyau):
    """
    Convolution 1D verticale, appliquée colonne par colonne.

    Même principe que _conv1d_lignes, mais vertical.
    C'est l'étape "verticale" du filtre séparable 2D.
    """
    H, W = image.shape
    resultat = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        resultat[:, x] = np.convolve(image[:, x].astype(np.float32), noyau, mode='same')
    return resultat


# =============================================================================
# S8 — FLOU GAUSSIEN
# =============================================================================
def flou_gaussien(image, taille):
    """
    Flou gaussien 2D par deux convolutions 1D séparables.

    Filtrage gaussien :
        G(x) = exp(−x² / (2σ²)),  normalisé pour que Σ G(x) = 1

    La séparabilité permet de faire deux passes 1D au lieu d'une passe 2D :
    résultat identique, mais complexité réduite de O(k²) à O(2k) par pixel.

    Étapes :
        1) Créer un noyau gaussien 1D de longueur taille
        2) Convoluer horizontalement (lignes) → lisse les variations horizontales
        3) Convoluer verticalement (colonnes)  → lisse les variations verticales
    """
    taille = max(3, int(taille))
    if taille % 2 == 0:
        taille += 1

    sigma = max(1.0, taille / 3.0)
    rayon = taille // 2
    axe   = np.arange(-rayon, rayon + 1, dtype=np.float32)
    noyau = np.exp(-(axe * axe) / (2.0 * sigma * sigma))
    noyau /= noyau.sum()   # normalisation : somme des poids = 1

    image = image.astype(np.float32)
    image = _conv1d_lignes(image, noyau)
    return _conv1d_colonnes(image, noyau)


# =============================================================================
# S10 — MORPHOLOGIE BINAIRE (VECTORISÉE SANS sliding_window_view)
# =============================================================================
def erosion_binaire(masque, taille):
    """
    Érosion binaire : un pixel reste True seulement si TOUS les pixels
    de sa fenêtre locale sont True.

    Morphologie mathématique :
        Érosion(A) = { p | fenêtre(p) ⊆ A }

    Effets : supprime les petits objets isolés, réduit les contours.

    Implémentation vectorisée :
    On démarre avec un tableau de True (résultat = 1 partout), puis on fait
    un AND cumulatif avec chaque décalage (dy, dx) du noyau carré.
    Un pixel final reste True seulement si TOUS les décalages sont True.
    """
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    H, W = masque.shape

    padded   = np.pad(masque.astype(bool), ((pad, pad), (pad, pad)),
                      mode='constant', constant_values=False)
    resultat = np.ones((H, W), dtype=bool)    # commence à True (identité pour AND)

    for dy in range(taille):
        for dx in range(taille):
            resultat &= padded[dy:dy+H, dx:dx+W]   # AND avec le décalage (dy, dx)

    return resultat


def dilatation_binaire(masque, taille):
    """
    Dilatation binaire : un pixel devient True dès qu'AU MOINS UN pixel
    de sa fenêtre locale est True.

    Morphologie mathématique :
        Dilatation(A) = { p | fenêtre(p) ∩ A ≠ ∅ }

    Effets : agrandit les objets, referme les petits trous.

    Implémentation vectorisée :
    Même idée que l'érosion mais avec OR cumulatif (au lieu de AND).
    Un pixel final devient True si AU MOINS UN décalage est True.
    """
    taille = max(1, int(taille))
    if taille % 2 == 0:
        taille += 1
    pad = taille // 2
    H, W = masque.shape

    padded   = np.pad(masque.astype(bool), ((pad, pad), (pad, pad)),
                      mode='constant', constant_values=False)
    resultat = np.zeros((H, W), dtype=bool)   # commence à False (identité pour OR)

    for dy in range(taille):
        for dx in range(taille):
            resultat |= padded[dy:dy+H, dx:dx+W]   # OR avec le décalage (dy, dx)

    return resultat


def ouverture_binaire(masque, taille):
    """
    Ouverture = Érosion puis Dilatation.
    Supprime les petits objets sans trop modifier les grands.
    S10.
    """
    return dilatation_binaire(erosion_binaire(masque, taille), taille)


def fermeture_binaire(masque, taille):
    """
    Fermeture = Dilatation puis Érosion.
    Bouche les petits trous sans trop modifier les contours extérieurs.
    S10.
    """
    return erosion_binaire(dilatation_binaire(masque, taille), taille)


# =============================================================================
# S9 — GRADIENT DE SOBEL (Gx ET Gy SÉPARÉS)
# =============================================================================
def gradient_sobel_xy(image):
    """
    Calcule les composantes Gx et Gy du gradient de Sobel SÉPARÉMENT.

    Opérateur de Sobel
    ──────────────────────────────
    On retourne Gx ET Gy séparément (pas seulement |G|) car la NMS de Canny
    a besoin de la DIRECTION du gradient pour savoir dans quelle direction
    comparer les pixels voisins.

    Séparabilité du noyau de Sobel 2D :
        Kx = [-1, 0, 1] ⊗ [1, 2, 1]ᵀ   → détecte les variations horizontales
        Ky = [1, 2, 1]  ⊗ [-1, 0, 1]ᵀ  → détecte les variations verticales

    Étapes pour Gx :
        1) Lissage vertical (noyau [1,2,1] sur les colonnes)
        2) Dérivée horizontale (noyau [-1,0,1] sur les lignes)

    Étapes pour Gy :
        1) Lissage horizontal (noyau [1,2,1] sur les lignes)
        2) Dérivée verticale (noyau [-1,0,1] sur les colonnes)
    """
    image   = image.astype(np.float32)
    lissage = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
    derivee = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    # Gx : lissage vertical → dérivée horizontale
    lisse_v = _conv1d_colonnes(image, lissage)
    gx      = _conv1d_lignes(lisse_v, derivee)

    # Gy : lissage horizontal → dérivée verticale
    lisse_h = _conv1d_lignes(image, lissage)
    gy      = _conv1d_colonnes(lisse_h, derivee)

    return gx, gy


# =============================================================================
# S9 — SUPPRESSION DES NON-MAXIMA (Canny étape 2)
# =============================================================================
def suppression_non_max(magnitude, gx, gy):
    """
    Canny étape 2 : amincit les bords en ne gardant que les maxima locaux
    dans la direction du gradient.

    Algorithme de Canny
    ────────────────────────────────
    Principe : un pixel de contour doit être le plus fort de ses voisins dans
    la direction perpendiculaire au contour (= direction du gradient).

    Si un pixel n'est pas un maximum local dans sa direction → il est supprimé.
    Résultat : des bords fins d'exactement 1 pixel d'épaisseur.

    Quantification de la direction en 4 angles :
        0°  (H)  : voisins gauche / droite
        45° (D+) : voisins haut-droite / bas-gauche
        90° (V)  : voisins haut / bas
       135° (D-) : voisins haut-gauche / bas-droite

    L'angle est calculé avec np.arctan2(Gy, Gx), converti en degrés
    puis pris modulo 180 (un gradient à 0° et 180° = même direction de contour).

    Implémentation vectorisée : slicing NumPy, pas de boucle pixel par pixel.
    """
    H, W = magnitude.shape

    # Angle du gradient dans [0°, 180°)
    # arctan2 retourne l'angle dans (−π, π] — on convertit et on prend modulo 180
    angle = np.arctan2(gy.astype(np.float32), gx.astype(np.float32))
    angle = (angle * (180.0 / math.pi)) % 180.0

    # Zone intérieure (on ignore le bord de 1 pixel car les voisins déborderaient)
    m = magnitude[1:-1, 1:-1]
    a = angle[1:-1, 1:-1]

    # Masques des 4 directions quantifiées (chaque pixel appartient à exactement 1 direction)
    masque_H    = (a <  22.5) | (a >= 157.5)           # 0°  : horizontal
    masque_D45  = (a >= 22.5) & (a <   67.5)           # 45° : diagonale montante
    masque_V    = (a >= 67.5) & (a <  112.5)           # 90° : vertical
    masque_D135 = (a >= 112.5) & (a < 157.5)           # 135°: diagonale descendante

    # Maximum local : vrai si le pixel est >= les 2 voisins dans sa direction
    # (opérations vectorisées par slicing décalé, pas de boucle pixel)
    max_H    = masque_H    & (m >= magnitude[1:-1, :-2]) & (m >= magnitude[1:-1,  2:])
    max_D45  = masque_D45  & (m >= magnitude[ :-2,  2:]) & (m >= magnitude[ 2:, :-2])
    max_V    = masque_V    & (m >= magnitude[ :-2, 1:-1]) & (m >= magnitude[ 2:, 1:-1])
    max_D135 = masque_D135 & (m >= magnitude[ :-2,  :-2]) & (m >= magnitude[ 2:,  2:])

    local_max = max_H | max_D45 | max_V | max_D135

    resultat = np.zeros_like(magnitude)
    resultat[1:-1, 1:-1] = m * local_max.astype(np.float32)
    return resultat


# =============================================================================
# S9 — DOUBLE SEUILLAGE AVEC HYSTÉRÉSIS (Canny étape 3)
# =============================================================================
def hysteresis_seuillage(nms, seuil_haut, seuil_bas):
    """
    Canny étape 3 : double seuillage avec propagation par hystérésis.

    Algorithme de Canny (suite)
    ────────────────────────────────────────
    Trois catégories de pixels après NMS :

        Bords FORTS   (nms ≥ seuil_haut) → toujours gardés, certainement des bords
        Bords FAIBLES (seuil_bas ≤ nms < seuil_haut) → gardés SEULEMENT s'ils
                       sont connectés (8-voisinage) à un bord fort
        Supprimés     (nms < seuil_bas)  → certainement du bruit

    Pourquoi deux seuils ?
    Un seul seuil haut → trop de bords manqués (contours discontinus).
    Un seul seuil bas  → trop de bruit gardé.
    Deux seuils : on accepte les bords faibles s'ils prolongent un bord fort.

    Implémentation : BFS depuis les bords forts.
    Même principe que le BFS des composantes connexes.
    """
    bords_forts  = nms >= seuil_haut
    bords_faibles = (nms >= seuil_bas) & ~bords_forts

    H, W     = nms.shape
    resultat = bords_forts.copy()
    file     = deque()

    # Initialiser la file avec tous les pixels forts (graines du BFS)
    ys, xs = np.where(bords_forts)
    for i in range(len(ys)):
        file.append((int(ys[i]), int(xs[i])))

    # BFS 8-connexité : propager vers les bords faibles voisins
    voisins = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while file:
        cy, cx = file.popleft()
        for dy, dx in voisins:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < H and 0 <= nx < W
                    and bords_faibles[ny, nx] and not resultat[ny, nx]):
                resultat[ny, nx] = True
                file.append((ny, nx))

    return resultat


# =============================================================================
# REMPLISSAGE DES INTÉRIEURS (BFS depuis le bord de l'image)
# =============================================================================
def remplir_interieurs(carte_bords):
    """
    Remplit les intérieurs des contours fermés par inondation depuis le bord.

    Principe (analogue au BFS des composantes connexes) :
    ──────────────────────────────────────────────────────
    1. Un pixel "libre" = tout pixel qui N'EST PAS un bord (carte_bords == False).

    2. Le FOND = ensemble des pixels libres accessibles depuis les 4 côtés
       de l'image en ne traversant que des pixels libres.
       → Le fond est la "mer" en dehors des contours fermés.

    3. INTÉRIEUR = pixel libre NON atteint depuis l'extérieur.
       → Il est enfermé à l'intérieur d'un contour fermé = région d'une pièce.

    Formule (S10 — analogie morphologique) :
        Intérieur = Pixels_libres − Fond

    Pourquoi BFS 4-connexité (et non 8) pour le fond :
    Avec la 8-connexité, l'inondation passerait en diagonale entre deux bords
    adjacents d'1 pixel, "fuyant" dans les cercles. La 4-connexité est stricte :
    un bord d'1 pixel stoppe complètement l'inondation.
    """
    H, W   = carte_bords.shape
    libre  = ~carte_bords          # True = pixel libre (intérieur possible OU fond)
    visite = np.zeros((H, W), dtype=bool)
    file   = deque()

    # ── Amorçage : tous les pixels libres du périmètre de l'image ─────────────
    for x in range(W):
        if libre[0,   x] and not visite[0,   x]: visite[0,   x] = True; file.append((0,   x))
        if libre[H-1, x] and not visite[H-1, x]: visite[H-1, x] = True; file.append((H-1, x))
    for y in range(1, H-1):
        if libre[y,   0] and not visite[y,   0]: visite[y,   0] = True; file.append((y,   0))
        if libre[y, W-1] and not visite[y, W-1]: visite[y, W-1] = True; file.append((y, W-1))

    # ── BFS 4-connexité : exploration du fond ─────────────────────────────────
    while file:
        cy, cx = file.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and libre[ny, nx] and not visite[ny, nx]:
                visite[ny, nx] = True
                file.append((ny, nx))

    # ── Résultat : pixels libres non atteints = intérieurs des cercles ─────────
    return libre & ~visite


# =============================================================================
# PIPELINE INTERNE (retourne toutes les étapes pour la visualisation)
# =============================================================================
def _pipeline_interne(chemin, taille_flou=(7, 7)):
    """
    Exécute le pipeline complet et retourne un dictionnaire avec toutes
    les étapes intermédiaires. Utilisé par visualiser_pipeline_new().
    """
    image_rgb = lire_image_rgb(chemin)
    if image_rgb is None:
        return None

    hauteur, largeur = image_rgb.shape[:2]

    # ── S4 : Niveaux de gris ──────────────────────────────────────────────────
    gris = rgb_vers_gris(image_rgb)

    # ── S7 : Égalisation d'histogramme ───────────────────────────────────────
    # Améliore le contraste global avant la détection de contours.
    # Sans cette étape, les pièces peu exposées ont un gradient trop faible pour Canny.
    gris_egal = egaliser_histogramme(gris)

    # ── S8 : Flou gaussien ────────────────────────────────────────────────────
    # Réduit le bruit haute fréquence pour éviter les faux contours.
    # Taille adaptive : proportionnelle à la taille demandée ET à la taille de l'image.
    taille_locale = max(int((taille_flou[0] + taille_flou[1]) / 2.0),
                        int(min(hauteur, largeur) / 90), 5)
    if taille_locale % 2 == 0:
        taille_locale += 1
    gris_flou = flou_gaussien(gris_egal, taille_locale)

    # ── S9 : Gradient Sobel (Gx, Gy séparés) ─────────────────────────────────
    gx, gy    = gradient_sobel_xy(gris_flou)
    magnitude = np.sqrt(gx * gx + gy * gy)
    magnitude_norm = magnitude / (magnitude.max() + 1e-8)   # normalisé [0, 1]

    # ── S9 : Suppression des non-maxima (Canny étape 2) ──────────────────────
    nms      = suppression_non_max(magnitude, gx, gy)
    nms_norm = nms / (nms.max() + 1e-8)

    # ── S9 : Hystérésis (Canny étape 3) ──────────────────────────────────────
    # seuil_haut = Otsu sur la carte NMS (automatique)
    # seuil_bas  = CANNY_SEUIL_BAS_RATIO × seuil_haut (convention standard)
    seuil_haut  = seuil_otsu(nms_norm)
    seuil_bas   = seuil_haut * CANNY_SEUIL_BAS_RATIO
    carte_bords = hysteresis_seuillage(nms_norm, seuil_haut, seuil_bas)

    # ── S10 : Fermeture morphologique ─────────────────────────────────────────
    # Les contours Canny d'une pièce ne sont jamais parfaitement fermés :
    # quelques pixels manquent toujours sur les arcs. La fermeture rebouche ces
    # petites interruptions pour que les cercles deviennent complets.
    carte_fermee = fermeture_binaire(carte_bords, CANNY_FERMETURE_TAILLE)

    # ── BFS : Remplissage des intérieurs ──────────────────────────────────────
    # Inondation depuis le bord de l'image → les zones non atteintes = intérieurs.
    interieurs = remplir_interieurs(carte_fermee)

    # ── S10 : Ouverture morphologique ─────────────────────────────────────────
    # Supprime les petits artefacts qui auraient échappé au remplissage
    # (ex : petites zones fermées par du bruit de gradient, non liées aux pièces).
    masque_propre = ouverture_binaire(interieurs, CANNY_OUVERTURE_TAILLE)

    # ── S7 : Composantes connexes + filtrage + estimation ─────────────────────
    aire_image = hauteur * largeur
    aire_min   = max(500, int(aire_image * 0.001))
    aire_max   = int(aire_image * 0.90)

    toutes_composantes = composantes_connexes(masque_propre)
    composantes = [
        c for c in toutes_composantes
        if aire_min <= c['area'] <= aire_max and not c['touche_bord']
    ]

    masque_final = masque_propre

    # Fallback : si aucune composante, essayer le masque complémentaire
    # (cas rare où les pièces créent des zones claires dans les bords de gradient)
    if not composantes:
        masque_inv = ouverture_binaire(~interieurs & ~carte_fermee, CANNY_OUVERTURE_TAILLE)
        toutes_inv = composantes_connexes(masque_inv)
        composantes_inv = [
            c for c in toutes_inv
            if aire_min <= c['area'] <= aire_max and not c['touche_bord']
        ]
        if composantes_inv:
            composantes  = composantes_inv
            masque_final = masque_inv

    prediction = estimer_nombre_depuis_composantes(composantes)

    return {
        'image_rgb'    : image_rgb,
        'gris_egal'    : gris_egal,
        'magnitude'    : magnitude_norm,
        'nms'          : nms_norm,
        'carte_bords'  : carte_bords,
        'carte_fermee' : carte_fermee,
        'interieurs'   : interieurs,
        'masque_propre': masque_final,
        'composantes'  : composantes,
        'prediction'   : prediction,
        'seuil_haut'   : seuil_haut,
        'seuil_bas'    : seuil_bas,
        'taille_locale': taille_locale,
    }


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================
def compter_pieces(chemin, taille_flou=(7, 7)):
    """
    Compte le nombre de pièces dans une image — pipeline Canny + remplissage.

    Interface identique à traitement.compter_pieces :
    peut remplacer directement l'import dans evaluation.py.

    Paramètres :
        chemin      : chemin vers l'image
        taille_flou : taille du noyau gaussien, ex. (7, 7)

    Retour :
        int — nombre estimé de pièces
    """
    result = _pipeline_interne(chemin, taille_flou)
    return result['prediction'] if result is not None else 0
