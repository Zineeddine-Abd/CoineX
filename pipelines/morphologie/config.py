import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np


# =============================================================================
# PIPELINE GLOBAL DU PROGRAMME
# =============================================================================
#
# 
#
# Chaîne globale de traitement :
# ------------------------------
# 1) lire l'image proprement et la normaliser (mettre ses valeurs de pixels dans une échelle standard pour faciliter les calculs)
# 2) éventuellement la redimensionner pour réduire le coût de calcul (parce qu’on traite moins de pixels)
# 3) Convertit RGB en HSL pour faire la détection à partir de la saturation (HSL)
# 4) Convertit en niveaux de gris avec la formule de luminance
# 4) lisser l'image avec un flou gaussien
# 5) segmenter automatiquement avec Otsu
# 6) nettoyer le masque avec une ouverture morphologique binaire
# 7) extraire les composantes connexes
# 8) mesurer leur forme (aire, circularité, remplissage, bbox, bord)
# 9) estimer le nombre de pièces à partir de ces composantes
# 10) lancer une détection secondaire spéciale "une seule pièce"
# 11) appliquer des règles correctives finales
#
#
# Exemples de cas difficiles que ce pipeline essaie de mieux gérer :
# ------------------------------------------------------------------
# - 2 pièces collées -> peuvent former une seule grosse composante
# - 1 seule pièce brillante -> peut être découpée en plusieurs régions
# - objet  au bord -> ne doit pas être compté
# - image très grande -> coût de calcul plus élevé si on ne redimensionne pas

# =============================================================================
# CONFIGURATION & HYPERPARAMÈTRES
# =============================================================================
# [RÈGLE D'OR] : Toutes les valeurs ci-dessous ont été réglées empiriquement
# sur la base de VALIDATION UNIQUEMENT, sans regarder la base de test.
# Cette approche garantit que les résultats finaux ne sont pas biaisés.
# =============================================================================

# ===== GROUPE 1 : DESCRIPTEURS DE FORMES POUR COIN DE RÉFÉRENCE =====
# Utilisé pour identifier une composante connexe qui ressemble vraiment à une pièce.
# Une pièce "de référence" nous permet d'estimer la taille typique d'une pièce.
#
# Descripteurs d'objets et composantes connexes
#
COIN_REFERENCE_CIRCULARITY_MIN = 0.45      # Circularité minimale (1.0 = cercle parfait)
COIN_REFERENCE_FILL_MIN = 0.45             # Remplissage minimum (aire / bbox)
COIN_REFERENCE_ASPECT_MIN = 0.65           # Ratio minimum hauteur/largeur (si < 1 : objet aplati)
COIN_REFERENCE_ASPECT_MAX = 1.55           # Ratio maximum hauteur/largeur (si > 1 : objet allongé)

# ===== GROUPE 2 : DÉTECTION DE PIÈCES FUSIONNÉES =====
# Quand deux pièces se touchent, elles forment une seule composante connexe.
# Ces paramètres permettent de détecter et de compter ces cas.
#
# Analyse d'objets connexes
#
MERGE_AREA_RATIO_THRESHOLD = 1.8           # Si aire > 1.8 × aire_typique, compte comme 2 pièces
MERGE_FILL_MIN = 0.35                      # Remplissage minimum pour détecter fusion

# ===== GROUPE 3 : DÉTECTION SPÉCIALE "UNE SEULE PIÈCE" =====
# Filet de sécurité : si le pipeline principal doute, on teste si c'est exactement 1 pièce.
# Critères très stricts pour ne faux-positif.
#
# Validation de composantes
#
SINGLE_COIN_FILL_MIN = 0.62                # Remplissage minimum pour une pièce unique
SINGLE_COIN_CIRCULARITY_MIN = 0.40         # Circularité minimum pour une pièce unique
SINGLE_COIN_ASPECT_RATIO_MIN = 0.78        # Ratio min hauteur/largeur
SINGLE_COIN_ASPECT_RATIO_MAX = 1.28        # Ratio max hauteur/largeur

# ===== GROUPE 4 : RÈGLES CORRECTIVES (TRÈS GRANDES PIÈCES) =====
# Si la prédiction principale est très grande mais qu'on observe une seule
# composante énorme et très circulaire, on corrige à 1.
#
# Analyse statistique des composantes
#
CORRECTION_LARGE_CIRCULARITY_MIN = 0.48    # Circularité pour très grande pièce
CORRECTION_LARGE_FILL_MIN = 0.68           # Remplissage pour très grande pièce
CORRECTION_LARGE_AREA_MULTIPLIER = 8.0     # Doit être > 8× aire typique

# ===== GROUPE 5 : RÈGLES CORRECTIVES (COMPOSANTES RARES) =====
# Si prediction est très grande mais peu de composantes, peut être 1 grosse pièce.
#
# Statistiques sur les composantes
#
CORRECTION_RARE_CIRCULARITY_MIN = 0.55     # Circularité stricte
CORRECTION_RARE_FILL_MIN = 0.72            # Remplissage strict

# ===== GROUPE 6 : PARAMÈTRES SYSTÈME =====
# Paramètres de performance et système, pas liés à la détection mathématique.
#
MAX_IMAGE_DIMENSION = 520                  # Redimensionner images > 520 pixels (performance)


