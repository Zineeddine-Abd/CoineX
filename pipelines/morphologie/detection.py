import math
from collections import deque
import matplotlib.image as mpimg
import numpy as np

from pipelines.morphologie.config import *
from utils.color import rgb_vers_gris, rgb_vers_hsl
from utils.io_utils import lire_image_rgb
from pipelines.morphologie.filters import flou_gaussien
from pipelines.morphologie.segmentation import seuil_otsu
from pipelines.morphologie.morphology import ouverture_binaire, fermeture_binaire, composantes_connexes



def extraire_composantes_utiles(masque, aire_min, aire_max):
    """
    Filtre les composantes détectées pour ne garder que les composantes plausibles.

    Étapes :
    --------
    1) extraire les composantes connexes du masque
    2) garder seulement celles dont :
       - l'aire est comprise entre aire_min et aire_max
       - la composante ne touche pas le bord

    Pourquoi ce filtre :
    --------------------
    Tout ce qui est segmenté en blanc n'est pas forcément une pièce.
    Il peut y avoir :
    - du bruit
    - une grande zone de fond mal segmentée
    - une pièce coupée au bord

    Exemple :
    ---------
    Si une composante a une aire plausible mais touche le bord,
    on la rejette, car elle peut être partiellement hors champ.
    """
    composantes = composantes_connexes(masque)
    return [
        comp
        for comp in composantes
        if aire_min <= comp["area"] <= aire_max and not comp["touche_bord"]
    ]


def estimer_nombre_depuis_composantes(composantes):
    """
    Estime le nombre de pièces réelles à partir des composantes détectées.
    
    COURS : Week 7 - Analyse statistique d'objets
    ============================================
    
    PROBLÈME À RÉSOUDRE :
    Une composante connexe ≠ une pièce dans 100% des cas :
    - Cas 1 : Deux pièces qui se touchent → 1 seule composante (sous-comptage)
    - Cas 2 : Une pièce brillante → 2-3 régions (sur-comptage)
    
    STRATÉGIE (exemple seuil + ratio) :
    1. Identifier des \"pièces de référence\" :
       - circularité >= COIN_REFERENCE_CIRCULARITY_MIN
       - remplissage >= COIN_REFERENCE_FILL_MIN
       - aspect ratio dans COIN_REFERENCE_ASPECT_{MIN,MAX}
       
    2. Calculer l'aire MÉDIANE de ces références = aire_typique
    
    3. Pour chaque composante :
       - Si aire < 1.8 * aire_typique : compter comme 1 pièce
       - Si aire >= 1.8 * aire_typique : compter comme round(aire/aire_typique) pièces
    
    EXEMPLE CONCRET :
    - Pièce typique = 12000 pixels
    - Composante trouvée = 24000 pixels
    - Ratio = 24000/12000 = 2.0 → compte comme 2 pièces
    
    AMÉLIORATIONS AU MAE :
    Réduit les grosses erreurs (prédire 1 au lieu de 4)

    Pourquoi cette fonction est cruciale :
    -------------------------------------
    Une composante n'est pas forcément une seule pièce.
    Si deux pièces se touchent, elles peuvent fusionner en une seule région.

    Idée :
    ------
    1) On choisit des composantes "de référence" qui ressemblent bien
       à des pièces normales :
       - circularité correcte
       - remplissage correct
       - ratio bbox raisonnable

    2) On calcule leur aire médiane :
       -> cela donne une aire typique d'une pièce

    3) Pour chaque composante :
       - si son aire vaut environ 2 fois l'aire typique, on compte 2
       - si elle vaut environ 3 fois, on peut compter 3
       - sinon on compte 1

    Exemple :
    ---------
    Si une pièce "typique" vaut 12000 pixels,
    une composante à 24000 pixels peut représenter 2 pièces collées.

    Pourquoi cela améliore le MAE :
    -------------------------------
    Parce que cela corrige des sous-comptages fréquents.
    """
    if not composantes:
        return 0

    # Utilisation des hyperparamètres définis en haut du fichier
    aires_reference = [
        comp["area"]
        for comp in composantes
        if comp["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
        and comp["remplissage"] >= COIN_REFERENCE_FILL_MIN
        and COIN_REFERENCE_ASPECT_MIN
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= COIN_REFERENCE_ASPECT_MAX
    ]

    if not aires_reference:
        return len(composantes)

    aire_reference = float(np.median(aires_reference))
    compteur = 0

    for comp in composantes:
        ratio = comp["area"] / max(1.0, aire_reference)
        if ratio >= MERGE_AREA_RATIO_THRESHOLD and comp["remplissage"] >= MERGE_FILL_MIN:
            compteur += max(1, int(round(ratio)))
        else:
            compteur += 1

    return compteur


# =============================================================================
# DÉTECTION PRINCIPALE
# =============================================================================
def detection_principale(image_rgb, taille_flou):
    """
    Détection principale basée sur la saturation.
    
    COURS : Semaines 3 à 8 - Pipeline complet de traitement d'image
    ===============================================================
    
    PIPELINE DÉTAILLÉ :
    
    Étape 1 : Transformation couleur (Week 3)
    ├─ RGB → HSL
    └─ Extraction du canal Saturation
    
    Étape 2 : Opération locale (Week 8)
    ├─ Flou gaussien à taille adaptative
    └─ Réduit le bruit avant segmentation
    
    Étape 3 : Seuillage (Week 5)
    ├─ Algorithme d'Otsu automatique
    └─ Crée image binaire sans paramètre manuel
    
    Étape 4 : Opérations morphologiques (Week 6)
    ├─ Ouverture binaire = Érosion puis Dilation
    └─ Nettoie les petits parasites
    
    Étape 5 : Analyse d'objets (Week 7)
    ├─ Extraction des composantes connexes
    ├─ Calcul des descripteurs de forme
    └─ Filtrage par aire et limites d'image
    
    Étape 6 : Estimation (Statistical analysis)
    └─ Compte le nombre de pièces probable

    But : Détecter les pièces via leur saturation (robuste aux ombres)

    Pipeline :
    ----------
    1) conversion RGB -> HSL, puis récupération de la saturation
    2) flou gaussien
    3) seuillage d'Otsu
    4) ouverture binaire
    5) calcul des seuils d'aire
    6) extraction des composantes utiles
    7) estimation du nombre de pièces

    Pourquoi cette détection :
    --------------------------
    La saturation est souvent utile pour faire ressortir les pièces
    par rapport au fond.

    Paramètre taille_flou :
    -----------------------
    Ici taille_flou est donné comme un tuple, par exemple (7,7).
    On en tire une taille moyenne locale.

    Pourquoi une taille locale adaptative :
    ---------------------------------------
    On combine :
    - la taille de flou demandée
    - la taille de l'image

    Cela permet d'éviter un flou trop faible sur une grande image
    ou trop fort sur une petite image.

    Exemple :
    ---------
    Si l'image est grande, on peut prendre un noyau un peu plus grand
    pour lisser correctement le bruit.
    """
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

    # On sauvegarde le masque binaire brut AVANT la morphologie.
    # Il servira à construire le masque inversé si la détection normale échoue.
    masque_brut = saturation_floue > seuil
    masque = masque_brut

    taille_morpho = max(3, int(round(min(image_rgb.shape[:2]) / 110)))
    if taille_morpho % 2 == 0:
        taille_morpho += 1

    # COURS Semaine 10 : Ouverture puis Fermeture = nettoyage complet du masque.
    #
    # Étape 1 — Ouverture (Érosion → Dilatation) :
    #   Supprime les petits points blancs parasites qui ne sont pas des pièces.
    #   Ex: un grain de poussière blanc sur le fond sera effacé.
    #
    # Étape 2 — Fermeture (Dilatation → Érosion) :
    #   Bouche les petits trous noirs à l'intérieur des pièces.
    #   Ex: un reflet brillant au centre d'une pièce crée un trou -> on le referme.
    masque = ouverture_binaire(masque, taille_morpho)
    masque = fermeture_binaire(masque, taille_morpho)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(500, int(aire_image * 0.0010))

    # -------------------------------------------------------------------------
    # COURS Semaine 3 — Règle d'or : paramètres réglés sur la base de VALIDATION
    # -------------------------------------------------------------------------
    # Problème : quand beaucoup de pièces se touchent, elles forment une seule
    # grande composante connexe qui dépasse le seuil d'aire maximum.
    #
    # Solution : on essaie plusieurs valeurs de aire_max, de la plus stricte
    # à la plus permissive. On s'arrête dès qu'on trouve au moins une composante.
    #
    #   Facteur 0.18 : réglage de base, correspond à ≈ 1 pièce bien isolée
    #   Facteur 0.35 : accepte des composantes plus grandes (2–3 pièces collées)
    #   Facteur 0.60 : accepte de très grandes zones (4–8 pièces collées)
    #   Facteur 0.90 : dernier recours, quasi toute l'image est acceptée
    #
    # Pourquoi pas d'emblée 0.90 ?
    # Car un facteur trop grand accepterait le fond mal segmenté comme une pièce.
    # On part donc du plus strict et on relâche seulement si nécessaire.
    #
    # IMPORTANT — Performance :
    # composantes_connexes (le BFS) est l'opération la plus coûteuse du pipeline.
    # On l'appelle UNE SEULE FOIS pour tout le masque, puis on filtre le résultat
    # avec différents seuils d'aire. C'est équivalent à appeler extraire_composantes_utiles
    # plusieurs fois, mais sans refaire le BFS à chaque itération.
    # -------------------------------------------------------------------------

    # Étape 1 : BFS unique — on extrait TOUTES les composantes du masque
    toutes_composantes = composantes_connexes(masque)

    # Étape 2 : filtrage progressif par aire_max (pas de nouveau BFS)
    composantes = []
    for facteur_max in [0.18, 0.35, 0.60, 0.90]:
        aire_max    = int(aire_image * facteur_max)
        composantes = [
            comp for comp in toutes_composantes
            if aire_min <= comp["area"] <= aire_max and not comp["touche_bord"]
        ]
        if composantes:
            break   # on a trouvé des composantes, pas besoin de relâcher davantage

    # -------------------------------------------------------------------------
    # DÉTECTION INVERSÉE — pièces peu saturées sur fond coloré
    # -------------------------------------------------------------------------
    # Problème : l'approche normale suppose que les pièces ont une saturation
    # PLUS HAUTE que le fond (fond blanc/neutre + pièces dorées/colorées).
    #
    # Mais certaines images ont l'inverse :
    #   - pièces argentées/métalliques → saturation BASSE (gris)
    #   - fond coloré (papier bleu, bois, tissu) → saturation HAUTE
    #
    # Dans ce cas, Otsu sépare bien les deux classes, mais on a segmenté
    # le FOND à la place des pièces. La solution : inverser le masque.
    #
    # Comment on détecte que la détection normale a échoué ?
    # On compte les composantes "de référence" (circulaires, bien remplies).
    # Si aucune n'est trouvée, les objets détectés ne ressemblent pas à des pièces
    # → on essaie avec le masque inversé.
    # -------------------------------------------------------------------------
    nb_ref_normal = sum(
        1 for c in composantes
        if c["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
        and c["remplissage"] >= COIN_REFERENCE_FILL_MIN
        and COIN_REFERENCE_ASPECT_MIN
            <= c["hauteur_bbox"] / max(1, c["largeur_bbox"])
            <= COIN_REFERENCE_ASPECT_MAX
    )

    if nb_ref_normal == 0:
        # Inverser le masque brut : les zones PEU saturées deviennent blanches
        # (= les pièces métalliques argentées, peu colorées)
        masque_inv = ~masque_brut
        masque_inv = ouverture_binaire(masque_inv, taille_morpho)
        masque_inv = fermeture_binaire(masque_inv, taille_morpho)

        toutes_inv = composantes_connexes(masque_inv)
        composantes_inv = []
        for facteur_max in [0.18, 0.35, 0.60, 0.90]:
            aire_max_inv = int(aire_image * facteur_max)
            composantes_inv = [
                c for c in toutes_inv
                if aire_min <= c["area"] <= aire_max_inv and not c["touche_bord"]
            ]
            if composantes_inv:
                break

        # On utilise le masque inversé seulement s'il donne de meilleures
        # composantes circulaires que l'approche normale
        nb_ref_inv = sum(
            1 for c in composantes_inv
            if c["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
            and c["remplissage"] >= COIN_REFERENCE_FILL_MIN
            and COIN_REFERENCE_ASPECT_MIN
                <= c["hauteur_bbox"] / max(1, c["largeur_bbox"])
                <= COIN_REFERENCE_ASPECT_MAX
        )

        if nb_ref_inv > nb_ref_normal:
            composantes = composantes_inv

    return estimer_nombre_depuis_composantes(composantes), composantes


# =============================================================================
# DÉTECTION SPÉCIALE : CAS "UNE SEULE PIÈCE"
# =============================================================================
def detection_piece_unique(image_rgb):
    """
    Détection spécialisée : \"Y a-t-il probablement exactement 1 pièce ?\"
    
    COURS : Week 7 - Validation & stratégies de détection
    =====================================================
    
    BUT : Filet de sécurité si la détection principale doute
    
    Cas où c'est utile :
    - Pièce unique mal segmentée par voie saturation → prédiction 0 ou 4
    - Pièce brillante qui crée plusieurs régions → sur-comptage
    
    STRATÉGIE ALTERNATIVE (Contraste avec fond) :
    
    Au lieu d'utiliser la saturation :
    1. Convertir en niveaux de gris (Week 3)
    2. Estimer la couleur du fond (médiane des bords)
    3. Soustraire le fond à l'image
    4. Déterminer où la différence est grande
    
    Cette approche :
    - Déteste les reflets (très différents du fond)
    - Déteste les zones ombragées (différentes du fond)
    - Isole mieux une pièce unique sur fond assez uniforme
    
    CRITÈRES TRÈS STRICTS (pour éviter les faux positifs) :
    - remplissage >= SINGLE_COIN_FILL_MIN
    - circularité >= SINGLE_COIN_CIRCULARITY_MIN  
    - ratio hauteur/largeur dans [SINGLE_COIN_ASPECT_RATIO_MIN, MAX]
    
    Pourquoi la médiane des bords pour le fond :
    -------------------------------------
    La détection principale peut parfois :
    - prédire 0 alors qu'il y a 1 pièce
    - ou prédire 4 alors qu'il n'y a qu'une seule pièce mal segmentée

    Cette fonction sert donc de filet de sécurité.

    Pipeline détaillé :
    -------------------
    1) conversion en gris
    2) estimation du fond à partir des bords
    3) soustraction au fond
    4) flou gaussien
    5) seuillage d'Otsu
    6) ouverture binaire
    7) extraction des composantes utiles
    8) filtrage strict de candidats plausibles
    9) si on trouve exactement un candidat -> True

    Pourquoi la soustraction au fond :
    ----------------------------------
    Si le fond est assez uniforme, alors :
    - les pixels du fond sont proches de "fond"
    - la pièce diffère plus fortement

    Exemple :
    ---------
    fond = 0.7
    pixel fond = 0.72 -> différence = 0.02
    pixel pièce = 0.35 -> différence = 0.35

    Donc la pièce ressort.

    Pourquoi prendre la médiane des bords :
    ---------------------------------------
    On suppose que le fond est visible sur les bords de l'image.
    La médiane est robuste aux petites perturbations.

    Pourquoi des critères stricts à la fin :
    ----------------------------------------
    On veut éviter de conclure trop facilement qu'il y a une seule pièce.
    On impose donc :
    - remplissage élevé
    - circularité correcte
    - bbox proche d'un carré
    """
    gris = rgb_vers_gris(image_rgb)

    # COURS Semaine 7 — Estimation du fond par les bords de l'image :
    # On suppose que le fond (la surface sur laquelle reposent les pièces)
    # est visible sur les bordures de l'image.
    # On prend la MÉDIANE (et non la moyenne) car elle est robuste :
    # si quelques pixels de bord appartiennent à une pièce, ils ne
    # faussent pas le résultat.
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

    # COURS Semaine 7 — Soustraction d'images (section 6) :
    # On soustrait la valeur du fond à chaque pixel de l'image.
    # Résultat : les pixels qui ressemblent au fond donnent une différence ~0,
    # les pixels qui appartiennent à une pièce donnent une différence élevée.
    #
    # Exemple concret :
    #   fond estimé = 0.75
    #   pixel de fond  = 0.73  ->  |0.73 - 0.75| = 0.02  (petit -> fond)
    #   pixel de pièce = 0.30  ->  |0.30 - 0.75| = 0.45  (grand -> pièce)
    #
    # La valeur absolue est importante : la pièce peut être plus sombre
    # OU plus claire que le fond (ex: pièce brillante sur fond sombre).
    difference = np.abs(gris - fond)
    difference_floue = flou_gaussien(difference, max(5, marge | 1))
    seuil = max(0.08, seuil_otsu(difference_floue))
    masque = ouverture_binaire(difference_floue > seuil, 5)

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(600, int(aire_image * 0.003))
    aire_max = int(aire_image * 0.30)
    composantes = extraire_composantes_utiles(masque, aire_min, aire_max)

    # Utilisation des hyperparamètres
    candidates = [
        comp
        for comp in composantes
        if comp["remplissage"] >= SINGLE_COIN_FILL_MIN
        and comp["circularite"] >= SINGLE_COIN_CIRCULARITY_MIN
        and SINGLE_COIN_ASPECT_RATIO_MIN
        <= comp["hauteur_bbox"] / max(1, comp["largeur_bbox"])
        <= SINGLE_COIN_ASPECT_RATIO_MAX
    ]

    return len(candidates) == 1


# =============================================================================
# FONCTION PRINCIPALE : COMPTER LES PIÈCES
# =============================================================================
def compter_pieces(chemin_image, taille_flou=(7, 7)):
    """
    Fonction principale : compte le nombre de pièces dans une image.
    
    APERÇU GLOBAL :
    
    Ce programme implémente un PIPELINE COMPLET de traitement d'image,
    démontrant les concepts de chaque semaine du cours.
    
    SEMAINE 1-2  : Représentation (lire l'image, normaliser en RGB propre)
    SEMAINE 3    : Espaces couleur (HSL saturation pour la robustesse)
    SEMAINE 5    : Seuillage (Otsu automatique, diviseur 2 classes)
    SEMAINE 6    : Morphologie (ouverture = érosion + dilatation)
    SEMAINE 7    : Composantes connexes (8-connectivity, descripteurs)
    SEMAINE 8    : Convolution & Filtrage (Gaussian blur séparable)
    
    FLUX DE CONTRÔLE :
    
    1. ENTRÉE : chemin_image + taille_flou=[7,7]
    
    2. DÉTECTION PRINCIPALE
       ├─ Voie saturation (robuste aux ombres)
       ├─ Applique tout le pipeline Weeks 1-8
       └─ Retourne : prédiction + composantes détaillées
       
    3. DÉTECTION SECONDAIRE (Filet de sécurité)
       ├─ Voie contraste gris (alternative)
       ├─ Cherche si exactement 1 pièce probable
       └─ Retourne : booléen True/False
       
    4. RÈGLES CORRECTIVES (Post-traitement)
       ├─ Règle 1 : Si prédiction >= 4 mais 1 grosse pièce → corrige à 1
       ├─ Règle 2 : Si prédiction >= 4 mais peu de composantes → corrige à 1
       └─ Règle 3 : Si détection secondaire positive → applique correction
       
    5. SORTIE : nombre final de pièces
    
    RAISON DES CORRECTIONS :
    Une erreur de 4 au lieu de 1 = MAE +3
    Corriger à 1 = MAE +0
    → Important pour réduire le MAE (metrics Week 10)

    Étapes globales :
    -----------------
    1) lecture et normalisation de l'image
    2) détection principale
    3) détection spéciale "une seule pièce"
    4) règles correctives finales
    5) retour de la prédiction finale

    Pourquoi cette structure :
    --------------------------
    On sépare :
    - la prédiction principale
    - la logique de correction

    Cela rend le pipeline plus robuste et plus lisible.

    Idée de la correction finale :
    ------------------------------
    Même si la détection principale se trompe,
    on peut parfois détecter qu'il s'agit en réalité d'une seule pièce
    à partir de la forme globale observée.

    Exemple :
    ---------
    Cas difficile :
    - vraie valeur = 1
    - détection principale = 4
    - mais on observe une seule grande composante très circulaire
    -> on corrige à 1

    Pourquoi c'est bon pour le MAE :
    --------------------------------
    Une erreur de 4 au lieu de 1 donne une erreur absolue de 3.
    Si on corrige à 1, l'erreur devient 0.
    """
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        return 0

    prediction_principale, composantes = detection_principale(image_rgb, taille_flou)
    piece_unique = detection_piece_unique(image_rgb)

    # Si on a des composantes, on peut appliquer des règles de cohérence globale.
    if composantes:
        aires = sorted(comp["area"] for comp in composantes)
        aire_mediane = float(np.median(aires))

        # On cherche des très grandes composantes bien rondes et bien remplies.
        #
        # Idée :
        # ------
        # Si la détection principale a beaucoup surcompté,
        # mais qu'on observe en réalité une seule grande forme circulaire,
        # cela suggère qu'il y a une seule grosse pièce.
        
        # Utilisation des hyperparamètres
        grandes_pieces_circulaires = [
            comp
            for comp in composantes
            if comp["circularite"] >= CORRECTION_LARGE_CIRCULARITY_MIN
            and comp["remplissage"] >= CORRECTION_LARGE_FILL_MIN
            and comp["area"] >= CORRECTION_LARGE_AREA_MULTIPLIER * max(1.0, aire_mediane)
        ]

        # Règle 1 :
        # Si la prédiction principale est très grande, mais qu'on voit une seule
        # énorme composante circulaire, on corrige à 1.
        if prediction_principale >= 4 and len(grandes_pieces_circulaires) == 1:
            return 1

        # Règle 2 :
        # Si la prédiction principale est grande, mais qu'il y a UNE SEULE
        # composante qui ressemble fortement à une pièce, on corrige à 1.
        #
        # NOTE : on utilise == 1 et non <= 2.
        # Raisonnement : si deux composantes DISTINCTES passent tous les filtres
        # (aire, bord, circularité, remplissage), c'est qu'il y a probablement
        # au moins 2 vraies pièces. La règle ne doit pas annuler cette information.
        # Avec <= 2, des images de 10 pièces fusionnées en 2 composantes étaient
        # incorrectement corrigées à 1 (erreur +9 au MAE).

        # Utilisation des hyperparamètres
        if (
            prediction_principale >= 4
            and len(composantes) == 1
            and any(comp["circularite"] >= CORRECTION_RARE_CIRCULARITY_MIN and comp["remplissage"] >= CORRECTION_RARE_FILL_MIN for comp in composantes)
        ):
            return 1

    # Règles liées à la détection spéciale "une seule pièce".
    if piece_unique:
        # Si la détection principale n'a rien vu, mais que la détection spéciale
        # voit clairement une seule pièce plausible, on corrige à 1.
        if prediction_principale == 0:
            return 1

        # Si la détection principale donne un nombre exagérément grand par rapport
        # au nombre de composantes, on corrige à 1.
        if prediction_principale >= 3 * max(1, len(composantes)):
            return 1

    return prediction_principale