import numpy as np

# =============================================================================
# HISTOGRAMME ET SEUIL D'OTSU
# =============================================================================
def histogramme_u8(image):
    """
    Convertit une image normalisée [0,1] en uint8 [0,255], puis calcule
    l'histogramme des intensités.

    Pourquoi on passe en uint8 :
    ----------------------------
    Otsu est ici implémenté sur 256 niveaux de gris.
    Cela simplifie le calcul de l'histogramme.

    Exemple :
    ---------
    Un pixel 0.0 devient 0
    Un pixel 0.5 devient environ 128
    Un pixel 1.0 devient 255
    """
    # rint arrondi à l'entier le plus proche, clip pour éviter les débordements, puis convertit en uint8
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    # np.bincount compte le nombre d'occurrences de chaque valeur de pixel (0 à 255)
    # ravel : aplati l'image 2D en 1D pour que bincount puisse compter tous les pixels
    return np.bincount(image_u8.ravel(), minlength=256).astype(np.float64)


# =============================================================================
# ÉGALISATION D'HISTOGRAMME
# =============================================================================
def egaliser_histogramme(image):
    """
    Égalise l'histogramme d'une image en niveaux de gris normalisée [0,1].

    Égalisation d'histogramme (Histogram Equalization)
    -----------------------------------------------------------------------
    But : améliorer le contraste d'une image sur- ou sous-exposée en
    redistribuant les intensités de façon plus uniforme.

    FORMULE DU         y = max(0,  256 × C_I(x) − 1)

    Où :
        x    = niveau d'intensité d'entrée (0 à 255)
        C_I  = histogramme CUMULÉ normalisé (entre 0 et 1)
        y    = nouveau niveau d'intensité de sortie (0 à 255)

    INTUITION :
    -----------
    Si une image est trop sombre, les pixels sont concentrés dans les basses
    intensités. L'histogramme cumulatif monte donc très vite au début, puis
    s'aplatit. La formule "étire" cette partie basse vers toute la plage [0,255].
    Résultat : les zones sombres deviennent plus contrastées.

    ÉTAPES PAS-À-PAS :
    ------------------
    1) Convertir l'image en uint8 [0,255] et calculer son histogramme
       hist[i] = nombre de pixels d'intensité i
    2) Normaliser : probabilites[i] = hist[i] / total_pixels
    3) Calculer l'histogramme cumulé C[i] = Σ_{k=0}^{i} probabilites[k]
       C[255] vaut toujours exactement 1.0
    4) Appliquer la transformation : sortie[i] = max(0, 256 × C[i] − 1)
       et clipper dans [0,255]
    5) Remapper chaque pixel de l'image avec cette table de correspondance
    6) Renormaliser le résultat en [0,1] pour rester compatible avec le pipeline

    EXEMPLE :
    ---------
    Image très sombre → la majorité des pixels ont une intensité entre 0 et 80.
    Après égalisation, ces pixels sont répartis entre 0 et 255.
    → les nuances dans les zones sombres deviennent visibles.

    Paramètre :
    -----------
    image : tableau 2D float32 ou float64, valeurs dans [0.0, 1.0]

    Retour :
    --------
    image_egalisee : tableau 2D float32, valeurs dans [0.0, 1.0]
    """
    # Étape 1 : histogramme des niveaux d'intensité (256 niveaux)
    hist = histogramme_u8(image)   # hist[i] = nombre de pixels d'intensité i
    total = hist.sum()
    if total == 0:
        return image.copy()

    # Étape 2 : histogramme cumulé normalisé C_I
    # C_I[i] = Σ_{k=0}^{i} hist[k] / total   (fraction de pixels ≤ i)
    cumul = np.cumsum(hist) / total   # np.cumsum = somme cumulée, de gauche à droite

    # Étape 3 : table de correspondance (LUT — Look-Up Table)
    # Pour chaque intensité d'entrée i, on calcule la nouvelle intensité de sortie.
    # y = max(0,  256 × C_I(i) − 1)
    # On clippe ensuite entre 0 et 255 pour rester dans la plage valide.
    lut = np.clip(256.0 * cumul - 1.0, 0.0, 255.0)   # LUT : 256 valeurs

    # Étape 4 : conversion de l'image en entiers 0–255 pour utiliser la LUT
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.int32)

    # Étape 5 : remapping — chaque pixel est remplacé par lut[valeur_pixel]
    # image_u8 sert ici d'INDEX dans la table lut.
    image_egalisee_u8 = lut[image_u8]   # indexation tableau : opération de base NumPy

    # Étape 6 : renormalisation en [0,1] pour rester compatible avec le reste du pipeline
    return (image_egalisee_u8 / 255.0).astype(np.float32)


def seuil_otsu(image):
    """
    Calcule automatiquement un seuil d'Otsu dans [0,1].
    
    Segmentation par seuillage & Algorithme d'Otsu
    ================================================================
    
    PRINCIPE MATHÉMATIQUE :
    Maximise la variance entre-classes σ²_B(t) pour tous seuils t ∈ [0,255]
    
    σ²_B(t) = w0(t) * w1(t) * (μ0(t) - μ1(t))²
    
    Où :
    - w0(t) = fraction de pixels en dessous du seuil t (fond)
    - w1(t) = fraction de pixels au-dessus du seuil t (objets)
    - μ0(t) = intensité moyenne du fond   
    - μ1(t) = intensité moyenne des objets
    
    ALGORITHME :
    1. Calculer histogramme de l'image (256 niveaux)
    2. Pour chaque seuil t, calculer la variance inter-classes
    3. Retourner le seuil avec variance maximale
    
    AVANTAGE : Seuil déterministe, s'adapte à l'image (pas de paramètre à régler)
    Comparaison avec K-Means : Otsu est OPTIMAL pour 2 classes, K-Means dépend de l'initialisation

    Principe d'Otsu :
    -----------------
    On cherche le seuil qui sépare au mieux deux classes :
    - le fond
    - les objets

    Pour chaque seuil possible, on mesure la séparation entre les deux classes
    avec la variance inter-classes, puis on prend le meilleur.

    Pourquoi c'est utile :
    ----------------------
    On évite de choisir un seuil à la main.
    Le seuil s'adapte à l'image.

    Exemple :
    ---------
    Si l'histogramme a :
    - un pic autour de 20 pour le fond
    - un pic autour de 170 pour les pièces
    Otsu choisira un seuil intermédiaire.

    Explication :
    -------------------------------------------------------------
    Pourquoi Otsu et pas les K-Moyennes ?
    Ici, nous voulons séparer exactement 2 classes (Fond vs Pièce).
    L'algorithme d'Otsu teste de manière exhaustive tous les seuils pour 
    minimiser la variance intra-classe. Il garantit donc une solution mathématiquement 
    OPTIMALE pour ce cas précis, contrairement aux K-Moyennes qui dépendent 
    de leur initialisation aléatoire.
    """
    hist = histogramme_u8(image)
    total = hist.sum()
    if total == 0:
        return 0.5

    # -------------------------------------------------------------------------
    # Algorithme d’Otsu, boucle explicite pas-à-pas
    # -------------------------------------------------------------------------
    # Objectif : trouver le seuil t* qui MAXIMISE la variance inter-classes :
    #
    #   σ²_B(t) = w0(t) · w1(t) · (μ0(t) − μ1(t))²
    #
    #   w0(t) = poids de la classe "fond"    (pixels d’intensité ≤ t)
    #   w1(t) = poids de la classe "objets"  (pixels d’intensité > t)
    #   μ0(t) = intensité moyenne de la classe fond
    #   μ1(t) = intensité moyenne de la classe objets
    #
    # On teste chaque seuil t de 0 à 255 et on garde le meilleur.
    # Calcul INCRÉMENTAL : on accumule w0 et la somme des intensités du fond
    # au fur et à mesure, évitant de tout recalculer depuis zéro à chaque t.
    # -------------------------------------------------------------------------

    # Précalcul : somme totale de toutes les intensités pondérées par leur fréquence
    # somme_totale = Σ_{i=0}^{255}  i · hist[i]
    somme_totale = 0.0
    for i in range(256):
        somme_totale += i * hist[i]

    poids_fond  = 0.0   # Σ hist[0..t]          — compte de pixels dans la classe 0
    somme_fond  = 0.0   # Σ i·hist[0..t]         — somme des intensités de la classe 0

    meilleure_variance = -1.0
    meilleur_seuil     = 128    # valeur par défaut de secours

    for t in range(256):
        # Ajout des pixels d’intensité t à la classe 0
        poids_fond += hist[t]
        somme_fond += t * hist[t]

        # Classe 0 vide → ce seuil ne sépare rien, on passe au suivant
        if poids_fond == 0:
            continue

        # Classe 1 vide → tous les pixels sont dans la classe 0, on s’arrête
        poids_objet = total - poids_fond
        if poids_objet == 0:
            break

        # Moyennes des deux classes
        moyenne_fond   = somme_fond / poids_fond
        moyenne_objet  = (somme_totale - somme_fond) / poids_objet

        # Variance inter-classes σ²_B = w0·w1·(μ0−μ1)²
        # (on utilise des comptes bruts ; la division par total² serait constante
        #  et n’affecterait pas l’argmax)
        variance = poids_fond * poids_objet * (moyenne_fond - moyenne_objet) ** 2

        if variance > meilleure_variance:
            meilleure_variance = variance
            meilleur_seuil     = t

    # Ramener le seuil en [0,1] (le pipeline travaille avec des valeurs normalisées)
    return meilleur_seuil / 255.0


