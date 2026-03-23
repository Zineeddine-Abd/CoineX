import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rgb_vers_saturation(img):
    """
    Remplace cv2.cvtColor(img, cv2.COLOR_BGR2HLS) et le split.
    Calcule mathématiquement le canal de Saturation depuis le RGB.
    """
    # Normalisation entre 0 et 1
    img = img.astype(float)
    if img.max() > 1.0:
        img /= 255.0
        
    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin
    luminosite = (cmax + cmin) / 2.0
    
    # Formule mathématique de la saturation HSL
    saturation = np.zeros_like(luminosite)
    mask = delta != 0
    saturation[mask] = delta[mask] / (1.0 - np.abs(2.0 * luminosite[mask] - 1.0))
    
    # Retour sur une échelle 0-255
    return (saturation * 255).astype(np.uint8)

def appliquer_flou_gaussien(img, ksize=15):
    """
    Remplace cv2.GaussianBlur. Utilise la convolution mathématique pure avec NumPy.
    """
    # 1. Création du noyau Gaussien 1D
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_1d = gauss / np.sum(gauss)
    
    # 2. Convolution séparable (horizontale puis verticale pour des performances optimales)
    flou_h = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=1, arr=img)
    flou_v = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode='same'), axis=0, arr=flou_h)
    return flou_v

def seuillage_otsu(img):
    """
    Remplace cv2.threshold(..., cv2.THRESH_OTSU).
    """
    # Calcul de l'histogramme de l'image
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    
    sumB, wB, max_var, seuil_optimal = 0.0, 0, 0.0, 0
    
    # Parcours de tous les seuils de 0 à 255
    for i in range(256):
        wB += hist[i]  # Poids du fond
        if wB == 0: continue
        wF = total - wB  # Poids du premier plan
        if wF == 0: break
        
        sumB += i * hist[i]
        mB = sumB / wB  # Moyenne du fond
        mF = (sum_total - sumB) / wF  # Moyenne du premier plan
        
        # Variance inter-classe (Between-class variance)
        var_between = wB * wF * (mB - mF) ** 2
        
        # On cherche à maximiser cette variance
        if var_between > max_var:
            max_var = var_between
            seuil_optimal = i
            
    # Binarisation
    binarisee = np.zeros_like(img)
    binarisee[img > seuil_optimal] = 255
    return binarisee

def ouverture_morphologique(bin_img, ksize=15):
    """
    Remplace cv2.morphologyEx(..., cv2.MORPH_OPEN).
    Une astuce de Data Science consiste à utiliser Pandas pour l'érosion/dilatation !
    """
    df = pd.DataFrame(bin_img)
    
    # 1. ÉROSION = Recherche du Minimum local avec une fenêtre glissante (rolling)
    # Lignes
    eroded = df.rolling(window=ksize, center=True, min_periods=1).min()
    # Colonnes (on transpose, on applique, on re-transpose)
    eroded = eroded.T.rolling(window=ksize, center=True, min_periods=1).min().T
    
    # 2. DILATATION = Recherche du Maximum local sur l'image érodée
    dilated = eroded.rolling(window=ksize, center=True, min_periods=1).max()
    dilated = dilated.T.rolling(window=ksize, center=True, min_periods=1).max().T
    
    return dilated.values

def extraire_aires_composantes(bin_img):
    """
    Remplace cv2.connectedComponentsWithStats.
    Algorithme de parcours en profondeur (DFS) pour trouver les objets connexes.
    """
    aires = []
    
    # On récupère tous les (y, x) des pixels blancs
    pixels_blancs = np.argwhere(bin_img == 255)
    ensemble_blancs = set(map(tuple, pixels_blancs))
    ensemble_visites = set()
    
    for r, c in pixels_blancs:
        if (r, c) in ensemble_visites:
            continue
            
        # Démarrage d'un nouveau bloc (Pile pour le DFS)
        pile = [(r, c)]
        ensemble_visites.add((r, c))
        aire_actuelle = 0
        
        while pile:
            curr_r, curr_c = pile.pop()
            aire_actuelle += 1
            
            # Vérification des 4 pixels voisins (connexité-4)
            voisins = [(curr_r-1, curr_c), (curr_r+1, curr_c), (curr_r, curr_c-1), (curr_r, curr_c+1)]
            for vr, vc in voisins:
                if (vr, vc) in ensemble_blancs and (vr, vc) not in ensemble_visites:
                    ensemble_visites.add((vr, vc))
                    pile.append((vr, vc))
                    
        aires.append(aire_actuelle)
        
    return aires

def compter_pieces(chemin_image, taille_flou=(15, 15)):
    # 0. Chargement via Matplotlib
    img = plt.imread(chemin_image)
    if img is None: return 0

    # 1. Conversion Colorimétrique Stratégique
    saturation = rgb_vers_saturation(img)

    # Vérification de sécurité : le noyau Gaussien doit toujours être impair
    t_x, t_y = taille_flou
    if t_x % 2 == 0: t_x += 1
    if t_y % 2 == 0: t_y += 1

    # 2. Opération Locale : Lissage
    floute = appliquer_flou_gaussien(saturation, ksize=t_x)

    # 3. Clustering : Algorithme d'Otsu
    binarisee = seuillage_otsu(floute)

    # 4. Opérations Locales : Post-traitement morphologique
    propre = ouverture_morphologique(binarisee, ksize=15)

    # 5. Extraction d'Information : Filtrage souple sur l'aire
    aires = extraire_aires_composantes(propre)
    
    compteur = 0
    aire_minimale = 10000  
    aire_maximale = 60000

    for aire in aires:
        if aire_minimale < aire < aire_maximale:
            compteur += 1

    return compteur