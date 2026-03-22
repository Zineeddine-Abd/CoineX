import cv2
import numpy as np

def compter_pieces(chemin_image, taille_flou=(15, 15)):
    img = cv2.imread(chemin_image)
    if img is None: return 0

    # 1. Conversion Colorimétrique Stratégique (Espace HSL)
    # On isole la saturation pour s'abstraire totalement des ombres
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _, _, saturation = cv2.split(hls)

    # Vérification de sécurité : le noyau Gaussien doit toujours être impair
    t_x, t_y = taille_flou
    if t_x % 2 == 0: t_x += 1
    if t_y % 2 == 0: t_y += 1

    # 2. Opération Locale : Lissage avec l'argument ajustable
    floute = cv2.GaussianBlur(saturation, (t_x, t_y), 0)

    # 3. Clustering : Algorithme d'Otsu sur la saturation
    _, binarisee = cv2.threshold(floute, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Opérations Locales : Post-traitement morphologique
    # On nettoie les petits bruits restants
    noyau = np.ones((15,15), np.uint8)
    propre = cv2.morphologyEx(binarisee, cv2.MORPH_OPEN, noyau)

    # 5. Extraction d'Information : Filtrage souple sur l'aire
    nb_labels, _, stats, _ = cv2.connectedComponentsWithStats(propre)
    
    compteur = 0
    # Des seuils très larges : on accepte les ellipses (pièces vues de biais)
    # tant qu'elles ont une taille cohérente avec une pièce.
    aire_minimale = 10000  
    aire_maximale = 60000

    for i in range(1, nb_labels):
        aire = stats[i, cv2.CC_STAT_AREA]
        if aire_minimale < aire < aire_maximale:
            compteur += 1

    return compteur