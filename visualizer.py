"""
VISUALIZER - Pipeline Processing Visualization
===============================================

This module visualizes each step of the coin detection pipeline.
It's completely independent and doesn't modify any existing code.

USAGE:
    from visualizer import visualiser_pipeline
    visualiser_pipeline("path/to/image.jpg")
    
    OR
    
    python visualizer.py "path/to/image.jpg"
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Import core functions from traitement_2
from traitement_2 import (
    lire_image_rgb,
    rgb_vers_hsl,
    rgb_vers_gris,
    flou_gaussien,
    seuil_otsu,
    ouverture_binaire,
    composantes_connexes,
    extraire_composantes_utiles,
    compter_pieces,
    # Hyperparameters
    COIN_REFERENCE_CIRCULARITY_MIN,
    COIN_REFERENCE_FILL_MIN,
    COIN_REFERENCE_ASPECT_MIN,
    COIN_REFERENCE_ASPECT_MAX,
)


def visualiser_pipeline(chemin_image, taille_flou=(7, 7), save_path=None):
    """
    Visualise chaque étape du pipeline de détection de pièces.
    
    OBJECTIF PÉDAGOGIQUE :
    Montrer visuellement comment le pipeline transforme l'image originale
    en comptage final de pièces, étape par étape.
    
    PARAMÈTRES :
    -----------
    chemin_image : str
        Chemin vers l'image à traiter
    
    taille_flou : tuple
        Taille du noyau Gaussien (par défaut : (7,7))
    
    save_path : str (optionnel)
        Si fourni, sauvegarde la figure à ce chemin au lieu d'afficher
    
    ÉTAPES VISUALISÉES :
    -------------------
    1. Image originale (RGB)
    2. Canal saturation (HSL)
    3. Image floutée (Gaussian blur)
    4. Image seuillée (Otsu binary)
    5. Après ouverture morphologique (nettoyage)
    6. Composantes connexes colorisées
    7. Statistiques finales
    
    EXEMPLE :
    --------
    >>> visualiser_pipeline("data/test/coin_image.jpg")
    >>> # Affiche une grille 2×4 avec chaque étape
    """
    
    # =========================================================================
    # ÉTAPE 0 : CHARGEMENT IMAGE ORIGINALE (Week 1-2)
    # =========================================================================
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        print(f"❌ ERREUR : Impossible de charger l'image {chemin_image}")
        return
    
    print(f"✓ Image chargée : {image_rgb.shape[0]}×{image_rgb.shape[1]}×{image_rgb.shape[2]}")
    
    # Créer la figure avec sous-graphiques
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        'Pipeline Complet de Détection de Pièces\n' +
        '(Cours Image Processing - Weeks 1-8)',
        fontsize=14, fontweight='bold'
    )
    
    # =========================================================================
    # ÉTAPE 1 : IMAGE ORIGINALE
    # =========================================================================
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Étape 1 : Image Originale (RGB)\nWeek 1-2', fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(
        0.5, -0.15, f'Shape: {image_rgb.shape}',
        ha='center', transform=axes[0, 0].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 2 : SATURATION (Week 3 - Color Spaces)
    # =========================================================================
    _, saturation = rgb_vers_hsl(image_rgb)
    axes[0, 1].imshow(saturation, cmap='gray')
    axes[0, 1].set_title('Étape 2 : Saturation (HSL)\nWeek 3 - Espaces couleur', fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 1].text(
        0.5, -0.15, 'RGB → HSL\nRobuste aux ombres',
        ha='center', transform=axes[0, 1].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 3 : FLOU GAUSSIEN (Week 8 - Filtrage)
    # =========================================================================
    if isinstance(taille_flou, tuple):
        taille_flou_val = taille_flou[0]
    else:
        taille_flou_val = taille_flou
    if taille_flou_val % 2 == 0:
        taille_flou_val += 1
    
    saturation_floue = flou_gaussien(saturation, taille_flou_val)
    axes[0, 2].imshow(saturation_floue, cmap='gray')
    axes[0, 2].set_title(f'Étape 3 : Gaussian Blur\nWeek 8 - Convolution\nKernel size: {taille_flou_val}×{taille_flou_val}', fontweight='bold')
    axes[0, 2].axis('off')
    axes[0, 2].text(
        0.5, -0.15, 'Réduit bruit\nOpération locale',
        ha='center', transform=axes[0, 2].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 4 : SEUILLAGE OTSU (Week 5 - Thresholding)
    # =========================================================================
    seuil = seuil_otsu(saturation_floue)
    image_binaire = saturation_floue > seuil
    axes[0, 3].imshow(image_binaire, cmap='gray')
    axes[0, 3].set_title(f'Étape 4 : Seuillage Otsu\nWeek 5 - Segmentation\nThreshold: {seuil:.3f}', fontweight='bold')
    axes[0, 3].axis('off')
    axes[0, 3].text(
        0.5, -0.15, 'Max variance\ninter-classes',
        ha='center', transform=axes[0, 3].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 5 : OUVERTURE MORPHOLOGIQUE (Week 6 - Morphology)
    # =========================================================================
    masque_cleané = ouverture_binaire(image_binaire, 3)
    axes[1, 0].imshow(masque_cleané, cmap='gray')
    axes[1, 0].set_title('Étape 5 : Ouverture Morphologique\nWeek 6 - Morphology\n(Érosion + Dilatation)', fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 0].text(
        0.5, -0.15, 'Nettoie petits\npara sites',
        ha='center', transform=axes[1, 0].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 6 : COMPOSANTES CONNEXES (Week 7 - Component Analysis)
    # =========================================================================
    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(500, int(aire_image * 0.0010))
    aire_max = int(aire_image * 0.18)
    composantes = extraire_composantes_utiles(masque_cleané, aire_min, aire_max)
    
    # Colorer les composantes
    composantes_map = np.zeros(masque_cleané.shape, dtype=np.int32)
    for i, comp in enumerate(composantes, 1):
        y_min, x_min, y_max, x_max = comp["bbox"]
        bbox_mask = np.zeros_like(masque_cleané, dtype=bool)
        bbox_mask[y_min:y_max+1, x_min:x_max+1] = True
        # Utiliser un label unique pour chaque composante
        composantes_map[bbox_mask & masque_cleané] = i
    
    # Afficher avec colormap
    axes[1, 1].imshow(composantes_map, cmap='tab20', interpolation='nearest')
    axes[1, 1].set_title(f'Étape 6 : Composantes Connexes\nWeek 7 - Component Labeling\n{len(composantes)} objets trouvés', fontweight='bold')
    axes[1, 1].axis('off')
    axes[1, 1].text(
        0.5, -0.15, f'8-connectivity\n{len(composantes)} régions',
        ha='center', transform=axes[1, 1].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 7 : DESSINER LES DESCRIPTEURS DE FORME
    # =========================================================================
    axes[1, 2].imshow(image_rgb)
    
    # Marquer les composantes de "référence" (vraies pièces)
    has_reference = False
    for comp in composantes:
        y_min, x_min, y_max, x_max = comp["bbox"]
        largeur_bbox = comp["largeur_bbox"]
        hauteur_bbox = comp["hauteur_bbox"]
        
        ratio = hauteur_bbox / max(1, largeur_bbox)
        is_reference = (
            comp["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
            and comp["remplissage"] >= COIN_REFERENCE_FILL_MIN
            and COIN_REFERENCE_ASPECT_MIN <= ratio <= COIN_REFERENCE_ASPECT_MAX
        )
        
        if is_reference:
            has_reference = True
            color = 'green'
            linewidth = 3
            label = '✓'
        else:
            color = 'red'
            linewidth = 1
            label = '✗'
        
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min, y_max - y_min,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        axes[1, 2].add_patch(rect)
    
    axes[1, 2].set_title('Étape 7 : Descripteurs de Forme\nWeek 7 - Shape Properties\n(vert=pièce, rouge=rejeté)', fontweight='bold')
    axes[1, 2].axis('off')
    axes[1, 2].text(
        0.5, -0.15, f'{sum(1 for c in composantes if c["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN)} pièces\nde référence',
        ha='center', transform=axes[1, 2].transAxes, fontsize=9
    )
    
    # =========================================================================
    # ÉTAPE 8 : RÉSULTAT FINAL
    # =========================================================================
    prediction = compter_pieces(chemin_image, taille_flou)
    
    # Afficher résumé
    axes[1, 3].axis('off')
    
    summary_text = f"""
    RÉSULTAT FINAL
    ══════════════════
    
    Pièces détectées : {prediction}
    
    ──────────────────
    Statistiques :
    • Composantes trouvées : {len(composantes)}
    • Aire image : {aire_image} pixels
    • Aire min : {aire_min}
    • Aire max : {aire_max}
    • Taille flou : {taille_flou}
    
    ──────────────────
    Références :
    • Circularité min : {COIN_REFERENCE_CIRCULARITY_MIN}
    • Remplissage min : {COIN_REFERENCE_FILL_MIN}
    • Ratio aspect : [{COIN_REFERENCE_ASPECT_MIN}, {COIN_REFERENCE_ASPECT_MAX}]
    """
    
    axes[1, 3].text(
        0.05, 0.95, summary_text,
        fontsize=10, family='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # =========================================================================
    # AFFICHAGE FINAL
    # =========================================================================
    plt.tight_layout()
    
    if save_path:
        print(f"✓ Sauvegarde figure à : {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    print(f"\n✓ Résultat : {prediction} pièce(s) détectée(s)")
    
    return prediction, composantes


def visualiser_descripteurs(chemin_image):
    """
    Affiche un graphique des descripteurs de formes trouvées.
    Utile pour comprendre les critères de sélection.
    
    EXEMPLE :
    --------
    >>> visualiser_descripteurs("data/test/coin_image.jpg")
    """
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        print(f"❌ ERREUR : Impossible de charger l'image {chemin_image}")
        return
    
    _, saturation = rgb_vers_hsl(image_rgb)
    saturation_floue = flou_gaussien(saturation, 7)
    seuil = seuil_otsu(saturation_floue)
    image_binaire = saturation_floue > seuil
    masque_cleané = ouverture_binaire(image_binaire, 3)
    
    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    aire_min = max(500, int(aire_image * 0.0010))
    aire_max = int(aire_image * 0.18)
    composantes = extraire_composantes_utiles(masque_cleané, aire_min, aire_max)
    
    if not composantes:
        print("Aucune composante trouvée")
        return
    
    # Extraire les descripteurs
    aires = [c["area"] for c in composantes]
    circularites = [c["circularite"] for c in composantes]
    remplissages = [c["remplissage"] for c in composantes]
    ratios = [c["hauteur_bbox"] / max(1, c["largeur_bbox"]) for c in composantes]
    
    # Créer figure avec 4 graphiques
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution des Descripteurs de Forme', fontsize=14, fontweight='bold')
    
    # Graphique 1 : Aire
    axes[0, 0].hist(aires, bins=10, edgecolor='black', color='skyblue')
    axes[0, 0].set_xlabel('Aire (pixels)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('1. Distribution des aires')
    axes[0, 0].axvline(np.median(aires), color='r', linestyle='--', label=f'Médiane: {np.median(aires):.0f}')
    axes[0, 0].legend()
    
    # Graphique 2 : Circularité
    axes[0, 1].scatter(range(len(circularites)), circularites, color='green', s=100, alpha=0.6)
    axes[0, 1].axhline(COIN_REFERENCE_CIRCULARITY_MIN, color='r', linestyle='--', label=f'Seuil: {COIN_REFERENCE_CIRCULARITY_MIN}')
    axes[0, 1].set_xlabel('Composante #')
    axes[0, 1].set_ylabel('Circularité')
    axes[0, 1].set_title('2. Circularité (1.0 = cercle parfait)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graphique 3 : Remplissage
    axes[1, 0].scatter(range(len(remplissages)), remplissages, color='orange', s=100, alpha=0.6)
    axes[1, 0].axhline(COIN_REFERENCE_FILL_MIN, color='r', linestyle='--', label=f'Seuil: {COIN_REFERENCE_FILL_MIN}')
    axes[1, 0].set_xlabel('Composante #')
    axes[1, 0].set_ylabel('Remplissage')
    axes[1, 0].set_title('3. Remplissage = aire / bbox_aire')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique 4 : Ratio hauteur/largeur
    axes[1, 1].scatter(range(len(ratios)), ratios, color='purple', s=100, alpha=0.6)
    axes[1, 1].axhline(COIN_REFERENCE_ASPECT_MIN, color='r', linestyle='--', label=f'Min: {COIN_REFERENCE_ASPECT_MIN}')
    axes[1, 1].axhline(COIN_REFERENCE_ASPECT_MAX, color='r', linestyle='--', label=f'Max: {COIN_REFERENCE_ASPECT_MAX}')
    axes[1, 1].set_xlabel('Composante #')
    axes[1, 1].set_ylabel('Ratio hauteur/largeur')
    axes[1, 1].set_title('4. Aspect Ratio (1.0 = carré)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Permettre d'utiliser depuis la ligne de commande
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n{'='*60}")
        print(f"Visualisation du pipeline : {image_path}")
        print(f"{'='*60}\n")
        visualiser_pipeline(image_path)
    else:
        print("USAGE: python visualizer.py <path_to_image>")
        print("\nEXEMPLE:")
        print("  python visualizer.py data/test/coin001.jpg")
        print("\nOU en Python:")
        print("  from visualizer import visualiser_pipeline")
        print("  visualiser_pipeline('data/test/coin001.jpg')")
