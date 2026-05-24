"""
visualizer.py — Visualisation pas-à-pas du pipeline de détection de pièces
===========================================================================

Ce fichier permet de voir ce que fait l'algorithme à chaque étape,
sous forme d'une grille d'images côte à côte.

Comment utiliser :
    from visualizer import visualiser_pipeline
    visualiser_pipeline("data/validation/img_001.jpg")

    OU directement depuis le terminal :
    python visualizer.py data/validation/img_001.jpg
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from traitement import (
    lire_image_rgb,
    rgb_vers_hsl,
    flou_gaussien,
    seuil_otsu,
    ouverture_binaire,
    fermeture_binaire,
    composantes_connexes,
    compter_pieces,
    COIN_REFERENCE_CIRCULARITY_MIN,
    COIN_REFERENCE_FILL_MIN,
    COIN_REFERENCE_ASPECT_MIN,
    COIN_REFERENCE_ASPECT_MAX,
    MERGE_AREA_RATIO_THRESHOLD,
    MERGE_FILL_MIN,
)


def visualiser_pipeline(chemin_image, taille_flou=(7, 7), save_path=None):
    """
    Affiche les 8 étapes du pipeline sous forme d'une grille 2x4.

    Paramètres :
        chemin_image : chemin vers l'image à analyser
        taille_flou  : taille du noyau gaussien, ex. (7, 7)
        save_path    : si fourni, sauvegarde l'image au lieu de l'afficher

    Ce que tu vas voir dans la grille :
        Ligne 1 :  Image originale | Saturation | Saturation floutée | Masque Otsu
        Ligne 2 :  Masque nettoyé  | Composantes colorées | Boites détectées | Résultat
    """

    # ------------------------------------------------------------------
    # CHARGEMENT DE L'IMAGE
    # On charge l'image une seule fois et on calcule le résultat final
    # en premier. Comme ca on peut l'afficher à la fin sans recalculer.
    # ------------------------------------------------------------------
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        print("Erreur : impossible de charger l'image '" + chemin_image + "'")
        return

    hauteur, largeur = image_rgb.shape[:2]
    aire_image = hauteur * largeur

    # Résultat final calculé UNE SEULE FOIS ici
    prediction_finale = compter_pieces(chemin_image, taille_flou)

    # ------------------------------------------------------------------
    # REPRODUCTION DU PIPELINE PAS-À-PAS (pour la visualisation)
    # On reproduit manuellement les mêmes étapes que detection_principale
    # afin de pouvoir afficher chaque résultat intermédiaire.
    # ------------------------------------------------------------------

    # Étape A : saturation
    _, saturation = rgb_vers_hsl(image_rgb)

    # Étape B : flou gaussien
    taille_locale = max(
        int(round((taille_flou[0] + taille_flou[1]) / 2.0)),
        int(round(min(hauteur, largeur) / 90)),
        5,
    )
    if taille_locale % 2 == 0:
        taille_locale += 1
    saturation_floue = flou_gaussien(saturation, taille_locale)

    # Étape C : seuillage Otsu
    seuil = seuil_otsu(saturation_floue)
    masque_brut = saturation_floue > seuil

    # Étape D : nettoyage morphologique (ouverture PUIS fermeture, comme le pipeline réel)
    taille_morpho = max(3, int(round(min(hauteur, largeur) / 110)))
    if taille_morpho % 2 == 0:
        taille_morpho += 1
    masque_ouvert  = ouverture_binaire(masque_brut, taille_morpho)
    masque_propre  = fermeture_binaire(masque_ouvert, taille_morpho)

    # Étape E : composantes connexes + filtrage progressif (identique à detection_principale)
    aire_min = max(500, int(aire_image * 0.0010))
    toutes_composantes = composantes_connexes(masque_propre)

    composantes = []
    aire_max_utilise = int(aire_image * 0.18)
    for facteur_max in [0.18, 0.35, 0.60, 0.90]:
        aire_max_utilise = int(aire_image * facteur_max)
        composantes = [
            c for c in toutes_composantes
            if aire_min <= c["area"] <= aire_max_utilise and not c["touche_bord"]
        ]
        if composantes:
            break

    # ------------------------------------------------------------------
    # CONSTRUCTION DE LA FIGURE
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Pipeline de détection de pieces — " + chemin_image.split("\\")[-1].split("/")[-1],
        fontsize=13, fontweight='bold'
    )

    # ---------- Case 1 : Image originale ----------
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Image originale", fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].set_xlabel(str(largeur) + " x " + str(hauteur) + " pixels", fontsize=9)

    # ---------- Case 2 : Canal saturation ----------
    # La saturation mesure à quel point chaque pixel est "coloré".
    # Les pièces métalliques ont souvent une saturation différente du fond.
    axes[0, 1].imshow(saturation, cmap='gray')
    axes[0, 1].set_title("2. Saturation (HSL)", fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 1].set_xlabel("Blanc = tres colore\nNoir = gris/blanc pur", fontsize=9)

    # ---------- Case 3 : Saturation après flou gaussien ----------
    # Le flou lisse les petites variations (bruit).
    # Résultat : les zones importantes ressortent mieux lors du seuillage.
    axes[0, 2].imshow(saturation_floue, cmap='gray')
    axes[0, 2].set_title("3. Flou gaussien (noyau " + str(taille_locale) + "x" + str(taille_locale) + ")", fontweight='bold')
    axes[0, 2].axis('off')
    axes[0, 2].set_xlabel("Reduit le bruit avant Otsu", fontsize=9)

    # ---------- Case 4 : Masque binaire (Otsu) ----------
    # Otsu choisit automatiquement le seuil qui sépare au mieux
    # les pixels "fond" des pixels "objet".
    # Blanc = objet (pièce probable), Noir = fond
    axes[0, 3].imshow(masque_brut, cmap='gray')
    axes[0, 3].set_title("4. Masque Otsu (seuil=" + str(round(seuil, 3)) + ")", fontweight='bold')
    axes[0, 3].axis('off')
    axes[0, 3].set_xlabel("Blanc = objet | Noir = fond", fontsize=9)

    # ---------- Case 5 : Masque après morphologie ----------
    # Ouverture : supprime les petits points parasites
    # Fermeture : bouche les petits trous dans les pièces (reflets)
    axes[1, 0].imshow(masque_propre, cmap='gray')
    axes[1, 0].set_title("5. Apres ouverture + fermeture", fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 0].set_xlabel("Ouverture supprime le bruit\nFermeture bouche les trous", fontsize=9)

    # ---------- Case 6 : Composantes connexes colorées ----------
    # Chaque region blanche connectee recoit une couleur differente.
    # On ne montre que les composantes qui passent le filtre d'aire.
    carte_couleurs = np.zeros(masque_propre.shape, dtype=np.int32)
    for i, comp in enumerate(composantes, 1):
        y_min, x_min, y_max, x_max = comp["bbox"]
        # On marque uniquement les pixels appartenant a cette composante
        # en se servant de la boite englobante + le masque propre
        zone = masque_propre[y_min:y_max + 1, x_min:x_max + 1]
        carte_couleurs[y_min:y_max + 1, x_min:x_max + 1][zone] = i

    axes[1, 1].imshow(carte_couleurs, cmap='tab20', interpolation='nearest')
    axes[1, 1].set_title("6. Composantes connexes", fontweight='bold')
    axes[1, 1].axis('off')
    axes[1, 1].set_xlabel(str(len(composantes)) + " region(s) apres filtrage aire", fontsize=9)

    # ---------- Case 7 : Boites englobantes avec compte estimé par boite ----------
    # On dessine une boite autour de chaque composante et on affiche
    # combien de pièces l'algorithme estime qu'il y a dedans.
    #
    # Vert  = 1 seule pièce isolée (composante de référence)
    # Orange = plusieurs pièces fusionnées (comptées via le ratio d'aire)
    #
    # NOTE : toutes les composantes montrées ici SONT comptées.
    # Il n'y a pas de composante "rejetée" dans cette étape —
    # le filtrage par aire a déjà eu lieu à l'étape précédente.

    axes[1, 2].imshow(image_rgb)

    # Reproduire la logique de estimer_nombre_depuis_composantes
    # pour savoir combien de pièces chaque boite représente
    aires_reference = [
        c["area"] for c in composantes
        if c["circularite"] >= COIN_REFERENCE_CIRCULARITY_MIN
        and c["remplissage"] >= COIN_REFERENCE_FILL_MIN
        and COIN_REFERENCE_ASPECT_MIN
            <= c["hauteur_bbox"] / max(1, c["largeur_bbox"])
            <= COIN_REFERENCE_ASPECT_MAX
    ]
    aire_reference = float(np.median(aires_reference)) if aires_reference else 0.0

    nb_references = len(aires_reference)
    for comp in composantes:
        y_min, x_min, y_max, x_max = comp["bbox"]

        # Calcul du nombre de pièces estimé pour cette composante
        if aire_reference > 0:
            ratio_aire = comp["area"] / aire_reference
            if ratio_aire >= MERGE_AREA_RATIO_THRESHOLD and comp["remplissage"] >= MERGE_FILL_MIN:
                nb_pieces_boite = max(1, int(round(ratio_aire)))
                couleur   = 'orange'
                epaisseur = 2
            else:
                nb_pieces_boite = 1
                couleur   = 'lime'
                epaisseur = 2
        else:
            # Pas de référence trouvée : on ne sait pas estimer
            nb_pieces_boite = 1
            couleur   = 'lime'
            epaisseur = 1

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=epaisseur,
            edgecolor=couleur,
            facecolor='none'
        )
        axes[1, 2].add_patch(rect)

        # Afficher le nombre estimé au coin de la boite
        axes[1, 2].text(
            x_min + 3, y_min + 14,
            "x" + str(nb_pieces_boite),
            color=couleur,
            fontsize=10,
            fontweight='bold'
        )

    axes[1, 2].set_title("7. Boites englobantes", fontweight='bold')
    axes[1, 2].axis('off')
    axes[1, 2].set_xlabel(
        "Vert = 1 piece isolee (" + str(nb_references) + ")\n"
        "Orange = pieces fusionnees (N > 1)",
        fontsize=9
    )

    # ---------- Case 8 : Résultat final ----------
    axes[1, 3].axis('off')
    texte = (
        "RESULTAT FINAL\n"
        "══════════════\n\n"
        "Pieces detectees : " + str(prediction_finale) + "\n\n"
        "──────────────────\n"
        "Composantes trouvees : " + str(len(composantes)) + "\n"
        "Aire image           : " + str(aire_image) + " px\n"
        "Aire min             : " + str(aire_min) + " px\n"
        "Aire max utilisee    : " + str(aire_max_utilise) + " px\n"
        "Taille flou          : " + str(taille_locale) + "x" + str(taille_locale) + "\n"
        "Seuil Otsu           : " + str(round(seuil, 3)) + "\n\n"
        "──────────────────\n"
        "Seuils de reference :\n"
        "  Circularite min : " + str(COIN_REFERENCE_CIRCULARITY_MIN) + "\n"
        "  Remplissage min : " + str(COIN_REFERENCE_FILL_MIN) + "\n"
        "  Ratio aspect    : ["
        + str(COIN_REFERENCE_ASPECT_MIN) + ", "
        + str(COIN_REFERENCE_ASPECT_MAX) + "]"
    )
    axes[1, 3].text(
        0.05, 0.95,
        texte,
        fontsize=9,
        family='monospace',
        verticalalignment='top',
        transform=axes[1, 3].transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    axes[1, 3].set_title("8. Resume", fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Figure sauvegardee : " + save_path)
    else:
        plt.show()

    print("Resultat : " + str(prediction_finale) + " piece(s) detectee(s)")
    return prediction_finale, composantes


def visualiser_descripteurs(chemin_image):
    """
    Affiche 4 graphiques montrant les descripteurs de forme de toutes
    les composantes trouvées dans l'image.

    Utile pour comprendre pourquoi certaines composantes sont acceptées
    ou rejetées comme "pièces de référence".

    Paramètre :
        chemin_image : chemin vers l'image à analyser
    """

    # --- Reproduction du pipeline jusqu'aux composantes ---
    image_rgb = lire_image_rgb(chemin_image)
    if image_rgb is None:
        print("Erreur : impossible de charger l'image.")
        return

    aire_image = image_rgb.shape[0] * image_rgb.shape[1]
    _, saturation = rgb_vers_hsl(image_rgb)
    saturation_floue = flou_gaussien(saturation, 7)
    seuil = seuil_otsu(saturation_floue)
    masque = ouverture_binaire(saturation_floue > seuil, 3)
    masque = fermeture_binaire(masque, 3)

    # Filtrage progressif (même logique que detection_principale)
    aire_min = max(500, int(aire_image * 0.0010))
    toutes_composantes = composantes_connexes(masque)
    composantes = []
    for facteur_max in [0.18, 0.35, 0.60, 0.90]:
        aire_max = int(aire_image * facteur_max)
        composantes = [
            c for c in toutes_composantes
            if aire_min <= c["area"] <= aire_max and not c["touche_bord"]
        ]
        if composantes:
            break

    if not composantes:
        print("Aucune composante trouvee apres filtrage.")
        return

    # --- Extraction des descripteurs ---
    aires        = [c["area"]                                    for c in composantes]
    circularites = [c["circularite"]                             for c in composantes]
    remplissages = [c["remplissage"]                             for c in composantes]
    ratios       = [c["hauteur_bbox"] / max(1, c["largeur_bbox"]) for c in composantes]
    indices      = list(range(len(composantes)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Descripteurs de forme — " + chemin_image.split("\\")[-1].split("/")[-1],
        fontsize=13, fontweight='bold'
    )

    # Graphique 1 : Aire de chaque composante
    # On veut voir si les composantes ont une taille cohérente (= une pièce)
    # ou si l'une est beaucoup plus grande (= plusieurs pièces fusionnées)
    axes[0, 0].bar(indices, aires, color='steelblue', edgecolor='black')
    axes[0, 0].axhline(np.median(aires), color='red', linestyle='--',
                       label="Mediane : " + str(int(np.median(aires))) + " px")
    axes[0, 0].set_xlabel("Composante #")
    axes[0, 0].set_ylabel("Aire (pixels)")
    axes[0, 0].set_title("Aire de chaque composante")
    axes[0, 0].legend()

    # Graphique 2 : Circularité (0 = forme quelconque, 1 = cercle parfait)
    # Formule : 4*pi*aire / perimetre²
    # Une pièce doit avoir une circularité suffisante (seuil = COIN_REFERENCE_CIRCULARITY_MIN)
    couleurs_circ = [
        'green' if c >= COIN_REFERENCE_CIRCULARITY_MIN else 'red'
        for c in circularites
    ]
    axes[0, 1].bar(indices, circularites, color=couleurs_circ, edgecolor='black')
    axes[0, 1].axhline(COIN_REFERENCE_CIRCULARITY_MIN, color='orange', linestyle='--',
                       label="Seuil min : " + str(COIN_REFERENCE_CIRCULARITY_MIN))
    axes[0, 1].set_xlabel("Composante #")
    axes[0, 1].set_ylabel("Circularite (0 a 1)")
    axes[0, 1].set_title("Circularite  (vert = passe le seuil)")
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].legend()

    # Graphique 3 : Remplissage = aire / aire_bbox
    # Mesure à quel point la composante remplit bien sa boite englobante.
    # Une pièce circulaire remplit environ 78% de son carré englobant.
    couleurs_fill = [
        'green' if r >= COIN_REFERENCE_FILL_MIN else 'red'
        for r in remplissages
    ]
    axes[1, 0].bar(indices, remplissages, color=couleurs_fill, edgecolor='black')
    axes[1, 0].axhline(COIN_REFERENCE_FILL_MIN, color='orange', linestyle='--',
                       label="Seuil min : " + str(COIN_REFERENCE_FILL_MIN))
    axes[1, 0].set_xlabel("Composante #")
    axes[1, 0].set_ylabel("Remplissage (0 a 1)")
    axes[1, 0].set_title("Remplissage = aire / aire_bbox  (vert = passe)")
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()

    # Graphique 4 : Ratio hauteur/largeur
    # Une pièce est presque ronde, donc ce ratio doit être proche de 1.
    # Trop éloigné de 1 = objet allongé ou pièce fortement inclinée.
    couleurs_ratio = [
        'green' if COIN_REFERENCE_ASPECT_MIN <= r <= COIN_REFERENCE_ASPECT_MAX else 'red'
        for r in ratios
    ]
    axes[1, 1].bar(indices, ratios, color=couleurs_ratio, edgecolor='black')
    axes[1, 1].axhline(COIN_REFERENCE_ASPECT_MIN, color='orange', linestyle='--',
                       label="Min : " + str(COIN_REFERENCE_ASPECT_MIN))
    axes[1, 1].axhline(COIN_REFERENCE_ASPECT_MAX, color='orange', linestyle='-.',
                       label="Max : " + str(COIN_REFERENCE_ASPECT_MAX))
    axes[1, 1].set_xlabel("Composante #")
    axes[1, 1].set_ylabel("Ratio hauteur / largeur")
    axes[1, 1].set_title("Ratio aspect  (vert = dans la plage acceptable)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        visualiser_pipeline(sys.argv[1])
    else:
        print("Usage  : python visualizer.py <chemin_image>")
        print("Exemple: python visualizer.py data/validation/img_001.jpg")
