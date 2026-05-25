"""
Visualiseur dédié au pipeline contours.

Lance :
    python visualiseur_contours.py                                   # première image
    python visualiseur_contours.py data/validation/img_007.jpg
    python visualiseur_contours.py data/validation/img_007.jpg --save out.png
    python visualiseur_contours.py data/validation --out-dir vis/    # mode batch

Affiche :
    - chaque étape intermédiaire de la chaîne (gris, équa, flou, Sobel X/Y,
      magnitude, binaire, fermée, masque plein, ouvert)
    - les histogrammes avant/après égalisation
    - le résultat final avec contours retenus (vert) et rejetés (rouge),
      annotés par leur aire et leur circularité
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from traitement import pipeline_contours


def _verite_terrain(chemin_image):
    """Cherche le compte annoté pour cette image dans data/*.json (si dispo)."""
    nom = os.path.basename(chemin_image)
    for jf in ("data/validation.json", "data/test.json"):
        if os.path.exists(jf):
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                if nom in data:
                    return data[nom], jf
            except Exception:
                pass
    return None, None


def _to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _sobel_pour_affichage(s):
    """Convertit une dérivée Sobel signée (float64) en image affichable [0,255]."""
    s_abs = np.abs(s)
    m = s_abs.max() if s_abs.max() > 0 else 1.0
    return np.uint8(255.0 * s_abs / m)


def _dessiner_overlay(image_bgr, retenus, rejetes):
    """Construit l'image finale annotée (BGR) pour affichage."""
    overlay = image_bgr.copy()
    # Rejetés en rouge fin
    for r in rejetes:
        cv2.drawContours(overlay, [r["contour"]], -1, (0, 0, 255), 1)
    # Retenus en vert épais + bbox + numéro
    for i, r in enumerate(retenus, start=1):
        cv2.drawContours(overlay, [r["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = r["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            overlay,
            str(i),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return overlay


def visualiser(chemin_image, sauvegarde=None, **kwargs):
    res = pipeline_contours(chemin_image, **kwargs)
    if "erreur" in res:
        print(res["erreur"])
        return

    overlay = _dessiner_overlay(res["originale"], res["retenus"], res["rejetes"])

    # Grille 4 x 4 d'étapes
    etapes = [
        ("1. Originale (réduite)", _to_rgb(res["originale"]), None),
        ("2. Niveaux de gris (luminance)", res["gris"], "gray"),
        ("3a. Pré-lissage gaussien", res["pre_floutee"], "gray"),
        ("3b. Égalisation d'histogramme", res["egalisee"], "gray"),
        ("4. Lissage avant Sobel", res["floutee"], "gray"),
        ("5a. Sobel |dI/dx|", _sobel_pour_affichage(res["sobel_x"]), "gray"),
        ("5b. Sobel |dI/dy|", _sobel_pour_affichage(res["sobel_y"]), "gray"),
        ("5c. Magnitude Sobel", res["magnitude"], "gray"),
        ("6. Otsu sur magnitude", res["binaire"], "gray"),
        ("7. Fermeture (petit noyau)", res["fermee"], "gray"),
        ("8. Remplissage des contours", res["masque_plein"], "gray"),
        ("9a. Ouverture (speckles)", res["propre"], "gray"),
        ("9b. Érosion (sépare pièces tangentes)", res["erodee"], "gray"),
        (f"10. Résultat : {res['compteur']} pièce(s)", _to_rgb(overlay), None),
    ]

    vt, vt_src = _verite_terrain(chemin_image)
    titre_vt = ""
    if vt is not None:
        diff = res["compteur"] - vt
        titre_vt = f" | Vérité : {vt} (diff={diff:+d})"

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"Pipeline contours — {os.path.basename(chemin_image)} — "
        f"Détectées : {res['compteur']} | Rejetées : {len(res['rejetes'])}{titre_vt}",
        fontsize=14,
        fontweight="bold",
    )

    cols = 5
    rows = 5  # 14 étapes + 2 histos + scatter + params + spare ≈ 20

    # Étapes (lignes 1-3)
    for i, (titre, img, cmap) in enumerate(etapes):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap=cmap)
        ax.set_title(titre, fontsize=9)
        ax.axis("off")

    # Ligne d'analyse (en bas)
    base = (rows - 1) * cols  # premier index de la dernière ligne (1-based ensuite)
    ax_h1 = fig.add_subplot(rows, cols, base + 1)
    ax_h1.hist(res["gris"].ravel(), bins=256, range=(0, 256), color="steelblue")
    ax_h1.set_title("Histogramme — gris brut", fontsize=9)
    ax_h1.set_xlim(0, 255)

    ax_h2 = fig.add_subplot(rows, cols, base + 2)
    ax_h2.hist(res["egalisee"].ravel(), bins=256, range=(0, 256), color="darkorange")
    ax_h2.set_title("Histogramme — après égalisation", fontsize=9)
    ax_h2.set_xlim(0, 255)

    # Diagramme de dispersion : circularité vs aire
    ax_sc = fig.add_subplot(rows, cols, base + 3)
    if res["retenus"]:
        ax_sc.scatter(
            [r["aire"] for r in res["retenus"]],
            [r["circularite"] for r in res["retenus"]],
            c="green",
            label="retenus",
            s=40,
        )
    if res["rejetes"]:
        ax_sc.scatter(
            [r["aire"] for r in res["rejetes"]],
            [r["circularite"] for r in res["rejetes"]],
            c="red",
            label="rejetés",
            s=20,
            alpha=0.6,
        )
    p = res["params"]
    ax_sc.axvline(p["aire_min"], color="gray", linestyle="--", linewidth=0.8)
    ax_sc.axvline(p["aire_max"], color="gray", linestyle="--", linewidth=0.8)
    ax_sc.axhline(p["circularite_min"], color="gray", linestyle="--", linewidth=0.8)
    ax_sc.set_xlabel("Aire (px)", fontsize=8)
    ax_sc.set_ylabel("Circularité 4πA/P²", fontsize=8)
    ax_sc.set_title("Filtrage géométrique", fontsize=9)
    ax_sc.legend(fontsize=7)
    ax_sc.grid(alpha=0.3)

    # Cartouche paramètres
    ax_p = fig.add_subplot(rows, cols, base + 4)
    ax_p.axis("off")
    texte = "Paramètres :\n" + "\n".join(f"  {k} = {v}" for k, v in p.items())
    texte += f"\n\nContours bruts : {len(res['retenus']) + len(res['rejetes'])}"
    texte += f"\nRetenus       : {len(res['retenus'])}"
    texte += f"\nRejetés       : {len(res['rejetes'])}"
    ax_p.text(
        0.0, 1.0, texte, fontsize=9, family="monospace",
        verticalalignment="top",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if sauvegarde:
        plt.savefig(sauvegarde, dpi=120, bbox_inches="tight")
        print(f"Figure sauvegardée : {sauvegarde}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualiseur du pipeline contours.")
    parser.add_argument(
        "image",
        nargs="?",
        default="data/validation/img_001.jpg",
        help="Chemin d'une image OU d'un dossier (mode batch).",
    )
    parser.add_argument("--save", default=None, help="Chemin PNG de sortie (optionnel).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Si 'image' est un dossier, dossier de destination des PNG (un par image).",
    )
    parser.add_argument("--largeur", type=int, default=800)
    parser.add_argument("--pre-flou", type=int, default=19)
    parser.add_argument("--flou", type=int, default=5)
    parser.add_argument("--fermeture", type=int, default=50)
    parser.add_argument("--ouverture", type=int, default=25)
    parser.add_argument("--erosion", type=int, default=7)
    # --- Anti-trou d'anneau (opt-in) ---
    parser.add_argument("--hull-fill", action="store_true",
                        help="Remplir l'enveloppe convexe (pour pièces isolées avec gap).")
    parser.add_argument("--n-fermetures", type=int, default=1,
                        help="Itérations de fermeture (>1 scelle des gaps plus larges).")
    parser.add_argument("--hyst-bas", type=int, default=None,
                        help="Seuil bas hystérésis 0-255 (union avec Otsu).")
    parser.add_argument("--aire-min", type=int, default=300)
    parser.add_argument("--aire-max", type=int, default=200000)
    parser.add_argument("--circ-min", type=float, default=0.55)
    parser.add_argument("--solidite-min", type=float, default=0.85)
    parser.add_argument("--ratio-max", type=float, default=1.8)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image introuvable : {args.image}", file=sys.stderr)
        sys.exit(1)

    # Mode batch : dossier → un PNG par image
    if os.path.isdir(args.image):
        out_dir = args.out_dir or "visualisations_contours"
        os.makedirs(out_dir, exist_ok=True)
        fichiers = sorted(
            f for f in os.listdir(args.image)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"Mode batch : {len(fichiers)} images -> {out_dir}/")
        for f in fichiers:
            out_path = os.path.join(out_dir, os.path.splitext(f)[0] + "_contours.png")
            visualiser(
                os.path.join(args.image, f),
                sauvegarde=out_path,
                largeur_cible=args.largeur,
                taille_pre_flou=(args.pre_flou, args.pre_flou),
                taille_flou=(args.flou, args.flou),
                taille_fermeture=args.fermeture,
                taille_ouverture=args.ouverture,
                taille_erosion=args.erosion,
                remplir_par_hull=args.hull_fill,
                n_fermetures=args.n_fermetures,
                seuil_bas_hysteresis=args.hyst_bas,
                aire_min=args.aire_min,
                aire_max=args.aire_max,
                circularite_min=args.circ_min,
                solidite_min=args.solidite_min,
                ratio_max=args.ratio_max,
            )
            plt.close("all")
        print("Terminé.")
        return

    visualiser(
        args.image,
        sauvegarde=args.save,
        largeur_cible=args.largeur,
        taille_pre_flou=(args.pre_flou, args.pre_flou),
        taille_flou=(args.flou, args.flou),
        taille_fermeture=args.fermeture,
        taille_ouverture=args.ouverture,
        taille_erosion=args.erosion,
        aire_min=args.aire_min,
        aire_max=args.aire_max,
        circularite_min=args.circ_min,
        solidite_min=args.solidite_min,
        ratio_max=args.ratio_max,
    )


if __name__ == "__main__":
    main()
