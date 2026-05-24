import os
import json
from traitement import compter_pieces
from traitement_contours import compter_pieces_contours

def evaluer_modele(dossier_images, fichier_json, taille_flou=(7, 7), pipeline="v1", **kwargs_contours):
    """
    Charge la vérité terrain, exécute l'algorithme et calcule la MAE et la MSE.

    pipeline = "v1" -> traitement.compter_pieces (HSV + Otsu, version originale)
    pipeline = "contours" -> traitement_contours.compter_pieces_contours (Gris + Sobel + fermeture)
    """
    # Chargement de la vérité terrain depuis le JSON
    with open(fichier_json, 'r') as f:
        verite_terrain = json.load(f)

    erreurs_absolues = []
    erreurs_quadratiques = []

    print(f"\nÉvaluation sur le dataset : {fichier_json}  (pipeline={pipeline})")
    print("-" * 40)

    for nom_fichier, vrai_nombre in verite_terrain.items():
        chemin = os.path.join(dossier_images, nom_fichier)

        # Prédiction (yi)
        if pipeline == "contours":
            prediction = compter_pieces_contours(chemin, **kwargs_contours)
        else:
            prediction = compter_pieces(chemin, taille_flou)
        
        # Vérité terrain (ŷi)
        # Calcul des écarts
        diff = prediction - vrai_nombre
        
        erreurs_absolues.append(abs(diff))
        erreurs_quadratiques.append(diff ** 2)
        
        # Affichage détaillé pour comprendre où l'algo se trompe
        if diff != 0:
            print(f"[ERREUR] {nom_fichier} | Prédit: {prediction} | Réel: {vrai_nombre} | Diff: {diff}")
        else:
            print(f"[OK] {nom_fichier} | Prédit: {prediction} | Réel: {vrai_nombre}")

    # Calcul des métriques de régression [cite: 289-291, 419, 423]
    N = len(erreurs_absolues)
    if N == 0:
        return

    mae = sum(erreurs_absolues) / N
    mse = sum(erreurs_quadratiques) / N

    print("-" * 40)
    # La MAE renseigne directement sur la distance moyenne aux prédictions[cite: 421].
    print(f"MAE (Erreur Absolue Moyenne)     : {mae:.2f}")
    # La MSE pénalise lourdement les grosses aberrations[cite: 424, 435].
    print(f"MSE (Erreur Quadratique Moyenne) : {mse:.2f}")
    print("-" * 40)