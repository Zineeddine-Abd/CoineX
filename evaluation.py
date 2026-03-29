import os
import json
from traitement_2 import compter_pieces

def evaluer_modele(dossier_images, fichier_json, taille_flou=(7, 7)):
    """
    Charge le JSON de la vérité terrain, exécute l'algorithme "compter_pieces" et calcule la MAE et la MSE.
    """
    # Chargement de la vérité terrain depuis le JSON
    # with : pour s'assurer que le fichier est correctement fermé après lecture
    with open(fichier_json, 'r') as f:
        verite_terrain = json.load(f)

    erreurs_absolues = []
    erreurs_quadratiques = []
    
    print(f"\nÉvaluation sur le dataset : {fichier_json}")
    print("-" * 40)

    for nom_fichier, vrai_nombre in verite_terrain.items():
        chemin = os.path.join(dossier_images, nom_fichier)
        
        # Prédiction (yi)
        # taille_flou : paramètre qui contrôle l'intensité du flou gaussien appliqué à l'image avant d'analyser les pièces.
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

    # Calcul des métriques de régression
    N = len(erreurs_absolues)
    if N == 0:
        return

    mae = sum(erreurs_absolues) / N
    mse = sum(erreurs_quadratiques) / N
    
    mean_real_amount = sum(verite_terrain.values()) / N
    mae_percentage = (mae / mean_real_amount) * 100 if mean_real_amount > 0 else 0

    print("-" * 40)
    # La MAE renseigne directement sur la distance moyenne aux prédictions.
    print(f"MAE (Erreur Absolue Moyenne)     : {mae:.2f}")

    # La MSE pénalise lourdement les grosses aberrations.
    print(f"MSE (Erreur Quadratique Moyenne) : {mse:.2f}")
    
    print(f"Nombre Moyen Réel de Pièces      : {mean_real_amount:.2f}")
    print(f"Pourcentage d'Erreur (MAE/Mean)  : {mae_percentage:.2f}%")
    print("-" * 40)

    