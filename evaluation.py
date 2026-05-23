import os
import json
from traitement import compter_pieces

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

    # -------------------------------------------------------------------------
    # COURS Semaine 3 — RMSE (Racine de l'Erreur Quadratique Moyenne)
    # -------------------------------------------------------------------------
    # Formule : RMSE = √MSE = √( Σ(yi − ŷi)² / N )
    #
    # Pourquoi ajouter la RMSE si on a déjà la MSE ?
    # La MSE est exprimée en "pièces²" (unité au carré), ce qui n'a pas de sens
    # physique direct. La RMSE ramène le résultat dans la même unité que la MAE
    # (nombre de pièces), donc elle est directement interprétable :
    #
    #   MAE  = 1.73  →  "en moyenne, on se trompe de 1.73 pièces"
    #   RMSE = 2.50  →  "erreur typique en tenant compte des pics = 2.50 pièces"
    #
    # Si RMSE >> MAE : il y a quelques grandes erreurs qui "tirent" la RMSE vers
    # le haut (images très difficiles avec beaucoup de pièces fusionnées).
    # Si RMSE ≈ MAE  : les erreurs sont uniformément réparties, pas de cas extrêmes.
    # -------------------------------------------------------------------------
    rmse = mse ** 0.5   # équivalent à √MSE, sans import math

    # -------------------------------------------------------------------------
    # COURS Semaine 3 — Taux de Prédictions Exactes (Accuracy)
    # -------------------------------------------------------------------------
    # Formule : Taux exact = (nombre d'images où prediction == vrai) / N × 100
    #
    # Sémantique :
    # C'est la fraction des images pour lesquelles l'algorithme a trouvé
    # EXACTEMENT le bon nombre de pièces (ni plus, ni moins).
    #
    # Différence avec la MAE :
    # La MAE mesure l'erreur MOYENNE sur toutes les images (y compris les
    # petites erreurs de ±1). Le taux exact est plus strict : une erreur de 1
    # compte comme un échec total.
    #
    # Exemple :
    #   MAE = 0.5  mais taux exact = 60%
    #   → On se trompe souvent de 1 pièce, mais la moitié du temps c'est exact.
    #
    # Pourquoi c'est utile pour notre projet :
    # Un jury ou un client veut souvent savoir "combien de fois votre système
    # donne la bonne réponse ?", pas juste "de combien vous vous trompez en
    # moyenne". Le taux exact répond directement à cette question.
    # -------------------------------------------------------------------------
    nb_exacts   = sum(1 for e in erreurs_absolues if e == 0)
    taux_exact  = (nb_exacts / N) * 100

    mean_real_amount = sum(verite_terrain.values()) / N
    mae_percentage = (mae / mean_real_amount) * 100 if mean_real_amount > 0 else 0

    print("-" * 40)
    # La MAE renseigne directement sur la distance moyenne aux prédictions.
    print(f"MAE  (Erreur Absolue Moyenne)        : {mae:.2f} pièces")

    # La MSE pénalise lourdement les grosses aberrations.
    print(f"MSE  (Erreur Quadratique Moyenne)    : {mse:.2f} pièces²")

    # La RMSE est dans la même unité que la MAE, plus lisible que la MSE.
    print(f"RMSE (Racine de la MSE)              : {rmse:.2f} pièces")

    # Le taux exact dit combien d'images sont parfaitement bien comptées.
    print(f"Taux de prédictions exactes          : {nb_exacts}/{N} ({taux_exact:.1f}%)")

    print(f"Nombre Moyen Réel de Pièces          : {mean_real_amount:.2f}")
    print(f"Pourcentage d'Erreur (MAE/Mean)      : {mae_percentage:.2f}%")
    print("-" * 40)

    