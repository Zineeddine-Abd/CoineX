import os
import json

def evaluer_modele(dossier_images, fichier_json, pipeline="morphologie", **kwargs):
    """
    Charge le JSON de la vérité terrain, exécute l'algorithme "compter_pieces" du pipeline choisi
    et calcule la MAE et la MSE.

    Pipelines disponibles : "morphologie", "contours", "opencv", "nn"
    """
    # Import dynamique selon le pipeline
    if pipeline == "morphologie":
        from pipelines.morphologie.traitement import compter_pieces
    elif pipeline == "contours":
        from pipelines.contours.traitement import compter_pieces
    elif pipeline == "opencv":
        from pipelines.opencv_test.traitement import compter_pieces
    elif pipeline == "nn":
        from pipelines.nn.traitement import compter_pieces
    else:
        raise ValueError(f"Pipeline non reconnu : {pipeline}")

    with open(fichier_json, 'r') as f:
        verite_terrain = json.load(f)

    erreurs_absolues = []
    erreurs_quadratiques = []
    
    print(f"\nÉvaluation sur le dataset : {fichier_json} (Pipeline : {pipeline})")
    print("-" * 50)

    for nom_fichier, vrai_nombre in verite_terrain.items():
        chemin = os.path.join(dossier_images, nom_fichier)
        
        prediction = compter_pieces(chemin, **kwargs)
        diff = prediction - vrai_nombre
        
        erreurs_absolues.append(abs(diff))
        erreurs_quadratiques.append(diff ** 2)
        
        if diff != 0:
            print(f"[ERREUR] {nom_fichier} | Prédit: {prediction} | Réel: {vrai_nombre} | Diff: {diff}")
        else:
            print(f"[OK] {nom_fichier} | Prédit: {prediction} | Réel: {vrai_nombre}")

    N = len(erreurs_absolues)
    if N == 0:
        return

    mae = sum(erreurs_absolues) / N
    mse = sum(erreurs_quadratiques) / N
    rmse = mse ** 0.5 

    nb_exacts   = sum(1 for e in erreurs_absolues if e == 0)
    taux_exact  = (nb_exacts / N) * 100

    mean_real_amount = sum(verite_terrain.values()) / N
    mae_percentage = (mae / mean_real_amount) * 100 if mean_real_amount > 0 else 0

    print("-" * 50)
    print(f"MAE  (Erreur Absolue Moyenne)        : {mae:.2f} pièces")
    print(f"MSE  (Erreur Quadratique Moyenne)    : {mse:.2f} pièces²")
    print(f"RMSE (Racine de la MSE)              : {rmse:.2f} pièces")
    print(f"Taux de prédictions exactes          : {nb_exacts}/{N} ({taux_exact:.1f}%)")
    print(f"Nombre Moyen Réel de Pièces          : {mean_real_amount:.2f}")
    print(f"Pourcentage d'Erreur (MAE/Mean)      : {mae_percentage:.2f}%")
    print("-" * 50)