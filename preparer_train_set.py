import os
import shutil
import json

def split_validation_to_train(dossier_val, dossier_train, fichier_val_json, fichier_train_json, nb_train=120):
    """
    Sépare le dossier validation pour créer un dossier d'entraînement.
    Prend nb_train (120) images de validation, les déplace vers train,
    et génère les fichiers JSON correspondants.
    """
    # Créer le dossier train s'il n'existe pas
    os.makedirs(dossier_train, exist_ok=True)
    
    # Lire le fichier validation.json d'origine
    if not os.path.exists(fichier_val_json):
        print(f"Erreur : Le fichier {fichier_val_json} n'existe pas.")
        return
        
    with open(fichier_val_json, 'r') as f:
        labels_val_complet = json.load(f)
        
    # Trier les fichiers pour avoir un comportement déterministe
    liste_fichiers = sorted(list(labels_val_complet.keys()))
    
    if len(liste_fichiers) < nb_train:
        print(f"Erreur : Pas assez d'images dans la validation ({len(liste_fichiers)} < {nb_train}).")
        return
        
    train_labels = {}
    val_labels = {}
    
    print(f"Séparation de {len(liste_fichiers)} images de validation en {nb_train} train / {len(liste_fichiers) - nb_train} validation...")
    
    for i, nom_fichier in enumerate(liste_fichiers):
        chemin_source = os.path.join(dossier_val, nom_fichier)
        
        if i < nb_train:
            # Déplacer vers le dossier d'entraînement
            chemin_dest = os.path.join(dossier_train, nom_fichier)
            if os.path.exists(chemin_source):
                shutil.move(chemin_source, chemin_dest)
            train_labels[nom_fichier] = labels_val_complet[nom_fichier]
        else:
            # Conserver dans le dossier de validation
            val_labels[nom_fichier] = labels_val_complet[nom_fichier]
            
    # Écrire les nouveaux fichiers JSON
    with open(fichier_train_json, 'w') as f:
        json.dump(train_labels, f, indent=4)
    with open(fichier_val_json, 'w') as f:
        json.dump(val_labels, f, indent=4)
        
    print(f"-> {len(train_labels)} images déplacées dans '{dossier_train}' et annotées dans '{fichier_train_json}'")
    print(f"-> {len(val_labels)} images restantes dans '{dossier_val}' et annotées dans '{fichier_val_json}'")

if __name__ == "__main__":
    split_validation_to_train(
        dossier_val="data/validation",
        dossier_train="data/train",
        fichier_val_json="data/validation.json",
        fichier_train_json="data/train.json",
        nb_train=120
    )
