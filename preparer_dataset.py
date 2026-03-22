import os
import shutil
import random
import json
import cv2

def preparer_et_diviser(dossier_brut, dossier_val, dossier_test, ratio_val=0.7):
    """Mélange, renomme et sépare les images brutes en Validation et Test."""
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(dossier_val, exist_ok=True)
    os.makedirs(dossier_test, exist_ok=True)

    # Récupérer toutes les images
    images = [f for f in os.listdir(dossier_brut) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images) # Mélange aléatoire indispensable pour éviter les biais

    nb_val = int(len(images) * ratio_val)
    
    print(f"Préparation de {len(images)} images...")
    
    for i, nom_fichier in enumerate(images):
        chemin_source = os.path.join(dossier_brut, nom_fichier)
        nouveau_nom = f"img_{str(i+1).zfill(3)}.jpg" # Format img_001.jpg, img_002.jpg...
        
        # Division 70% Validation / 30% Test
        if i < nb_val:
            chemin_dest = os.path.join(dossier_val, nouveau_nom)
        else:
            chemin_dest = os.path.join(dossier_test, nouveau_nom)
            
        shutil.copy(chemin_source, chemin_dest)
        
    print(f"-> {nb_val} images dans '{dossier_val}'")
    print(f"-> {len(images) - nb_val} images dans '{dossier_test}'")


def annoter_dossier(dossier_images, chemin_json):
    """Affiche chaque image et te demande le nombre de pièces pour créer le JSON."""
    verite_terrain = {}
    images = sorted(os.listdir(dossier_images))
    
    print(f"\n=== Lancement de l'annotation pour {dossier_images} ===")
    print("Tape le nombre de pièces dans la console et appuie sur Entrée.")
    print("Tape 'q' pour quitter et sauvegarder l'avancement.\n")

    for nom_fichier in images:
        chemin_img = os.path.join(dossier_images, nom_fichier)
        
        # Afficher l'image avec OpenCV
        img = cv2.imread(chemin_img)
        # Redimensionner si l'image est trop grande pour ton écran
        img_affichee = cv2.resize(img, (800, 600)) 
        cv2.imshow("Annotation (Regarde ici, tape dans la console !)", img_affichee)
        cv2.waitKey(1) # Force l'affichage de la fenêtre
        
        # Demander la saisie dans la console
        saisie = input(f"Image {nom_fichier} - Nombre de pièces ? : ")
        
        if saisie.lower() == 'q':
            print("Arrêt de l'annotation. Sauvegarde en cours...")
            break
            
        try:
            verite_terrain[nom_fichier] = int(saisie)
        except ValueError:
            print("Entrée invalide, valeur par défaut (0) attribuée. Tu pourras corriger le JSON.")
            verite_terrain[nom_fichier] = 0

    # Fermer la fenêtre et sauvegarder le JSON
    cv2.destroyAllWindows()
    
    with open(chemin_json, 'w') as f:
        json.dump(verite_terrain, f, indent=4)
        
    print(f"Fichier {chemin_json} sauvegardé avec succès ! 🎉")


# --- Exécution du Script ---
if __name__ == "__main__":
    # 1. Exécute cette ligne UNE SEULE FOIS pour préparer tes dossiers
    preparer_et_diviser("data_brute", "data/validation", "data/test")
    
    # 2. Ensuite, annote tes deux dossiers très rapidement
    annoter_dossier("data/validation", "data/validation.json")
    annoter_dossier("data/test", "data/test.json")