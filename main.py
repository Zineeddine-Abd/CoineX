import argparse
import os
from evaluation import evaluer_modele

def main():
    parser = argparse.ArgumentParser(description="Évaluation de CoineX")
    parser.add_argument("--pipeline", type=str, default="morphologie",
                        choices=["morphologie", "contours", "nn"],
                        help="Choix du pipeline à évaluer (morphologie par défaut)")
    parser.add_argument("--mode", type=str, default="validation", 
                        choices=["validation", "test"], 
                        help="Dataset à utiliser (validation ou test)")
    parser.add_argument("--image", type=str, default=None,
                        help="Chemin vers une image spécifique à tester (ignore l'évaluation complète si fourni)")
    
    args = parser.parse_args()
    meilleur_flou = (7, 7)

    # 1. Tester une seule image si l'argument --image est fourni
    if args.image:
        if args.pipeline == "morphologie":
            from pipelines.morphologie.traitement import compter_pieces
        elif args.pipeline == "contours":
            from pipelines.contours.traitement import compter_pieces
        elif args.pipeline == "nn":
            from pipelines.nn.traitement import compter_pieces
        
        if not os.path.exists(args.image):
            print(f"[ERREUR] L'image {args.image} est introuvable.")
            return
            
        print(f"--- Traitement de l'image : {args.image} avec le pipeline '{args.pipeline}' ---")
        nombre = compter_pieces(args.image, taille_flou=meilleur_flou)
        print(f"\n[RÉSULTAT] L'algorithme a détecté : {nombre} pièces.\n")
        return

    # 2. Sinon, évaluer un dataset entier
    DOSSIER_VALIDATION = "data/validation"
    JSON_VALIDATION = "data/validation.json"
    DOSSIER_TEST = "data/test"
    JSON_TEST = "data/test.json"

    if args.mode == "validation":
        print(f"PHASE DE VALIDATION (Réglage des hyperparamètres) - Pipeline : {args.pipeline}")
        evaluer_modele(DOSSIER_VALIDATION, JSON_VALIDATION, pipeline=args.pipeline, taille_flou=meilleur_flou)
    elif args.mode == "test":
        print(f"\n\nPHASE DE TEST (Évaluation finale) - Pipeline : {args.pipeline}")
        evaluer_modele(DOSSIER_TEST, JSON_TEST, pipeline=args.pipeline, taille_flou=meilleur_flou)

if __name__ == "__main__":
    main()