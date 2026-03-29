from evaluation import evaluer_modele

def main():
    DOSSIER_VALIDATION = "data/validation"
    JSON_VALIDATION = "data/validation.json"
    
    DOSSIER_TEST = "data/test"
    JSON_TEST = "data/test.json"

    # 1- PHASE DE VALIDATION
    # En ajustant la taille du flou, on cherche à obtenir la MAE et la MSE les plus basses possibles.
    print("PHASE DE VALIDATION (Réglage des hyperparamètres)")
    meilleur_flou = (7, 7)
    evaluer_modele(DOSSIER_VALIDATION, JSON_VALIDATION, taille_flou=meilleur_flou)

    # 2- PHASE DE TEST (RÈGLE D'OR)
    # print("\n\nPHASE DE TEST (Évaluation finale)")
    # evaluer_modele(DOSSIER_TEST, JSON_TEST, taille_flou=meilleur_flou)

if __name__ == "__main__":
    main()