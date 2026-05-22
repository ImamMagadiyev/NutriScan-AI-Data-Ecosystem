# ================================
# Outil Nutrition
# ================================

ORDRE_GRADES = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def premierNombre(donnees: dict, *cles: str):
    for cle in cles:
        valeur = donnees.get(cle)
        if valeur in [None, "", "null", "None"]:
            continue
        try:
            return float(valeur)
        except (TypeError, ValueError):
            continue
    return None


def premiereValeur(donnees: dict, *cles: str):
    for cle in cles:
        valeur = donnees.get(cle)
        if valeur not in [None, "", "null", "None"]:
            return valeur
    return None


def normaliserInfosProduit(infosProduit: dict) -> dict:
    nutriments = infosProduit.get("nutriments", {}) if isinstance(infosProduit, dict) else {}
    return {
        "product_name": premiereValeur(infosProduit, "product_name"),
        "nutrition_grade_fr": str(
            premiereValeur(infosProduit, "nutrition_grade_fr", "nutriscore_grade", "nutriscore") or ""
        ).lower(),
        "nutrition_score_fr": premierNombre(
            infosProduit,
            "nutrition-score-fr_100g",
            "nutrition_score_fr",
            "nutriscore_score",
        ),
        "sugars_100g": premierNombre(infosProduit, "sugars_100g") or premierNombre(nutriments, "sugars_100g"),
        "saturated_fat_100g": premierNombre(infosProduit, "saturated-fat_100g") or premierNombre(nutriments, "saturated-fat_100g"),
        "salt_100g": premierNombre(infosProduit, "salt_100g") or premierNombre(nutriments, "salt_100g"),
        "fiber_100g": premierNombre(infosProduit, "fiber_100g") or premierNombre(nutriments, "fiber_100g"),
        "proteins_100g": premierNombre(infosProduit, "proteins_100g") or premierNombre(nutriments, "proteins_100g"),
        "fat_100g": premierNombre(infosProduit, "fat_100g") or premierNombre(nutriments, "fat_100g"),
    }


def scoreVersGrade(scoreNutritionnel: float | None) -> str | None:
    if scoreNutritionnel is None:
        return None
    if scoreNutritionnel <= -1:
        return "a"
    if scoreNutritionnel <= 2:
        return "b"
    if scoreNutritionnel <= 10:
        return "c"
    if scoreNutritionnel <= 18:
        return "d"
    return "e"


def estimerNutriscore(infosProduit: dict) -> dict:
    infosNormalisees = normaliserInfosProduit(infosProduit)
    if infosNormalisees["nutrition_score_fr"] is not None:
        gradeEstime = scoreVersGrade(infosNormalisees["nutrition_score_fr"])
        return {
            "grade_estime": gradeEstime,
            "score_estime": infosNormalisees["nutrition_score_fr"],
            "methode": "score_officiel_present",
            "justification": (
                f"Le score nutritionnel numerique vaut {infosNormalisees['nutrition_score_fr']}, "
                f"ce qui correspond au Nutri-Score {gradeEstime.upper()}."
            ),
        }

    score = 0.0
    detailsCalcul = []

    sucres = infosNormalisees["sugars_100g"]
    if sucres is not None:
        if sucres >= 45:
            score += 10
        elif sucres >= 31:
            score += 7
        elif sucres >= 18:
            score += 5
        elif sucres >= 9:
            score += 2
        detailsCalcul.append(f"sucres={sucres}g/100g")

    graissesSaturees = infosNormalisees["saturated_fat_100g"]
    if graissesSaturees is not None:
        if graissesSaturees >= 10:
            score += 10
        elif graissesSaturees >= 5:
            score += 7
        elif graissesSaturees >= 2:
            score += 4
        elif graissesSaturees >= 1:
            score += 2
        detailsCalcul.append(f"graisses saturees={graissesSaturees}g/100g")

    sel = infosNormalisees["salt_100g"]
    if sel is not None:
        if sel >= 1.5:
            score += 8
        elif sel >= 1:
            score += 6
        elif sel >= 0.6:
            score += 4
        elif sel >= 0.3:
            score += 2
        detailsCalcul.append(f"sel={sel}g/100g")

    fibres = infosNormalisees["fiber_100g"]
    if fibres is not None:
        if fibres >= 7:
            score -= 5
        elif fibres >= 4:
            score -= 3
        elif fibres >= 2:
            score -= 1
        detailsCalcul.append(f"fibres={fibres}g/100g")

    proteines = infosNormalisees["proteins_100g"]
    if proteines is not None:
        if proteines >= 8:
            score -= 2
        elif proteines >= 4:
            score -= 1
        detailsCalcul.append(f"proteines={proteines}g/100g")

    gradeEstime = scoreVersGrade(score)
    return {
        "grade_estime": gradeEstime,
        "score_estime": round(score, 2),
        "methode": "heuristique_nutriments",
        "justification": (
            "Estimation heuristique basee sur les nutriments disponibles : "
            + ", ".join(detailsCalcul)
            + f". Score estime = {round(score, 2)}."
        ),
    }


def evaluerCoherenceNutriscore(infosProduit: dict) -> dict:
    if not infosProduit:
        return {"statut": "inconnu", "message": "Informations produit manquantes."}

    infosNormalisees = normaliserInfosProduit(infosProduit)
    gradeActuel = infosNormalisees["nutrition_grade_fr"] or None
    estimation = estimerNutriscore(infosProduit)

    if not gradeActuel:
        return {
            "statut": "manquant",
            "message": "Nutri-Score absent dans les donnees.",
            "estimation": estimation,
        }

    rangActuel = ORDRE_GRADES.get(gradeActuel)
    rangEstime = ORDRE_GRADES.get(estimation["grade_estime"])
    if rangActuel is None or rangEstime is None:
        return {
            "statut": "inconnu",
            "message": "Impossible de comparer le Nutri-Score avec certitude.",
            "estimation": estimation,
        }

    ecart = abs(rangActuel - rangEstime)
    if ecart <= 1:
        statut = "coherent"
        message = (
            f"Le Nutri-Score {gradeActuel.upper()} parait coherent avec l'estimation "
            f"{estimation['grade_estime'].upper()}."
        )
    else:
        statut = "incoherent"
        message = (
            f"Le Nutri-Score {gradeActuel.upper()} semble peu coherent avec l'estimation "
            f"{estimation['grade_estime'].upper()}."
        )

    return {
        "statut": statut,
        "message": message,
        "grade_actuel": gradeActuel,
        "estimation": estimation,
    }


def evaluerNutriscore(infosProduit: dict) -> str:
    """
    Évalue la cohérence du Nutri-Score d'un produit.

    Args:
        infosProduit (dict): informations produit (ex: calories, nutriscore)

    Returns:
        str: "Cohérent" ou "Incohérent"
    """
    resultat = evaluerCoherenceNutriscore(infosProduit)
    return resultat["message"]