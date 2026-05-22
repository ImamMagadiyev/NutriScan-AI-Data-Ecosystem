# ================================
# Outil Similarité
# ================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from RAG_environment import obtenirDataframe, trouverProduitParCodebarres, trouverProduitsParNom


ORDRE_GRADES = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def construireTexteSimilarite(ligne: pd.Series) -> str:
    return " ".join(
        [
            str(ligne.get("product_name", "")),
            str(ligne.get("categories_en", "")),
            str(ligne.get("ingredients_text", "")),
            str(ligne.get("brands_tags", "")),
        ]
    )


def resoudreReferenceProduit(referenceProduit: str) -> dict | None:
    produit = trouverProduitParCodebarres(referenceProduit)
    if produit:
        return produit

    correspondances = trouverProduitsParNom(referenceProduit, limite=1)
    if correspondances:
        return correspondances[0]
    return None


def trouverProduitsSimilaires(referenceProduit: str, nombreResultats: int = 5, mieuxNotesSeulement: bool = False) -> list[dict]:
    tableau = obtenirDataframe().copy()
    cible = resoudreReferenceProduit(referenceProduit)
    if not cible:
        return []

    codeCible = str(cible.get("code", "")).strip()
    textes = tableau.apply(construireTexteSimilarite, axis=1)
    vectoriseur = TfidfVectorizer(stop_words="english")
    vecteurs = vectoriseur.fit_transform(textes)

    listeIndicesCible = tableau.index[tableau["code"].astype(str).str.strip() == codeCible].tolist()
    if not listeIndicesCible:
        return []

    indiceCible = listeIndicesCible[0]
    scoresSimilarite = cosine_similarity(vecteurs[indiceCible], vecteurs).flatten()
    candidats = tableau.copy()
    candidats["similarite"] = scoresSimilarite
    candidats = candidats[candidats["code"].astype(str).str.strip() != codeCible]

    gradeCible = str(cible.get("nutrition_grade_fr", "")).lower().strip()
    if mieuxNotesSeulement and gradeCible in ORDRE_GRADES:
        rangCible = ORDRE_GRADES[gradeCible]
        candidats = candidats[
            candidats["nutrition_grade_fr"].astype(str).str.lower().map(ORDRE_GRADES).fillna(99) < rangCible
        ]

    candidats = candidats.sort_values(["similarite", "nutrition-score-fr_100g"], ascending=[False, True])
    return candidats.head(nombreResultats).to_dict(orient="records")


def trouverProduitsSimilairesMieuxNotes(referenceProduit: str, nombreResultats: int = 5) -> list[dict]:
    return trouverProduitsSimilaires(referenceProduit, nombreResultats=nombreResultats, mieuxNotesSeulement=True)


def calculerSimilarite(texte1: str, texte2: str) -> float:
    """
    Calcule la similarité cosinus entre deux textes.

    Args:
        texte1 (str): premier texte
        texte2 (str): second texte

    Returns:
        float: score de similarité entre 0 et 1
    """
    vectoriseur = TfidfVectorizer().fit([texte1, texte2])
    vecteurs = vectoriseur.transform([texte1, texte2])
    scoreSimilarite = cosine_similarity(vecteurs[0], vecteurs[1])[0][0]
    return round(scoreSimilarite, 3)