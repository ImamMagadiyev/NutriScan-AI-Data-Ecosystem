# ================================
# Outil API
# ================================

import requests


URL_API_PRODUIT_OFF_PAR_DEFAUT = "https://world.openfoodfacts.org/api/v2/product"
URL_API_RECHERCHE_OFF_PAR_DEFAUT = "https://world.openfoodfacts.org/cgi/search.pl"
EN_TETES_REQUETE = {
    "User-Agent": "SAES4-Agent/1.0 (OpenFoodFacts educational project)",
}


def effectuerRequeteGet(url: str, parametres: dict | None = None) -> dict:
    reponse = requests.get(url, params=parametres, headers=EN_TETES_REQUETE, timeout=10)
    reponse.raise_for_status()
    return reponse.json()


def obtenirInfosProduit(codebarres: str, urlApi: str = URL_API_PRODUIT_OFF_PAR_DEFAUT) -> dict:
    """
    Interroge une API externe pour récupérer les informations d'un produit via son code-barres.

    Args:
        codebarres (str): code-barres du produit
        urlApi (str): URL de l'API

    Returns:
        dict: informations produit (JSON)
    """
    try:
        donnees = effectuerRequeteGet(f"{urlApi}/{codebarres}.json")
        if donnees.get("status") != 1:
            return {
                "codebarres": codebarres,
                "erreur": donnees.get("status_verbose", "Produit introuvable sur Open Food Facts."),
            }
        return donnees.get("product", {})
    except requests.exceptions.RequestException as erreur:
        return {"erreur": str(erreur)}


def rechercherProduits(requete: str, taillePage: int = 5) -> list[dict]:
    try:
        donnees = effectuerRequeteGet(
            URL_API_RECHERCHE_OFF_PAR_DEFAUT,
            parametres={
                "search_terms": requete,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": taillePage,
                "fields": "code,product_name,countries,categories,nutriscore_grade",
            },
        )
    except requests.exceptions.RequestException:
        return []

    return donnees.get("products", [])


def obtenirOrigineProduit(codebarres: str) -> dict:
    produit = obtenirInfosProduit(codebarres)
    if produit.get("erreur"):
        return produit

    return {
        "codebarres": codebarres,
        "pays": produit.get("countries") or produit.get("countries_tags") or [],
        "source": "Open Food Facts API",
    }