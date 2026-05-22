# ================================
# Outil SQL
# ================================

from pathlib import Path
import sqlite3

from RAG_environment import obtenirDataframe


DOSSIER_BASE = Path(__file__).resolve().parents[1]
CHEMIN_BASE_PAR_DEFAUT = DOSSIER_BASE / "ma_base_openfoodfacts" / "openfoodfacts.db"
NOM_TABLE = "products"


def echapperLitteralSql(value: str) -> str:
    return value.replace("'", "''")


def assurerCacheSqlite(cheminBase: str | Path = CHEMIN_BASE_PAR_DEFAUT, forcerRafraichissement: bool = False) -> str:
    """Crée (ou met à jour) une base SQLite locale à partir du dataset chargé par le RAG."""
    fichierBase = Path(cheminBase)
    fichierBase.parent.mkdir(parents=True, exist_ok=True)

    if fichierBase.exists() and not forcerRafraichissement:
        return str(fichierBase)

    tableau = obtenirDataframe().copy()
    with sqlite3.connect(fichierBase) as connexion:
        tableau.to_sql(NOM_TABLE, connexion, if_exists="replace", index=False)
    return str(fichierBase)

def interrogerBase(requete: str, cheminBase: str | Path = CHEMIN_BASE_PAR_DEFAUT) -> list:
    """
    Exécute une requête SQL et retourne les résultats.
    
    Args:
        requete (str): requête SQL
        cheminBase (str): chemin vers la base SQLite
    
    Returns:
        list: liste de tuples contenant les résultats
    """
    requeteSecurisee = requete.strip()
    if not requeteSecurisee.lower().startswith("select"):
        return [("Erreur SQL", "Seules les requetes SELECT sont autorisees.")]

    cheminBaseResolue = assurerCacheSqlite(cheminBase)
    try:
        with sqlite3.connect(cheminBaseResolue) as connexion:
            curseur = connexion.cursor()
            curseur.execute(requeteSecurisee)
            resultats = curseur.fetchall()
            return resultats
    except sqlite3.Error as erreur:
        return [("Erreur SQL", str(erreur))]


def interrogerBaseCommeDict(requete: str, cheminBase: str | Path = CHEMIN_BASE_PAR_DEFAUT) -> list[dict]:
    """Exécute une requête SELECT et retourne une liste de dictionnaires."""
    requeteSecurisee = requete.strip()
    if not requeteSecurisee.lower().startswith("select"):
        return [{"error": "Seules les requetes SELECT sont autorisees."}]

    cheminBaseResolue = assurerCacheSqlite(cheminBase)
    try:
        with sqlite3.connect(cheminBaseResolue) as connexion:
            connexion.row_factory = sqlite3.Row
            curseur = connexion.cursor()
            curseur.execute(requeteSecurisee)
            lignes = curseur.fetchall()
            return [dict(ligne) for ligne in lignes]
    except sqlite3.Error as erreur:
        return [{"error": str(erreur)}]


def rechercherProduitSql(referenceProduit: str, limite: int = 5) -> list[dict]:
    """Recherche SQL par code-barres exact ou nom produit partiel."""
    valeur = echapperLitteralSql(str(referenceProduit).strip())
    if not valeur:
        return []

    requete = f"""
SELECT *
FROM {NOM_TABLE}
WHERE TRIM(CAST(code AS TEXT)) = '{valeur}'
   OR LOWER(CAST(product_name AS TEXT)) LIKE LOWER('%{valeur}%')
LIMIT {int(limite)}
"""
    return interrogerBaseCommeDict(requete)


def obtenirProduitsSansPaysSql(limite: int = 5) -> list[dict]:
    requete = f"""
SELECT code, product_name, countries_en
FROM {NOM_TABLE}
WHERE countries_en IS NULL
   OR TRIM(CAST(countries_en AS TEXT)) = ''
   OR LOWER(TRIM(CAST(countries_en AS TEXT))) IN ('nan', 'null', 'none')
LIMIT {int(limite)}
"""
    return interrogerBaseCommeDict(requete)