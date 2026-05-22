# ================================
# RAG_ENVIRONMENT
# ================================

from pathlib import Path
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOSSIER_BASE = Path(__file__).resolve().parent
CHEMIN_JEU_DONNEES = DOSSIER_BASE / "derniere_versionn.csv"
DOSSIER_INDEX = Path("C:/faiss_openfoodfacts")
CHEMIN_ENV = DOSSIER_BASE / ".env"
MAX_LIGNES = int(os.getenv("OFF_MAX_ROWS", "500"))

if CHEMIN_ENV.exists():
    load_dotenv(dotenv_path=CHEMIN_ENV)
else:
    load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def valeurSecurisee(valeur):
    if pd.isna(valeur) or str(valeur).strip().lower() in ["nan", "", "null", "none"]:
        return "[DONNEE MANQUANTE]"
    return str(valeur)


def chargerJeuDonnees(nombreLignes: int = MAX_LIGNES) -> pd.DataFrame:
    tableau = pd.read_csv(CHEMIN_JEU_DONNEES, low_memory=False, nrows=nombreLignes)
    tableau = tableau.fillna("")
    return tableau


tableauProduits = chargerJeuDonnees()


def formater_ligne_pour_agent(row: pd.Series) -> str:
    infos = [f"{col}: {valeurSecurisee(row[col])}" for col in tableauProduits.columns]
    return "PRODUIT SOURCE | " + " | ".join(infos)


if "connaissance_produit" not in tableauProduits.columns:
    tableauProduits["connaissance_produit"] = tableauProduits.apply(formater_ligne_pour_agent, axis=1)


def construirePlongements() -> HuggingFaceEmbeddings:
    nomModele = "sentence-transformers/all-MiniLM-L6-v2"

    # Priorise le cache local pour eviter les appels reseau et les warnings HF non authentifies.
    try:
        return HuggingFaceEmbeddings(
            model_name=nomModele,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(model_name=nomModele)


plongements = construirePlongements()


documents = [
    Document(
        page_content=text,
        metadata={
            "source": "openfoodfacts",
            "product_name": row.get("product_name", ""),
            "code": str(row.get("code", "")),
        },
    )
    for text, (_, row) in zip(tableauProduits["connaissance_produit"].tolist(), tableauProduits.iterrows())
]


def chargerOuCreerMagasinVecteurs() -> FAISS:
    cheminFaiss = DOSSIER_INDEX / "index.faiss"
    cheminPkl = DOSSIER_INDEX / "index.pkl"

    if cheminFaiss.exists() and cheminPkl.exists():
        return FAISS.load_local(
            str(DOSSIER_INDEX),
            plongements,
            allow_dangerous_deserialization=True,
        )
    
    DOSSIER_INDEX.mkdir(parents=True, exist_ok=True)


    baseVecteurs = FAISS.from_documents(documents, plongements)
    baseVecteurs.save_local(str(DOSSIER_INDEX))
    return baseVecteurs


magasinVecteurs = chargerOuCreerMagasinVecteurs()
recuperateur = magasinVecteurs.as_retriever(search_kwargs={"k": 5})


def formaterDocuments(documentsTrouves) -> str:
    """Transforme une liste de documents FAISS en texte brut."""
    return "\n\n".join(document.page_content for document in documentsTrouves)


def obtenirDataframe() -> pd.DataFrame:
    return tableauProduits.copy()


def trouverProduitParCodebarres(codebarres: str):
    texteCodebarres = str(codebarres).strip()
    if not texteCodebarres:
        return None

    correspondances = tableauProduits[tableauProduits["code"].astype(str).str.strip() == texteCodebarres]
    if correspondances.empty:
        return None
    return correspondances.iloc[0].to_dict()


def trouverProduitsParNom(nom: str, limite: int = 5) -> list[dict]:
    requete = str(nom).strip()
    if not requete:
        return []

    correspondances = tableauProduits[
        tableauProduits["product_name"].astype(str).str.contains(requete, case=False, na=False)
    ].head(limite)
    return correspondances.to_dict(orient="records")


def trouverProduitsSansPays(limite: int = 5) -> list[dict]:
    pays = tableauProduits["countries_en"].astype(str).str.strip().str.lower()
    correspondances = tableauProduits[pays.isin(["", "null", "none", "nan"])].head(limite)
    return correspondances.to_dict(orient="records")