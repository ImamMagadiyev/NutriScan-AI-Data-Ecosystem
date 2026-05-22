# ================================
# Outils RAG
# ================================

from langchain_core.vectorstores import VectorStoreRetriever

from RAG_environment import formaterDocuments, recuperateur


def recupererDocuments(question: str, nombreDocuments: int = 5):
    recuperateurLocal = recuperateur
    if isinstance(recuperateur, VectorStoreRetriever):
        recuperateurLocal = recuperateur.vectorstore.as_retriever(search_kwargs={"k": nombreDocuments})
    return recuperateurLocal.invoke(question)


def rechercherRag(question: str, nombreDocuments: int = 5) -> str:
    """
    Interroge le RAG pour récupérer les k documents les plus pertinents.

    Args:
        question (str): question utilisateur
        nombreDocuments (int): nombre de documents à récupérer

    Returns:
        str: contexte textuel
    """
    documents = recupererDocuments(question, nombreDocuments=nombreDocuments)

    if not documents:
        return "Aucune information trouvée dans le dataset."

    return formaterDocuments(documents)