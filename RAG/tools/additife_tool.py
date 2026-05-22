# ================================
# Outil Additif
# ================================

import re


baseAdditifs = {
    "E100": {
        "nom": "Curcumine",
        "role": "colorant",
        "description": "Colorant jaune d'origine naturelle, souvent extrait du curcuma.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e100-curcumine",
    },
    "E200": {
        "nom": "Acide sorbique",
        "role": "conservateur",
        "description": "Conservateur utilisé pour limiter le développement de levures et moisissures.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e200-acide-sorbique",
    },
    "E202": {
        "nom": "Sorbate de potassium",
        "role": "conservateur",
        "description": "Sel de l'acide sorbique, utilisé pour prolonger la conservation des aliments.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e202-sorbate-de-potassium",
    },
    "E300": {
        "nom": "Acide ascorbique",
        "role": "antioxydant",
        "description": "Vitamine C utilisée comme antioxydant pour limiter l'oxydation.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e300-acide-ascorbique",
    },
    "E322": {
        "nom": "Lécithines",
        "role": "émulsifiant",
        "description": "Famille d'émulsifiants utilisés pour stabiliser les mélanges eau/gras.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e322-lecithines",
    },
    "E330": {
        "nom": "Acide citrique",
        "role": "acidifiant",
        "description": "Acidifiant très courant qui ajuste l'acidité et améliore parfois la conservation ou le goût.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e330-acide-citrique",
    },
    "E331": {
        "nom": "Citrates de sodium",
        "role": "correcteur d'acidité",
        "description": "Régule l'acidité et peut stabiliser certaines boissons et préparations.",
        "source_nom": "Open Food Facts",
        "url_source": "https://fr.openfoodfacts.org/additif/e331-citrates-de-sodium",
    },
}


def extraireCodeAdditif(texte: str) -> str | None:
    correspondance = re.search(r"\bE\s?(\d{3,4}[a-z]?)\b", texte.upper())
    if not correspondance:
        return None
    return f"E{correspondance.group(1).upper()}"


def expliquerAdditif(code: str) -> dict:
    codeAdditif = extraireCodeAdditif(code) or code.upper().strip()
    infos = baseAdditifs.get(codeAdditif)
    if infos:
        return {"code": codeAdditif, **infos}

    return {
        "code": codeAdditif,
        "nom": "Additif non repertorie localement",
        "role": "inconnu",
        "description": "Aucune fiche locale n'est disponible. Une verification externe est necessaire.",
        "source_nom": "Open Food Facts",
        "url_source": f"https://fr.openfoodfacts.org/recherche?search_terms={codeAdditif}",
    }


def obtenirInfoAdditif(code: str) -> str:
    """
    Retourne les informations sur un additif alimentaire.

    Args:
        code (str): code additif (ex: "E100")

    Returns:
        str: description de l'additif
    """
    infos = expliquerAdditif(code)
    return (
        f"{infos['code']} - {infos['nom']} ({infos['role']}) : "
        f"{infos['description']} Source : {infos['source_nom']} - {infos['url_source']}"
    )