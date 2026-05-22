from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from RAG_environment import (
    trouverProduitParCodebarres,
    trouverProduitsParNom,
)
from tools.additife_tool import expliquerAdditif, extraireCodeAdditif
from tools.api_tool import obtenirInfosProduit, obtenirOrigineProduit
from tools.nutrition_tool import estimerNutriscore, evaluerCoherenceNutriscore
from tools.rag_tool import rechercherRag
from tools.similarity_tool import trouverProduitsSimilaires, trouverProduitsSimilairesMieuxNotes
from tools.sql_tool import (
    assurerCacheSqlite,
    obtenirProduitsSansPaysSql,
    rechercherProduitSql,
)


load_dotenv()


class NutritionAgent:
    def __init__(self) -> None:
        self.cleApi = os.getenv("MISTRALAI_API_KEY")
        self.maxToursMemoire = int(os.getenv("AGENT_MAX_MEMORY_TURNS", "12"))
        self.maxEtapesReact = int(os.getenv("AGENT_MAX_REACT_STEPS", "4"))
        self.memoire: list[dict[str, Any]] = []
        self.cheminBaseSqlite = assurerCacheSqlite()
        self.modele = None
        if self.cleApi:
            self.modele = ChatMistralAI(
                model="mistral-small",
                temperature=0.2,
                api_key=self.cleApi,
            )
        self.gabaritPrompt = ChatPromptTemplate.from_template(
            """
Tu es un agent nutritionnel pour Open Food Facts.
Tu reçois une question utilisateur et un dossier de preuves.

Réponds en français avec exactement trois sections :
1. Réponse
2. Proposition d'enrichissement
3. Justification

Contraintes :
- N'invente aucune donnée absente des preuves.
- Si une information manque, dis-le explicitement.
- Cite les chiffres utiles lorsqu'ils sont disponibles.
- Si une source externe est utilisée, mentionne-la clairement.
- Si les preuves montrent une ambiguite (plusieurs produits possibles), demande une clarification explicite.

Historique recent : {contexte_memoire}
Question : {question}
Preuves : {preuves}
""".strip()
        )

    def extraireCodebarres(self, question: str, codebarres: str | None = None) -> str | None:
        if codebarres:
            return codebarres.strip()
        correspondance = re.search(r"\b\d{8,14}\b", question)
        return correspondance.group(0) if correspondance else None

    def detecterIntention(self, question: str) -> str:
        questionMinuscule = question.lower()
        if extraireCodeAdditif(question) or "additif" in questionMinuscule:
            return "additif"
        if "similaire" in questionMinuscule or "mieux not" in questionMinuscule:
            return "similarite"
        if "catégorie" in questionMinuscule or "categorie" in questionMinuscule:
            return "categorie"
        if "pays" in questionMinuscule or "origine" in questionMinuscule:
            return "origine"
        if "cohérent" in questionMinuscule or "coherent" in questionMinuscule:
            return "coherence"
        if "nutri-score" in questionMinuscule or "nutriscore" in questionMinuscule:
            return "estimer_nutriscore"
        return "rag"

    def resoudreProduitLocal(
        self,
        question: str,
        indiceProduit: str | None = None,
        codebarres: str | None = None,
    ) -> dict | None:
        codebarresEffectif = self.extraireCodebarres(question, codebarres)
        if codebarresEffectif:
            produit = trouverProduitParCodebarres(codebarresEffectif)
            if produit:
                return produit

        if indiceProduit:
            correspondances = trouverProduitsParNom(indiceProduit, limite=1)
            if correspondances:
                return correspondances[0]

        textesEntreGuillemets = re.findall(r'"([^"]+)"', question)
        for candidat in textesEntreGuillemets:
            correspondances = trouverProduitsParNom(candidat, limite=1)
            if correspondances:
                return correspondances[0]

        return None

    def resumerCategorie(self, referenceProduit: str) -> dict:
        produitsSimilaires = trouverProduitsSimilaires(referenceProduit, nombreResultats=5, mieuxNotesSeulement=False)
        listeCategories = []
        for produit in produitsSimilaires:
            categoriesBrutes = str(produit.get("categories_en", ""))
            listeCategories.extend(
                [categorie.strip() for categorie in categoriesBrutes.split(",") if categorie.strip() and categorie.strip().lower() not in {"null", "undefined"}]
            )

        categoriesCommunes = Counter(listeCategories).most_common(3)
        categorieSuggeree = categoriesCommunes[0][0] if categoriesCommunes else "Categorie indeterminee"
        return {
            "categorie_suggeree": categorieSuggeree,
            "categories_preuves": categoriesCommunes,
            "produits_similaires": [
                {
                    "product_name": produit.get("product_name"),
                    "categories_en": produit.get("categories_en"),
                    "nutrition_grade_fr": produit.get("nutrition_grade_fr"),
                }
                for produit in produitsSimilaires
            ],
        }

    def construirePayload(
        self,
        question: str,
        intention: str,
        produitLocal: dict | None,
        produitExterne: dict | None,
        analyse: dict,
        contexteRag: str,
        contexteSql: dict,
        traceReact: list[dict],
    ) -> dict:
        return {
            "intention": intention,
            "produit_local": produitLocal,
            "produit_externe": produitExterne,
            "analyse": analyse,
            "contexte_rag": contexteRag[:2000],
            "contexte_sql": contexteSql,
            "trace_react": traceReact,
        }

    def synthetiser(self, question: str, chargeUtile: dict) -> str:
        preuves = json.dumps(chargeUtile, ensure_ascii=False, indent=2)
        contexteMemoire = self.contexteMemoire()
        if not self.modele:
            return self.reponseSecours(question, chargeUtile)

        try:
            pipelineMessage = self.gabaritPrompt | self.modele
            return pipelineMessage.invoke(
                {
                    "question": question,
                    "preuves": preuves,
                    "contexte_memoire": contexteMemoire,
                }
            ).content
        except Exception:
            return self.reponseSecours(question, chargeUtile)

    def reponseSecours(self, question: str, chargeUtile: dict) -> str:
        analyse = chargeUtile["analyse"]
        reponse = analyse.get("reponse", "Aucune reponse exploitable n'a pu etre construite.")
        enrichissement = analyse.get("enrichissement", "Aucune proposition d'enrichissement supplementaire.")
        justification = analyse.get("justification", "Justification indisponible.")
        return (
            f"Réponse\n{reponse}\n\n"
            f"Proposition d'enrichissement\n{enrichissement}\n\n"
            f"Justification\n{justification}"
        )

    def contexteMemoire(self) -> str:
        if not self.memoire:
            return "Aucun historique disponible."

        toursRecents = self.memoire[-5:]
        lignes = []
        for position, tour in enumerate(toursRecents, start=1):
            lignes.append(
                f"{position}. Q={tour['question']} | intention={tour['intention']} | resume={tour['resume']}"
            )
        return "\n".join(lignes)

    def memoriser(self, question: str, intention: str, texteReponse: str, chargeUtile: dict) -> None:
        analyse = chargeUtile.get("analyse", {}) if isinstance(chargeUtile, dict) else {}
        resume = str(analyse.get("reponse") or texteReponse).strip().replace("\n", " ")
        resume = resume[:260]
        self.memoire.append(
            {
                "question": question,
                "intention": intention,
                "resume": resume,
                "reponse": texteReponse,
                "charge_utile": chargeUtile,
            }
        )
        if len(self.memoire) > self.maxToursMemoire:
            self.memoire = self.memoire[-self.maxToursMemoire :]

    def obtenirMemoire(self) -> list[dict[str, Any]]:
        return self.memoire.copy()

    def viderMemoire(self) -> None:
        self.memoire = []

    def executerBoucleReact(
        self,
        question: str,
        intention: str,
        indiceProduit: str | None,
        codebarres: str | None,
    ) -> tuple[dict | None, dict | None, str, dict, list[dict], list[dict]]:
        produitLocal = None
        produitExterne = None
        contexteRag = ""
        contexteSql: dict[str, Any] = {}
        traceReact: list[dict] = []
        candidats: list[dict] = []

        referenceDeduite = (
            (codebarres or "").strip()
            or (indiceProduit or "").strip()
            or (self.extraireCodebarres(question) or "").strip()
        )

        for etape in range(1, self.maxEtapesReact + 1):
            if etape == 1:
                if referenceDeduite:
                    candidats = rechercherProduitSql(referenceDeduite, limite=5)
                    contexteSql["reference"] = referenceDeduite
                    contexteSql["candidats"] = candidats
                    traceReact.append(
                        {
                            "etape": etape,
                            "reflexion": "Verifier d'abord la base SQL locale issue du RAG.",
                            "action": "rechercherProduitSql",
                            "observation": f"{len(candidats)} candidat(s) trouves.",
                        }
                    )
                    if len(candidats) == 1:
                        produitLocal = candidats[0]
                else:
                    traceReact.append(
                        {
                            "etape": etape,
                            "reflexion": "Aucune reference explicite produit/code-barres.",
                            "action": "rechercherProduitSql",
                            "observation": "Recherche SQL non lancee faute de reference.",
                        }
                    )

            elif etape == 2:
                if not produitLocal:
                    produitLocal = self.resoudreProduitLocal(
                        question,
                        indiceProduit=indiceProduit,
                        codebarres=codebarres,
                    )
                traceReact.append(
                    {
                        "etape": etape,
                        "reflexion": "Completer avec la resolution locale existante.",
                        "action": "resoudreProduitLocal",
                        "observation": "Produit local trouve." if produitLocal else "Aucun produit local resolu.",
                    }
                )

            elif etape == 3:
                codebarresEffectif = (
                    self.extraireCodebarres(question, codebarres)
                    or str((produitLocal or {}).get("code", "")).strip()
                    or None
                )
                if codebarresEffectif:
                    produitExterne = obtenirInfosProduit(codebarresEffectif)
                contexteRag = rechercherRag(question, nombreDocuments=4)
                traceReact.append(
                    {
                        "etape": etape,
                        "reflexion": "Fusionner preuves externes et contexte RAG.",
                        "action": "obtenirInfosProduit + rechercherRag",
                        "observation": (
                            f"API={'ok' if produitExterne else 'non utilisee'}, "
                            f"RAG_len={len(contexteRag)}"
                        ),
                    }
                )

            else:
                if intention == "origine" and not self.extraireCodebarres(question, codebarres):
                    contexteSql["exemples_sans_pays"] = obtenirProduitsSansPaysSql(limite=5)
                    traceReact.append(
                        {
                            "etape": etape,
                            "reflexion": "Sans code-barres, proposer des produits incomplets detectes en SQL.",
                            "action": "obtenirProduitsSansPaysSql",
                            "observation": f"{len(contexteSql['exemples_sans_pays'])} exemple(s) recuperes.",
                        }
                    )
                else:
                    traceReact.append(
                        {
                            "etape": etape,
                            "reflexion": "Arret de boucle: preuves suffisantes.",
                            "action": "stop",
                            "observation": "Fin de l'orchestration ReAct.",
                        }
                    )
                break

        return produitLocal, produitExterne, contexteRag, contexteSql, traceReact, candidats

    def demander(
        self,
        question: str,
        indiceProduit: str | None = None,
        codebarres: str | None = None,
    ) -> dict:
        intention = self.detecterIntention(question)
        (
            produitLocal,
            produitExterne,
            contexteRag,
            contexteSql,
            traceReact,
            candidats,
        ) = self.executerBoucleReact(question, intention, indiceProduit, codebarres)

        codebarresEffectif = (
            self.extraireCodebarres(question, codebarres)
            or str((produitLocal or {}).get("code", "")).strip()
            or None
        )

        candidatsAmbigus = []
        if len(candidats) > 1:
            candidatsAmbigus = [
                {
                    "code": str(candidat.get("code", "")),
                    "product_name": candidat.get("product_name", ""),
                }
                for candidat in candidats[:5]
            ]

        if intention == "additif":
            codeAdditif = extraireCodeAdditif(question) or extraireCodeAdditif(contexteRag or "") or "E330"
            additif = expliquerAdditif(codeAdditif)
            analyse = {
                "reponse": (
                    f"{additif['code']} correspond a {additif['nom']}. Son role principal est : {additif['role']}. "
                    f"{additif['description']}"
                ),
                "enrichissement": (
                    f"Ajouter une fiche additif structuree avec code, nom, role et source fiable. "
                    f"Source retenue : {additif['source_nom']} ({additif['url_source']})."
                ),
                "justification": (
                    f"La reponse s'appuie sur la fiche locale de l'additif et sur une source externe referencee : "
                    f"{additif['url_source']}."
                ),
                "additif": additif,
            }
        elif intention == "categorie":
            referenceProduit = indiceProduit or str((produitLocal or {}).get("product_name", ""))
            if not referenceProduit and candidatsAmbigus:
                analyse = {
                    "reponse": "Plusieurs produits correspondent a la demande, je ne peux pas proposer une categorie unique sans precision.",
                    "enrichissement": "Preciser le code-barres ou le nom exact du produit parmi les candidats proposes.",
                    "justification": f"Ambiguite detectee en SQL local: {candidatsAmbigus}",
                    "candidats_ambigus": candidatsAmbigus,
                }
            elif not referenceProduit:
                analyse = {
                    "reponse": "Je n'ai pas trouve de produit cible explicite pour proposer une categorie.",
                    "enrichissement": "Fournir un nom produit ou un code-barres.",
                    "justification": "Aucune reference exploitable dans la question.",
                }
            else:
                donneesCategorie = self.resumerCategorie(referenceProduit)
                nomProduit = (produitLocal or {}).get("product_name", referenceProduit)
                analyse = {
                    "reponse": (
                        f"Pour {nomProduit}, la categorie la plus plausible est : {donneesCategorie['categorie_suggeree']}."
                    ),
                    "enrichissement": (
                        f"Renseigner le champ categories_en avec {donneesCategorie['categorie_suggeree']} puis demander une validation humaine."
                    ),
                    "justification": (
                        f"La proposition vient des produits les plus proches dans la base, dont les categories dominantes sont : "
                        f"{donneesCategorie['categories_preuves']}."
                    ),
                    **donneesCategorie,
                }
        elif intention == "origine":
            if codebarresEffectif:
                origine = obtenirOrigineProduit(codebarresEffectif)
                pays = origine.get("pays") or []
                textePays = ", ".join(pays) if isinstance(pays, list) else str(pays)
                analyse = {
                    "reponse": f"Le code-barres {codebarresEffectif} renvoie le pays d'origine suivant : {textePays or 'information non trouvee' }.",
                    "enrichissement": "Completer le champ countries_en avec la valeur retournee par l'API Open Food Facts si elle est validee.",
                    "justification": "Le pays provient de l'API Open Food Facts interrogee a partir du code-barres.",
                    "origine": origine,
                }
            else:
                exemplesSansPays = contexteSql.get("exemples_sans_pays") or obtenirProduitsSansPaysSql(limite=3)
                analyse = {
                    "reponse": "Aucun code-barres explicite n'a ete detecte dans la question.",
                    "enrichissement": "Fournir un code-barres ou un produit cible pour completer le pays d'origine via l'API Open Food Facts.",
                    "justification": f"Exemples locaux de produits a completer : {[exemple.get('product_name') for exemple in exemplesSansPays]}",
                    "exemples_sans_pays": exemplesSansPays,
                }
        elif intention == "similarite":
            referenceProduit = indiceProduit or str((produitLocal or {}).get("product_name", ""))
            if not referenceProduit and candidatsAmbigus:
                analyse = {
                    "reponse": "Je detecte plusieurs produits possibles, donc je ne peux pas calculer une similarite ciblee sans precision.",
                    "enrichissement": "Donner le code-barres exact du produit a comparer.",
                    "justification": f"Ambiguite detectee en SQL local: {candidatsAmbigus}",
                    "candidats_ambigus": candidatsAmbigus,
                }
            elif not referenceProduit:
                analyse = {
                    "reponse": "Je n'ai pas trouve de produit de reference pour la recherche de similarite.",
                    "enrichissement": "Fournir un nom de produit ou un code-barres.",
                    "justification": "Aucune reference exploitable dans la question.",
                }
            else:
                produitsSimilaires = trouverProduitsSimilairesMieuxNotes(referenceProduit, nombreResultats=5)
                produitsFormates = [
                    f"{produit.get('product_name')} (Nutri-Score {str(produit.get('nutrition_grade_fr', '?')).upper()}, similarite={round(float(produit.get('similarite', 0)), 3)})"
                    for produit in produitsSimilaires
                ]
                analyse = {
                    "reponse": "Produits similaires mieux notes trouves : " + "; ".join(produitsFormates) if produitsFormates else "Aucun produit similaire mieux note n'a ete trouve.",
                    "enrichissement": "Ajouter au produit courant une liste de recommandations alternatives mieux notees.",
                    "justification": "La similarite est calculee sur le nom, la categorie, les ingredients et la marque via TF-IDF.",
                    "produits_similaires": produitsSimilaires,
                }
        elif intention == "coherence":
            donneesProduit = produitLocal or produitExterne or {}
            coherence = evaluerCoherenceNutriscore(donneesProduit)
            analyse = {
                "reponse": coherence["message"],
                "enrichissement": "Conserver le Nutri-Score actuel s'il est coherent, sinon signaler une verification manuelle.",
                "justification": coherence.get("estimation", {}).get("justification", "Comparaison realisee a partir des nutriments disponibles."),
                "coherence": coherence,
            }
        elif intention == "estimer_nutriscore":
            donneesProduit = produitLocal or produitExterne or {}
            estimation = estimerNutriscore(donneesProduit)
            nomProduit = (donneesProduit or {}).get("product_name", "ce produit")
            analyse = {
                "reponse": (
                    f"Pour {nomProduit}, le Nutri-Score estime est {str(estimation['grade_estime']).upper()} "
                    f"(score {estimation['score_estime']})."
                ),
                "enrichissement": "Ajouter le Nutri-Score estime comme proposition d'enrichissement en attendant validation humaine.",
                "justification": estimation["justification"],
                "estimation": estimation,
            }
        else:
            analyse = {
                "reponse": "Voici la meilleure synthese disponible a partir du contexte local.",
                "enrichissement": "Enrichir le produit avec les champs manquants identifies dans le contexte.",
                "justification": contexteRag or "Aucun contexte local supplementaire disponible.",
            }

        chargeUtile = self.construirePayload(
            question,
            intention,
            produitLocal,
            produitExterne,
            analyse,
            contexteRag,
            contexteSql,
            traceReact,
        )
        texteReponse = self.synthetiser(question, chargeUtile)
        self.memoriser(question, intention, texteReponse, chargeUtile)
        return {
            "question": question,
            "intention": intention,
            "reponse": texteReponse,
            "charge_utile": chargeUtile,
        }


def executerDemo() -> list[dict]:
    agentNutrition = NutritionAgent()
    casDeTest = [
        {
            "question": "Ce produit n'a pas de categorie claire. Peux-tu en proposer une ?",
            "indice_produit": "Organic Oat Groats",
        },
        {
            "question": "Explique l'additif E330 en t'appuyant sur une source fiable.",
        },
        {
            "question": "Trouve des produits similaires mieux notes dans la base.",
            "indice_produit": "Chili Mango",
        },
        {
            "question": "Ce Nutri-Score est-il coherent avec les valeurs nutritionnelles ?",
            "indice_produit": "Chili Mango",
        },
        {
            "question": "Ce produit n'a pas de Nutri-Score. Peux-tu estimer un score a partir de ses nutriments ?",
            "indice_produit": "Organic Oat Groats",
        },
        {
            "question": "Le pays d'origine est manquant pour le code-barres 3017620422003. Peux-tu le completer a partir du code-barres ?",
            "codebarres": "3017620422003",
        },
    ]

    sorties = []
    for cas in casDeTest:
        sorties.append(
            agentNutrition.demander(
                cas["question"],
                indiceProduit=cas.get("indice_produit"),
                codebarres=cas.get("codebarres"),
            )
        )
    return sorties


if __name__ == "__main__":
    for item in executerDemo():
        print("=" * 80)
        print("QUESTION:", item["question"])
        print(item["reponse"])