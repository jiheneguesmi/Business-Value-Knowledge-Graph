"""
Approche 1 : Chain-of-Thought avec auto-critique
Adapté pour fonctionner avec les segments JSON
Utilise Groq (modèle gratuit) avec gestion des rate limits
"""

import os
import re
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments\Idex"
OUTPUT_DIR = Path(r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY non définie")

client = Groq(api_key=GROQ_API_KEY)

# Modèle plus stable et moins sujet aux rate limits
MODEL = "llama-3.1-8b-instant"   # ou "llama-3.1-70b-versatile"

# Paramètres anti-rate-limit
BASE_DELAY = 2.0          # délai de base entre appels (secondes)
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

CATEGORIES = ["ROI", "Notoriété", "Obligation", "Description"]

SYSTEM_PROMPT = """Tu es un expert en analyse de documents commerciaux.
Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON."""

# ============================================================
# PROMPTS (inchangés mais on réduit la longueur des paragraphes)
# ============================================================
PROMPT_COT = """Section parente : "{section_title}"

Paragraphe :
\"\"\"
{paragraph}
\"\"\"

Classifie ce paragraphe en suivant ce raisonnement pas à pas :

ÉTAPE 1 — Identifie les mots-clés porteurs de sens.
ÉTAPE 2 — Évalue chaque question (oui/non) :
  [ROI-1] Gain financier, réduction de coût, rentabilité mesurable ?
  [ROI-2] Amélioration fonctionnelle (temps, charge, automatisation) comme avantage opérationnel ?
  [ROI-3] Impact sur résultats, ressources ou performance d'une organisation ?
  [NOT-1] Amélioration bien-être, confort, qualité de vie d'un usager ?
  [NOT-2] Label, reconnaissance, attractivité, image positive ?
  [NOT-3] Impact visible ou perçu positivement dans l'environnement ou l'expérience utilisateur ?
  [OBL-1] Nécessité de respecter une norme, loi ou exigence réglementaire ?
  [OBL-2] Mesure de sécurité ou prévention des risques ?
  [OBL-3] Action nécessaire pour éviter danger, sanction ou garantir protection minimale ?
ÉTAPE 3 — Déduis la catégorie principale :
  ROI        → majorité oui sur ROI-1/2/3
  Notoriété  → majorité oui sur NOT-1/2/3
  Obligation → majorité oui sur OBL-1/2/3
  Description → aucune majorité claire

Retourne ce JSON :
{{
  "raisonnement": "<résumé 2-3 phrases>",
  "mots_cles": ["<mot1>", "<mot2>"],
  "reponses": {{
    "roi_1": "<oui|non>", "roi_2": "<oui|non>", "roi_3": "<oui|non>",
    "not_1": "<oui|non>", "not_2": "<oui|non>", "not_3": "<oui|non>",
    "obl_1": "<oui|non>", "obl_2": "<oui|non>", "obl_3": "<oui|non>"
  }},
  "categorie": "<ROI|Notoriété|Obligation|Description>",
  "confiance": <0.0-1.0>,
  "justification": "<une phrase>"
}}"""

PROMPT_CRITIQUE = """Tu as classifié ce paragraphe comme "{categorie}" (confiance : {confiance}).

Paragraphe :
\"\"\"
{paragraph}
\"\"\"

Raisonnement initial : {raisonnement}

Réponses aux questions :
{reponses_str}

Critique : vérifie la cohérence entre tes réponses et ta classification.
- Y a-t-il une contradiction entre tes "oui" et la catégorie choisie ?
- As-tu sous-estimé une dimension implicite ?
- La valeur exprimée est-elle directe ou implicite ?

Retourne ce JSON :
{{
  "contradiction_detectee": <true|false>,
  "analyse_critique": "<ce que tu as potentiellement mal évalué>",
  "categorie_revisee": "<ROI|Notoriété|Obligation|Description>",
  "confiance_revisee": <0.0-1.0>,
  "justification_revisee": "<une phrase>"
}}"""


# ============================================================
# FONCTIONS
# ============================================================

def find_all_segments_files(root_dir: Path) -> list[Path]:
    return list(root_dir.rglob("*_segments.json"))


def call_llm_groq_with_retry(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Appelle Groq avec backoff exponentiel sur les erreurs 429"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=700,
            )
            raw = response.choices[0].message.content.strip()
            # Nettoyage des backticks
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                wait_time = BASE_DELAY * (BACKOFF_FACTOR ** attempt) + random.uniform(0, 1)
                print(f"      ⚠️ Rate limit ({MODEL}), attente {wait_time:.1f}s (tentative {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                continue
            elif attempt < MAX_RETRIES:
                wait_time = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                print(f"      ⚠️ Erreur {MODEL}: {e}, nouvel essai dans {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Échec après {MAX_RETRIES} tentatives")


def reponses_to_str(reponses: dict) -> str:
    labels = {
        "roi_1": "[ROI-1]", "roi_2": "[ROI-2]", "roi_3": "[ROI-3]",
        "not_1": "[NOT-1]", "not_2": "[NOT-2]", "not_3": "[NOT-3]",
        "obl_1": "[OBL-1]", "obl_2": "[OBL-2]", "obl_3": "[OBL-3]",
    }
    return "\n".join(f"  {labels.get(k, k)} : {v}" for k, v in reponses.items())


def classify_cot(segment: dict, source_info: dict = None) -> dict:
    """Retourne un dictionnaire de résultats (pour éviter dataclass non sérialisable)"""
    result = {
        "section_title": segment["section_title"],
        "paragraph": segment["paragraph"],
        "paragraph_index": segment["paragraph_index"],
        "source_file": source_info.get("source_file", "") if source_info else "",
        "source_folder": source_info.get("source_folder", "") if source_info else "",
        "categorie_initiale": "",
        "confiance_initiale": 0.0,
        "raisonnement": "",
        "reponses": {},
        "mots_cles": [],
        "contradiction_detectee": False,
        "analyse_critique": "",
        "categorie_finale": "",
        "confiance_finale": 0.0,
        "justification_finale": "",
        "nb_appels_api": 0,
        "erreur": None,
    }

    try:
        # Tronquer le paragraphe à 600 caractères pour éviter les timeouts
        truncated_paragraph = segment["paragraph"][:600]
        seg_for_prompt = {
            "section_title": segment["section_title"],
            "paragraph": truncated_paragraph,
            "paragraph_index": segment["paragraph_index"],
        }

        # Étape 1
        r1 = call_llm_groq_with_retry(PROMPT_COT.format(**seg_for_prompt))
        result["nb_appels_api"] += 1

        cat_init = r1.get("categorie", "Description")
        if cat_init not in CATEGORIES:
            cat_init = "Description"

        result["categorie_initiale"] = cat_init
        result["confiance_initiale"] = float(r1.get("confiance", 0.5))
        result["raisonnement"] = r1.get("raisonnement", "")
        result["reponses"] = r1.get("reponses", {})
        result["mots_cles"] = r1.get("mots_cles", [])
        time.sleep(BASE_DELAY)

        # Étape 2
        r2 = call_llm_groq_with_retry(PROMPT_CRITIQUE.format(
            categorie=cat_init,
            confiance=result["confiance_initiale"],
            paragraph=truncated_paragraph,
            raisonnement=result["raisonnement"],
            reponses_str=reponses_to_str(result["reponses"]),
        ))
        result["nb_appels_api"] += 1

        result["contradiction_detectee"] = bool(r2.get("contradiction_detectee", False))
        result["analyse_critique"] = r2.get("analyse_critique", "")

        cat_rev = r2.get("categorie_revisee", cat_init)
        if cat_rev not in CATEGORIES:
            cat_rev = cat_init

        result["categorie_finale"] = cat_rev
        result["confiance_finale"] = float(r2.get("confiance_revisee", result["confiance_initiale"]))
        result["justification_finale"] = r2.get("justification_revisee", "")

    except Exception as e:
        result["categorie_finale"] = result["categorie_initiale"] or "Description"
        result["confiance_finale"] = result["confiance_initiale"]
        result["justification_finale"] = ""
        result["erreur"] = str(e)

    return result


def aggregate(results: list) -> dict:
    total = len(results)
    if not total:
        return {}
    counts = {c: 0 for c in CATEGORIES}
    conf_sum = {c: 0.0 for c in CATEGORIES}
    revised = sum(1 for r in results if r["contradiction_detectee"])
    for r in results:
        counts[r["categorie_finale"]] += 1
        conf_sum[r["categorie_finale"]] += r["confiance_finale"]
    return {
        "total_segments": total,
        "segments_revises": revised,
        "pct_revises": round(revised / total * 100, 1),
        "distribution": {
            c: {
                "count": counts[c],
                "pct": round(counts[c] / total * 100, 1),
                "avg_confidence": round(conf_sum[c] / counts[c], 3) if counts[c] else 0.0,
            }
            for c in CATEGORIES
        },
        "dominant_category": max(counts, key=counts.get),
    }


# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 80)
    print("CoT + Auto-critique - Classification (Groq) - Rate limit géré")
    print("=" * 80)

    seg_root = Path(SEGMENTS_INPUT_DIR)
    if not seg_root.exists():
        print(f"[ERREUR] Dossier introuvable : {SEGMENTS_INPUT_DIR}")
        return

    segment_files = find_all_segments_files(seg_root)
    if not segment_files:
        print("[ERREUR] Aucun fichier *_segments.json trouvé")
        return

    print(f"📁 {len(segment_files)} fichier(s) de segments")
    print(f"🤖 Modèle : {MODEL}")
    print(f"⏱️  Délai entre appels : {BASE_DELAY}s\n")

    all_results = []

    for seg_path in segment_files:
        rel_path = seg_path.relative_to(seg_root)
        print(f"\n📄 {rel_path}")

        with open(seg_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)

        print(f"   {len(segments_data)} segments")

        source_info = {
            "source_file": seg_path.name,
            "source_folder": str(seg_path.parent.relative_to(seg_root)),
        }

        for seg_idx, seg in enumerate(segments_data):
            print(f"   Segment {seg_idx+1}/{len(segments_data)} : '{seg['section_title'][:40]}'...")
            res = classify_cot(seg, source_info)
            rev = " [RÉVISÉ]" if res["contradiction_detectee"] else ""
            print(f"      → {res['categorie_finale']} (conf:{res['confiance_finale']:.2f}){rev}")
            if res["erreur"]:
                print(f"      ERREUR: {res['erreur'][:100]}")
            all_results.append(res)
            time.sleep(BASE_DELAY)  # pause entre segments

        # Sauvegarde
        out_subdir = OUTPUT_DIR / seg_path.parent.relative_to(seg_root)
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / f"{seg_path.stem.replace('_segments', '')}_cot.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([r for r in all_results if r["source_file"] == seg_path.name], f, ensure_ascii=False, indent=2)

    # Statistiques finales
    total = len(all_results)
    if total:
        final_cats = defaultdict(int)
        for r in all_results:
            final_cats[r["categorie_finale"]] += 1
        print("\n" + "=" * 80)
        print("RÉSUMÉ FINAL")
        print("=" * 80)
        for cat, cnt in sorted(final_cats.items()):
            print(f"  {cat:<12} : {cnt} ({cnt/total*100:.1f}%)")
    print(f"\n📁 Résultats dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption")
    except Exception as e:
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()