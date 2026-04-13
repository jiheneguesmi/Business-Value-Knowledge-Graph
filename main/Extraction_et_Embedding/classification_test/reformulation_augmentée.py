"""
Approche 2 : Reformulation augmentée
Étape 1 → le LLM explicite la valeur implicite du paragraphe
Étape 2 → classification sur la version explicitée (plus facile à classifier)
Étape 3 → résultat ancré sur l'original + la reformulation

Adapté pour fonctionner avec les segments JSON (sortie du découpage)
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

# Dossier racine contenant les fichiers *_segments.json (sortie du découpage)
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments\Idex"

# Dossier de sortie pour les résultats
OUTPUT_DIR = Path(r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_reformulation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY non définie. Créez un fichier .env")

client = Groq(api_key=GROQ_API_KEY)

# Modèle gratuit recommandé (stable, moins de rate limits)
MODEL = "llama-3.1-8b-instant"   # ou "llama-3.3-70b-versatile"

# Paramètres anti-rate-limit
BASE_DELAY = 2.0          # secondes entre appels
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

CATEGORIES = ["ROI", "Notoriété", "Obligation", "Description"]

SYSTEM_PROMPT = """Tu es un expert en analyse de documents commerciaux.
Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON."""

# ============================================================
# PROMPTS (inchangés)
# ============================================================

PROMPT_EXPLICITATION = """Section parente : "{section_title}"

Paragraphe original :
\"\"\"
{paragraph}
\"\"\"

Ce paragraphe provient d'une brochure commerciale. Il peut exprimer sa valeur de manière implicite.

Reformule ce paragraphe en rendant EXPLICITE :
1. Ce que cela apporte concrètement à l'organisation ou à l'usager
2. Le type de valeur sous-entendu (financière, image, conformité, description neutre)
3. Les conséquences implicites si cette valeur est ignorée ou réalisée

Retourne ce JSON :
{{
  "valeur_implicite_detectee": "<ce que le paragraphe dit sans le dire>",
  "reformulation_explicite": "<le même message rendu totalement explicite en 2-3 phrases>",
  "type_valeur_pressenti": "<ROI|Notoriété|Obligation|Description>",
  "indicateurs_detectes": ["<indice1>", "<indice2>"],
  "niveau_implicite": "<faible|moyen|élevé>"
}}"""

PROMPT_CLASSIFICATION = """Tu dois classifier ce contenu commercial.

Paragraphe original :
\"\"\"
{paragraph}
\"\"\"

Reformulation explicite de sa valeur :
\"\"\"
{reformulation}
\"\"\"

Valeur implicite détectée : {valeur_implicite}
Indicateurs : {indicateurs}

En te basant SUR LES DEUX versions (original + reformulation explicite), réponds aux questions :

[ROI-1] Gain financier, réduction de coût, rentabilité mesurable ?
[ROI-2] Amélioration fonctionnelle (temps, charge, automatisation) comme avantage opérationnel ?
[ROI-3] Impact sur résultats, ressources ou performance d'une organisation ?
[NOT-1] Amélioration bien-être, confort, qualité de vie d'un usager ?
[NOT-2] Label, reconnaissance, attractivité, image positive ?
[NOT-3] Impact visible ou perçu positivement dans l'environnement ou l'expérience utilisateur ?
[OBL-1] Nécessité de respecter une norme, loi ou exigence réglementaire ?
[OBL-2] Mesure de sécurité ou prévention des risques ?
[OBL-3] Action nécessaire pour éviter danger, sanction ou garantir protection minimale ?

Retourne ce JSON :
{{
  "reponses": {{
    "roi_1": "<oui|non>", "roi_2": "<oui|non>", "roi_3": "<oui|non>",
    "not_1": "<oui|non>", "not_2": "<oui|non>", "not_3": "<oui|non>",
    "obl_1": "<oui|non>", "obl_2": "<oui|non>", "obl_3": "<oui|non>"
  }},
  "categorie": "<ROI|Notoriété|Obligation|Description>",
  "confiance": <0.0-1.0>,
  "justification": "<une phrase>",
  "apport_reformulation": "<en quoi la reformulation a changé ou confirmé ta classification>"
}}"""


# ============================================================
# FONCTIONS DE LECTURE DES SEGMENTS
# ============================================================

def find_all_segments_files(root_dir: Path) -> list[Path]:
    """Trouve tous les fichiers *_segments.json dans l'arborescence"""
    return list(root_dir.rglob("*_segments.json"))


# ============================================================
# APPEL API GROQ AVEC GESTION DES RATE LIMITS
# ============================================================

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


# ============================================================
# CLASSIFICATION AVEC REFORMULATION
# ============================================================

@dataclass
class ReformResult:
    section_title: str
    paragraph: str
    paragraph_index: int
    source_file: str = ""
    source_folder: str = ""
    # Étape 1
    valeur_implicite: str = ""
    reformulation: str = ""
    type_pressenti: str = ""
    indicateurs: list = None
    niveau_implicite: str = ""
    # Étape 2
    reponses: dict = None
    categorie_finale: str = ""
    confiance_finale: float = 0.0
    justification_finale: str = ""
    apport_reformulation: str = ""
    nb_appels_api: int = 0
    erreur: str = None


def classify_with_reformulation(segment: dict, source_info: dict = None) -> ReformResult:
    result = ReformResult(
        section_title=segment["section_title"],
        paragraph=segment["paragraph"],
        paragraph_index=segment["paragraph_index"],
        source_file=source_info.get("source_file", "") if source_info else "",
        source_folder=source_info.get("source_folder", "") if source_info else "",
        indicateurs=[], reponses={},
    )

    try:
        # Troncature pour éviter les timeouts
        truncated_paragraph = segment["paragraph"][:600]
        seg_for_prompt = {
            "section_title": segment["section_title"],
            "paragraph": truncated_paragraph,
            "paragraph_index": segment["paragraph_index"],
        }

        # Étape 1 : Explicitation
        r1 = call_llm_groq_with_retry(PROMPT_EXPLICITATION.format(**seg_for_prompt))
        result.nb_appels_api += 1

        result.valeur_implicite = r1.get("valeur_implicite_detectee", "")
        result.reformulation = r1.get("reformulation_explicite", "")
        result.type_pressenti = r1.get("type_valeur_pressenti", "")
        result.indicateurs = r1.get("indicateurs_detectes", [])
        result.niveau_implicite = r1.get("niveau_implicite", "")
        time.sleep(BASE_DELAY)

        # Étape 2 : Classification
        r2 = call_llm_groq_with_retry(PROMPT_CLASSIFICATION.format(
            paragraph=truncated_paragraph,
            reformulation=result.reformulation,
            valeur_implicite=result.valeur_implicite,
            indicateurs=", ".join(result.indicateurs),
        ))
        result.nb_appels_api += 1

        cat = r2.get("categorie", "Description")
        if cat not in CATEGORIES:
            cat = "Description"

        result.reponses = r2.get("reponses", {})
        result.categorie_finale = cat
        result.confiance_finale = float(r2.get("confiance", 0.5))
        result.justification_finale = r2.get("justification", "")
        result.apport_reformulation = r2.get("apport_reformulation", "")

    except Exception as e:
        result.categorie_finale = "Description"
        result.confiance_finale = 0.0
        result.erreur = str(e)

    return result


def aggregate_results(results: list) -> dict:
    total = len(results)
    if not total:
        return {}
    
    counts = {c: 0 for c in CATEGORIES}
    conf_sum = {c: 0.0 for c in CATEGORIES}
    implicite_counts = {"faible": 0, "moyen": 0, "élevé": 0}

    for r in results:
        counts[r.categorie_finale] += 1
        conf_sum[r.categorie_finale] += r.confiance_finale
        if r.niveau_implicite in implicite_counts:
            implicite_counts[r.niveau_implicite] += 1

    return {
        "total_segments": total,
        "distribution_implicite": implicite_counts,
        "pct_implicite_eleve": round(implicite_counts.get("élevé", 0) / total * 100, 1),
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
    print("APPROCHE 2 : REFORMULATION AUGMENTÉE (Groq)")
    print("=" * 80)

    seg_root = Path(SEGMENTS_INPUT_DIR)
    if not seg_root.exists():
        print(f"[ERREUR] Dossier des segments introuvable : {SEGMENTS_INPUT_DIR}")
        return

    segment_files = find_all_segments_files(seg_root)
    if not segment_files:
        print("[ERREUR] Aucun fichier *_segments.json trouvé")
        return

    print(f"📁 {len(segment_files)} fichier(s) de segments")
    print(f"🤖 Modèle Groq : {MODEL}")
    print(f"⚖️  Approche : Reformulation augmentée\n")

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

        results_for_file = []
        for seg_idx, seg in enumerate(segments_data):
            print(f"   Segment {seg_idx+1}/{len(segments_data)} : '{seg['section_title'][:50]}'...")
            res = classify_with_reformulation(seg, source_info)
            results_for_file.append(res)
            impl = f" [impl:{res.niveau_implicite}]" if res.niveau_implicite else ""
            print(f"      → {res.categorie_finale} (conf:{res.confiance_finale:.2f}){impl}")
            if res.erreur:
                print(f"      ERREUR: {res.erreur[:100]}")
            time.sleep(BASE_DELAY)

        agg = aggregate_results(results_for_file)

        # Sauvegarde
        out_subdir = OUTPUT_DIR / seg_path.parent.relative_to(seg_root)
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / f"{seg_path.stem.replace('_segments', '')}_reform.json"
        doc_output = {
            "fichier": str(rel_path),
            "dossier_client": seg_path.parent.parent.name if seg_path.parent.parent != seg_root else "",
            "dossier_pdf": seg_path.parent.name,
            "approche": "reformulation_augmentee_groq",
            "segments": [asdict(r) for r in results_for_file],
            "agregation": agg,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc_output, f, ensure_ascii=False, indent=2)

        print(f"\n   Résumé {seg_path.stem.replace('_segments', '')}:")
        print(f"      Niveau implicite : faible={agg['distribution_implicite'].get('faible',0)}, "
              f"moyen={agg['distribution_implicite'].get('moyen',0)}, "
              f"élevé={agg['distribution_implicite'].get('élevé',0)}")
        print(f"      {agg['pct_implicite_eleve']}% de segments à haute implicité")
        for cat, v in agg["distribution"].items():
            bar = "█" * int(v["pct"] / 5)
            print(f"      {cat:<12} {v['count']:>3} ({v['pct']:>5.1f}%)  {bar}")

        all_results.extend(results_for_file)

    # Statistiques finales
    total = len(all_results)
    if total:
        implicite_total = {"faible": 0, "moyen": 0, "élevé": 0}
        final_cats = defaultdict(int)
        for r in all_results:
            if r.niveau_implicite in implicite_total:
                implicite_total[r.niveau_implicite] += 1
            final_cats[r.categorie_finale] += 1

        print("\n" + "=" * 80)
        print("RÉSULTATS FINAUX")
        print("=" * 80)
        print(f"📊 Segments analysés : {total}")
        print(f"\n📊 Niveau d'implicité :")
        for level, cnt in implicite_total.items():
            print(f"   {level:<6} : {cnt} ({cnt/total*100:.1f}%)")
        print(f"\n📊 Distribution par catégorie :")
        for cat, cnt in sorted(final_cats.items()):
            print(f"   {cat:<12} : {cnt} ({cnt/total*100:.1f}%)")

    print(f"\n📁 Résultats sauvegardés dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        print(f"\nErreur fatale : {e}")
        import traceback
        traceback.print_exc()