"""
Approche 1 : Chain-of-Thought avec auto-critique
Adapté pour fonctionner avec la structure de Marker
"""

import os
import re
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import anthropic

# ============================================================
# CONFIGURATION
# ============================================================

# Dossier racine contenant les fichiers .md (sortie de Marker)
MARKDOWN_ROOT = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_per_client\Docs"

# Dossier de sortie pour les résultats
OUTPUT_DIR = Path(r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API Claude
ANTHROPIC_API_KEY = "sk-ant-YOUR_KEY"
MODEL = "claude-sonnet-4-20250514"
DELAY = 0.3

CATEGORIES = ["ROI", "Notoriété", "Obligation", "Description"]

SYSTEM = "Tu es un expert en analyse de documents commerciaux. Réponds UNIQUEMENT en JSON valide, sans markdown."

# ============================================================
# PROMPTS
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
# FONCTIONS DE PARSING (IGNORE TABLEAUX/IMAGES)
# ============================================================

def clean_text_from_markdown(line: str) -> str:
    """Extrait le texte brut d'une ligne Markdown"""
    line = re.sub(r'^#{1,6}\s+', '', line)
    line = re.sub(r'!\[.*?\]\(.*?\)', '', line)
    line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    line = re.sub(r'\*(.+?)\*', r'\1', line)
    line = re.sub(r'`(.+?)`', r'\1', line)
    return line.strip()


def parse_markdown_to_segments(markdown_text: str) -> list[dict]:
    """Découpe le Markdown en segments, ignore les tableaux/images"""
    segments = []
    current_title = "Introduction"
    paragraph_index = 0
    lines = markdown_text.split("\n")
    buffer = []
    in_code_block = False
    in_table = False

    def flush_buffer(title, buf, idx):
        text = "\n".join(buf).strip()
        if len(text) >= 30:
            return {
                "section_title": title,
                "paragraph": text,
                "paragraph_index": idx,
            }, idx + 1
        return None, idx

    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        
        heading_match = re.match(r'^(#{1,3})\s+(.+)', line_stripped)
        if heading_match:
            seg, paragraph_index = flush_buffer(current_title, buffer, paragraph_index)
            if seg:
                segments.append(seg)
            buffer = []
            current_title = heading_match.group(2).strip()
            continue
        
        if line_stripped.startswith('|') or re.match(r'^[\|\-\s]+$', line_stripped):
            in_table = True
            continue
        if in_table and line_stripped and not line_stripped.startswith('|'):
            in_table = False
        if in_table:
            continue
        
        if line_stripped.startswith('!['):
            continue
        
        if line_stripped == "":
            seg, paragraph_index = flush_buffer(current_title, buffer, paragraph_index)
            if seg:
                segments.append(seg)
            buffer = []
        else:
            cleaned_line = clean_text_from_markdown(line)
            if cleaned_line:
                buffer.append(cleaned_line)

    seg, _ = flush_buffer(current_title, buffer, paragraph_index)
    if seg:
        segments.append(seg)

    return segments


def find_all_markdown_files(root_dir: Path) -> list:
    """Parcourt récursivement tous les sous-dossiers et retourne la liste de tous les .md"""
    return list(root_dir.rglob("*.md"))


# ============================================================
# CLASSIFICATION COT
# ============================================================

@dataclass
class CoTResult:
    section_title: str
    paragraph: str
    paragraph_index: int
    source_file: str = ""
    source_folder: str = ""
    categorie_initiale: str = ""
    confiance_initiale: float = 0.0
    raisonnement: str = ""
    reponses: dict = None
    mots_cles: list = None
    contradiction_detectee: bool = False
    analyse_critique: str = ""
    categorie_finale: str = ""
    confiance_finale: float = 0.0
    justification_finale: str = ""
    nb_appels_api: int = 0
    erreur: str = None


def call_llm(client, prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL, max_tokens=700, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def reponses_to_str(reponses: dict) -> str:
    labels = {
        "roi_1": "[ROI-1]", "roi_2": "[ROI-2]", "roi_3": "[ROI-3]",
        "not_1": "[NOT-1]", "not_2": "[NOT-2]", "not_3": "[NOT-3]",
        "obl_1": "[OBL-1]", "obl_2": "[OBL-2]", "obl_3": "[OBL-3]",
    }
    return "\n".join(f"  {labels.get(k, k)} : {v}" for k, v in reponses.items())


def classify_cot(segment: dict, client, source_info: dict = None) -> CoTResult:
    result = CoTResult(
        section_title=segment["section_title"],
        paragraph=segment["paragraph"],
        paragraph_index=segment["paragraph_index"],
        source_file=source_info.get("source_file", "") if source_info else "",
        source_folder=source_info.get("source_folder", "") if source_info else "",
        reponses={}, mots_cles=[],
    )
    try:
        # Étape 1 : Classification CoT
        r1 = call_llm(client, PROMPT_COT.format(**segment))
        result.nb_appels_api += 1

        cat_init = r1.get("categorie", "Description")
        if cat_init not in CATEGORIES:
            cat_init = "Description"

        result.categorie_initiale = cat_init
        result.confiance_initiale = float(r1.get("confiance", 0.5))
        result.raisonnement = r1.get("raisonnement", "")
        result.reponses = r1.get("reponses", {})
        result.mots_cles = r1.get("mots_cles", [])
        time.sleep(DELAY)

        # Étape 2 : Auto-critique
        r2 = call_llm(client, PROMPT_CRITIQUE.format(
            categorie=cat_init,
            confiance=result.confiance_initiale,
            paragraph=segment["paragraph"],
            raisonnement=result.raisonnement,
            reponses_str=reponses_to_str(result.reponses),
        ))
        result.nb_appels_api += 1

        result.contradiction_detectee = bool(r2.get("contradiction_detectee", False))
        result.analyse_critique = r2.get("analyse_critique", "")

        cat_rev = r2.get("categorie_revisee", cat_init)
        if cat_rev not in CATEGORIES:
            cat_rev = cat_init

        result.categorie_finale = cat_rev
        result.confiance_finale = float(r2.get("confiance_revisee", result.confiance_initiale))
        result.justification_finale = r2.get("justification_revisee", "")

    except Exception as e:
        result.categorie_finale = result.categorie_initiale or "Description"
        result.confiance_finale = result.confiance_initiale
        result.justification_finale = ""
        result.erreur = str(e)

    return result


def aggregate(results: list) -> dict:
    total = len(results)
    if not total:
        return {}
    counts = {c: 0 for c in CATEGORIES}
    conf_sum = {c: 0.0 for c in CATEGORIES}
    revised = sum(1 for r in results if r.contradiction_detectee)
    for r in results:
        counts[r.categorie_finale] += 1
        conf_sum[r.categorie_finale] += r.confiance_finale
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
    print("="*80)
    print("CoT + Auto-critique - Classification de valeur commerciale")
    print("="*80)
    
    root_path = Path(MARKDOWN_ROOT)
    if not root_path.exists():
        print(f"[ERREUR] Dossier introuvable : {MARKDOWN_ROOT}")
        return
    
    md_files = find_all_markdown_files(root_path)
    if not md_files:
        print(f"[ERREUR] Aucun fichier .md trouvé dans {MARKDOWN_ROOT}")
        return
    
    print(f" {len(md_files)} fichier(s) .md trouvé(s) dans l'arborescence")
    print(f" Modèle : {MODEL}")
    print(f"  Approche : CoT + Auto-critique\n")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    global_summary = []
    all_results = []
    
    for i, md_path in enumerate(md_files, 1):
        try:
            rel_path = md_path.relative_to(root_path)
        except ValueError:
            rel_path = md_path.name
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(md_files)}] {rel_path}")
        print(f"{'='*60}")
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        
        segments = parse_markdown_to_segments(md_text)
        print(f"   {len(segments)} segments textuels détectés (tableaux/images ignorés)")
        
        source_info = {
            "source_file": md_path.name,
            "source_folder": str(md_path.parent.relative_to(root_path)) if md_path.parent != root_path else "."
        }
        
        results = []
        for seg_idx, seg in enumerate(segments):
            print(f"   Segment {seg_idx+1}/{len(segments)} : '{seg['section_title'][:50]}'...", end=" ", flush=True)
            res = classify_cot(seg, client, source_info)
            results.append(res)
            rev = " [RÉVISÉ]" if res.contradiction_detectee else ""
            print(f"{res.categorie_finale} ({res.confiance_finale:.2f}){rev}")
            time.sleep(DELAY)
        
        agg = aggregate(results)
        
        # Sauvegarde
        output_filename = f"{md_path.stem}_cot.json"
        output_subdir = OUTPUT_DIR / md_path.parent.relative_to(root_path)
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / output_filename
        
        doc_output = {
            "fichier": str(rel_path),
            "dossier_client": md_path.parent.parent.name if md_path.parent.parent != root_path else "",
            "dossier_pdf": md_path.parent.name,
            "approche": "cot_autocritique",
            "segments": [asdict(r) for r in results],
            "agregation": agg,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n   Résumé {md_path.stem}:")
        print(f"      {agg['segments_revises']}/{agg['total_segments']} segments révisés ({agg['pct_revises']}%)")
        for cat, v in agg["distribution"].items():
            bar = "█" * int(v["pct"] / 5)
            print(f"      {cat:<12} {v['count']:>3} ({v['pct']:>5.1f}%)  {bar}")
        
        for r in results:
            all_results.append(r)
        
        global_summary.append({
            "fichier": str(rel_path),
            "dossier_client": md_path.parent.parent.name if md_path.parent.parent != root_path else "",
            "dossier_pdf": md_path.parent.name,
            **agg
        })
    
    # Sauvegarde résumé global
    summary_path = OUTPUT_DIR / "_summary_global.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)
    
    # Affichage final
    print("\n" + "="*80)
    print("RÉSULTATS FINAUX")
    print("="*80)
    print(f"📁 Fichiers traités : {len(md_files)}")
    print(f"📊 Segments analysés : {len(all_results)}")
    
    if all_results:
        total_revised = sum(1 for r in all_results if r.contradiction_detectee)
        print(f" Segments révisés : {total_revised} ({total_revised/len(all_results)*100:.1f}%)")
        
        final_cats = defaultdict(int)
        for r in all_results:
            final_cats[r.categorie_finale] += 1
        
        print(f"\n Distribution finale :")
        for cat, count in sorted(final_cats.items()):
            pct = count / len(all_results) * 100
            print(f"   {cat:<12} : {count} ({pct:.1f}%)")
    
    print(f"\n Résultats sauvegardés dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n\n Erreur fatale : {e}")
        import traceback
        traceback.print_exc()