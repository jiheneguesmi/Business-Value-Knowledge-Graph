"""
Approche 2 : Reformulation augmentée
Étape 1 → le LLM explicite la valeur implicite du paragraphe
Étape 2 → classification sur la version explicitée (plus facile à classifier)
Étape 3 → résultat ancré sur l'original + la reformulation

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
OUTPUT_DIR = Path(r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_reformulation")
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
    """
    Découpe le Markdown en segments.
    IGNORE les tableaux, images, schémas, blocs de code.
    """
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
        
        # Blocs de code
        if line_stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        
        # Titres
        heading_match = re.match(r'^(#{1,3})\s+(.+)', line_stripped)
        if heading_match:
            seg, paragraph_index = flush_buffer(current_title, buffer, paragraph_index)
            if seg:
                segments.append(seg)
            buffer = []
            current_title = heading_match.group(2).strip()
            continue
        
        # Tableaux
        if line_stripped.startswith('|'):
            in_table = True
            continue
        if re.match(r'^[\|\-\s]+$', line_stripped):
            in_table = True
            continue
        if in_table and line_stripped and not line_stripped.startswith('|'):
            in_table = False
        if in_table:
            continue
        
        # Images
        if line_stripped.startswith('!['):
            continue
        
        # Ligne vide = séparateur
        if line_stripped == "":
            seg, paragraph_index = flush_buffer(current_title, buffer, paragraph_index)
            if seg:
                segments.append(seg)
            buffer = []
        else:
            cleaned_line = clean_text_from_markdown(line)
            if cleaned_line:
                buffer.append(cleaned_line)

    # Flush final
    seg, paragraph_index = flush_buffer(current_title, buffer, paragraph_index)
    if seg:
        segments.append(seg)

    return segments


def find_all_markdown_files(root_dir: Path) -> list:
    """Parcourt récursivement tous les sous-dossiers et retourne la liste de tous les .md"""
    return list(root_dir.rglob("*.md"))


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ReformResult:
    section_title: str
    paragraph: str
    paragraph_index: int
    source_file: str = ""
    source_folder: str = ""
    # Étape 1 : explicitation
    valeur_implicite: str = ""
    reformulation: str = ""
    type_pressenti: str = ""
    indicateurs: list = None
    niveau_implicite: str = ""
    # Étape 2 : classification sur reformulation
    reponses: dict = None
    categorie_finale: str = ""
    confiance_finale: float = 0.0
    justification_finale: str = ""
    apport_reformulation: str = ""
    nb_appels_api: int = 0
    erreur: str = None


# ============================================================
# FONCTIONS PRINCIPALES
# ============================================================

def call_llm(client, prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL, max_tokens=700, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def classify_with_reformulation(segment: dict, client, source_info: dict = None) -> ReformResult:
    result = ReformResult(
        section_title=segment["section_title"],
        paragraph=segment["paragraph"],
        paragraph_index=segment["paragraph_index"],
        source_file=source_info.get("source_file", "") if source_info else "",
        source_folder=source_info.get("source_folder", "") if source_info else "",
        indicateurs=[], reponses={},
    )
    try:
        # Étape 1 : Explicitation de la valeur implicite
        r1 = call_llm(client, PROMPT_EXPLICITATION.format(**segment))
        result.nb_appels_api += 1

        result.valeur_implicite = r1.get("valeur_implicite_detectee", "")
        result.reformulation = r1.get("reformulation_explicite", "")
        result.type_pressenti = r1.get("type_valeur_pressenti", "")
        result.indicateurs = r1.get("indicateurs_detectes", [])
        result.niveau_implicite = r1.get("niveau_implicite", "")
        time.sleep(DELAY)

        # Étape 2 : Classification sur original + reformulation
        r2 = call_llm(client, PROMPT_CLASSIFICATION.format(
            paragraph=segment["paragraph"],
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
    print("APPROCHE 2 : REFORMULATION AUGMENTÉE")
    print("=" * 80)
    
    root_path = Path(MARKDOWN_ROOT)
    if not root_path.exists():
        print(f"[ERREUR] Dossier introuvable : {MARKDOWN_ROOT}")
        return
    
    # Trouver tous les fichiers .md (récursivement)
    md_files = find_all_markdown_files(root_path)
    
    if not md_files:
        print(f"[ERREUR] Aucun fichier .md trouvé dans {MARKDOWN_ROOT}")
        return
    
    print(f"📁 {len(md_files)} fichier(s) .md trouvé(s) dans l'arborescence")
    print(f"🤖 Modèle : {MODEL}")
    print(f"⚖️  Approche : Reformulation augmentée\n")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    all_results = []
    global_summary = []
    
    for i, md_path in enumerate(md_files, 1):
        try:
            rel_path = md_path.relative_to(root_path)
        except ValueError:
            rel_path = md_path.name
        
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(md_files)}] {rel_path}")
        print(f"{'=' * 60}")
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        
        # Parser le Markdown en segments
        segments = parse_markdown_to_segments(md_text)
        print(f"   {len(segments)} segments textuels détectés (tableaux/images ignorés)")
        
        source_info = {
            "source_file": md_path.name,
            "source_folder": str(md_path.parent.relative_to(root_path)) if md_path.parent != root_path else "."
        }
        
        results = []
        for seg_idx, seg in enumerate(segments):
            print(f"   Segment {seg_idx+1}/{len(segments)} : '{seg['section_title'][:50]}'...", end=" ", flush=True)
            res = classify_with_reformulation(seg, client, source_info)
            results.append(res)
            impl = f" [impl:{res.niveau_implicite}]" if res.niveau_implicite else ""
            print(f"{res.categorie_finale} ({res.confiance_finale:.2f}){impl}")
            time.sleep(DELAY)
        
        agg = aggregate_results(results)
        
        # Sauvegarde dans un dossier qui reflète la structure source
        output_subdir = OUTPUT_DIR / md_path.parent.relative_to(root_path)
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{md_path.stem}_reform.json"
        
        doc_output = {
            "fichier": str(rel_path),
            "dossier_client": md_path.parent.parent.name if md_path.parent.parent != root_path else "",
            "dossier_pdf": md_path.parent.name,
            "approche": "reformulation_augmentee",
            "segments": [asdict(r) for r in results],
            "agregation": agg,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n   Résumé {md_path.stem}:")
        print(f"      Niveau implicite : faible={agg['distribution_implicite'].get('faible',0)}, "
              f"moyen={agg['distribution_implicite'].get('moyen',0)}, "
              f"élevé={agg['distribution_implicite'].get('élevé',0)}")
        print(f"      {agg['pct_implicite_eleve']}% de segments à haute implicité")
        for cat, v in agg["distribution"].items():
            bar = "█" * int(v["pct"] / 5)
            print(f"      {cat:<12} {v['count']:>3} ({v['pct']:>5.1f}%)  {bar}")
        
        all_results.extend(results)
        global_summary.append({
            "fichier": str(rel_path),
            "dossier_client": md_path.parent.parent.name if md_path.parent.parent != root_path else "",
            "dossier_pdf": md_path.parent.name,
            **agg
        })
    
    # Sauvegarde du résumé global
    summary_path = OUTPUT_DIR / "_summary_global.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)
    
    # Affichage final
    print("\n" + "=" * 80)
    print("RÉSULTATS FINAUX")
    print("=" * 80)
    print(f"📁 Fichiers traités : {len(md_files)}")
    print(f"📊 Segments analysés : {len(all_results)}")
    
    # Distribution de l'implicité globale
    implicite_total = {"faible": 0, "moyen": 0, "élevé": 0}
    for r in all_results:
        if r.niveau_implicite in implicite_total:
            implicite_total[r.niveau_implicite] += 1
    
    if len(all_results) > 0:
        print(f"\n📊 Niveau d'implicité détecté :")
        print(f"   Faible  : {implicite_total['faible']} ({implicite_total['faible']/len(all_results)*100:.1f}%)")
        print(f"   Moyen   : {implicite_total['moyen']} ({implicite_total['moyen']/len(all_results)*100:.1f}%)")
        print(f"   Élevé   : {implicite_total['élevé']} ({implicite_total['élevé']/len(all_results)*100:.1f}%)")
    
    # Distribution par catégorie finale
    final_cats = defaultdict(int)
    for r in all_results:
        final_cats[r.categorie_finale] += 1
    
    print(f"\n📊 Distribution par catégorie :")
    for cat, count in sorted(final_cats.items()):
        pct = count / len(all_results) * 100 if all_results else 0
        print(f"   {cat:<12} : {count} ({pct:.1f}%)")
    
    print(f"\n📁 Résultats sauvegardés dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()