"""
Script : Classification Multi-LLM — OpenRouter (5 modèles gratuits)
v3 — export par fichier :
  - Chaque JSON lu → résultats sauvegardés DANS LE MÊME DOSSIER que le JSON source
  - Si le fichier de résultat existe déjà → SKIP (reprise en cas d'interruption)
  - Export global final conservé dans OUTPUT_DIR
"""

import os
import re
import json
import time
import random
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import requests

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments\1km"
OUTPUT_DIR         = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante dans le fichier .env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Modèles ────────────────────────────────────────────────────────────────────
ENSEMBLE_A = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-24b-instruct-2501",
    "qwen/qwen3-8b",
    "openai/gpt-4o-mini",
]
MODELS_TO_USE = ENSEMBLE_A

# ── Paramètres ────────────────────────────────────────────────────────────────
MAX_TOKENS        = 1024
MAX_RETRIES       = 3
BACKOFF_BASE      = 3.0
INTER_MODEL_DELAY = 2.0

CATEGORIES     = ["ROI", "Notoriété", "Obligation", "Description"]
PRIORITY_ORDER = ["ROI", "Obligation", "Notoriété", "Description"]
QUESTION_KEYS  = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un expert en analyse de documents commerciaux. "
    "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON."
)

PROMPT_TEMPLATE = """Section parente : "{section_title}"

Paragraphe :
\"\"\"{paragraph}\"\"\"

Questions (oui/non) :
[ROI-1] Gain financier, réduction de coût, rentabilité mesurable ?
[ROI-2] Amélioration fonctionnelle claire (temps, charge, automatisation) comme avantage opérationnel ?
[ROI-3] Impact sur résultats, ressources ou performance d'une organisation ?
[NOT-1] Amélioration bien-être, confort, qualité de vie ou cadre de travail d'un usager (interne ou externe)?
[NOT-2] Label, reconnaissance, attractivité, image positive d'un service, d'un lieu ou de l'organisation ?
[NOT-3] Impact visible ou perçu positivement dans l'environnement, les usages ou l'expérience utilisateur ?
[OBL-1] Nécessité de respecter une norme, loi ou exigence réglementaire ?
[OBL-2] Mesure de sécurité ou prévention des risques(physiques, numériques, juridiques...) ?
[OBL-3] Action nécessaire pour éviter danger, sanction ou garantir protection minimale ?

Réponds avec ce JSON (sans markdown) :
{{"reponses_questions":{{"roi_1":"<oui|non>","roi_2":"<oui|non>","roi_3":"<oui|non>","not_1":"<oui|non>","not_2":"<oui|non>","not_3":"<oui|non>","obl_1":"<oui|non>","obl_2":"<oui|non>","obl_3":"<oui|non>"}}}}"""


# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class Segment:
    section_title:   str
    paragraph:       str
    paragraph_index: int
    source_file:     str
    source_folder:   str


@dataclass
class ModelPrediction:
    model_name:         str
    reponses_questions: dict
    erreur:             str = None


@dataclass
class FinalClassification:
    section_title:          str
    paragraph:              str
    paragraph_index:        int
    source_file:            str
    source_folder:          str
    categorie:              str
    scores_roi:             int
    scores_notoriete:       int
    scores_obligation:      int
    reponses_questions:     dict
    predictions_par_modele: list = field(default_factory=list)
    ensemble_modeles:       list = field(default_factory=list)


# ── API OpenRouter ─────────────────────────────────────────────────────────────
def _extract_content(response_data: dict) -> str:
    choice  = response_data["choices"][0]
    message = choice.get("message", {})
    content   = message.get("content")
    reasoning = message.get("reasoning")
    if content and content.strip():
        return content.strip()
    if reasoning and reasoning.strip():
        return reasoning.strip()
    raise ValueError("Réponse vide : ni 'content' ni 'reasoning' disponibles")


def _call_openrouter(model_name: str, messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jiheneguesmi/Business-Value-Knowledge-Graph",
        "X-Title": "Business Value Classification",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                print(f"        Rate limit ({model_name.split('/')[-1]}), attente {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                raise ValueError(f"Modèle introuvable : {model_name}")
            if resp.status_code == 400:
                err_detail = resp.json().get("error", {}).get("message", resp.text[:200])
                raise ValueError(f"Requête invalide (400) : {err_detail}")
            resp.raise_for_status()
            return _extract_content(resp.json())
        except (ValueError, KeyError) as e:
            raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * attempt)
            else:
                raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * attempt
                print(f"        {model_name.split('/')[-1]}: {e}, retry {attempt}/{MAX_RETRIES} dans {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Échec après {MAX_RETRIES} tentatives pour {model_name}")


def _parse_json_response(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        matches = list(re.finditer(r'\{[^{}]*\}', raw, re.DOTALL))
        for match in sorted(matches, key=lambda m: len(m.group()), reverse=True):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Aucun JSON valide trouvé dans : {raw[:300]}")


def _normalize_reponses(parsed: dict) -> dict:
    rq = parsed.get("reponses_questions", {})
    return {
        k: ("oui" if str(rq.get(k, "non")).lower().strip() in ("oui","yes","true","1") else "non")
        for k in QUESTION_KEYS
    }


def classify_with_model(segment: Segment, model_name: str) -> ModelPrediction:
    prompt = PROMPT_TEMPLATE.format(
        section_title=segment.section_title,
        paragraph=segment.paragraph[:800],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        raw      = _call_openrouter(model_name, messages)
        parsed   = _parse_json_response(raw)
        reponses = _normalize_reponses(parsed)
        return ModelPrediction(model_name=model_name, reponses_questions=reponses)
    except Exception as e:
        return ModelPrediction(
            model_name=model_name,
            reponses_questions={k: "non" for k in QUESTION_KEYS},
            erreur=str(e),
        )


# ── Agrégation ────────────────────────────────────────────────────────────────
def aggregate_responses(predictions: list[ModelPrediction]) -> dict:
    valid = [p for p in predictions if not p.erreur]
    n = len(valid)
    if n == 0:
        return {k: "non" for k in QUESTION_KEYS}
    return {
        k: ("oui" if sum(1 for p in valid if p.reponses_questions.get(k) == "oui") > n / 2 else "non")
        for k in QUESTION_KEYS
    }


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi  = sum(1 for k in ["roi_1","roi_2","roi_3"] if reponses.get(k) == "oui")
    not_ = sum(1 for k in ["not_1","not_2","not_3"] if reponses.get(k) == "oui")
    obl  = sum(1 for k in ["obl_1","obl_2","obl_3"] if reponses.get(k) == "oui")
    return roi, not_, obl


def determine_label(roi: int, not_: int, obl: int) -> str:
    if roi == 0 and not_ == 0 and obl == 0 :
        return "Description"
    scores = {"ROI": roi, "Obligation": obl, "Notoriété": not_}
    max_score = max(scores.values())
    tied = [c for c, s in scores.items() if s == max_score]
    for priority in PRIORITY_ORDER:
        if priority in tied:
            return priority
    return tied[0]


# ── Export d'un fichier (JSON + Excel) ────────────────────────────────────────
def export_file_results(results: list[FinalClassification], dest_dir: Path,
                        base_name: str, models: list):
    """
    Sauvegarde JSON + Excel dans dest_dir (= dossier du fichier source).
    Nom de fichier : <base_name>_classification.json / .xlsx
    """
    json_out = dest_dir / f"{base_name}_classification.json"
    xl_out   = dest_dir / f"{base_name}_classification.xlsx"

    # JSON
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Excel
    rows = []
    for r in results:
        row = {
            "Source_File":      r.source_file,
            "Section_Title":    r.section_title,
            "Paragraph":        r.paragraph[:500],
            "Category":         r.categorie,
            "ROI_Score":        r.scores_roi,
            "Notoriété_Score":  r.scores_notoriete,
            "Obligation_Score": r.scores_obligation,
        }
        for qk, qv in r.reponses_questions.items():
            row[f"Q_{qk}"] = qv
        for pred_d in r.predictions_par_modele:
            short = pred_d["model_name"].split("/")[-1][:25]
            rq    = pred_d["reponses_questions"]
            row[f"resp_{short}"] = (
                f"R:{rq['roi_1'][0]}{rq['roi_2'][0]}{rq['roi_3'][0]} "
                f"N:{rq['not_1'][0]}{rq['not_2'][0]}{rq['not_3'][0]} "
                f"O:{rq['obl_1'][0]}{rq['obl_2'][0]}{rq['obl_3'][0]}"
            )
            row[f"err_{short}"] = pred_d.get("erreur") or ""
        rows.append(row)

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(xl_out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classification", index=False)

    print(f"      💾 Sauvegardé → {json_out.name}")
    print(f"      💾 Sauvegardé → {xl_out.name}")
    return json_out, xl_out


# ── Traitement d'un fichier ────────────────────────────────────────────────────
def process_segments_file(seg_path: Path, models: list) -> list[FinalClassification]:
    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = [Segment(**s) for s in data]

    print(f"\n   Fichier : {seg_path.name}  ({len(segments)} segments)")
    results = []

    for idx, seg in enumerate(segments):
        print(f"\n   [{idx+1}/{len(segments)}] '{seg.paragraph[:60]}...'")

        preds = []
        for model in models:
            pred  = classify_with_model(seg, model)
            preds.append(pred)
            short = model.split("/")[-1][:30]
            if pred.erreur:
                print(f"      {short}: [ERR] {pred.erreur[:60]}")
            else:
                rq = pred.reponses_questions
                print(f"      {short}: "
                      f"ROI:{rq['roi_1'][0]}{rq['roi_2'][0]}{rq['roi_3'][0]} "
                      f"NOT:{rq['not_1'][0]}{rq['not_2'][0]}{rq['not_3'][0]} "
                      f"OBL:{rq['obl_1'][0]}{rq['obl_2'][0]}{rq['obl_3'][0]}")
            time.sleep(INTER_MODEL_DELAY)

        agg_resp             = aggregate_responses(preds)
        roi_s, not_s, obl_s  = compute_scores(agg_resp)
        label                = determine_label(roi_s, not_s, obl_s)

        n_valides = sum(1 for p in preds if not p.erreur)
        n_erreurs = len(preds) - n_valides
        err_flag  = f"  {n_erreurs} modèle(s) en erreur" if n_erreurs else ""
        print(f"      → FINAL: {label}  (ROI:{roi_s} NOT:{not_s} OBL:{obl_s} | {n_valides}/5 votes{err_flag})")

        results.append(FinalClassification(
            section_title=seg.section_title,
            paragraph=seg.paragraph,
            paragraph_index=seg.paragraph_index,
            source_file=seg.source_file,
            source_folder=seg.source_folder,
            categorie=label,
            scores_roi=roi_s,
            scores_notoriete=not_s,
            scores_obligation=obl_s,
            reponses_questions=agg_resp,
            predictions_par_modele=[asdict(p) for p in preds],
            ensemble_modeles=models,
        ))

    return results


# ── Export global (synthèse de tous les fichiers) ─────────────────────────────
def export_global_summary(all_results: list[FinalClassification],
                          models: list, timestamp: str):
    """Exporte une synthèse globale dans OUTPUT_DIR."""
    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    rows = []
    for r in all_results:
        rows.append({
            "Source_Folder":    r.source_folder,
            "Source_File":      r.source_file,
            "Paragraph_Index":  r.paragraph_index,
            "Section_Title":    r.section_title,
            "Category":         r.categorie,
            "ROI_Score":        r.scores_roi,
            "Notoriété_Score":  r.scores_notoriete,
            "Obligation_Score": r.scores_obligation,
        })
    df = pd.DataFrame(rows)

    # Synthèse par fichier
    summary_rows = []
    for key, grp in df.groupby("Source_File"):
        t = len(grp)
        summary_rows.append({
            "Fichier":     key,
            "Total":       t,
            "ROI":         (grp.Category == "ROI").sum(),
            "Notoriété":   (grp.Category == "Notoriété").sum(),
            "Obligation":  (grp.Category == "Obligation").sum(),
            "Description": (grp.Category == "Description").sum(),
        })
    df_summary = pd.DataFrame(summary_rows)

    df_global = pd.DataFrame([{
        "Timestamp":      timestamp,
        "Priorité":       " > ".join(PRIORITY_ORDER),
        "Modèles":        " | ".join(m.split("/")[-1] for m in models),
        "Total_Segments": total,
        **{f"Total_{c}": cats[c] for c in CATEGORIES},
        **{f"Pct_{c}": f"{round(cats[c]/total*100,1)}%" if total else "0%" for c in CATEGORIES},
    }])

    xl_path = Path(OUTPUT_DIR) / f"synthese_globale_{timestamp}.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        df.to_excel(writer,         sheet_name="Tous_Segments",        index=False)
        df_summary.to_excel(writer, sheet_name="Synthese_Par_Fichier", index=False)
        df_global.to_excel(writer,  sheet_name="Statistiques_Globales",index=False)

    print(f"\n   Synthèse globale → {xl_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(models: list = None):
    if models is None:
        models = MODELS_TO_USE

    print("=" * 80)
    print("CLASSIFICATION MULTI-LLM — Export par fichier (v3)")
    print("=" * 80)
    for m in models:
        print(f"  • {m}")
    print(f"\n  Règle de priorité : {' > '.join(PRIORITY_ORDER)}")
    print(f"  Délai inter-appels : {INTER_MODEL_DELAY}s")
    print()

    seg_root  = Path(SEGMENTS_INPUT_DIR)
    seg_files = sorted(seg_root.rglob("*_segments.json"))
    if not seg_files:
        print("[ERREUR] Aucun *_segments.json trouvé")
        return

    print(f"  {len(seg_files)} fichier(s) de segments trouvés\n")

    all_results = []
    ok, skipped, err = 0, 0, 0

    for i, seg_path in enumerate(seg_files, 1):
        rel       = seg_path.relative_to(seg_root)
        dest_dir  = seg_path.parent                        # même dossier que le JSON source
        base_name = seg_path.stem                          # ex: "1km_segments"
        json_out  = dest_dir / f"{base_name}_classification.json"

        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")

        # ── SKIP si déjà traité ───────────────────────────────────────────────
        if json_out.exists():
            print(f"   Déjà traité ({json_out.name}), chargement et skip.")
            with open(json_out, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Recharge les résultats existants pour la synthèse globale
            for item in cached:
                all_results.append(FinalClassification(**{
                    k: item[k] for k in FinalClassification.__dataclass_fields__
                    if k in item
                }))
            skipped += 1
            continue

        # ── Traitement ────────────────────────────────────────────────────────
        try:
            results = process_segments_file(seg_path, models)

            # Export immédiat dans le dossier source
            export_file_results(results, dest_dir, base_name, models)

            all_results.extend(results)
            ok += 1

        except Exception as e:
            print(f"   ERREUR : {e}")
            err += 1
            continue   # on passe au fichier suivant sans bloquer

    # ── Synthèse globale ──────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    if all_results:
        export_global_summary(all_results, models, ts)

    # ── Résumé console ────────────────────────────────────────────────────────
    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"  Traités : {ok}  |  Skippés : {skipped}  |  Erreurs : {err}  |  Total fichiers : {len(seg_files)}")
    print(f"  Segments total : {total}")
    for c in CATEGORIES:
        pct = round(cats[c]/total*100, 1) if total else 0
        print(f"  {c:<12} : {cats[c]:>4}  ({pct}%)")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        import traceback
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()