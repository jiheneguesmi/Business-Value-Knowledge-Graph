"""
Script 2 : Classification Multi-LLM avec vote pondéré — v4 (Groq)
- 3 modèles gratuits disponibles (vérifiés avril 2026)
- Gestion des rate limits (backoff exponentiel)
- Mode square (confiance²)
"""

import os
import re
import json
import time
import random
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments\Idex"
OUTPUT_DIR         = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clé API Groq (depuis variable d'environnement)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie. "
                     "Créez un fichier .env avec GROQ_API_KEY=votre_clé")
client = Groq(api_key=GROQ_API_KEY)

# ── 3 MODÈLES GRATUITS ACTIFS SUR GROQ (avril 2026) ───────────────────────────
MODELS_TO_USE = [
    "llama-3.3-70b-versatile",      # Meta Llama 3.3 70B – recommandé
    "llama-3.1-8b-instant",         # Meta Llama 3.1 8B – rapide
    "llama-3.1-70b-versatile",      # Meta Llama 3.1 70B – fallback robuste
]

# ── Mode de pondération ────────────────────────────────────────────────────────
VOTE_MODE            = "square"
CONFIDENCE_THRESHOLD = 0.65

MAX_TOKENS     = 512
BASE_DELAY     = 2.0       # secondes entre les appels à différents modèles
MAX_RETRIES    = 3
BACKOFF_FACTOR = 2

CATEGORIES = ["ROI", "Notoriété", "Obligation", "Description"]

# ── Prompt unique ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un expert en analyse de documents commerciaux. "
    "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON."
)

PROMPT_UNIQUE = """Section parente : "{section_title}"

Paragraphe :
\"\"\"{paragraph}\"\"\"

Questions (oui/non) :
[ROI-1] Gain financier, réduction de coût, rentabilité mesurable ?
[ROI-2] Amélioration fonctionnelle (temps, charge, automatisation) comme avantage opérationnel ?
[ROI-3] Impact sur résultats, ressources ou performance d'une organisation ?
[NOT-1] Amélioration bien-être, confort, qualité de vie d'un usager ?
[NOT-2] Label, reconnaissance, attractivité, image positive ?
[NOT-3] Impact visible ou perçu positivement dans l'environnement ou l'expérience utilisateur ?
[OBL-1] Nécessité de respecter une norme, loi ou exigence réglementaire ?
[OBL-2] Mesure de sécurité ou prévention des risques ?
[OBL-3] Action nécessaire pour éviter danger, sanction ou garantir protection minimale ?

Réponds avec ce JSON (sans markdown) :
{{"categorie":"<ROI|Notoriété|Obligation|Description>","confiance":<0.0-1.0>,"justification":"<phrase>","reponses_questions":{{"roi_1":"<oui|non>","roi_2":"<oui|non>","roi_3":"<oui|non>","not_1":"<oui|non>","not_2":"<oui|non>","not_3":"<oui|non>","obl_1":"<oui|non>","obl_2":"<oui|non>","obl_3":"<oui|non>"}}}}"""


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
    categorie:          str
    confiance:          float
    justification:      str
    reponses_questions: dict
    poids:              float = 0.0
    erreur:             str   = None


@dataclass
class FinalClassification:
    section_title:          str
    paragraph:              str
    paragraph_index:        int
    source_file:            str
    source_folder:          str
    categorie:              str
    confiance:              float
    scores_roi:             int
    scores_notoriete:       int
    scores_obligation:      int
    vote_mode:              str
    reponses_questions:     dict
    votes_ponderes:         dict
    predictions_par_modele: list


# ── Appel API Groq avec retry et backoff ───────────────────────────────────────

def _call_groq_with_retry(model_name: str, messages: list) -> str:
    """Appelle Groq avec gestion des rate limits et des erreurs transitoires."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Détection des rate limits (429) ou modèles décommissionnés (400)
            if "rate limit" in error_str or "429" in error_str:
                wait_time = BASE_DELAY * (BACKOFF_FACTOR ** attempt) + random.uniform(0, 1)
                print(f"      ⚠️ Rate limit ({model_name}), attente {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            elif "decommissioned" in error_str or "400" in error_str:
                # Modèle indisponible → on le remplace temporairement par un fallback
                print(f"      ⚠️ Modèle {model_name} décommissionné. Utilisation de llama-3.1-70b-versatile comme fallback.")
                # Remplacer le modèle pour cette prédiction seulement
                fallback_model = "llama-3.1-70b-versatile"
                try:
                    response = client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=0,
                        max_tokens=MAX_TOKENS,
                    )
                    return response.choices[0].message.content.strip()
                except:
                    raise e
            elif attempt < MAX_RETRIES:
                wait_time = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                print(f"      ⚠️ Erreur {model_name}: {e}, nouvel essai dans {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise
    raise Exception(f"Échec après {MAX_RETRIES} tentatives pour {model_name}")


def _parse_json_response(raw: str) -> dict:
    """Parse robuste du JSON (supprime les backticks markdown)."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Impossible de parser : {raw[:200]}")


def _normalize_prediction(parsed: dict) -> tuple[str, float, str, dict]:
    categorie = parsed.get("categorie", "Description")
    if categorie not in CATEGORIES:
        mapping = {"ROI": "ROI", "NOTORIETE": "Notoriété", "NOTORIÉTÉ": "Notoriété",
                   "OBLIGATION": "Obligation", "DESCRIPTION": "Description"}
        categorie = mapping.get(str(categorie).upper(), "Description")
    raw_conf = parsed.get("confiance", 0.5)
    try:
        confiance = max(0.0, min(1.0, float(raw_conf)))
    except (TypeError, ValueError):
        confiance = 0.5
    justification = str(parsed.get("justification", ""))
    rq_raw = parsed.get("reponses_questions", {})
    question_keys = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]
    reponses = {}
    for k in question_keys:
        val = str(rq_raw.get(k, "non")).lower().strip()
        reponses[k] = "oui" if val in ("oui","yes","true","1") else "non"
    return categorie, confiance, justification, reponses


def classify_with_model(segment: Segment, model_name: str) -> ModelPrediction:
    prompt = PROMPT_UNIQUE.format(
        section_title=segment.section_title,
        paragraph=segment.paragraph[:800],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        raw = _call_groq_with_retry(model_name, messages)
        parsed = _parse_json_response(raw)
        cat, conf, just, rep = _normalize_prediction(parsed)
        return ModelPrediction(
            model_name=model_name,
            categorie=cat,
            confiance=conf,
            justification=just,
            reponses_questions=rep,
        )
    except Exception as e:
        return ModelPrediction(
            model_name=model_name,
            categorie="Description",
            confiance=0.0,
            justification="",
            reponses_questions={k:"non" for k in ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]},
            erreur=str(e),
        )


# ── Pondération et vote ────────────────────────────────────────────────────────

def compute_weight(confiance: float, mode: str) -> float:
    if mode == "square":
        return confiance ** 2
    elif mode == "threshold":
        return confiance if confiance >= CONFIDENCE_THRESHOLD else 0.0
    return confiance


def weighted_majority_vote(predictions: list[ModelPrediction], mode: str) -> tuple:
    scores = defaultdict(float)
    for pred in predictions:
        if pred.erreur:
            pred.poids = 0.0
            continue
        w = compute_weight(pred.confiance, mode)
        pred.poids = round(w, 4)
        if w > 0:
            scores[pred.categorie] += w
    if not scores:
        return "Description", 0.0, {}
    winner = max(scores, key=scores.get)
    total_w = sum(scores.values())
    conf_fin = round(scores[winner] / total_w, 3) if total_w > 0 else 0.0
    return winner, conf_fin, dict(scores)


def aggregate_responses_majority(predictions: list[ModelPrediction]) -> dict:
    question_keys = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]
    aggregated = {}
    for qk in question_keys:
        oui_count = sum(1 for p in predictions if not p.erreur and p.reponses_questions.get(qk) == "oui")
        aggregated[qk] = "oui" if oui_count > len(predictions) / 2 else "non"
    return aggregated


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi = sum(1 for k in ["roi_1","roi_2","roi_3"] if reponses.get(k) == "oui")
    not_ = sum(1 for k in ["not_1","not_2","not_3"] if reponses.get(k) == "oui")
    obl = sum(1 for k in ["obl_1","obl_2","obl_3"] if reponses.get(k) == "oui")
    return roi, not_, obl


# ── Traitement des fichiers ────────────────────────────────────────────────────

def process_segments_file(seg_path: Path) -> list[FinalClassification]:
    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = [Segment(**s) for s in data]

    print(f"\n   Fichier : {seg_path.name}")
    print(f"   Dossier : {seg_path.parent}")
    print(f"   {len(segments)} segments")

    results = []
    for idx, seg in enumerate(segments):
        print(f"\n   Segment {idx+1}/{len(segments)} : '{seg.paragraph[:50]}'")

        preds = []
        for model in MODELS_TO_USE:
            short = model.split("/")[-1]
            print(f"       {short}...", end=" ", flush=True)
            pred = classify_with_model(seg, model)
            preds.append(pred)
            err_tag = f" [ERR:{pred.erreur[:30]}]" if pred.erreur else ""
            print(f"{pred.categorie} (conf:{pred.confiance:.2f}){err_tag}")
            time.sleep(BASE_DELAY)

        winner, conf_fin, votes_pond = weighted_majority_vote(preds, VOTE_MODE)
        agg_resp = aggregate_responses_majority(preds)
        roi_s, not_s, obl_s = compute_scores(agg_resp)

        roi_v = sum(1 for p in preds if p.categorie == "ROI" and not p.erreur)
        not_v = sum(1 for p in preds if p.categorie == "Notoriété" and not p.erreur)
        obl_v = sum(1 for p in preds if p.categorie == "Obligation" and not p.erreur)
        print(f"       VOTE [{VOTE_MODE}]: {winner} (conf:{conf_fin:.2f}) | ROI:{roi_v} NOT:{not_v} OBL:{obl_v}")

        results.append(FinalClassification(
            section_title=seg.section_title,
            paragraph=seg.paragraph,
            paragraph_index=seg.paragraph_index,
            source_file=seg.source_file,
            source_folder=seg.source_folder,
            categorie=winner,
            confiance=conf_fin,
            scores_roi=roi_s,
            scores_notoriete=not_s,
            scores_obligation=obl_s,
            vote_mode=VOTE_MODE,
            reponses_questions=agg_resp,
            votes_ponderes=votes_pond,
            predictions_par_modele=[asdict(p) for p in preds],
        ))
    return results


# ── Export Excel ───────────────────────────────────────────────────────────────

def export_to_excel(all_results: list[FinalClassification], out_path: Path, global_stats: dict):
    rows = []
    for r in all_results:
        parts = Path(r.source_folder).parts
        row = {
            "Client":          parts[0] if parts else "",
            "Dossier_PDF":     parts[1] if len(parts) > 1 else "",
            "Source_Folder":   r.source_folder,
            "Source_File":     r.source_file,
            "Section_Title":   r.section_title,
            "Paragraph":       r.paragraph[:500],
            "Category":        r.categorie,
            "Confidence":      round(r.confiance, 3),
            "Vote_Mode":       r.vote_mode,
            "ROI_Score":       r.scores_roi,
            "Notoriété_Score": r.scores_notoriete,
            "Obligation_Score":r.scores_obligation,
            "Votes_Pondérés":  json.dumps(r.votes_ponderes, ensure_ascii=False),
        }
        for qk, qv in r.reponses_questions.items():
            row[f"Q_{qk}"] = qv
        for pred_d in r.predictions_par_modele:
            short = pred_d["model_name"].split("/")[-1]
            row[f"pred_{short}"]   = pred_d["categorie"]
            row[f"conf_{short}"]   = round(pred_d["confiance"], 2)
            row[f"poids_{short}"]  = round(pred_d.get("poids", 0.0), 4)
            row[f"erreur_{short}"] = pred_d.get("erreur") or ""
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_rows = []
    for key, grp in df.groupby("Source_File"):
        total = len(grp)
        summary_rows.append({
            "Fichier":      key,
            "Total":        total,
            "ROI":          (grp.Category == "ROI").sum(),
            "Notoriété":    (grp.Category == "Notoriété").sum(),
            "Obligation":   (grp.Category == "Obligation").sum(),
            "Description":  (grp.Category == "Description").sum(),
            "%_ROI":        round((grp.Category=="ROI").sum()/total*100, 1),
            "%_Notoriété":  round((grp.Category=="Notoriété").sum()/total*100, 1),
            "%_Obligation": round((grp.Category=="Obligation").sum()/total*100, 1),
            "Conf_Moy":     round(grp.Confidence.mean(), 3),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_global = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in global_stats.items()}])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Details_Par_Segment", index=False)
        df_summary.to_excel(writer, sheet_name="Synthese_Par_Fichier", index=False)
        df_global.to_excel(writer, sheet_name="Statistiques_Globales", index=False)
    return df, df_summary


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 80)
    print("CLASSIFICATION MULTI-LLM (Groq – 3 modèles gratuits)")
    print("=" * 80)
    print(f"  Modèles : {MODELS_TO_USE}")
    print(f"  Mode vote : {VOTE_MODE} (poids=confiance²)")
    print(f"  Délai inter-appels : {BASE_DELAY}s")
    print()

    seg_root = Path(SEGMENTS_INPUT_DIR)
    if not seg_root.exists():
        print(f"[ERREUR] Dossier introuvable : {SEGMENTS_INPUT_DIR}")
        return
    seg_files = sorted(seg_root.rglob("*_segments.json"))
    if not seg_files:
        print("[ERREUR] Aucun *_segments.json trouvé")
        return
    print(f"  {len(seg_files)} fichier(s) de segments\n")

    all_results = []
    ok_count = 0
    err_count = 0

    for i, seg_path in enumerate(seg_files, 1):
        rel = seg_path.relative_to(seg_root)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")
        try:
            res = process_segments_file(seg_path)
            all_results.extend(res)
            ok_count += 1
        except Exception as e:
            print(f"  ERREUR : {e}")
            err_count += 1

    total = len(all_results)
    cats = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}
    global_stats = {
        "fichiers_traites": ok_count,
        "fichiers_erreurs": err_count,
        "total_segments":   total,
        "vote_mode":        VOTE_MODE,
        **{f"total_{c}": cats[c] for c in CATEGORIES},
        **{f"pct_{c}": round(cats[c]/total*100, 1) if total else 0 for c in CATEGORIES},
        "modeles": MODELS_TO_USE,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    xl_path   = Path(OUTPUT_DIR) / f"classification_{VOTE_MODE}_{ts}.xlsx"
    json_path = Path(OUTPUT_DIR) / f"classification_{VOTE_MODE}_{ts}.json"
    export_to_excel(all_results, xl_path, global_stats)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"  Fichiers traités : {ok_count}/{len(seg_files)}")
    print(f"  Total segments   : {total}")
    print(f"  Mode vote        : {VOTE_MODE}")
    for c in CATEGORIES:
        print(f"  {c:<12} : {cats[c]:>4}  ({global_stats[f'pct_{c}']}%)")
    print(f"\n  Excel : {xl_path}")
    print(f"  JSON  : {json_path}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        print(f"\nErreur fatale : {e}")
        import traceback
        traceback.print_exc()