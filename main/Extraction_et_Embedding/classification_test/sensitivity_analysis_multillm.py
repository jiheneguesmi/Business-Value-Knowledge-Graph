"""
Script : Analyse de sensibilité Multi-LLM
- 5 modèles via OpenRouter
- Vote pondéré (confiance²) sur les 5 modèles → label de référence
- Analyse : si on retire/change 1, 2, 3, 4 modèles → le label change-t-il ?
- Export : dataset labellisé final + rapport de stabilité complet
"""

import os
import re
import json
import time
import itertools
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
import requests
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments"
OUTPUT_DIR         = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_API_KEY = "sk-or-v1-VOTRE_CLE"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── 5 modèles ──────────────────────────────────────────────────────────────────
ALL_MODELS = [
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen2.5-72b-instruct",
    "mistralai/mistral-large-2411",
]

CATEGORIES         = ["ROI", "Notoriété", "Obligation", "Description"]
MAX_TOKENS         = 512
DELAY              = 0.4
MAX_RETRIES        = 2

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un expert en analyse de documents commerciaux. "
    "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON."
)

PROMPT_STANDARD = """Section parente : "{section_title}"

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

PROMPT_QWEN = """Classifie ce paragraphe commercial en une seule catégorie.

Section : {section_title}
Texte : {paragraph}

Catégories :
- ROI : gain financier, réduction coût, performance, automatisation
- Notoriété : image, bien-être, attractivité, label, expérience positive
- Obligation : norme, loi, sécurité, risque, conformité réglementaire
- Description : information neutre sans valeur commerciale claire

Réponds uniquement avec ce JSON (pas de markdown) :
{{"categorie":"ROI|Notoriété|Obligation|Description","confiance":0.0,"justification":"phrase","reponses_questions":{{"roi_1":"oui|non","roi_2":"oui|non","roi_3":"oui|non","not_1":"oui|non","not_2":"oui|non","not_3":"oui|non","obl_1":"oui|non","obl_2":"oui|non","obl_3":"oui|non"}}}}"""


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
    erreur:             str = None


# ── API ────────────────────────────────────────────────────────────────────────

def _call_api(model_name: str, messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    data = {
        "model":       model_name,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"JSON invalide : {raw[:150]}")


def _normalize(parsed: dict) -> tuple:
    cat = parsed.get("categorie", "Description")
    mapping = {
        "ROI": "ROI", "NOTORIETE": "Notoriété", "NOTORIÉTÉ": "Notoriété",
        "OBLIGATION": "Obligation", "DESCRIPTION": "Description",
    }
    if cat not in CATEGORIES:
        cat = mapping.get(str(cat).upper(), "Description")

    try:
        conf = float(parsed.get("confiance", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5

    justif = str(parsed.get("justification", ""))
    rq_raw = parsed.get("reponses_questions", {})
    keys   = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]
    rq     = {k: ("oui" if str(rq_raw.get(k,"non")).lower() in ("oui","yes","true","1") else "non")
              for k in keys}
    return cat, conf, justif, rq


def classify_with_model(seg: Segment, model_name: str) -> ModelPrediction:
    is_qwen = "qwen" in model_name.lower()
    prompt  = (PROMPT_QWEN if is_qwen else PROMPT_STANDARD).format(
        section_title=seg.section_title,
        paragraph=seg.paragraph[:800],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = _call_api(model_name, messages)
            parsed = _parse_json(raw)
            cat, conf, justif, rq = _normalize(parsed)
            return ModelPrediction(model_name=model_name, categorie=cat,
                                   confiance=conf, justification=justif,
                                   reponses_questions=rq)
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(1)

    blank_rq = {k: "non" for k in ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]}
    return ModelPrediction(model_name=model_name, categorie="Description",
                           confiance=0.0, justification="", reponses_questions=blank_rq,
                           erreur=last_err)


# ── Vote pondéré (confiance²) ──────────────────────────────────────────────────

def vote_square(preds: list[ModelPrediction]) -> tuple[str, float, dict]:
    """Retourne (catégorie_gagnante, confiance_normalisée, scores_par_cat)."""
    scores = defaultdict(float)
    for p in preds:
        if p.erreur:
            continue
        scores[p.categorie] += p.confiance ** 2

    if not scores:
        return "Description", 0.0, {}

    total  = sum(scores.values())
    winner = max(scores, key=scores.get)
    conf   = round(scores[winner] / total, 3) if total > 0 else 0.0
    return winner, conf, dict(scores)


# ── Analyse de sensibilité ─────────────────────────────────────────────────────

def sensitivity_analysis(
    preds: list[ModelPrediction],
    reference_label: str,
) -> dict:
    """
    Pour chaque sous-ensemble possible de modèles (taille 1 à N-1),
    calcule le label et vérifie s'il diffère du label de référence (N modèles).

    Retourne un dict :
      {
        "nb_models_removed": {
          1: {"nb_combinations": X, "nb_changes": Y, "pct_change": Z,
              "details": [{"models_kept": [...], "label": "...", "changed": bool}]},
          2: {...},
          ...
        }
      }
    """
    n      = len(preds)
    result = {}

    for nb_removed in range(1, n):
        nb_kept      = n - nb_removed
        combos       = list(itertools.combinations(preds, nb_kept))
        nb_changes   = 0
        details      = []

        for combo in combos:
            combo_preds = list(combo)
            lbl, _, _   = vote_square(combo_preds)
            changed     = lbl != reference_label
            if changed:
                nb_changes += 1
            details.append({
                "models_kept":  [p.model_name.split("/")[-1] for p in combo_preds],
                "label":         lbl,
                "changed":       changed,
            })

        result[nb_removed] = {
            "nb_combinations": len(combos),
            "nb_changes":      nb_changes,
            "pct_change":      round(nb_changes / len(combos) * 100, 1) if combos else 0.0,
            "details":         details,
        }

    return result


# ── Traitement d'un fichier ────────────────────────────────────────────────────

def process_file(seg_path: Path) -> list[dict]:
    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = [Segment(**s) for s in data]

    print(f"\n   {seg_path.name}  —  {len(segments)} segments")
    records = []

    for idx, seg in enumerate(segments):
        print(f"\n   [{idx+1}/{len(segments)}] {seg.paragraph[:55]}...")

        # ── Appels aux 5 modèles ──────────────────────────────────────────
        preds = []
        for model in ALL_MODELS:
            short = model.split("/")[-1]
            print(f"       {short}...", end=" ", flush=True)
            pred = classify_with_model(seg, model)
            preds.append(pred)
            err_tag = f" ERR" if pred.erreur else ""
            print(f"{pred.categorie} ({pred.confiance:.2f}){err_tag}")
            time.sleep(DELAY)

        # ── Label de référence (5 modèles) ────────────────────────────────
        ref_label, ref_conf, ref_scores = vote_square(preds)

        # ── Analyse de sensibilité ────────────────────────────────────────
        sensitivity = sensitivity_analysis(preds, ref_label)

        # ── Agrégation des réponses aux questions ─────────────────────────
        keys = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]
        agg_rq = {}
        for k in keys:
            oui = sum(1 for p in preds if not p.erreur and p.reponses_questions.get(k) == "oui")
            agg_rq[k] = "oui" if oui > len([p for p in preds if not p.erreur]) / 2 else "non"

        roi_s  = sum(1 for k in ["roi_1","roi_2","roi_3"] if agg_rq[k] == "oui")
        not_s  = sum(1 for k in ["not_1","not_2","not_3"] if agg_rq[k] == "oui")
        obl_s  = sum(1 for k in ["obl_1","obl_2","obl_3"] if agg_rq[k] == "oui")

        # Stabilité globale du segment : % moyen de changement sur toutes tailles
        all_pcts   = [v["pct_change"] for v in sensitivity.values()]
        stab_score = round(100 - (sum(all_pcts) / len(all_pcts)), 1) if all_pcts else 100.0

        print(f"       RÉFÉRENCE : {ref_label} ({ref_conf:.2f}) | stabilité={stab_score}%")

        records.append({
            # ── Identifiant ──────────────────────────────────────────────
            "source_folder":    seg.source_folder,
            "source_file":      seg.source_file,
            "paragraph_index":  seg.paragraph_index,
            "section_title":    seg.section_title,
            "paragraph":        seg.paragraph,
            # ── Label final ──────────────────────────────────────────────
            "label":            ref_label,
            "confidence":       ref_conf,
            "scores":           ref_scores,
            "roi_score":        roi_s,
            "not_score":        not_s,
            "obl_score":        obl_s,
            "reponses_questions": agg_rq,
            # ── Prédictions par modèle ────────────────────────────────────
            "predictions": {
                p.model_name.split("/")[-1]: {
                    "categorie": p.categorie,
                    "confiance": p.confiance,
                    "erreur":    p.erreur,
                }
                for p in preds
            },
            # ── Sensibilité ───────────────────────────────────────────────
            "sensitivity":      sensitivity,
            "stability_score":  stab_score,
        })

    return records


# ── Rapport de stabilité global ────────────────────────────────────────────────

def build_stability_report(all_records: list[dict]) -> dict:
    """
    Agrège la sensibilité sur tout le corpus :
    pour chaque nombre de modèles retirés (1, 2, 3, 4),
    calcule le % de segments dont le label change.
    """
    n_total = len(all_records)
    if n_total == 0:
        return {}

    report = {}
    max_removed = len(ALL_MODELS) - 1   # 4

    for nb_removed in range(1, max_removed + 1):
        segments_with_change = 0
        combo_details        = defaultdict(lambda: {"changes": 0, "total": 0})

        for rec in all_records:
            sens = rec.get("sensitivity", {}).get(nb_removed, {})
            if not sens:
                continue
            if sens["nb_changes"] > 0:
                segments_with_change += 1
            # Détail par combinaison de modèles
            for detail in sens.get("details", []):
                key = " + ".join(sorted(detail["models_kept"]))
                combo_details[key]["total"]   += 1
                combo_details[key]["changes"] += int(detail["changed"])

        # Trier les combinaisons par taux de changement
        combo_ranked = sorted(
            [
                {
                    "models_kept": k,
                    "nb_changes":  v["changes"],
                    "total":       v["total"],
                    "pct_change":  round(v["changes"] / v["total"] * 100, 1) if v["total"] else 0,
                }
                for k, v in combo_details.items()
            ],
            key=lambda x: x["pct_change"],
            reverse=True,
        )

        report[f"remove_{nb_removed}_model(s)"] = {
            "segments_affected":     segments_with_change,
            "segments_total":        n_total,
            "pct_segments_affected": round(segments_with_change / n_total * 100, 1),
            "combinations_ranked":   combo_ranked[:20],   # top 20
        }

    return report


# ── Export Excel ───────────────────────────────────────────────────────────────

def export_excel(all_records: list[dict], report: dict, out_path: Path):

    # ── Feuille 1 : Dataset labellisé ────────────────────────────────────────
    rows_dataset = []
    for r in all_records:
        parts = Path(r["source_folder"]).parts
        row = {
            "client":           parts[0] if parts else "",
            "dossier":          parts[1] if len(parts) > 1 else "",
            "source_file":      r["source_file"],
            "paragraph_index":  r["paragraph_index"],
            "section_title":    r["section_title"],
            "paragraph":        r["paragraph"][:500],
            "label":            r["label"],
            "confidence":       r["confidence"],
            "stability_score":  r["stability_score"],
            "roi_score":        r["roi_score"],
            "not_score":        r["not_score"],
            "obl_score":        r["obl_score"],
        }
        # Prédictions par modèle
        for m_short, pred in r["predictions"].items():
            row[f"pred_{m_short}"] = pred["categorie"]
            row[f"conf_{m_short}"] = pred["confiance"]
        # Réponses questions agrégées
        for qk, qv in r["reponses_questions"].items():
            row[f"Q_{qk}"] = qv
        rows_dataset.append(row)

    df_dataset = pd.DataFrame(rows_dataset)

    # ── Feuille 2 : Synthèse par fichier ─────────────────────────────────────
    rows_synth = []
    for key, grp in df_dataset.groupby("source_file"):
        total = len(grp)
        rows_synth.append({
            "source_file":   key,
            "total":         total,
            "ROI":           (grp.label == "ROI").sum(),
            "Notoriété":     (grp.label == "Notoriété").sum(),
            "Obligation":    (grp.label == "Obligation").sum(),
            "Description":   (grp.label == "Description").sum(),
            "%_ROI":         round((grp.label=="ROI").sum()/total*100, 1),
            "%_Notoriété":   round((grp.label=="Notoriété").sum()/total*100, 1),
            "%_Obligation":  round((grp.label=="Obligation").sum()/total*100, 1),
            "conf_moy":      round(grp.confidence.mean(), 3),
            "stabilité_moy": round(grp.stability_score.mean(), 1),
        })
    df_synth = pd.DataFrame(rows_synth)

    # ── Feuille 3 : Rapport de sensibilité global ─────────────────────────────
    rows_report = []
    for scenario, data in report.items():
        rows_report.append({
            "scenario":               scenario,
            "segments_affectés":      data["segments_affected"],
            "total_segments":         data["segments_total"],
            "pct_segments_affectés":  data["pct_segments_affected"],
        })
    df_report = pd.DataFrame(rows_report)

    # ── Feuille 4 : Détail combinaisons (top instables) ───────────────────────
    rows_combo = []
    for scenario, data in report.items():
        for combo in data["combinations_ranked"]:
            rows_combo.append({
                "scenario":        scenario,
                "modèles_gardés":  combo["models_kept"],
                "nb_changements":  combo["nb_changes"],
                "total":           combo["total"],
                "pct_changement":  combo["pct_change"],
            })
    df_combo = pd.DataFrame(rows_combo)

    # ── Feuille 5 : Segments instables (stability < 80%) ─────────────────────
    df_unstable = df_dataset[df_dataset["stability_score"] < 80].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_dataset.to_excel(writer,  sheet_name="Dataset_Labellisé",        index=False)
        df_synth.to_excel(writer,    sheet_name="Synthèse_par_fichier",      index=False)
        df_report.to_excel(writer,   sheet_name="Rapport_Sensibilité",       index=False)
        df_combo.to_excel(writer,    sheet_name="Détail_Combinaisons",        index=False)
        df_unstable.to_excel(writer, sheet_name="Segments_Instables",        index=False)

    print(f"\n  Excel exporté : {out_path}")
    return df_dataset, df_synth, df_report


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 80)
    print("ANALYSE DE SENSIBILITÉ MULTI-LLM (5 modèles)")
    print("=" * 80)
    print(f"  Modèles : {[m.split('/')[-1] for m in ALL_MODELS]}")
    print(f"  Scénarios analysés : retrait de 1, 2, 3 ou 4 modèles\n")

    seg_root  = Path(SEGMENTS_INPUT_DIR)
    seg_files = sorted(seg_root.rglob("*_segments.json"))
    if not seg_files:
        print("[ERREUR] Aucun *_segments.json trouvé")
        return

    print(f"  {len(seg_files)} fichier(s) de segments\n")

    all_records = []
    ok, ko = 0, 0

    for i, seg_path in enumerate(seg_files, 1):
        rel = seg_path.relative_to(seg_root)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")
        try:
            records = process_file(seg_path)
            all_records.extend(records)
            ok += 1
        except Exception as e:
            print(f"  ERREUR : {e}")
            ko += 1

    # ── Rapport global ────────────────────────────────────────────────────────
    report = build_stability_report(all_records)

    # ── Export ────────────────────────────────────────────────────────────────
    ts        = time.strftime("%Y%m%d_%H%M%S")
    xl_path   = Path(OUTPUT_DIR) / f"sensitivity_analysis_{ts}.xlsx"
    json_path = Path(OUTPUT_DIR) / f"dataset_labellise_{ts}.json"

    df_dataset, df_synth, df_report = export_excel(all_records, report, xl_path)

    # JSON dataset propre (sans les détails de sensibilité)
    dataset_clean = [
        {
            "source_folder":      r["source_folder"],
            "source_file":        r["source_file"],
            "paragraph_index":    r["paragraph_index"],
            "section_title":      r["section_title"],
            "paragraph":          r["paragraph"],
            "label":              r["label"],
            "confidence":         r["confidence"],
            "stability_score":    r["stability_score"],
            "reponses_questions": r["reponses_questions"],
        }
        for r in all_records
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_clean, f, ensure_ascii=False, indent=2)

    # ── Résumé console ────────────────────────────────────────────────────────
    total = len(all_records)
    cats  = {c: sum(1 for r in all_records if r["label"] == c) for c in CATEGORIES}
    stab_moy = round(sum(r["stability_score"] for r in all_records) / total, 1) if total else 0

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"  Fichiers traités : {ok}/{len(seg_files)}")
    print(f"  Total segments   : {total}")
    print(f"  Stabilité moyenne: {stab_moy}%\n")

    print("  Distribution des labels (5 modèles, vote pondéré) :")
    for c in CATEGORIES:
        pct = round(cats[c]/total*100, 1) if total else 0
        print(f"    {c:<12} : {cats[c]:>4}  ({pct}%)")

    print("\n  Impact du retrait de modèles sur le corpus :")
    for scenario, data in report.items():
        print(f"    {scenario:<25} → {data['pct_segments_affected']}% des segments changent de label")

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