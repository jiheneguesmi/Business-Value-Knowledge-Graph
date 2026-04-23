"""
Script : Classification Multi-LLM — Par phrase (v6)
—————————————————————————————————————————————————————————————————
Changements vs v1 :
  - Le segment est uniquement un contexte pour le LLM
  - L'unité de sortie est la PHRASE (plus le segment)
  - 1 appel = 1 question posée sur TOUTES les phrases du paragraphe
  - Le LLM retourne {"phrase_1": "oui", "phrase_2": "non", ...}
  - Agrégation vote majoritaire par phrase × par question × 5 modèles
  - Tensor shape : (n_phrases_total, 5_modèles, 9_questions)
  - Export Excel/JSON : une ligne = une phrase
"""

import os
import re
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import requests

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments-phrases\1km\1km à Pied - Plaquette PDME"
OUTPUT_DIR         = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\resultats_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante dans le fichier .env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Modèles ────────────────────────────────────────────────────────────────────
MODELS_TO_USE = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-24b-instruct-2501",
    "qwen/qwen3-8b",
    "openai/gpt-4o-mini",
]

# ── Paramètres ────────────────────────────────────────────────────────────────
MAX_TOKENS        = 256     # plusieurs phrases à répondre
MAX_RETRIES       = 3
BACKOFF_BASE      = 3.0
INTER_CALL_DELAY  = 1.0

CATEGORIES     = ["ROI", "Notoriété", "Obligation", "Description"]
PRIORITY_ORDER = ["ROI", "Obligation", "Notoriété", "Description"]
MIN_SCORE      = 1

QUESTION_KEYS  = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]
MODEL_INDEX    = {m: i for i, m in enumerate(MODELS_TO_USE)}
QUESTION_INDEX = {q: i for i, q in enumerate(QUESTION_KEYS)}

# ── Variantes OUI ─────────────────────────────────────────────────────────────
OUI_VARIANTS = {
    "oui", "yes", "true", "1", "vrai", "si", "да", "ja",
    "oui.", "yes.", "true.", "oui!", "yes!",
}

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTIONS = {
    "roi_1": "Cette phrase parle-t-elle d'un gain financier, d'une réduction de coût ou d'une rentabilité mesurable ?",
    "roi_2": "Cette phrase décrit-elle une amélioration fonctionnelle claire (temps, charge, automatisation) comme avantage opérationnel ?",
    "roi_3": "Cette phrase mentionne-t-elle un impact sur les résultats, les ressources ou la performance d'une organisation ?",
    "not_1": "Cette phrase parle-t-elle d'une amélioration du bien-être, du confort, de la qualité de vie ou du cadre de travail d'un usager ?",
    "not_2": "Cette phrase mentionne-t-elle un label, une reconnaissance, une attractivité ou une image positive d'un service ou d'une organisation ?",
    "not_3": "Cette phrase décrit-elle un impact visible ou perçu positivement dans l'environnement, les usages ou l'expérience utilisateur ?",
    "obl_1": "Cette phrase parle-t-elle de la nécessité de respecter une norme, une loi ou une exigence réglementaire ?",
    "obl_2": "Cette phrase mentionne-t-elle une mesure de sécurité ou une prévention des risques (physiques, numériques, juridiques) ?",
    "obl_3": "Cette phrase décrit-elle une action nécessaire pour éviter un danger, une sanction ou garantir une protection minimale ?",
}

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un expert en analyse de documents commerciaux. "
    "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON. "
    "Pour chaque phrase, tu réponds uniquement par 'oui' ou 'non'."
)

PROMPT_TEMPLATE = """Section parente : "{section_title}"

Contexte complet du paragraphe :
\"\"\"{full_paragraph}\"\"\"

Question : {question}

Réponds pour chacune des phrases suivantes par 'oui' ou 'non' :
{phrases_list}

Réponds avec ce JSON (sans markdown) :
{expected_json}"""

SYSTEM_PROMPT_FALLBACK = (
    "Tu dois répondre à une question par 'oui' ou 'non' pour chaque phrase listée. "
    "Ta réponse doit être UNIQUEMENT un JSON valide du type {{\"phrase_1\": \"oui\", \"phrase_2\": \"non\"}}. "
    "Rien d'autre."
)

PROMPT_FALLBACK = """Question : {question}

Contexte : {full_paragraph}

Réponds uniquement par 'oui' ou 'non' pour chaque phrase :
{phrases_list}

JSON attendu :
{expected_json}"""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """Structure d'entrée — le segment lu depuis le JSON de segmentation."""
    section_title:   str
    paragraph:       dict   # {"phrase_1": "...", "phrase_2": "..."}
    paragraph_index: int
    source_file:     str
    source_folder:   str


@dataclass
class PhraseClassification:
    """Unité de sortie — une ligne par phrase dans l'export final."""
    phrase_key:             str     # "phrase_1", "phrase_2", ...
    phrase_text:            str
    paragraph_index:        int     # traçabilité vers le segment d'origine
    section_title:          str
    source_file:            str
    source_folder:          str
    categorie:              str
    scores_roi:             int
    scores_notoriete:       int
    scores_obligation:      int
    reponses_questions:     dict    # {roi_1: "oui"|"non", ...} — agrégé
    predictions_par_modele: list = field(default_factory=list)
    ensemble_modeles:       list = field(default_factory=list)
    n_fallbacks_total:      int  = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def reconstruct_full_paragraph(paragraph: dict) -> str:
    """Reconstitue le texte complet du paragraphe depuis le dict de phrases."""
    return " ".join(paragraph.values())


def build_phrases_list(paragraph: dict) -> str:
    """Formate la liste des phrases pour le prompt."""
    return "\n".join(f'- {key} : "{text}"' for key, text in paragraph.items())


def build_expected_json(paragraph: dict) -> str:
    """Construit le JSON attendu en exemple pour le LLM."""
    inner = ", ".join(f'"{k}": "<oui|non>"' for k in paragraph.keys())
    return "{" + inner + "}"


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
    raise ValueError("Réponse vide : ni 'content' ni 'reasoning'")


def _call_openrouter(model_name: str, messages: list, max_tokens: int = MAX_TOKENS) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jiheneguesmi/Business-Value-Knowledge-Graph",
        "X-Title": "Business Value Classification",
    }
    payload = {
        "model":       model_name,
        "messages":    messages,
        "temperature": 0,
        "max_tokens":  max_tokens,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                print(f"          Rate limit, attente {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code in (404, 400):
                err_detail = resp.json().get("error", {}).get("message", resp.text[:100])
                raise ValueError(f"Erreur API {resp.status_code}: {err_detail}")
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
                time.sleep(BACKOFF_BASE * attempt)
            else:
                raise
    raise Exception(f"Échec après {MAX_RETRIES} tentatives")


# ── Parsing de la réponse JSON du LLM ─────────────────────────────────────────

def _parse_phrase_response(raw: str, expected_keys: list[str]) -> tuple[dict, bool]:
    """
    Parse la réponse du LLM qui doit être un JSON {phrase_N: oui/non}.
    Retourne (dict_normalisé, is_ambiguous).
    is_ambiguous = True si au moins une phrase est ambiguë.
    """
    # Nettoyer les backticks markdown
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw).strip()

    parsed = {}
    is_ambiguous = False

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Tentative d'extraction du premier JSON trouvé
        matches = list(re.finditer(r'\{[^{}]*\}', raw, re.DOTALL))
        data = {}
        for match in sorted(matches, key=lambda m: len(m.group()), reverse=True):
            try:
                data = json.loads(match.group())
                break
            except json.JSONDecodeError:
                continue

    for key in expected_keys:
        val = str(data.get(key, "")).lower().strip().rstrip(".,!?;:")
        if val in OUI_VARIANTS or val.startswith("oui") or val.startswith("yes"):
            parsed[key] = "oui"
        elif val in ("non", "no", "false", "0", "faux") or val.startswith("non") or val.startswith("no "):
            parsed[key] = "non"
        else:
            # Valeur manquante ou ambiguë
            parsed[key] = "non"
            is_ambiguous = True

    return parsed, is_ambiguous


# ── Appel pour une question sur toutes les phrases ────────────────────────────

def ask_single_question(
    segment: Segment,
    model_name: str,
    question_key: str,
) -> dict:
    """
    Pose UNE question au modèle pour TOUTES les phrases du paragraphe.
    Retourne {phrase_key: "oui"|"non"} pour chaque phrase.
    """
    phrase_keys      = list(segment.paragraph.keys())
    full_paragraph   = reconstruct_full_paragraph(segment.paragraph)
    phrases_list_str = build_phrases_list(segment.paragraph)
    expected_json    = build_expected_json(segment.paragraph)
    question_text    = QUESTIONS[question_key]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            section_title=segment.section_title,
            full_paragraph=full_paragraph[:800],
            question=question_text,
            phrases_list=phrases_list_str,
            expected_json=expected_json,
        )},
    ]

    try:
        raw = _call_openrouter(model_name, messages)
        result, is_ambiguous = _parse_phrase_response(raw, phrase_keys)

        # Fallback si réponse ambiguë
        if is_ambiguous:
            fallback_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_FALLBACK},
                {"role": "user", "content": PROMPT_FALLBACK.format(
                    question=question_text,
                    full_paragraph=full_paragraph[:400],
                    phrases_list=phrases_list_str,
                    expected_json=expected_json,
                )},
            ]
            try:
                raw_fb = _call_openrouter(model_name, fallback_messages)
                result_fb, still_ambiguous = _parse_phrase_response(raw_fb, phrase_keys)
                if not still_ambiguous:
                    return result_fb
                # Toujours ambigu → garder "non" par défaut
                return result_fb
            except Exception:
                return result   # on garde ce qu'on a

        return result

    except Exception as e:
        # En cas d'erreur complète → tout à "non"
        print(f"          ERREUR {model_name.split('/')[-1]} / {question_key}: {e}")
        return {k: "non" for k in phrase_keys}


# ── Toutes les questions pour un modèle ───────────────────────────────────────

def classify_segment_with_model(segment: Segment, model_name: str) -> dict:
    """
    Pose les 9 questions une par une au modèle pour toutes les phrases.
    Retourne {question_key: {phrase_key: "oui"|"non"}}.
    """
    results = {}
    for qkey in QUESTION_KEYS:
        phrase_responses = ask_single_question(segment, model_name, qkey)
        results[qkey] = phrase_responses
        time.sleep(INTER_CALL_DELAY)
    return results


# ── Agrégation vote majoritaire par phrase × question ─────────────────────────

def aggregate_by_phrase(
    all_model_results: list[dict],
    phrase_keys: list[str],
) -> dict:
    """
    Vote majoritaire parmi les 5 modèles, par phrase et par question.
    Retourne {phrase_key: {question_key: "oui"|"non"}}.
    """
    n_models = len(all_model_results)
    aggregated = {}

    for pkey in phrase_keys:
        aggregated[pkey] = {}
        for qkey in QUESTION_KEYS:
            votes_oui = sum(
                1 for model_result in all_model_results
                if model_result.get(qkey, {}).get(pkey, "non") == "oui"
            )
            aggregated[pkey][qkey] = "oui" if votes_oui > n_models / 2 else "non"

    return aggregated


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi  = sum(1 for k in ["roi_1","roi_2","roi_3"] if reponses.get(k) == "oui")
    not_ = sum(1 for k in ["not_1","not_2","not_3"] if reponses.get(k) == "oui")
    obl  = sum(1 for k in ["obl_1","obl_2","obl_3"] if reponses.get(k) == "oui")
    return roi, not_, obl


def determine_label(roi: int, not_: int, obl: int) -> str:
    if roi < MIN_SCORE and not_ < MIN_SCORE and obl < MIN_SCORE:
        return "Description"
    scores = {"ROI": roi, "Obligation": obl, "Notoriété": not_}
    max_score = max(scores.values())
    tied = [c for c, s in scores.items() if s == max_score]
    for priority in PRIORITY_ORDER:
        if priority in tied:
            return priority
    return tied[0]


# ── Construction tensor pour un segment (shape: n_phrases × 5 × 9) ────────────

def build_segment_tensor(
    all_model_results: list[dict],
    phrase_keys: list[str],
    models: list[str],
) -> np.ndarray:
    """
    Construit un array shape (n_phrases, n_models, n_questions).
    Valeurs : 1=oui, 0=non.
    """
    n_phrases  = len(phrase_keys)
    n_models   = len(models)
    n_questions = len(QUESTION_KEYS)

    tensor = np.zeros((n_phrases, n_models, n_questions), dtype=np.int8)

    for mi, model_result in enumerate(all_model_results):
        for pi, pkey in enumerate(phrase_keys):
            for qi, qkey in enumerate(QUESTION_KEYS):
                val = model_result.get(qkey, {}).get(pkey, "non")
                tensor[pi, mi, qi] = 1 if val == "oui" else 0

    return tensor


# ── Traitement d'un fichier ────────────────────────────────────────────────────

def process_segments_file(
    seg_path: Path,
    models: list,
) -> tuple[list[PhraseClassification], np.ndarray]:
    """
    Retourne (liste de PhraseClassification, tensor shape (n_phrases_total, 5, 9)).
    """
    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = [Segment(**s) for s in data]
    n_segments = len(segments)
    n_phrases_total = sum(len(s.paragraph) for s in segments)

    print(f"\n   Fichier : {seg_path.name}")
    print(f"   {n_segments} segments | {n_phrases_total} phrases | "
          f"{n_phrases_total * len(models) * 9} appels API max")

    all_phrase_results: list[PhraseClassification] = []
    all_tensors: list[np.ndarray] = []

    for seg_idx, seg in enumerate(segments):
        phrase_keys = list(seg.paragraph.keys())
        n_phrases   = len(phrase_keys)
        full_para   = reconstruct_full_paragraph(seg.paragraph)

        print(f"\n   Segment [{seg_idx+1}/{n_segments}] — {n_phrases} phrases")
        print(f"   Contexte : '{full_para[:60]}...'")

        # ── Appels par modèle ──────────────────────────────────────────────
        all_model_results = []
        for model in models:
            short = model.split("/")[-1][:28]
            print(f"      {short} ...", end=" ", flush=True)
            model_result = classify_segment_with_model(seg, model)
            all_model_results.append(model_result)

            # Affichage compact par phrase
            parts = []
            for pkey in phrase_keys:
                r = {qk: model_result.get(qk, {}).get(pkey, "n") for qk in QUESTION_KEYS}
                parts.append(
                    f"{pkey}: R:{r['roi_1'][0]}{r['roi_2'][0]}{r['roi_3'][0]} "
                    f"N:{r['not_1'][0]}{r['not_2'][0]}{r['not_3'][0]} "
                    f"O:{r['obl_1'][0]}{r['obl_2'][0]}{r['obl_3'][0]}"
                )
            print(" | ".join(parts))

        # ── Agrégation par phrase ──────────────────────────────────────────
        aggregated = aggregate_by_phrase(all_model_results, phrase_keys)

        # ── Tensor du segment : (n_phrases, 5, 9) ─────────────────────────
        seg_tensor = build_segment_tensor(all_model_results, phrase_keys, models)
        all_tensors.append(seg_tensor)

        # ── Créer un PhraseClassification par phrase ───────────────────────
        for pkey in phrase_keys:
            reponses = aggregated[pkey]
            roi_s, not_s, obl_s = compute_scores(reponses)
            label = determine_label(roi_s, not_s, obl_s)

            # Réponses par modèle pour cette phrase
            preds_par_modele = []
            for mi, model in enumerate(models):
                model_reponses = {
                    qk: all_model_results[mi].get(qk, {}).get(pkey, "non")
                    for qk in QUESTION_KEYS
                }
                preds_par_modele.append({
                    "model_name":         model,
                    "reponses_questions": model_reponses,
                })

            print(f"      → {pkey} : {label}  (ROI:{roi_s} NOT:{not_s} OBL:{obl_s})")

            all_phrase_results.append(PhraseClassification(
                phrase_key=pkey,
                phrase_text=seg.paragraph[pkey],
                paragraph_index=seg.paragraph_index,
                section_title=seg.section_title,
                source_file=seg.source_file,
                source_folder=seg.source_folder,
                categorie=label,
                scores_roi=roi_s,
                scores_notoriete=not_s,
                scores_obligation=obl_s,
                reponses_questions=reponses,
                predictions_par_modele=preds_par_modele,
                ensemble_modeles=models,
            ))

    # Tensor complet du fichier : (n_phrases_total, 5, 9)
    file_tensor = np.concatenate(all_tensors, axis=0) if all_tensors else np.empty((0, len(models), 9), dtype=np.int8)
    return all_phrase_results, file_tensor


# ── Export fichier : JSON + Excel + tensor .npy ────────────────────────────────

def export_file_results(
    results: list[PhraseClassification],
    file_tensor: np.ndarray,
    dest_dir: Path,
    base_name: str,
    models: list,
):
    json_out = dest_dir / f"{base_name}_classification.json"
    xl_out   = dest_dir / f"{base_name}_classification.xlsx"
    npy_out  = dest_dir / f"{base_name}_tensor.npy"
    meta_out = dest_dir / f"{base_name}_tensor_meta.json"

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # ── Excel : une ligne = une phrase ────────────────────────────────────────
    rows = []
    for r in results:
        row = {
            "Source_File":      r.source_file,
            "Source_Folder":    r.source_folder,
            "Paragraph_Index":  r.paragraph_index,
            "Section_Title":    r.section_title,
            "Phrase_Key":       r.phrase_key,
            "Phrase_Text":      r.phrase_text,
            "Category":         r.categorie,
            "ROI_Score":        r.scores_roi,
            "Notoriété_Score":  r.scores_notoriete,
            "Obligation_Score": r.scores_obligation,
        }
        # Réponses agrégées
        for qk, qv in r.reponses_questions.items():
            row[f"AGG_{qk}"] = qv

        # Réponses par modèle
        for pred_d in r.predictions_par_modele:
            short = pred_d["model_name"].split("/")[-1][:20]
            rq    = pred_d["reponses_questions"]
            row[f"resp_{short}"] = (
                f"R:{rq['roi_1'][0]}{rq['roi_2'][0]}{rq['roi_3'][0]} "
                f"N:{rq['not_1'][0]}{rq['not_2'][0]}{rq['not_3'][0]} "
                f"O:{rq['obl_1'][0]}{rq['obl_2'][0]}{rq['obl_3'][0]}"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(xl_out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classification", index=False)

    # ── Tensor .npy : (n_phrases, n_models, n_questions) ─────────────────────
    np.save(str(npy_out), file_tensor)

    meta = {
        "shape":           list(file_tensor.shape),
        "dtype":           "int8",
        "axes":            ["phrase", "model", "question"],
        "values":          {"1": "oui", "0": "non"},
        "models":          models,
        "model_index":     {m: i for i, m in enumerate(models)},
        "questions":       QUESTION_KEYS,
        "question_index":  {q: i for i, q in enumerate(QUESTION_KEYS)},
        "phrases": [
            {
                "index":           i,
                "phrase_key":      r.phrase_key,
                "paragraph_index": r.paragraph_index,
                "source_file":     r.source_file,
                "phrase_text":     r.phrase_text[:100],
                "categorie":       r.categorie,
            }
            for i, r in enumerate(results)
        ],
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"      💾 JSON   → {json_out.name}")
    print(f"      💾 Excel  → {xl_out.name}")
    print(f"      💾 Tensor → {npy_out.name}  shape={file_tensor.shape}")
    print(f"      💾 Meta   → {meta_out.name}")


# ── Export global ─────────────────────────────────────────────────────────────

def export_global_summary(
    all_results: list[PhraseClassification],
    all_tensors: list[np.ndarray],
    models: list,
    timestamp: str,
):
    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    rows = [{
        "Source_Folder":    r.source_folder,
        "Source_File":      r.source_file,
        "Paragraph_Index":  r.paragraph_index,
        "Section_Title":    r.section_title,
        "Phrase_Key":       r.phrase_key,
        "Phrase_Text":      r.phrase_text[:300],
        "Category":         r.categorie,
        "ROI_Score":        r.scores_roi,
        "Notoriété_Score":  r.scores_notoriete,
        "Obligation_Score": r.scores_obligation,
    } for r in all_results]
    df = pd.DataFrame(rows)

    # Synthèse par fichier
    summary_rows = []
    for key, grp in df.groupby("Source_File"):
        t = len(grp)
        summary_rows.append({
            "Fichier":     key,
            "Total_Phrases": t,
            "ROI":         (grp.Category == "ROI").sum(),
            "Notoriété":   (grp.Category == "Notoriété").sum(),
            "Obligation":  (grp.Category == "Obligation").sum(),
            "Description": (grp.Category == "Description").sum(),
        })
    df_summary = pd.DataFrame(summary_rows)

    df_global = pd.DataFrame([{
        "Timestamp":       timestamp,
        "Priorité":        " > ".join(PRIORITY_ORDER),
        "Modèles":         " | ".join(m.split("/")[-1] for m in models),
        "Total_Phrases":   total,
        "Seuil_Min_Score": MIN_SCORE,
        **{f"Total_{c}": cats[c] for c in CATEGORIES},
        **{f"Pct_{c}": f"{round(cats[c]/total*100,1)}%" if total else "0%" for c in CATEGORIES},
    }])

    xl_path = Path(OUTPUT_DIR) / f"synthese_globale_{timestamp}.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        df.to_excel(writer,         sheet_name="Toutes_Phrases",        index=False)
        df_summary.to_excel(writer, sheet_name="Synthese_Par_Fichier",  index=False)
        df_global.to_excel(writer,  sheet_name="Statistiques_Globales", index=False)

    # Tensor global : (n_phrases_total, 5, 9)
    if all_tensors:
        global_tensor = np.concatenate(all_tensors, axis=0)
        npy_global  = Path(OUTPUT_DIR) / f"tensor_global_{timestamp}.npy"
        meta_global = Path(OUTPUT_DIR) / f"tensor_global_{timestamp}_meta.json"
        np.save(str(npy_global), global_tensor)

        meta = {
            "shape":          list(global_tensor.shape),
            "dtype":          "int8",
            "axes":           ["phrase", "model", "question"],
            "values":         {"1": "oui", "0": "non"},
            "models":         models,
            "model_index":    {m: i for i, m in enumerate(models)},
            "questions":      QUESTION_KEYS,
            "question_index": {q: i for i, q in enumerate(QUESTION_KEYS)},
            "n_phrases":      len(all_results),
        }
        with open(meta_global, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n   Tensor global → {npy_global.name}  shape={global_tensor.shape}")

    print(f"   Synthèse Excel → {xl_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(models: list = None):
    if models is None:
        models = MODELS_TO_USE

    print("=" * 80)
    print("CLASSIFICATION MULTI-LLM — Par phrase (v6)")
    print("=" * 80)
    for m in models:
        print(f"  • {m}")
    print(f"\n  9 questions × {len(models)} modèles = {9 * len(models)} appels/segment")
    print(f"  1 appel = 1 question posée sur toutes les phrases du paragraphe")
    print(f"  Seuil minimum score : {MIN_SCORE}/3")
    print()

    seg_root  = Path(SEGMENTS_INPUT_DIR)
    seg_files = sorted(seg_root.rglob("*_segments.json"))
    if not seg_files:
        print("[ERREUR] Aucun *_segments.json trouvé")
        return

    print(f"  {len(seg_files)} fichier(s) de segments trouvés\n")

    all_results: list[PhraseClassification] = []
    all_tensors: list[np.ndarray] = []
    ok, skipped, err = 0, 0, 0

    for i, seg_path in enumerate(seg_files, 1):
        rel       = seg_path.relative_to(seg_root)
        dest_dir  = seg_path.parent
        base_name = seg_path.stem
        json_out  = dest_dir / f"{base_name}_classification.json"
        npy_out   = dest_dir / f"{base_name}_tensor.npy"

        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")

        # Skip si déjà traité
        if json_out.exists() and npy_out.exists():
            print(f"  ⏭️  Déjà traité, chargement...")
            with open(json_out, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for item in cached:
                all_results.append(PhraseClassification(**{
                    k: item[k] for k in PhraseClassification.__dataclass_fields__ if k in item
                }))
            tensor_cached = np.load(str(npy_out))
            all_tensors.append(tensor_cached)
            skipped += 1
            continue

        try:
            results, file_tensor = process_segments_file(seg_path, models)
            export_file_results(results, file_tensor, dest_dir, base_name, models)
            all_results.extend(results)
            all_tensors.append(file_tensor)
            ok += 1
        except Exception as e:
            print(f"  ❌ ERREUR : {e}")
            import traceback; traceback.print_exc()
            err += 1
            continue

    # Synthèse globale
    ts = time.strftime("%Y%m%d_%H%M%S")
    if all_results:
        export_global_summary(all_results, all_tensors, models, ts)

    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"  Traités : {ok}  |  Skippés : {skipped}  |  Erreurs : {err}")
    print(f"  Phrases total : {total}")
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