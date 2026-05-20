"""
Script : Classification Multi-LLM — OpenRouter (5 modeles)
v2 — Classification phrase par phrase avec contexte du paragraphe
Avec export Excel complet et tenseur NPY
Ajout du taux d'accord inter-modeles par phrase

ARCHITECTURE :
  - Avant (v1) : 1 appel LLM = toutes les phrases d'un paragraphe
  - Apres (v2) : 1 appel LLM = 1 phrase (avec paragraphe comme contexte)
  
AMELIORATIONS :
  - Meilleure precision : le LLM se concentre sur UNE phrase
  - Contexte explicit : le paragraphe entier est fourni comme contexte
  - ThreadPoolExecutor : les 5 modeles sont lances EN PARALLELE par phrase
  - _call_openrouter : retry exponentiel ameliore + lecture header X-RateLimit-Reset
  - Ajout jitter aleatoire sur les 429 pour desynchroniser les threads
  - MAX_RETRIES 3->5
  - Ajout threading.Lock() sur StatsAccumulator (thread-safe)

FIXES :
  - FIX 1 : _process_one_phrase_one_model top-level (evite bug de closure)
  - FIX 2 : titre de section traite separement (phrase_index=-1)
"""

import os
import re
import json
import time
import random
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"F:\Jihene\business_value_classification\classification_test\Decoupage\paragraphes"
OUTPUT_DIR         = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_paragraphes"
SYSTEM_PROMPT_FILE = r"F:\Jihene\business_value_classification\classification_test\Classification\system_prompt_paragraph.txt"
USER_PROMPT_FILE   = r"F:\Jihene\business_value_classification\classification_test\Classification\user_prompt_paragraph.txt"
QUESTIONS_FILE     = r"F:\Jihene\business_value_classification\classification_test\Classification\liste_questions.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante dans le fichier .env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Taux de change EUR/USD ─────────────────────────────────────────────────────
EUR_USD_RATE = 1.08

def usd_to_eur(usd: float) -> float:
    return usd / EUR_USD_RATE

# ── Prix par modele (USD par million de tokens) ────────────────────────────────
MODEL_PRICING = {
    "meta-llama/llama-3.3-70b-instruct":          {"input": 0.10, "output": 0.32},
    "google/gemma-3-27b-it":                       {"input": 0.08, "output": 0.16},
    "mistralai/mistral-nemo":                      {"input": 0.02, "output": 0.03},
    "qwen/qwen3-8b":                               {"input": 0.05, "output": 0.40},
    "openai/gpt-4o-mini":                          {"input": 0.15, "output": 0.60},
}
DEFAULT_PRICING = {"input": 0.10, "output": 0.10}

def compute_call_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

# ── Modeles ────────────────────────────────────────────────────────────────────
ENSEMBLE_A = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-nemo",
    "qwen/qwen3-8b",
    "openai/gpt-4o-mini",
]
MODELS_TO_USE = ENSEMBLE_A

# ── Parametres ─────────────────────────────────────────────────────────────────
MAX_TOKENS        = 2048
MAX_RETRIES       = 5
BACKOFF_BASE      = 2.0
INTER_MODEL_DELAY = 0.5
INTER_CALL_DELAY  = 0.3
MAX_WORKERS       = 2

CATEGORIES     = ["ROI", "Notoriete", "Obligation", "Description"]
PRIORITY_ORDER = ["ROI", "Obligation", "Notoriete", "Description"]

# ── Chargement du System Prompt ───────────────────────────────────────────────
def load_system_prompt(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"System Prompt introuvable : {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ── Chargement du User Prompt template ────────────────────────────────────────
def load_user_prompt_template(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"User Prompt template introuvable : {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ── Chargement des questions ───────────────────────────────────────────────────
def load_questions(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier questions introuvable : {file_path}")
    questions = {}
    category_counters = {"ROI": 1, "NOT": 1, "OBL": 1}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\[(ROI|NOT|OBL)\]\s*(.*)$', line)
            if match:
                category = match.group(1)
                question_text = match.group(2).strip()
                key_category = "notoriete" if category == "NOT" else category.lower()
                key = f"{key_category}_{category_counters[category]}"
                questions[key] = {
                    "text": question_text,
                    "category": category,
                    "display": "Notoriete" if category == "NOT" else category
                }
                category_counters[category] += 1
            else:
                print(f"  Avertissement: Ligne {line_num} ignoree (format non reconnu): {line[:50]}...")
    return questions

# ── Chargement effectif ───────────────────────────────────────────────────────
SYSTEM_PROMPT = load_system_prompt(SYSTEM_PROMPT_FILE)
USER_PROMPT_TEMPLATE = load_user_prompt_template(USER_PROMPT_FILE)
QUESTIONS_DATA  = load_questions(QUESTIONS_FILE)
QUESTION_KEYS   = list(QUESTIONS_DATA.keys())

def get_question_text(question_key: str) -> str:
    return QUESTIONS_DATA.get(question_key, {}).get("text", "")

def get_question_category(question_key: str) -> str:
    return QUESTIONS_DATA.get(question_key, {}).get("category", "UNKNOWN")

def get_question_display(question_key: str) -> str:
    cat = get_question_category(question_key)
    return "Notoriete" if cat == "NOT" else cat

ROI_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "ROI"]
NOT_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "NOT"]
OBL_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "OBL"]

# ── Construction du prompt pour UNE SEULE PHRASE ──────────────────────────────
def build_single_phrase_prompt(full_paragraph: str, single_phrase: str, 
                                question_key: str) -> str:
    """
    Construit un prompt pour classer UNE phrase avec le contexte du paragraphe.
    Envoie : contexte (paragraphe) + phrase a classer + question
    """
    question_text = get_question_text(question_key)

    return USER_PROMPT_TEMPLATE.format(
        full_paragraph=full_paragraph[:1200],
        phrase=single_phrase,
        question_text=question_text
    )

# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class Paragraph:
    section_title:   str
    paragraph_index: int
    source_file:     str
    source_folder:   str
    phrases:         list

    @property
    def full_text(self) -> str:
        return " ".join(p["phrase"] for p in self.phrases)


@dataclass
class PhraseModelPrediction:
    model_name:         str
    phrase_index:       int
    reponses_questions: dict
    input_tokens:       int   = 0
    output_tokens:      int   = 0
    cost_usd:           float = 0.0
    cost_eur:           float = 0.0
    duration_s:         float = 0.0
    erreur:             str   = None


@dataclass
class FinalClassification:
    section_title:          str
    phrase:                 str
    phrase_index:           int
    paragraph_index:        int
    source_file:            str
    source_folder:          str
    categorie:              str
    scores_roi:             int
    scores_notoriete:       int
    scores_obligation:      int
    reponses_questions:     dict
    agreement_rates:        dict = field(default_factory=dict)
    cost_usd_phrase:        float = 0.0
    cost_eur_phrase:        float = 0.0
    duration_s_paragraph:   float = 0.0
    predictions_par_modele: list = field(default_factory=list)
    ensemble_modeles:       list = field(default_factory=list)


# ── Accumulateur de statistiques global ───────────────────────────────────────
class StatsAccumulator:
    def __init__(self, models: list):
        self.models = models
        self.start_time_global = time.time()
        self._lock = threading.Lock()
        self.cost_usd_by_model   = {m: 0.0 for m in models}
        self.tokens_in_by_model  = {m: 0   for m in models}
        self.tokens_out_by_model = {m: 0   for m in models}
        self.calls_by_model      = {m: 0   for m in models}
        self.errors_by_model     = {m: 0   for m in models}
        self.agree_counts        = {k: 0   for k in QUESTION_KEYS}
        self.total_votes         = {k: 0   for k in QUESTION_KEYS}
        self.cost_usd_by_file:   dict[str, float] = {}
        self.phrase_stats:       list[dict]       = []

    def record_call(self, model: str, input_tokens: int, output_tokens: int,
                    error: bool, source_file: str):
        cost_usd = compute_call_cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            self.cost_usd_by_model[model]   += cost_usd
            self.tokens_in_by_model[model]  += input_tokens
            self.tokens_out_by_model[model] += output_tokens
            self.calls_by_model[model]      += 1
            if error:
                self.errors_by_model[model] += 1
            self.cost_usd_by_file[source_file] = (
                self.cost_usd_by_file.get(source_file, 0.0) + cost_usd
            )

    def record_agreement(self, phrase_predictions: list[PhraseModelPrediction]):
        valid = [p for p in phrase_predictions if not p.erreur]
        n = len(valid)
        if n < 2:
            return
        with self._lock:
            for qk in QUESTION_KEYS:
                votes_oui = sum(1 for p in valid if p.reponses_questions.get(qk) == "oui")
                agreed = (votes_oui == n or votes_oui == 0)
                self.agree_counts[qk] += (1 if agreed else 0)
                self.total_votes[qk]  += 1

    def compute_phrase_agreement(self, phrase_predictions: list[PhraseModelPrediction]) -> dict:
        valid = [p for p in phrase_predictions if not p.erreur]
        n = len(valid)
        if n < 2:
            return {qk: 0.5 for qk in QUESTION_KEYS}
        agreement = {}
        for qk in QUESTION_KEYS:
            votes_oui = sum(1 for p in valid if p.reponses_questions.get(qk) == "oui")
            majority = max(votes_oui, n - votes_oui)
            agreement[qk] = majority / n
        return agreement

    def record_phrase(self, phrase_index: int, source_file: str,
                      n_errors: int, label: str):
        with self._lock:
            self.phrase_stats.append({
                "phrase_index": phrase_index,
                "source_file":  source_file,
                "n_errors":     n_errors,
                "label":        label,
            })

    @property
    def total_cost_usd(self) -> float:
        return sum(self.cost_usd_by_model.values())

    @property
    def total_cost_eur(self) -> float:
        return usd_to_eur(self.total_cost_usd)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time_global

    def error_rate(self, model: str) -> float:
        t = self.calls_by_model.get(model, 0)
        return self.errors_by_model.get(model, 0) / t if t else 0.0

    def agreement_rate(self, question_key: str) -> float:
        t = self.total_votes.get(question_key, 0)
        return self.agree_counts.get(question_key, 0) / t if t else 0.0

    def cost_per_phrase_eur(self) -> float:
        n = len(self.phrase_stats)
        return usd_to_eur(self.total_cost_usd) / n if n else 0.0

    def print_summary(self):
        elapsed = self.elapsed_s
        n_done  = len(self.phrase_stats)
        print("\n" + "-" * 70)
        print("STATISTIQUES GLOBALES")
        print("-" * 70)
        print(f"\n  Cout total   : ${self.total_cost_usd:.4f} USD  /  EUR{self.total_cost_eur:.4f} EUR")
        if n_done:
            print(f"  Cout/phrase  : ${self.total_cost_usd/n_done:.6f} USD  /  EUR{self.cost_per_phrase_eur():.6f} EUR")
        print("\n  Cout par modele :")
        for m in self.models:
            c_usd = self.cost_usd_by_model[m]
            c_eur = usd_to_eur(c_usd)
            tok_i = self.tokens_in_by_model[m]
            tok_o = self.tokens_out_by_model[m]
            err_r = self.error_rate(m) * 100
            short = m.split("/")[-1][:30]
            print(f"    {short:<30}  ${c_usd:.4f} / EUR{c_eur:.4f}  "
                  f"(in:{tok_i:,} out:{tok_o:,})  erreurs:{err_r:.1f}%")
        print(f"\n  Temps total : {elapsed/60:.1f} min")
        if n_done:
            print(f"  Temps moyen/paragraphe-modele : {elapsed/n_done:.1f}s")
        print(f"\n  Accord inter-modeles par question :")
        for qk in QUESTION_KEYS:
            rate = self.agreement_rate(qk) * 100
            text = get_question_text(qk)[:55]
            print(f"    {qk:<20}  {rate:5.1f}%  {text}")
        print("-" * 70)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        rows_models = []
        for m in self.models:
            c_usd = self.cost_usd_by_model[m]
            rows_models.append({
                "Modele":          m.split("/")[-1],
                "Modele_complet":  m,
                "Tokens_Input":    self.tokens_in_by_model[m],
                "Tokens_Output":   self.tokens_out_by_model[m],
                "Cout_USD":        round(c_usd, 6),
                "Cout_EUR":        round(usd_to_eur(c_usd), 6),
                "Nb_Appels":       self.calls_by_model[m],
                "Nb_Erreurs":      self.errors_by_model[m],
                "Taux_Erreur_%":   round(self.error_rate(m) * 100, 2),
            })
        rows_files = [
            {"Fichier": f, "Cout_USD": round(c, 6), "Cout_EUR": round(usd_to_eur(c), 6)}
            for f, c in self.cost_usd_by_file.items()
        ]
        rows_agree = []
        for qk in QUESTION_KEYS:
            rows_agree.append({
                "Question_Key":   qk,
                "Categorie":      get_question_display(qk),
                "Texte_Question": get_question_text(qk),
                "Taux_Accord_%":  round(self.agreement_rate(qk) * 100, 2),
                "N_Comparaisons": self.total_votes.get(qk, 0),
            })
        n_done = len(self.phrase_stats)
        df_global = pd.DataFrame([{
            "Cout_Total_USD":       round(self.total_cost_usd, 6),
            "Cout_Total_EUR":       round(self.total_cost_eur, 6),
            "Cout_par_Phrase_USD":  round(self.total_cost_usd / n_done, 7) if n_done else 0,
            "Cout_par_Phrase_EUR":  round(self.cost_per_phrase_eur(), 7),
            "Taux_EUR_USD":         EUR_USD_RATE,
            "Nb_Phrases":           n_done,
            "Duree_Totale_min":     round(self.elapsed_s / 60, 2),
        }])
        return {
            "Stats_Par_Model":   pd.DataFrame(rows_models),
            "Stats_Par_Fichier": pd.DataFrame(rows_files),
            "Accord_Questions":  pd.DataFrame(rows_agree),
            "Stats_Globales":    df_global,
        }


# ── API OpenRouter ─────────────────────────────────────────────────────────────
def _extract_content_and_usage(response_data: dict) -> tuple[str, int, int]:
    choice    = response_data["choices"][0]
    message   = choice.get("message", {})
    content   = message.get("content")
    reasoning = message.get("reasoning")
    usage        = response_data.get("usage", {})
    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    if content and content.strip():
        return content.strip(), input_tokens, output_tokens
    if reasoning and reasoning.strip():
        return reasoning.strip(), input_tokens, output_tokens
    raise ValueError("Reponse vide")


def _call_openrouter(model_name: str, messages: list) -> tuple[str, int, int]:
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
        "max_tokens":  MAX_TOKENS,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                reset_ms = resp.headers.get("X-RateLimit-Reset")
                if reset_ms:
                    wait = max(1.0, (int(reset_ms) - time.time() * 1000) / 1000)
                    wait = min(wait, 60.0)
                else:
                    wait = BACKOFF_BASE ** attempt
                wait += random.uniform(0, 2.0)
                print(f"        Rate limit ({model_name.split('/')[-1]}), attente {wait:.1f}s... (tentative {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                raise ValueError(f"Modele introuvable : {model_name}")
            if resp.status_code == 400:
                err_detail = resp.json().get("error", {}).get("message", resp.text[:200])
                raise ValueError(f"Requete invalide (400) : {err_detail}")
            resp.raise_for_status()
            return _extract_content_and_usage(resp.json())
        except (ValueError, KeyError) as e:
            raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** attempt)
            else:
                raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                print(f"        {model_name.split('/')[-1]}: {e}, retry {attempt}/{MAX_RETRIES} dans {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Echec apres {MAX_RETRIES} tentatives pour {model_name}")


def _parse_json_response(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        matches = list(re.finditer(r'\{.*\}', raw, re.DOTALL))
        for match in sorted(matches, key=lambda m: len(m.group()), reverse=True):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Aucun JSON valide trouve dans : {raw[:300]}")


def _normalize_reponses(rq: dict) -> dict:
    return {
        k: ("oui" if str(rq.get(k, "non")).lower().strip() in ("oui", "yes", "true", "1") else "non")
        for k in QUESTION_KEYS
    }


def _empty_reponses() -> dict:
    return {k: "non" for k in QUESTION_KEYS}


# ── Appel modele pour UNE SEULE PHRASE ────────────────────────────────────────
def classify_single_phrase_with_model(
    full_paragraph: str,
    phrase_text: str,
    phrase_index: int,
    model_name: str,
    question_key: str,
    source_file: str,
    stats: StatsAccumulator,
) -> tuple[PhraseModelPrediction, str | None]:
    """
    Classifie UNE phrase pour UNE question avec UN modele.
    Envoie : paragraphe (contexte) + phrase + question
    """
    user_prompt = build_single_phrase_prompt(
        full_paragraph=full_paragraph,
        single_phrase=phrase_text,
        question_key=question_key,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    t0 = time.time()
    try:
        raw, tok_in, tok_out = _call_openrouter(model_name, messages)
        duration = time.time() - t0
        parsed = _parse_json_response(raw)
        cost_usd = compute_call_cost_usd(model_name, tok_in, tok_out)
        cost_eur = usd_to_eur(cost_usd)

        stats.record_call(model_name, tok_in, tok_out, error=False,
                          source_file=source_file)

        reponse = str(parsed.get("reponse", "non")).lower()
        reponse = "oui" if reponse in ("oui", "yes", "true", "1") else "non"

        return PhraseModelPrediction(
            model_name=model_name,
            phrase_index=phrase_index,
            reponses_questions={question_key: reponse},
            input_tokens=tok_in,
            output_tokens=tok_out,
            cost_usd=cost_usd,
            cost_eur=cost_eur,
            duration_s=round(duration, 2),
        ), None

    except Exception as e:
        duration = time.time() - t0
        stats.record_call(model_name, 0, 0, error=True,
                          source_file=source_file)

        return PhraseModelPrediction(
            model_name=model_name,
            phrase_index=phrase_index,
            reponses_questions={question_key: "non"},
            duration_s=round(duration, 2),
            erreur=str(e),
        ), str(e)


# ── Agregation ────────────────────────────────────────────────────────────────
def aggregate_responses(predictions: list[PhraseModelPrediction]) -> dict:
    valid = [p for p in predictions if not p.erreur]
    n = len(valid)
    if n == 0:
        return _empty_reponses()
    result = {}
    for k in QUESTION_KEYS:
        votes_oui = sum(1 for p in valid if p.reponses_questions.get(k) == "oui")
        result[k] = "oui" if votes_oui > n / 2 else "non"
    return result


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi  = sum(1 for k in ROI_KEYS if reponses.get(k) == "oui")
    not_ = sum(1 for k in NOT_KEYS if reponses.get(k) == "oui")
    obl  = sum(1 for k in OBL_KEYS if reponses.get(k) == "oui")
    return roi, not_, obl


def determine_label(roi: int, not_: int, obl: int) -> str:
    if roi == 0 and not_ == 0 and obl == 0:
        return "Description"
    scores = {"ROI": roi, "Obligation": obl, "Notoriete": not_}
    max_score = max(scores.values())
    tied = [c for c, s in scores.items() if s == max_score]
    for priority in PRIORITY_ORDER:
        if priority in tied:
            return priority
    return tied[0]


# ── FIX 1 : process_one_phrase_one_model definie au niveau module ────────────────
#    pour eviter le bug de capture par reference dans ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────
def _process_one_phrase_one_model(args):
    """
    Fonction top-level (non-closure) pour ThreadPoolExecutor.
    Classifie UNE phrase avec UN modele pour TOUTES les questions.
    Reçoit (model, full_paragraph, phrase_text, phrase_index, source_file, stats) 
    """
    model, full_paragraph, phrase_text, phrase_index, source_file, stats = args

    phrase_predictions_by_model = {}
    total_tok_in = 0
    total_tok_out = 0
    total_duration = 0
    reponses_aggregated = {}

    for q_idx, qkey in enumerate(QUESTION_KEYS):
        pred, err = classify_single_phrase_with_model(
            full_paragraph=full_paragraph,
            phrase_text=phrase_text,
            phrase_index=phrase_index,
            model_name=model,
            question_key=qkey,
            source_file=source_file,
            stats=stats,
        )
        total_tok_in += pred.input_tokens
        total_tok_out += pred.output_tokens
        total_duration += pred.duration_s
        reponses_aggregated.update(pred.reponses_questions)
        
        if q_idx < len(QUESTION_KEYS) - 1:
            time.sleep(INTER_CALL_DELAY)

    return model, PhraseModelPrediction(
        model_name=model,
        phrase_index=phrase_index,
        reponses_questions=reponses_aggregated,
        input_tokens=total_tok_in,
        output_tokens=total_tok_out,
        cost_usd=compute_call_cost_usd(model, total_tok_in, total_tok_out),
        cost_eur=usd_to_eur(compute_call_cost_usd(model, total_tok_in, total_tok_out)),
        duration_s=total_duration,
    )


# ── Traitement d'un fichier (paragraphes → phrases) ───────────────────────────
def process_paragraphs_file(
    seg_path: Path,
    models: list,
    stats: StatsAccumulator,
) -> list[FinalClassification]:
    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paragraphs = []
    for item in data:
        phrases = item.get("phrases", [])
        if not phrases:
            continue
        paragraphs.append(Paragraph(
            section_title=item.get("section_title", ""),
            paragraph_index=item.get("paragraph_index", 0),
            source_file=item.get("source_file", seg_path.name),
            source_folder=item.get("source_folder", ""),
            phrases=phrases,
        ))

    total_phrases = sum(len(p.phrases) for p in paragraphs)
    print(f"\n   Fichier : {seg_path.name}  "
          f"({len(paragraphs)} paragraphes, {total_phrases} phrases)")

    all_phrase_results: list[FinalClassification] = []

    for p_idx, paragraph in enumerate(paragraphs):
        t_para = time.time()
        print(f"\n   [Para {p_idx+1}/{len(paragraphs)}] "
              f"Section: '{paragraph.section_title[:50]}' "
              f"({len(paragraph.phrases)} phrases)")

        # Traiter le titre
        if paragraph.section_title:
            model_predictions_title = {}
            args_list = [
                (model, paragraph.full_text, paragraph.section_title, -1, 
                 paragraph.source_file, stats)
                for model in models
            ]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(_process_one_phrase_one_model, args): args[0] 
                          for args in args_list}
                for future in as_completed(futures):
                    model, pred = future.result()
                    model_predictions_title[model] = pred

            phrase_preds_all = list(model_predictions_title.values())
            agg_resp = aggregate_responses(phrase_preds_all)
            roi_s, not_s, obl_s = compute_scores(agg_resp)
            label = determine_label(roi_s, not_s, obl_s)
            agreement_rates = stats.compute_phrase_agreement(phrase_preds_all)
            phrase_cost_usd = sum(p.cost_usd for p in phrase_preds_all)
            phrase_cost_eur = usd_to_eur(phrase_cost_usd)
            n_valides = sum(1 for p in phrase_preds_all if not p.erreur)
            n_erreurs = len(phrase_preds_all) - n_valides
            stats.record_agreement(phrase_preds_all)
            stats.record_phrase(-1, paragraph.source_file, n_erreurs, label)

            print(f"        Phrase [TITRE] -> {label}  "
                  f"(ROI:{roi_s} NOT:{not_s} OBL:{obl_s} | {n_valides}/5)  "
                  f"'{paragraph.section_title[:50]}...'")

            all_phrase_results.append(FinalClassification(
                section_title=paragraph.section_title,
                phrase=paragraph.section_title,
                phrase_index=-1,
                paragraph_index=paragraph.paragraph_index,
                source_file=paragraph.source_file,
                source_folder=paragraph.source_folder,
                categorie=label,
                scores_roi=roi_s,
                scores_notoriete=not_s,
                scores_obligation=obl_s,
                reponses_questions=agg_resp,
                agreement_rates=agreement_rates,
                cost_usd_phrase=phrase_cost_usd,
                cost_eur_phrase=phrase_cost_eur,
                duration_s_paragraph=round(time.time() - t_para, 2),
                predictions_par_modele=[asdict(p) for p in phrase_preds_all],
                ensemble_modeles=models,
            ))

        # Traiter chaque phrase
        for ph_idx, phrase_obj in enumerate(paragraph.phrases):
            pi = phrase_obj["phrase_index"]
            phrase_text = phrase_obj["phrase"]

            model_predictions = {}
            args_list = [
                (model, paragraph.full_text, phrase_text, pi, 
                 paragraph.source_file, stats)
                for model in models
            ]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(_process_one_phrase_one_model, args): args[0] 
                          for args in args_list}
                for future in as_completed(futures):
                    model, pred = future.result()
                    model_predictions[model] = pred

            phrase_preds_all = list(model_predictions.values())
            agg_resp = aggregate_responses(phrase_preds_all)
            roi_s, not_s, obl_s = compute_scores(agg_resp)
            label = determine_label(roi_s, not_s, obl_s)
            agreement_rates = stats.compute_phrase_agreement(phrase_preds_all)
            phrase_cost_usd = sum(p.cost_usd for p in phrase_preds_all)
            phrase_cost_eur = usd_to_eur(phrase_cost_usd)
            n_valides = sum(1 for p in phrase_preds_all if not p.erreur)
            n_erreurs = len(phrase_preds_all) - n_valides

            stats.record_agreement(phrase_preds_all)
            stats.record_phrase(pi, paragraph.source_file, n_erreurs, label)

            agree_display = " | ".join([f"{qk}:{agreement_rates[qk]*100:.0f}%" 
                                       for qk in QUESTION_KEYS[:3]]) + "..."
            err_flag = f"  {n_erreurs} erreur(s)" if n_erreurs else ""
            print(f"        Phrase [{pi}] -> {label}  "
                  f"(ROI:{roi_s} NOT:{not_s} OBL:{obl_s} | {n_valides}/5{err_flag})  "
                  f"[Accord: {agree_display}]  "
                  f"'{phrase_text[:50]}...'")

            all_phrase_results.append(FinalClassification(
                section_title=paragraph.section_title,
                phrase=phrase_text,
                phrase_index=pi,
                paragraph_index=paragraph.paragraph_index,
                source_file=paragraph.source_file,
                source_folder=paragraph.source_folder,
                categorie=label,
                scores_roi=roi_s,
                scores_notoriete=not_s,
                scores_obligation=obl_s,
                reponses_questions=agg_resp,
                agreement_rates=agreement_rates,
                cost_usd_phrase=phrase_cost_usd,
                cost_eur_phrase=phrase_cost_eur,
                duration_s_paragraph=round(time.time() - t_para, 2),
                predictions_par_modele=[asdict(p) for p in phrase_preds_all],
                ensemble_modeles=models,
            ))

    return all_phrase_results


# ── Export d'un fichier (JSON + Excel) ────────────────────────────────────────
def export_file_results(
    results: list[FinalClassification],
    dest_dir: Path,
    base_name: str,
    models: list,
):
    json_out = dest_dir / f"{base_name}_classification.json"
    xl_out   = dest_dir / f"{base_name}_classification.xlsx"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    rows = []
    for r in results:
        row = {
            "Source_File":       r.source_file,
            "Paragraph_Index":   r.paragraph_index,
            "Phrase_Index":      r.phrase_index,
            "Section_Title":     r.section_title,
            "Phrase":            r.phrase,
            "Is_Title":          r.phrase_index == -1,
            "Category":          r.categorie,
            "ROI_Score":         f"{r.scores_roi}/{len(ROI_KEYS)}",
            "Notoriete_Score":   f"{r.scores_notoriete}/{len(NOT_KEYS)}",
            "Obligation_Score":  f"{r.scores_obligation}/{len(OBL_KEYS)}",
            "Cost_USD":          round(r.cost_usd_phrase, 7),
            "Cost_EUR":          round(r.cost_eur_phrase, 7),
        }
        for qk, qv in r.reponses_questions.items():
            row[f"Q_{qk}"] = qv
        for pred_d in r.predictions_par_modele:
            short = pred_d["model_name"].split("/")[-1][:25]
            rq    = pred_d["reponses_questions"]
            roi_str = "".join("1" if rq.get(k) == "oui" else "0" for k in ROI_KEYS)
            not_str = "".join("1" if rq.get(k) == "oui" else "0" for k in NOT_KEYS)
            obl_str = "".join("1" if rq.get(k) == "oui" else "0" for k in OBL_KEYS)
            row[f"resp_{short}"] = f"R:{roi_str} N:{not_str} O:{obl_str}"
            row[f"err_{short}"]  = pred_d.get("erreur") or ""
            row[f"cost_eur_{short}"] = round(pred_d.get("cost_eur", 0.0), 7)
        rows.append(row)

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(xl_out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classification", index=False)

    print(f"      Sauvegarde -> {json_out.name}")
    print(f"      Sauvegarde -> {xl_out.name}")
    return json_out, xl_out


# ── Export global complet (Excel + NPY) ───────────────────────────────────────
def export_complete_summary(
    all_results: list[FinalClassification],
    models: list,
    timestamp: str,
    stats: StatsAccumulator,
    output_dir: Path
):
    rows = []
    for r in all_results:
        row = {
            "Source_File":     r.source_file,
            "Source_Folder":   r.source_folder,
            "Phrase_Index":    r.phrase_index,
            "Is_Title":        r.phrase_index == -1,
            "Phrase_Text":     r.phrase,
            "Category":        r.categorie,
            "ROI_Score":       f"{r.scores_roi}/{len(ROI_KEYS)}",
            "Notoriete_Score": f"{r.scores_notoriete}/{len(NOT_KEYS)}",
            "Obligation_Score":f"{r.scores_obligation}/{len(OBL_KEYS)}",
            "Cost_USD":        round(r.cost_usd_phrase, 7),
            "Cost_EUR":        round(r.cost_eur_phrase, 7),
            "Duration_Seconds":r.duration_s_paragraph,
        }
        for qk in QUESTION_KEYS:
            row[f"ACCORD_{qk}_%"] = round(r.agreement_rates.get(qk, 0.5) * 100, 1)
        for qk in QUESTION_KEYS:
            row[f"AGG_{qk}"] = r.reponses_questions.get(qk, "?")
        for model in models:
            short_name = model.split("/")[-1][:20]
            pred_match = next(
                (p for p in r.predictions_par_modele if p["model_name"] == model),
                None
            )
            if pred_match:
                rq = pred_match.get("reponses_questions", {})
                for qk in QUESTION_KEYS:
                    col_name = f"{short_name}_{qk}"
                    row[col_name] = rq.get(qk, "?")
                    row[f"{col_name}_cost_eur"] = round(pred_match.get("cost_eur", 0), 7)
        rows.append(row)

    df_main = pd.DataFrame(rows)

    summary_by_file = []
    for file_name, grp in df_main.groupby("Source_File"):
        summary_by_file.append({
            "Fichier":                 file_name,
            "Total_Phrases":           len(grp),
            "ROI":                     (grp.Category == "ROI").sum(),
            "Notoriete":               (grp.Category == "Notoriete").sum(),
            "Obligation":              (grp.Category == "Obligation").sum(),
            "Description":             (grp.Category == "Description").sum(),
            "Cout_Total_EUR":          grp["Cost_EUR"].sum(),
            "Cout_Moyen_EUR_Phrase":   grp["Cost_EUR"].mean(),
            "Duree_Totale_sec":        grp["Duration_Seconds"].sum(),
            "Duree_Moyenne_sec_Phrase":grp["Duration_Seconds"].mean(),
            "Accord_Moyen_ROI_%":      grp["ACCORD_roi_1_%"].mean(),
            "Accord_Moyen_Notoriete_%":grp["ACCORD_notoriete_1_%"].mean(),
            "Accord_Moyen_Obligation_%":grp["ACCORD_obl_1_%"].mean(),
        })
    df_summary_file = pd.DataFrame(summary_by_file)

    rows_models = []
    for m in models:
        short = m.split("/")[-1][:30]
        rows_models.append({
            "Modele":       short,
            "Modele_complet": m,
            "Tokens_Input": stats.tokens_in_by_model.get(m, 0),
            "Tokens_Output":stats.tokens_out_by_model.get(m, 0),
            "Cout_USD":     round(stats.cost_usd_by_model.get(m, 0), 6),
            "Cout_EUR":     round(usd_to_eur(stats.cost_usd_by_model.get(m, 0)), 6),
            "Nb_Appels":    stats.calls_by_model.get(m, 0),
            "Nb_Erreurs":   stats.errors_by_model.get(m, 0),
            "Taux_Erreur_%":round(stats.error_rate(m) * 100, 2) if stats.calls_by_model.get(m, 0) > 0 else 0,
        })
    df_models = pd.DataFrame(rows_models)

    rows_agreement = []
    for qk in QUESTION_KEYS:
        rows_agreement.append({
            "Question_Key":   qk,
            "Categorie":      get_question_display(qk),
            "Texte_Question": get_question_text(qk),
            "Taux_Accord_%":  round(stats.agreement_rate(qk) * 100, 2),
            "Nb_Comparaisons":stats.total_votes.get(qk, 0),
        })
    df_agreement = pd.DataFrame(rows_agreement)

    category_counts = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}
    df_category = pd.DataFrame([{
        "Categorie":     c,
        "Nombre":        category_counts[c],
        "Pourcentage_%": round(category_counts[c] / len(all_results) * 100, 1) if all_results else 0
    } for c in CATEGORIES])

    total_phrases = len(all_results)
    total_cost_eur = stats.total_cost_eur
    df_global = pd.DataFrame([{
        "Timestamp":               timestamp,
        "Total_Phrases":           total_phrases,
        "Total_Cout_EUR":          round(total_cost_eur, 6),
        "Cout_Moyen_EUR_Phrase":   round(total_cost_eur / total_phrases, 7) if total_phrases > 0 else 0,
        "Duree_Totale_min":        round(stats.elapsed_s / 60, 2),
        "Duree_Moyenne_sec_Phrase":round(stats.elapsed_s / total_phrases, 2) if total_phrases > 0 else 0,
        "Taux_EUR_USD":            EUR_USD_RATE,
        "Nb_Modeles":              len(models),
        "Nb_Questions":            len(QUESTION_KEYS),
        "Modeles_Utilises":        " | ".join(m.split("/")[-1] for m in models),
        "Accord_Moyen_Global_%":   round(stats.agreement_rate(QUESTION_KEYS[0]) * 100, 1) if QUESTION_KEYS else 0,
    }])

    df_agreement_phrases = pd.DataFrame([{
        "Source_File":  r.source_file,
        "Phrase_Index": r.phrase_index,
        "Is_Title":     r.phrase_index == -1,
        "Phrase_Text":  r.phrase[:100],
        **{f"Accord_{qk}_%": round(r.agreement_rates.get(qk, 0.5) * 100, 1) for qk in QUESTION_KEYS}
    } for r in all_results])

    excel_path = output_dir / f"recapitulatif_complet_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_main.to_excel(writer,             sheet_name="Toutes_Phrases",         index=False)
        df_summary_file.to_excel(writer,     sheet_name="Synthese_Par_Fichier",   index=False)
        df_models.to_excel(writer,           sheet_name="Stats_Par_Model",        index=False)
        df_agreement.to_excel(writer,        sheet_name="Accord_Questions_Global",index=False)
        df_category.to_excel(writer,         sheet_name="Distribution_Categories",index=False)
        df_global.to_excel(writer,           sheet_name="Stats_Globales",         index=False)
        df_agreement_phrases.to_excel(writer,sheet_name="Accord_Par_Phrase",      index=False)

    print(f"\n   Export Excel complet -> {excel_path.name}")

    # Export Tensor NPY — titres inclus (phrase_index=-1 = titre de section)
    clients = sorted({r.source_folder for r in all_results})
    client_to_index = {client: idx for idx, client in enumerate(clients)}
    phrases_by_client = {
        client: []
        for client in clients
    }
    for result in sorted(
        all_results,
        key=lambda r: (r.source_folder, r.source_file, r.paragraph_index, r.phrase_index)
    ):
        phrases_by_client[result.source_folder].append(result)

    n_clients = len(clients)
    n_models = len(models)
    n_questions = len(QUESTION_KEYS)
    max_phrases_per_client = max(len(v) for v in phrases_by_client.values()) if phrases_by_client else 0

    tensor = np.zeros((n_clients, max_phrases_per_client, n_models, n_questions), dtype=np.int8)

    for client, phrase_results in phrases_by_client.items():
        ci = client_to_index[client]
        for pi, result in enumerate(phrase_results):
            for mi, model in enumerate(models):
                pred_match = next(
                    (p for p in result.predictions_par_modele if p["model_name"] == model),
                    None
                )
                if pred_match:
                    rq = pred_match.get("reponses_questions", {})
                    for qi, qk in enumerate(QUESTION_KEYS):
                        tensor[ci, pi, mi, qi] = 1 if rq.get(qk, "non") == "oui" else 0
                else:
                    tensor[ci, pi, mi, :] = 0

    npy_path = output_dir / f"tensor_complet_{timestamp}.npy"
    np.save(npy_path, tensor)

    meta = {
        "shape":          list(tensor.shape),
        "dtype":          "int8",
        "axes":           ["client", "phrase", "model", "question"],
        "values":         {"1": "oui", "0": "non"},
        "note":           "phrase_index=-1 indique un titre de section (inclus dans le tenseur). La dimension client est basee sur source_folder.",
        "clients":        clients,
        "models":         [m.split("/")[-1] for m in models],
        "questions":      QUESTION_KEYS,
        "questions_text": {k: get_question_text(k) for k in QUESTION_KEYS},
        "n_clients":      n_clients,
        "max_phrases_per_client": max_phrases_per_client,
        "n_models":       n_models,
        "n_questions":    n_questions,
        "timestamp":      timestamp,
    }

    meta_path = output_dir / f"tensor_complet_{timestamp}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"   Tensor NPY -> {npy_path.name} shape={tensor.shape}")
    print(f"   Metadonnees -> {meta_path.name}")

    return excel_path, npy_path


# ── Export global ─────────────────────────────────────────────────────────────
def export_global_summary(
    all_results: list[FinalClassification],
    models: list,
    timestamp: str,
    stats: StatsAccumulator,
):
    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    rows = []
    for r in all_results:
        rows.append({
            "Source_Folder":    r.source_folder,
            "Source_File":      r.source_file,
            "Paragraph_Index":  r.paragraph_index,
            "Phrase_Index":     r.phrase_index,
            "Is_Title":         r.phrase_index == -1,
            "Section_Title":    r.section_title,
            "Phrase":           r.phrase,
            "Category":         r.categorie,
            "ROI_Score":        f"{r.scores_roi}/{len(ROI_KEYS)}",
            "Notoriete_Score":  f"{r.scores_notoriete}/{len(NOT_KEYS)}",
            "Obligation_Score": f"{r.scores_obligation}/{len(OBL_KEYS)}",
            "Cout_EUR_Phrase":  round(r.cost_eur_phrase, 7),
        })
    df = pd.DataFrame(rows)

    summary_rows = []
    for key, grp in df.groupby("Source_File"):
        t = len(grp)
        summary_rows.append({
            "Fichier":          key,
            "Total_Phrases":    t,
            "ROI":              (grp.Category == "ROI").sum(),
            "Notoriete":        (grp.Category == "Notoriete").sum(),
            "Obligation":       (grp.Category == "Obligation").sum(),
            "Description":      (grp.Category == "Description").sum(),
            "Cout_USD_Fichier": round(stats.cost_usd_by_file.get(key, 0.0), 6),
            "Cout_EUR_Fichier": round(usd_to_eur(stats.cost_usd_by_file.get(key, 0.0)), 6),
        })
    df_summary = pd.DataFrame(summary_rows)

    df_global = pd.DataFrame([{
        "Timestamp":       timestamp,
        "Priorite":        " > ".join(PRIORITY_ORDER),
        "Modeles":         " | ".join(m.split("/")[-1] for m in models),
        "Total_Phrases":   total,
        "Nb_Q_ROI":        len(ROI_KEYS),
        "Nb_Q_Notoriete":  len(NOT_KEYS),
        "Nb_Q_Obligation": len(OBL_KEYS),
        **{f"Total_{c}": cats[c] for c in CATEGORIES},
        **{f"Pct_{c}": f"{round(cats[c]/total*100,1)}%" if total else "0%" for c in CATEGORIES},
    }])

    xl_path = Path(OUTPUT_DIR) / f"synthese_globale_{timestamp}.xlsx"
    stats_dfs = stats.to_dataframes()

    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        df.to_excel(writer,         sheet_name="Toutes_Phrases",              index=False)
        df_summary.to_excel(writer, sheet_name="Synthese_Par_Fichier",        index=False)
        df_global.to_excel(writer,  sheet_name="Statistiques_Globales",       index=False)
        stats_dfs["Stats_Par_Model"].to_excel(writer,   sheet_name="Stats_Par_Model",         index=False)
        stats_dfs["Stats_Par_Fichier"].to_excel(writer, sheet_name="Stats_Par_Fichier",        index=False)
        stats_dfs["Accord_Questions"].to_excel(writer,  sheet_name="Accord_Questions_Global",  index=False)
        stats_dfs["Stats_Globales"].to_excel(writer,    sheet_name="Stats_Globales",            index=False)

    print(f"\n   Synthese globale -> {xl_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(models: list = None):
    if models is None:
        models = MODELS_TO_USE

    stats = StatsAccumulator(models)

    print("=" * 80)
    print("CLASSIFICATION MULTI-LLM — Phrase par phrase (contexte = paragraphe)")
    print("=" * 80)
    print(f"  Architecture : 1 appel = 1 phrase + 1 question + 1 modele")
    print(f"                + contexte du paragraphe pour chaque phrase")
    print(f"  System Prompt  : {SYSTEM_PROMPT_FILE}")
    print(f"  User Prompt    : {USER_PROMPT_FILE}")
    print(f"  Questions file : {QUESTIONS_FILE}")
    print(f"  {len(QUESTION_KEYS)} questions chargees  "
          f"(ROI:{len(ROI_KEYS)} NOT:{len(NOT_KEYS)} OBL:{len(OBL_KEYS)})")
    print(f"  MAX_WORKERS    : {MAX_WORKERS} modeles en parallele par phrase")
    for m in models:
        pricing = MODEL_PRICING.get(m, DEFAULT_PRICING)
        print(f"  * {m}  (in:${pricing['input']}/M  out:${pricing['output']}/M)")
    print(f"\n  Taux EUR/USD : {EUR_USD_RATE}")
    print(f"  Regle de priorite : {' > '.join(PRIORITY_ORDER)}")
    print(f"  MAX_TOKENS : {MAX_TOKENS}  |  MAX_RETRIES : {MAX_RETRIES}  |  BACKOFF_BASE : {BACKOFF_BASE}")
    print()

    seg_root  = Path(SEGMENTS_INPUT_DIR)
    seg_files = sorted(seg_root.rglob("*.json"))
    if not seg_files:
        print("[ERREUR] Aucun fichier .json trouve")
        return

    print(f"  {len(seg_files)} fichier(s) trouve(s)\n")

    all_results = []
    ok, skipped, err_count = 0, 0, 0

    for i, seg_path in enumerate(seg_files, 1):
        rel       = seg_path.relative_to(seg_root)
        dest_dir  = seg_path.parent
        base_name = seg_path.stem
        json_out  = dest_dir / f"{base_name}_classification.json"

        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")

        if json_out.exists():
            print(f"   Deja traite ({json_out.name}), chargement et skip.")
            with open(json_out, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for item in cached:
                all_results.append(FinalClassification(**{
                    k: item[k]
                    for k in FinalClassification.__dataclass_fields__
                    if k in item
                }))
            skipped += 1
            continue

        try:
            t_file  = time.time()
            results = process_paragraphs_file(seg_path, models, stats)
            export_file_results(results, dest_dir, base_name, models)
            all_results.extend(results)
            ok += 1
            file_dur     = time.time() - t_file
            file_cost_eur = usd_to_eur(
                stats.cost_usd_by_file.get(
                    results[0].source_file if results else "", 0.0
                )
            )
            print(f"   Fichier termine en {file_dur/60:.1f}min  "
                  f"cout fichier: EUR{file_cost_eur:.4f}  "
                  f"{len(results)} phrases classifiees")
        except Exception as e:
            import traceback
            print(f"   ERREUR : {e}")
            traceback.print_exc()
            err_count += 1
            continue

    ts = time.strftime("%Y%m%d_%H%M%S")
    if all_results:
        export_global_summary(all_results, models, ts, stats)
        export_complete_summary(all_results, models, ts, stats, Path(OUTPUT_DIR))

    stats.print_summary()

    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    print("\n" + "=" * 80)
    print("RESUME FINAL")
    print("=" * 80)
    print(f"  Traites : {ok}  |  Skipes : {skipped}  |  Erreurs : {err_count}  "
          f"|  Total fichiers : {len(seg_files)}")
    print(f"  Phrases total : {total}")
    for c in CATEGORIES:
        pct = round(cats[c] / total * 100, 1) if total else 0
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