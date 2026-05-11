"""
Script : Classification Multi-LLM — OpenRouter (5 modeles)
v5-parallel — Input plat (liste de phrases sans paragraphe)
     1 appel = 1 question x 1 phrase x 1 modele
     Version avec system_prompt_phrase.txt et user_prompt_phrase.txt
     Ajout du taux d'accord inter-modeles par phrase

MODIFICATIONS v5 -> v5-parallel :
  - ThreadPoolExecutor : les 5 modeles sont lances EN PARALLELE par question
  - Pour chaque question, les appels aux modeles se font simultanement
  - Ajout threading.Lock() sur calls_by_question et all_calls (thread-safe)
  - AUCUNE autre logique modifiee
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
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Decoupage\phrases\NGE"
OUTPUT_DIR         = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Classification\resultats_classification_phrases\NGE"
SYSTEM_PROMPT_FILE = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Classification\system_prompt_phrase.txt"
USER_PROMPT_FILE   = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Classification\user_prompt_phrase.txt"
QUESTIONS_FILE     = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Classification\liste_questions.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante dans le fichier .env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EUR_USD_RATE = 1.08

def usd_to_eur(usd: float) -> float:
    return usd / EUR_USD_RATE

MODEL_PRICING = {
    "meta-llama/llama-3.3-70b-instruct":        {"input": 0.10, "output": 0.32},
    "google/gemma-3-27b-it":                     {"input": 0.08, "output": 0.16},
    "mistralai/mistral-small-24b-instruct-2501": {"input": 0.05, "output": 0.08},
    "qwen/qwen3-8b":                             {"input": 0.05, "output": 0.40},
    "openai/gpt-4o-mini":                        {"input": 0.15, "output": 0.60},
}
DEFAULT_PRICING = {"input": 0.10, "output": 0.10}

def compute_call_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

ENSEMBLE_A = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-24b-instruct-2501",
    "qwen/qwen3-8b",
    "openai/gpt-4o-mini",
]
MODELS_TO_USE = ENSEMBLE_A

MAX_TOKENS        = 64
MAX_RETRIES       = 5
BACKOFF_BASE      = 2.0
INTER_MODEL_DELAY = 0.5
INTER_CALL_DELAY  = 0.3
MAX_WORKERS       = 2

CATEGORIES     = ["ROI", "Notoriete", "Obligation", "Description"]
PRIORITY_ORDER = ["ROI", "Obligation", "Notoriete", "Description"]


# ── Chargement du System Prompt ────────────────────────────────────────────────
def load_system_prompt(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"System Prompt introuvable : {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ── Chargement du User Prompt template ─────────────────────────────────────────
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
        raise FileNotFoundError(f"Questions introuvables : {file_path}")
    questions = {}
    category_counters = {"ROI": 1, "NOT": 1, "OBL": 1}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\[(ROI|NOT|OBL)\]\s*(.*)$', line)
            if match:
                category      = match.group(1)
                question_text = match.group(2).strip()
                key_category  = "notoriete" if category == "NOT" else category.lower()
                key = f"{key_category}_{category_counters[category]}"
                questions[key] = {
                    "text":     question_text,
                    "category": category,
                    "display":  "Notoriete" if category == "NOT" else category,
                }
                category_counters[category] += 1
            else:
                print(f"  Avertissement: Ligne {line_num} ignoree : {line[:50]}")
    return questions


# ── Chargement effectif ────────────────────────────────────────────────────────
SYSTEM_PROMPT = load_system_prompt(SYSTEM_PROMPT_FILE)
USER_PROMPT_TEMPLATE = load_user_prompt_template(USER_PROMPT_FILE)
QUESTIONS_DATA  = load_questions(QUESTIONS_FILE)
QUESTION_KEYS   = list(QUESTIONS_DATA.keys())

def get_question_text(k):     return QUESTIONS_DATA[k]["text"]
def get_question_category(k): return QUESTIONS_DATA[k]["category"]
def get_question_display(k):  return QUESTIONS_DATA[k]["display"]

ROI_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "ROI"]
NOT_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "NOT"]
OBL_KEYS = [k for k in QUESTION_KEYS if get_question_category(k) == "OBL"]


# ── Construction du prompt ─────────────────────────────────────────────────────
def build_prompt(phrase: str, question_key: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        phrase=phrase,
        question_text=get_question_text(question_key),
    )


# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class Phrase:
    phrase:       str
    phrase_index: int
    source_file:  str
    source_folder: str


@dataclass
class CallResult:
    """Resultat d'un appel : 1 modele x 1 question x 1 phrase."""
    model_name:    str
    question_key:  str
    reponse:       str
    input_tokens:  int   = 0
    output_tokens: int   = 0
    cost_usd:      float = 0.0
    cost_eur:      float = 0.0
    duration_s:    float = 0.0
    erreur:        str   = None


@dataclass
class FinalClassification:
    phrase:                 str
    phrase_index:           int
    source_file:            str
    source_folder:          str
    categorie:              str
    scores_roi:             int
    scores_notoriete:       int
    scores_obligation:      int
    reponses_questions:     dict
    agreement_rates:        dict = field(default_factory=dict)  # NOUVEAU
    cost_usd_phrase:        float = 0.0
    cost_eur_phrase:        float = 0.0
    duration_s_phrase:      float = 0.0
    detail_appels:          list = field(default_factory=list)
    ensemble_modeles:       list = field(default_factory=list)


# ── Accumulateur de stats ──────────────────────────────────────────────────────
class StatsAccumulator:
    def __init__(self, models: list):
        self.models              = models
        self.start_time          = time.time()
        self._lock               = threading.Lock()
        self.cost_usd_by_model   = {m: 0.0 for m in models}
        self.tokens_in_by_model  = {m: 0   for m in models}
        self.tokens_out_by_model = {m: 0   for m in models}
        self.calls_by_model      = {m: 0   for m in models}
        self.errors_by_model     = {m: 0   for m in models}
        self.agree_counts        = {k: 0   for k in QUESTION_KEYS}
        self.total_votes         = {k: 0   for k in QUESTION_KEYS}
        self.cost_usd_by_file:   dict[str, float] = {}
        self.phrase_count:       int = 0

    def record_call(self, model: str, question_key: str,
                    tok_in: int, tok_out: int,
                    error: bool, source_file: str):
        cost = compute_call_cost_usd(model, tok_in, tok_out)
        with self._lock:
            self.cost_usd_by_model[model]   += cost
            self.tokens_in_by_model[model]  += tok_in
            self.tokens_out_by_model[model] += tok_out
            self.calls_by_model[model]      += 1
            if error:
                self.errors_by_model[model] += 1
            self.cost_usd_by_file[source_file] = (
                self.cost_usd_by_file.get(source_file, 0.0) + cost
            )

    def record_agreement(self, question_key: str, call_results: list[CallResult]):
        """Accord inter-modeles pour une question sur une phrase (global)."""
        valid = [r for r in call_results if not r.erreur]
        n = len(valid)
        if n < 2:
            return
        votes_oui = sum(1 for r in valid if r.reponse == "oui")
        agreed = (votes_oui == n or votes_oui == 0)
        with self._lock:
            self.agree_counts[question_key] += (1 if agreed else 0)
            self.total_votes[question_key]  += 1

    def compute_phrase_agreement(self, call_results: list[CallResult]) -> dict:
        """
        Calcule le taux d'accord pour UNE phrase et UNE question.
        Retourne un dict avec pour chaque question le niveau d'accord (entre 0.5 et 1.0).
        """
        valid = [r for r in call_results if not r.erreur]
        n = len(valid)
        if n < 2:
            return {qk: 0.5 for qk in QUESTION_KEYS}
        
        agreement = {}
        # Grouper par question_key
        calls_by_qk: dict[str, list[CallResult]] = {qk: [] for qk in QUESTION_KEYS}
        for r in call_results:
            if r.question_key in calls_by_qk:
                calls_by_qk[r.question_key].append(r)
        
        for qk in QUESTION_KEYS:
            qk_calls = calls_by_qk.get(qk, [])
            valid_qk = [r for r in qk_calls if not r.erreur]
            n_qk = len(valid_qk)
            if n_qk < 2:
                agreement[qk] = 0.5
            else:
                votes_oui = sum(1 for r in valid_qk if r.reponse == "oui")
                majority = max(votes_oui, n_qk - votes_oui)
                agreement[qk] = majority / n_qk
        
        return agreement

    @property
    def total_cost_usd(self) -> float:
        return sum(self.cost_usd_by_model.values())

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def error_rate(self, model: str) -> float:
        t = self.calls_by_model.get(model, 0)
        return self.errors_by_model.get(model, 0) / t if t else 0.0

    def agreement_rate(self, qk: str) -> float:
        t = self.total_votes.get(qk, 0)
        return self.agree_counts.get(qk, 0) / t if t else 0.0

    def eta_str(self, done: int, total: int) -> str:
        if done == 0:
            return ""
        avg = self.elapsed_s / done
        remaining = (total - done) * avg
        return f"ETA {remaining/60:.1f}min"

    def print_summary(self):
        n     = self.phrase_count
        elaps = self.elapsed_s
        total_eur = usd_to_eur(self.total_cost_usd)
        print("\n" + "-" * 70)
        print("STATISTIQUES GLOBALES")
        print("-" * 70)
        print(f"  Cout total  : ${self.total_cost_usd:.4f} / EUR{total_eur:.4f}")
        if n:
            print(f"  Cout/phrase : ${self.total_cost_usd/n:.6f} / EUR{usd_to_eur(self.total_cost_usd)/n:.6f}")
        print(f"  Temps total : {elaps/60:.1f} min")
        if n:
            print(f"  Temps/phrase: {elaps/n:.1f}s")
        print("\n  Par modele :")
        for m in self.models:
            c = self.cost_usd_by_model[m]
            print(f"    {m.split('/')[-1]:<32} "
                  f"${c:.4f}/EUR{usd_to_eur(c):.4f}  "
                  f"appels:{self.calls_by_model[m]}  "
                  f"err:{self.error_rate(m)*100:.1f}%")
        print(f"\n  Accord inter-modeles par question (global) :")
        for qk in QUESTION_KEYS:
            print(f"    {qk:<20} {self.agreement_rate(qk)*100:5.1f}%  "
                  f"{get_question_text(qk)[:52]}")
        print("-" * 70)


# ── API OpenRouter ─────────────────────────────────────────────────────────────
def _extract_content_and_usage(data: dict) -> tuple[str, int, int]:
    choice  = data["choices"][0]
    msg     = choice.get("message", {})
    content = msg.get("content") or msg.get("reasoning") or ""
    if not content.strip():
        raise ValueError("Reponse vide")
    usage = data.get("usage", {})
    return (content.strip(),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0))


def _call_openrouter(model_name: str, messages: list) -> tuple[str, int, int]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/jiheneguesmi/Business-Value-Knowledge-Graph",
        "X-Title":       "Business Value Classification",
    }
    payload = {
        "model":       model_name,
        "messages":    messages,
        "temperature": 0,
        "max_tokens":  MAX_TOKENS,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=120)
            if resp.status_code == 429:
                # Lire le header reset si disponible, sinon backoff exponentiel
                reset_ms = resp.headers.get("X-RateLimit-Reset")
                if reset_ms:
                    wait = max(1.0, (int(reset_ms) - time.time() * 1000) / 1000)
                    wait = min(wait, 60.0)
                else:
                    wait = BACKOFF_BASE ** attempt
                # Jitter pour desynchroniser les threads paralleles
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
        except (ValueError, KeyError):
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


def _parse_reponse(raw: str) -> str:
    """Extrait 'oui' ou 'non' depuis la reponse brute."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        parsed = json.loads(raw)
        rep = str(parsed.get("reponse", "non")).lower().strip()
        return "oui" if rep in ("oui", "yes", "true", "1") else "non"
    except json.JSONDecodeError:
        lower = raw.lower()
        if "oui" in lower:
            return "oui"
        return "non"


# ── Appel : 1 modele x 1 question x 1 phrase ─────────────────────────────────
def call_one(
    phrase_obj: Phrase,
    question_key: str,
    model_name: str,
    stats: StatsAccumulator,
) -> CallResult:

    prompt   = build_prompt(phrase_obj.phrase, question_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    t0 = time.time()
    try:
        raw, tok_in, tok_out = _call_openrouter(model_name, messages)
        duration = time.time() - t0
        reponse  = _parse_reponse(raw)
        cost_usd = compute_call_cost_usd(model_name, tok_in, tok_out)

        stats.record_call(model_name, question_key, tok_in, tok_out,
                          error=False, source_file=phrase_obj.source_file)

        return CallResult(
            model_name=model_name,
            question_key=question_key,
            reponse=reponse,
            input_tokens=tok_in,
            output_tokens=tok_out,
            cost_usd=cost_usd,
            cost_eur=usd_to_eur(cost_usd),
            duration_s=round(duration, 2),
        )

    except Exception as e:
        duration = time.time() - t0
        stats.record_call(model_name, question_key, 0, 0,
                          error=True, source_file=phrase_obj.source_file)
        return CallResult(
            model_name=model_name,
            question_key=question_key,
            reponse="non",
            duration_s=round(duration, 2),
            erreur=str(e),
        )


# ── Agregation ────────────────────────────────────────────────────────────────
def majority_vote(call_results: list[CallResult]) -> str:
    valid = [r for r in call_results if not r.erreur]
    n = len(valid)
    if n == 0:
        return "non"
    votes_oui = sum(1 for r in valid if r.reponse == "oui")
    return "oui" if votes_oui > n / 2 else "non"


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi  = sum(1 for k in ROI_KEYS if reponses.get(k) == "oui")
    not_ = sum(1 for k in NOT_KEYS if reponses.get(k) == "oui")
    obl  = sum(1 for k in OBL_KEYS if reponses.get(k) == "oui")
    return roi, not_, obl


def determine_label(roi: int, not_: int, obl: int) -> str:
    if roi == 0 and not_ == 0 and obl == 0:
        return "Description"
    scores    = {"ROI": roi, "Obligation": obl, "Notoriete": not_}
    max_score = max(scores.values())
    tied      = [c for c, s in scores.items() if s == max_score]
    for priority in PRIORITY_ORDER:
        if priority in tied:
            return priority
    return tied[0]


# ── Traitement d'une phrase ───────────────────────────────────────────────────
def process_phrase(
    phrase_obj: Phrase,
    models: list,
    stats: StatsAccumulator,
    phrase_num: int,
    total_phrases: int,
) -> FinalClassification:
    """
    Pour chaque question x chaque modele : 1 appel.
    Les modeles sont lances EN PARALLELE par question.
    Retourne 1 FinalClassification.
    """
    t_phrase = time.time()

    # calls_by_question[question_key] = liste de CallResult (un par modele)
    calls_by_question: dict[str, list[CallResult]] = {
        qk: [] for qk in QUESTION_KEYS
    }

    # Stocker tous les appels pour calculer l'accord par phrase
    all_calls: list[CallResult] = []

    # ── MODIFIE : modeles en parallele via ThreadPoolExecutor ─────────────
    _lock = threading.Lock()

    for question_key in QUESTION_KEYS:
        def process_one_model_for_question(model):
            result = call_one(phrase_obj, question_key, model, stats)
            return model, result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one_model_for_question, model): model for model in models}
            for future in as_completed(futures):
                model, result = future.result()
                with _lock:
                    calls_by_question[question_key].append(result)
                    all_calls.append(result)

        # Accord inter-modeles pour cette question (global)
        stats.record_agreement(question_key, calls_by_question[question_key])

    # Calculer l'accord par phrase
    phrase_agreement = stats.compute_phrase_agreement(all_calls)

    # Agreger par question (vote majoritaire)
    reponses_aggregees = {
        qk: majority_vote(calls_by_question[qk])
        for qk in QUESTION_KEYS
    }

    roi_s, not_s, obl_s = compute_scores(reponses_aggregees)
    label               = determine_label(roi_s, not_s, obl_s)

    phrase_cost_usd = sum(r.cost_usd for r in all_calls)
    phrase_cost_eur = usd_to_eur(phrase_cost_usd)
    phrase_duration = time.time() - t_phrase

    # Detail brut pour tracabilite
    detail = [
        {
            "model":        r.model_name.split("/")[-1],
            "question_key": r.question_key,
            "reponse":      r.reponse,
            "erreur":       r.erreur,
            "cost_eur":     round(r.cost_eur, 7),
            "duration_s":   r.duration_s,
        }
        for r in all_calls
    ]

    n_errors = sum(1 for d in detail if d["erreur"])

    # Affichage console avec accord
    eta = stats.eta_str(phrase_num, total_phrases)
    err_flag = f"  {n_errors} err" if n_errors else ""
    agree_display = " | ".join([f"{qk}:{phrase_agreement.get(qk, 0.5)*100:.0f}%" for qk in QUESTION_KEYS[:3]]) + "..."
    print(f"  [{phrase_num:>4}/{total_phrases}] "
          f"'{phrase_obj.phrase[:55]:<55}'  "
          f"{label:<12}  "
          f"R:{roi_s} N:{not_s} O:{obl_s}  "
          f"EUR{phrase_cost_eur:.6f}  "
          f"{phrase_duration:.1f}s  "
          f"[Accord: {agree_display}]  "
          f"{eta}{err_flag}")

    stats.phrase_count += 1

    return FinalClassification(
        phrase=phrase_obj.phrase,
        phrase_index=phrase_obj.phrase_index,
        source_file=phrase_obj.source_file,
        source_folder=phrase_obj.source_folder,
        categorie=label,
        scores_roi=roi_s,
        scores_notoriete=not_s,
        scores_obligation=obl_s,
        reponses_questions=reponses_aggregees,
        agreement_rates=phrase_agreement,
        cost_usd_phrase=round(phrase_cost_usd, 7),
        cost_eur_phrase=round(phrase_cost_eur, 7),
        duration_s_phrase=round(phrase_duration, 2),
        detail_appels=detail,
        ensemble_modeles=models,
    )


# ── Traitement d'un fichier ────────────────────────────────────────────────────
def process_file(
    seg_path: Path,
    models: list,
    stats: StatsAccumulator,
) -> list[FinalClassification]:

    with open(seg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phrases = [
        Phrase(
            phrase=item["phrase"],
            phrase_index=item["phrase_index"],
            source_file=item.get("source_file", seg_path.name),
            source_folder=item.get("source_folder", ""),
        )
        for item in data
        if item.get("phrase", "").strip()
    ]

    n_phrases = len(phrases)
    n_calls   = n_phrases * len(QUESTION_KEYS) * len(models)
    print(f"\n   {seg_path.name}")
    print(f"   {n_phrases} phrases  |  "
          f"{len(QUESTION_KEYS)} questions  |  "
          f"{len(models)} modeles  |  "
          f"~{n_calls} appels\n")

    results = []
    for i, phrase_obj in enumerate(phrases, 1):
        result = process_phrase(phrase_obj, models, stats, i, n_phrases)
        results.append(result)

    return results


# ── Export fichier (JSON + Excel) ─────────────────────────────────────────────
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
            "Source_File":      r.source_file,
            "Phrase_Index":     r.phrase_index,
            "Phrase":           r.phrase,
            "Category":         r.categorie,
            "ROI_Score":        f"{r.scores_roi}/{len(ROI_KEYS)}",
            "Notoriete_Score":  f"{r.scores_notoriete}/{len(NOT_KEYS)}",
            "Obligation_Score": f"{r.scores_obligation}/{len(OBL_KEYS)}",
            "Cout_EUR_Phrase":  r.cost_eur_phrase,
            "Duree_s":          r.duration_s_phrase,
        }
        
        # Taux d'accord par question pour cette phrase
        for qk in QUESTION_KEYS:
            row[f"ACCORD_{qk}_%"] = round(r.agreement_rates.get(qk, 0.5) * 100, 1)
        
        # Reponses agreegees par question
        for qk in QUESTION_KEYS:
            row[f"AGG_{qk}"] = r.reponses_questions.get(qk, "?")

        # Reponses detaillees par modele x question
        for model in models:
            short = model.split("/")[-1][:20]
            for qk in QUESTION_KEYS:
                match = next(
                    (d for d in r.detail_appels
                     if d["model"] == short and d["question_key"] == qk),
                    None,
                )
                if match:
                    row[f"{short}__{qk}"] = (
                        match["reponse"] if not match["erreur"] else "ERR"
                    )
                else:
                    row[f"{short}__{qk}"] = "?"

        rows.append(row)

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(xl_out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Classification", index=False)

    print(f"\n   Sauvegarde -> {json_out.name}")
    print(f"   Sauvegarde -> {xl_out.name}")


# ── Export global complet (Excel + NPY) ───────────────────────────────────────
def export_complete_summary(
    all_results: list[FinalClassification],
    models: list,
    timestamp: str,
    stats: StatsAccumulator,
    output_dir: Path
):
    """
    Exporte un fichier Excel recapitulatif complet avec :
    - Toutes les phrases avec leurs classifications
    - Reponses par question et par modele
    - Taux d'accord par phrase
    - Couts et durees
    - Metriques globales
    """
    
    rows = []
    for r in all_results:
        row = {
            "Source_File": r.source_file,
            "Source_Folder": r.source_folder,
            "Phrase_Index": r.phrase_index,
            "Phrase_Text": r.phrase,
            "Category": r.categorie,
            "ROI_Score": f"{r.scores_roi}/{len(ROI_KEYS)}",
            "Notoriete_Score": f"{r.scores_notoriete}/{len(NOT_KEYS)}",
            "Obligation_Score": f"{r.scores_obligation}/{len(OBL_KEYS)}",
            "Cost_USD": round(r.cost_usd_phrase, 7),
            "Cost_EUR": round(r.cost_eur_phrase, 7),
            "Duration_Seconds": r.duration_s_phrase,
        }
        
        # Taux d'accord par question pour cette phrase
        for qk in QUESTION_KEYS:
            row[f"ACCORD_{qk}_%"] = round(r.agreement_rates.get(qk, 0.5) * 100, 1)
        
        # Reponses agreegees par question
        for qk in QUESTION_KEYS:
            row[f"AGG_{qk}"] = r.reponses_questions.get(qk, "?")
        
        # Reponses par modele pour chaque question
        for model in models:
            short_name = model.split("/")[-1][:20]
            for qk in QUESTION_KEYS:
                match = next(
                    (d for d in r.detail_appels
                     if d["model"] == short_name and d["question_key"] == qk),
                    None,
                )
                if match:
                    col_name = f"{short_name}_{qk}"
                    row[col_name] = match["reponse"] if not match["erreur"] else "ERR"
                    row[f"{col_name}_cost_eur"] = round(match.get("cost_eur", 0), 7)
                else:
                    row[f"{short_name}_{qk}"] = "?"
        
        rows.append(row)
    
    df_main = pd.DataFrame(rows)
    
    # Synthese par fichier source
    summary_by_file = []
    for file_name, grp in df_main.groupby("Source_File"):
        summary_by_file.append({
            "Fichier": file_name,
            "Total_Phrases": len(grp),
            "ROI": (grp.Category == "ROI").sum(),
            "Notoriete": (grp.Category == "Notoriete").sum(),
            "Obligation": (grp.Category == "Obligation").sum(),
            "Description": (grp.Category == "Description").sum(),
            "Cout_Total_EUR": grp["Cost_EUR"].sum(),
            "Cout_Moyen_EUR_Phrase": grp["Cost_EUR"].mean(),
            "Duree_Totale_sec": grp["Duration_Seconds"].sum(),
            "Duree_Moyenne_sec_Phrase": grp["Duration_Seconds"].mean(),
            "Accord_Moyen_ROI_%": grp["ACCORD_roi_1_%"].mean() if "ACCORD_roi_1_%" in grp.columns else 0,
            "Accord_Moyen_Notoriete_%": grp["ACCORD_notoriete_1_%"].mean() if "ACCORD_notoriete_1_%" in grp.columns else 0,
            "Accord_Moyen_Obligation_%": grp["ACCORD_obl_1_%"].mean() if "ACCORD_obl_1_%" in grp.columns else 0,
        })
    df_summary_file = pd.DataFrame(summary_by_file)
    
    # Synthese par modele
    rows_models = []
    for m in models:
        short = m.split("/")[-1][:30]
        rows_models.append({
            "Modele": short,
            "Modele_complet": m,
            "Tokens_Input": stats.tokens_in_by_model.get(m, 0),
            "Tokens_Output": stats.tokens_out_by_model.get(m, 0),
            "Cout_USD": round(stats.cost_usd_by_model.get(m, 0), 6),
            "Cout_EUR": round(usd_to_eur(stats.cost_usd_by_model.get(m, 0)), 6),
            "Nb_Appels": stats.calls_by_model.get(m, 0),
            "Nb_Erreurs": stats.errors_by_model.get(m, 0),
            "Taux_Erreur_%": round(stats.error_rate(m) * 100, 2) if stats.calls_by_model.get(m, 0) > 0 else 0,
        })
    df_models = pd.DataFrame(rows_models)
    
    # Taux d'accord par question (global)
    rows_agreement = []
    for qk in QUESTION_KEYS:
        rows_agreement.append({
            "Question_Key": qk,
            "Categorie": get_question_display(qk),
            "Texte_Question": get_question_text(qk),
            "Taux_Accord_%": round(stats.agreement_rate(qk) * 100, 2),
            "Nb_Comparaisons": stats.total_votes.get(qk, 0),
        })
    df_agreement = pd.DataFrame(rows_agreement)
    
    # Distribution des categories
    category_counts = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}
    df_category = pd.DataFrame([{
        "Categorie": c,
        "Nombre": category_counts[c],
        "Pourcentage_%": round(category_counts[c] / len(all_results) * 100, 1) if all_results else 0
    } for c in CATEGORIES])
    
    # Statistiques globales
    total_phrases = len(all_results)
    total_cost_eur = stats.total_cost_eur
    total_duration_min = stats.elapsed_s / 60
    
    df_global = pd.DataFrame([{
        "Timestamp": timestamp,
        "Total_Phrases": total_phrases,
        "Total_Cout_EUR": round(total_cost_eur, 6),
        "Cout_Moyen_EUR_Phrase": round(total_cost_eur / total_phrases, 7) if total_phrases > 0 else 0,
        "Duree_Totale_min": round(total_duration_min, 2),
        "Duree_Moyenne_sec_Phrase": round(stats.elapsed_s / total_phrases, 2) if total_phrases > 0 else 0,
        "Taux_EUR_USD": EUR_USD_RATE,
        "Nb_Modeles": len(models),
        "Nb_Questions": len(QUESTION_KEYS),
        "Modeles_Utilises": " | ".join(m.split("/")[-1] for m in models),
    }])
    
    # Accord par phrase (detail)
    df_agreement_phrases = pd.DataFrame([{
        "Source_File": r.source_file,
        "Phrase_Index": r.phrase_index,
        "Phrase_Text": r.phrase[:100],
        **{f"Accord_{qk}_%": round(r.agreement_rates.get(qk, 0.5) * 100, 1) for qk in QUESTION_KEYS}
    } for r in all_results])
    
    # Export Excel
    excel_path = output_dir / f"recapitulatif_complet_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="Toutes_Phrases", index=False)
        df_summary_file.to_excel(writer, sheet_name="Synthese_Par_Fichier", index=False)
        df_models.to_excel(writer, sheet_name="Stats_Par_Model", index=False)
        df_agreement.to_excel(writer, sheet_name="Accord_Questions_Global", index=False)
        df_category.to_excel(writer, sheet_name="Distribution_Categories", index=False)
        df_global.to_excel(writer, sheet_name="Stats_Globales", index=False)
        df_agreement_phrases.to_excel(writer, sheet_name="Accord_Par_Phrase", index=False)
    
    print(f"\n   Export Excel complet -> {excel_path.name}")
    
    # Export Tensor NPY
    n_phrases = len(all_results)
    n_models = len(models)
    n_questions = len(QUESTION_KEYS)
    
    tensor = np.zeros((n_phrases, n_models, n_questions), dtype=np.int8)
    model_short_names = [m.split("/")[-1][:20] for m in models]
    
    for i, result in enumerate(all_results):
        for mi, model_name in enumerate(model_short_names):
            for qi, qk in enumerate(QUESTION_KEYS):
                match = next(
                    (d for d in result.detail_appels
                     if d["model"] == model_name and d["question_key"] == qk),
                    None
                )
                if match and not match.get("erreur"):
                    tensor[i, mi, qi] = 1 if match["reponse"] == "oui" else 0
                else:
                    tensor[i, mi, qi] = 0
    
    npy_path = output_dir / f"tensor_complet_{timestamp}.npy"
    np.save(npy_path, tensor)
    
    meta = {
        "shape": list(tensor.shape),
        "dtype": "int8",
        "axes": ["phrase", "model", "question"],
        "values": {"1": "oui", "0": "non"},
        "models": model_short_names,
        "questions": QUESTION_KEYS,
        "questions_text": {k: get_question_text(k) for k in QUESTION_KEYS},
        "n_phrases": n_phrases,
        "n_models": n_models,
        "n_questions": n_questions,
        "timestamp": timestamp,
    }
    
    meta_path = output_dir / f"tensor_complet_{timestamp}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"   Tensor NPY -> {npy_path.name} shape={tensor.shape}")
    print(f"   Metadonnees -> {meta_path.name}")
    
    return excel_path, npy_path


# ── Export global ──────────────────────────────────────────────────────────────
def export_global_summary(
    all_results: list[FinalClassification],
    models: list,
    timestamp: str,
    stats: StatsAccumulator,
):
    total = len(all_results)
    cats  = {c: sum(1 for r in all_results if r.categorie == c) for c in CATEGORIES}

    rows = [{
        "Source_Folder":    r.source_folder,
        "Source_File":      r.source_file,
        "Phrase_Index":     r.phrase_index,
        "Phrase":           r.phrase,
        "Category":         r.categorie,
        "ROI_Score":        f"{r.scores_roi}/{len(ROI_KEYS)}",
        "Notoriete_Score":  f"{r.scores_notoriete}/{len(NOT_KEYS)}",
        "OBL_Score":        f"{r.scores_obligation}/{len(OBL_KEYS)}",
        "Cout_EUR_Phrase":  r.cost_eur_phrase,
        **{f"AGG_{qk}": r.reponses_questions.get(qk, "?") for qk in QUESTION_KEYS},
    } for r in all_results]
    df = pd.DataFrame(rows)

    # Synthese par fichier
    summary = []
    for key, grp in df.groupby("Source_File"):
        summary.append({
            "Fichier": key,
            "Total":   len(grp),
            **{c: (grp.Category == c).sum() for c in CATEGORIES},
            "Cout_EUR": round(
                usd_to_eur(stats.cost_usd_by_file.get(key, 0.0)), 4
            ),
        })
    df_sum = pd.DataFrame(summary)

    # Stats modeles
    df_models = pd.DataFrame([{
        "Modele":      m.split("/")[-1],
        "Appels":      stats.calls_by_model[m],
        "Erreurs":     stats.errors_by_model[m],
        "Taux_Err_%":  round(stats.error_rate(m) * 100, 2),
        "Tokens_In":   stats.tokens_in_by_model[m],
        "Tokens_Out":  stats.tokens_out_by_model[m],
        "Cout_USD":    round(stats.cost_usd_by_model[m], 6),
        "Cout_EUR":    round(usd_to_eur(stats.cost_usd_by_model[m]), 6),
    } for m in models])

    # Accord questions
    df_q = pd.DataFrame([{
        "Question_Key": qk,
        "Categorie":    get_question_display(qk),
        "Texte":        get_question_text(qk),
        "Accord_%":     round(stats.agreement_rate(qk) * 100, 2),
        "N_votes":      stats.total_votes.get(qk, 0),
    } for qk in QUESTION_KEYS])

    # Globale
    n = stats.phrase_count
    df_glob = pd.DataFrame([{
        "Timestamp":       timestamp,
        "Modeles":         " | ".join(m.split("/")[-1] for m in models),
        "Nb_Questions":    len(QUESTION_KEYS),
        "Total_Phrases":   total,
        "Cout_Total_USD":  round(stats.total_cost_usd, 6),
        "Cout_Total_EUR":  round(usd_to_eur(stats.total_cost_usd), 6),
        "Cout_Phrase_EUR": round(usd_to_eur(stats.total_cost_usd) / n, 7) if n else 0,
        "Duree_min":       round(stats.elapsed_s / 60, 2),
        **{f"Total_{c}": cats[c] for c in CATEGORIES},
        **{f"Pct_{c}": f"{round(cats[c]/total*100,1)}%" if total else "0%"
           for c in CATEGORIES},
    }])

    xl_path = Path(OUTPUT_DIR) / f"synthese_globale_{timestamp}.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        df.to_excel(writer,        sheet_name="Toutes_Phrases",       index=False)
        df_sum.to_excel(writer,    sheet_name="Synthese_Par_Fichier", index=False)
        df_models.to_excel(writer, sheet_name="Stats_Modeles",        index=False)
        df_q.to_excel(writer,      sheet_name="Accord_Questions",     index=False)
        df_glob.to_excel(writer,   sheet_name="Stats_Globales",       index=False)

    print(f"\n   Synthese globale -> {xl_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(models: list = None):
    if models is None:
        models = MODELS_TO_USE

    stats = StatsAccumulator(models)

    print("=" * 80)
    print("CLASSIFICATION  —  1 appel = 1 question x 1 phrase  (v5)")
    print("=" * 80)
    print(f"  System Prompt  : {SYSTEM_PROMPT_FILE}")
    print(f"  User Prompt    : {USER_PROMPT_FILE}")
    print(f"  Questions file : {QUESTIONS_FILE}")
    print(f"  {len(QUESTION_KEYS)} questions  |  {len(models)} modeles")
    print(f"  Appels par phrase : {len(QUESTION_KEYS) * len(models)}")
    for m in models:
        p = MODEL_PRICING.get(m, DEFAULT_PRICING)
        print(f"  * {m}  in:${p['input']}/M  out:${p['output']}/M")
    print(f"\n  Priorite : {' > '.join(PRIORITY_ORDER)}")
    print(f"  Delai inter-appels : {INTER_CALL_DELAY}s")
    print()

    seg_root  = Path(SEGMENTS_INPUT_DIR)
    seg_files = sorted(seg_root.rglob("*.json"))
    if not seg_files:
        print("[ERREUR] Aucun .json trouve")
        return

    print(f"  {len(seg_files)} fichier(s) trouve(s)\n")

    all_results         = []
    ok, skipped, errors = 0, 0, 0

    for i, seg_path in enumerate(seg_files, 1):
        rel       = seg_path.relative_to(seg_root)
        dest_dir  = seg_path.parent
        base_name = seg_path.stem
        json_out  = dest_dir / f"{base_name}_classification.json"

        print(f"\n{'='*60}")
        print(f"[{i}/{len(seg_files)}] {rel}")
        print(f"{'='*60}")

        if json_out.exists():
            print(f"   Skip — {json_out.name} existe deja.")
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
            results = process_file(seg_path, models, stats)
            export_file_results(results, dest_dir, base_name, models)
            all_results.extend(results)
            ok += 1
            file_eur = usd_to_eur(
                stats.cost_usd_by_file.get(
                    results[0].source_file if results else "", 0.0
                )
            )
            print(f"\n   Fichier termine : {len(results)} phrases  EUR{file_eur:.4f}")
        except Exception as e:
            import traceback
            print(f"   ERREUR : {e}")
            traceback.print_exc()
            errors += 1

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
    print(f"  Traites:{ok}  Skipes:{skipped}  Erreurs:{errors}  "
          f"Fichiers:{len(seg_files)}")
    print(f"  Phrases : {total}")
    for c in CATEGORIES:
        pct = round(cats[c] / total * 100, 1) if total else 0
        print(f"  {c:<12} {cats[c]:>4}  ({pct}%)")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        import traceback
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()