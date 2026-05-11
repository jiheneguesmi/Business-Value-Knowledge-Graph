"""
script4_phrases.py — Découpage Markdown → Phrases plates
═════════════════════════════════════════════════════════

Sortie JSON par fichier : liste plate de phrases, chaque élément = {
    "phrase":        str,
    "phrase_index":  int,
    "source_file":   str,
    "source_folder": str,
}

Threshold unique MIN_MERGE_CHARS (défaut 80) :
  • Phrases spaCy < 80 chars → fusionnées avec la suivante (sauf séparateur structurel).
  • Titres et bullets : émis tels quels (atomiques), jamais filtrés par la longueur.
  • Seul le bruit pur est ignoré : URLs seules, séparateurs horizontaux, chiffres/ponctuation seuls.

Installation :
  pip install spacy langdetect
  python -m spacy download fr_core_news_sm
  python -m spacy download en_core_web_sm
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import spacy
from langdetect import LangDetectException, detect

# ── Configuration ──────────────────────────────────────────────────────────────
CLEAN_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\decoupage\clean_markdown"
PHRASES_OUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\decoupage\phrases"

# Threshold unique : en dessous de cette longueur, une phrase spaCy est courte.
MIN_MERGE_CHARS = 80

# ── Patterns Markdown ─────────────────────────────────────────────────────────
_RE_HEADING    = re.compile(r"^(#{1,6})\s+(.+)")
_RE_BULLET     = re.compile(r"^[-*+]\s+(.+)")
_RE_NUM_LIST   = re.compile(r"^\d+\.\s+(.+)")
_RE_TABLE      = re.compile(r"^\|")
_RE_IMAGE      = re.compile(r"^!\[")
_RE_HORIZONTAL = re.compile(r"^[-*_]{3,}\s*$")
_RE_URL_ONLY   = re.compile(r"^https?://\S+$|^www\.\S+$")
_RE_PURE_PUNCT = re.compile(r"^[\d\s.\-:|/\\+()]+$")
_RE_STRUCT_SEP = re.compile(
    r"(\n\s*\n|\n\s*[-*+] |\n\s*\d+\. |\n\s*#{1,6} |\n\s*> )"
)

# ── Modèles spaCy ─────────────────────────────────────────────────────────────
_NLP: dict = {}

def _load_models() -> None:
    for lang, name in [("fr", "fr_core_news_sm"), ("en", "en_core_web_sm")]:
        try:
            m = spacy.load(name)
            m.max_length = 2_000_000
            _NLP[lang] = m
            print(f"  [OK] spaCy {name}")
        except OSError:
            print(f"  [MISSING] {name} — python -m spacy download {name}")
    if not _NLP:
        raise SystemExit("[ERREUR] Aucun modèle spaCy disponible.")

def _get_nlp(lang: str):
    return _NLP.get(lang) or next(iter(_NLP.values()))

# ── Utilitaires ───────────────────────────────────────────────────────────────

def _detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in _NLP else next(iter(_NLP))
    except LangDetectException:
        return next(iter(_NLP))

def _fingerprint(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower().strip()).encode()).hexdigest()

def _clean_line(line: str) -> str:
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", line)
    line = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", line)
    line = re.sub(r"`[^`]+`", "", line)
    return line.strip()

def _is_noise(text: str) -> bool:
    """Bruit pur uniquement : URL seule, séparateur horizontal, chiffres/ponctuation seuls, vide."""
    s = text.strip()
    if not s:                   return True
    if _RE_URL_ONLY.match(s):   return True
    if _RE_HORIZONTAL.match(s): return True
    if _RE_PURE_PUNCT.match(s): return True
    return False

# ── Découpage spaCy avec fusion des phrases courtes ───────────────────────────

def _has_struct_sep(text: str, a: str, b: str) -> bool:
    try:
        pa = text.index(a)
        pb = text.index(b, pa + len(a))
        return bool(_RE_STRUCT_SEP.search(text[pa + len(a):pb]))
    except ValueError:
        return True

def _split_to_sentences(text: str, lang: str) -> list[str]:
    doc = _get_nlp(lang)(text)
    raw = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not raw:
        return [text.strip()] if text.strip() else []
    merged, i = [], 0
    while i < len(raw):
        cur = raw[i]
        while len(cur) < MIN_MERGE_CHARS and i + 1 < len(raw) and not _has_struct_sep(text, cur, raw[i + 1]):
            i += 1
            cur = cur + " " + raw[i]
        merged.append(cur.strip())
        i += 1
    return merged

# ── Tokeniseur Markdown (RÉFÉRENCE — utilisé par les deux scripts) ─────────────
# Tokens : ("title", level:int, text) | ("bullet", text) | ("text", text)
# IMPORTANT : les bullets sont atomiques (un token par bullet),
#             jamais regroupés en bullet_block.

def _iter_tokens(md_text: str) -> list[tuple]:
    result, buffer = [], []
    in_code = in_table = False

    def flush():
        if not buffer:
            return
        block = " ".join(buffer).strip()
        buffer.clear()
        if block and not _is_noise(block):
            result.append(("text", block))

    for line in md_text.split("\n"):
        s = line.strip()

        if s.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue

        if _RE_TABLE.match(s):
            flush()
            in_table = True
            continue
        if in_table:
            if s and (_RE_TABLE.match(s) or re.match(r"^[\|\-:\s]+$", s)):
                continue
            in_table = False

        if _RE_IMAGE.match(s) or _RE_URL_ONLY.match(s):
            continue

        if not s:
            flush()
            continue

        m = _RE_HEADING.match(s)
        if m:
            flush()
            t = _clean_line(m.group(2).strip())
            if t:
                result.append(("title", len(m.group(1)), t))
            continue

        m = _RE_BULLET.match(s) or _RE_NUM_LIST.match(s)
        if m:
            flush()
            t = _clean_line(m.group(1).strip())
            if t:
                result.append(("bullet", t))
            continue

        cl = _clean_line(s)
        if cl:
            buffer.append(cl)

    flush()
    return result

# ── Dataclass ──────────────────────────────────────────────────────────────────
@dataclass
class Phrase:
    phrase:        str
    phrase_index:  int
    source_file:   str
    source_folder: str

# ── Pipeline ───────────────────────────────────────────────────────────────────

def parse_to_phrases(md_text: str, source_file: str, source_folder: str, lang: str) -> list[Phrase]:
    phrases, seen, idx = [], set(), 0

    def emit(text: str) -> None:
        nonlocal idx
        text = text.strip()
        if not text or _is_noise(text):
            return
        fp = _fingerprint(text)
        if fp in seen:
            return
        seen.add(fp)
        phrases.append(Phrase(text, idx, source_file, source_folder))
        idx += 1

    for token in _iter_tokens(md_text):
        kind = token[0]

        if kind == "title":
            title_text = token[2] if token[2].strip() else "Introduction"
            emit(title_text)

        elif kind == "bullet":
            emit(token[1])

        elif kind == "text":
            for sent in _split_to_sentences(token[1], lang):
                emit(sent)

    return phrases

# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 72)
    print("SCRIPT 4 — PHRASES PLATES (liste flat, sans section_title)")
    print(f"  MIN_MERGE_CHARS = {MIN_MERGE_CHARS}")
    print("=" * 72)

    _load_models()

    root_path = Path(CLEAN_INPUT_DIR)
    if not root_path.exists():
        raise SystemExit(f"[ERREUR] Dossier introuvable : {CLEAN_INPUT_DIR}")

    md_files = sorted(root_path.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"[ERREUR] Aucun .md dans {CLEAN_INPUT_DIR}")

    print(f"\n  {len(md_files)} fichier(s) .md trouvé(s)\n")

    total_phrases = 0
    fichiers_ok   = 0
    fichiers_ko   = 0

    for i, md_path in enumerate(md_files, 1):
        try:
            md_text = md_path.read_text(encoding="utf-8")
            lang    = _detect_lang(md_text)
            folder  = str(md_path.parent.relative_to(root_path)) if md_path.parent != root_path else "."

            phrases = parse_to_phrases(md_text, md_path.name, folder, lang)

            out_dir = Path(PHRASES_OUT_DIR) / md_path.parent.relative_to(root_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{md_path.stem}_phrases.json"
            out_path.write_text(
                json.dumps([asdict(p) for p in phrases], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            nb_phrases = len(phrases)
            total_phrases += nb_phrases
            fichiers_ok   += 1

            print(f"[{i}/{len(md_files)}] {md_path.name} → {nb_phrases} phrases")

        except Exception as e:
            print(f"[{i}/{len(md_files)}] ERREUR sur {md_path.name}: {e}")
            fichiers_ko += 1

    print("\n" + "=" * 72)
    print("RÉSUMÉ PHRASES PLATES")
    print("=" * 72)
    print(f"  Fichiers OK   : {fichiers_ok}/{len(md_files)}  |  KO : {fichiers_ko}")
    print(f"  TOTAL PHRASES : {total_phrases}")
    print(f"  Sortie        : {PHRASES_OUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur.")
    except Exception as e:
        import traceback
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()