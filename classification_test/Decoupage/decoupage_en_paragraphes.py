"""
Script : Découpage Markdown → Paragraphes → Phrases

Format JSON de sortie :
{
  "paragraph_index": 0,
  "section_title":   "Introduction",
  "source_file":     "document.md",
  "source_folder":   "client1",
  "phrases": [
    {"phrase_index": 0, "phrase": "phrase 1"},
    {"phrase_index": 1, "phrase": "phrase 2"}
  ]
}

RÈGLES :
- Un titre qui a du contenu (phrases) → paragraphe normal avec ses phrases.
- Un titre qui n'a PAS de contenu    → segment conservé avec phrases = [].
- Le titre n'est PAS répété dans les phrases du paragraphe.

COMPTAGE 
  Pour chaque paragraphe :
    • section_title non vide  → +1  
    • chaque phrase de contenu → +1
  ⇒ total_phrases_global 
"""

import os
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

import spacy
from langdetect import LangDetectException, detect

# ── Configuration ──────────────────────────────────────────────────────────────
MARKDOWN_ROOT       = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\decoupage\clean_markdown"
SEGMENTS_OUTPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\decoupage\paragraphes"

MIN_MERGE_CHARS = 80   # identique au script 4

os.makedirs(SEGMENTS_OUTPUT_DIR, exist_ok=True)

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
            print(f"  [MISSING] {name}")
    if not _NLP:
        raise SystemExit("[ERREUR] Aucun modèle spaCy disponible.")

def _get_nlp(lang: str):
    return _NLP.get(lang) or next(iter(_NLP.values()))

def _detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in _NLP else next(iter(_NLP))
    except LangDetectException:
        return next(iter(_NLP))


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

# ── Patterns ───────────────────────────────────────────────────────────────────
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

# ── Fonctions de nettoyage ────────────────────────────────────────────────────
def _clean_line(line: str) -> str:
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", line)
    line = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", line)
    line = re.sub(r"`[^`]+`", "", line)
    return line.strip()

def _is_noise(text: str) -> bool:
    s = text.strip()
    if not s:                   return True
    if _RE_URL_ONLY.match(s):   return True
    if _RE_HORIZONTAL.match(s): return True
    if _RE_PURE_PUNCT.match(s): return True
    return False

def fingerprint(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

# ── Découpage spaCy ───────────────────────────────────────────────────────────
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
        while (
            len(cur) < MIN_MERGE_CHARS
            and i + 1 < len(raw)
            and not _has_struct_sep(text, cur, raw[i + 1])
        ):
            i += 1
            cur = cur + " " + raw[i]
        merged.append(cur.strip())
        i += 1
    return merged

# ── Tokeniseur Markdown (IDENTIQUE au script 4) ───────────────────────────────
# Tokens : ("title", level:int, text) | ("bullet", text) | ("text", text)
# Les bullets sont ATOMIQUES — un token ("bullet", text) par ligne de bullet.
# Pas de bullet_block. Garantit le même ordre de tokens que le script 4.

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

# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class PhraseItem:
    phrase_index: int
    phrase:       str

@dataclass
class Paragraph:
    paragraph_index: int
    section_title:   str
    source_file:     str
    source_folder:   str
    phrases:         list[dict]

# ── Parser principal ──────────────────────────────────────────────────────────
def parse_markdown(
    markdown_text: str,
    source_file:   str,
    source_folder: str,
    lang:          str,
) -> list[Paragraph]:
    """
    Retourne la liste des paragraphes.

    Logique d'émission alignée sur le script 4 :
      - Un token "title"  → ouvre un nouveau paragraphe (section_title = texte du titre).
                            Le titre lui-même N'EST PAS ajouté aux phrases du paragraphe
                            (il sera compté séparément dans le résumé, comme emit(title)
                            le fait dans le script 4).
      - Un token "bullet" → ajouté aux phrases du paragraphe courant (atomique).
      - Un token "text"   → découpé en phrases spaCy, ajoutées au paragraphe courant.

    Titres sans contenu : conservés avec phrases = [].
    Déduplication phrase par phrase via fingerprint (set seen global au fichier).
    """
    tokens = _iter_tokens(markdown_text)

    paragraphs:  list[Paragraph] = []
    seen:        set[str]        = set()
    para_idx                     = 0
    current_section              = ""
    current_phrases: list[str]   = []
    current_has_content          = False  # au moins un token non-titre reçu

    def emit_phrase(text: str) -> bool:
        nonlocal current_phrases, current_has_content
        phrase = text.strip()
        if not phrase or _is_noise(phrase):
            return False
        fp = fingerprint(phrase)
        if fp in seen:
            return False
        seen.add(fp)
        current_phrases.append(phrase)
        current_has_content = True
        return True

    def flush_paragraph():
        """Valide et enregistre le paragraphe courant s'il a des phrases uniques."""
        nonlocal para_idx, current_phrases, current_has_content

        if not current_phrases:
            return

        if current_section.strip():
            title_fp = fingerprint(current_section)
            if title_fp not in seen:
                seen.add(title_fp)

        paragraphs.append(Paragraph(
            paragraph_index=para_idx,
            section_title=current_section,
            source_file=source_file,
            source_folder=source_folder,
            phrases=[asdict(PhraseItem(phrase_index=j, phrase=p)) for j, p in enumerate(current_phrases)],
        ))
        para_idx += 1

        current_phrases.clear()
        current_has_content = False

    def flush_title_only():
        """
        Enregistre un paragraphe vide pour un titre sans contenu,
        UNIQUEMENT si le titre n'a pas déjà été vu (déduplication).
        """
        nonlocal para_idx

        if not current_section.strip():
            return
        if current_has_content:
            # Il y a du contenu → flush_paragraph() s'en chargera
            return

        fp = fingerprint(current_section)
        if fp not in seen:
            seen.add(fp)
            paragraphs.append(Paragraph(
                paragraph_index=para_idx,
                section_title=current_section,
                source_file=source_file,
                source_folder=source_folder,
                phrases=[],
            ))
            para_idx += 1

    for token in tokens:
        kind = token[0]

        if kind == "title":
            title_text = token[2].strip()
            if _is_noise(title_text):
                # Titre purement numérique / bruitux : ne pas le compter.
                if current_has_content:
                    flush_paragraph()
                else:
                    flush_title_only()
                current_section = ""
                current_phrases = []
                current_has_content = False
                continue

            # — Fermer le paragraphe précédent —
            if current_has_content:
                flush_paragraph()
            else:
                flush_title_only()

            # — Ouvrir le nouveau paragraphe —
            current_section     = title_text
            current_phrases     = []
            current_has_content = False

        elif kind == "bullet":
            emit_phrase(token[1])

        elif kind == "text":
            for sent in _split_to_sentences(token[1], lang):
                emit_phrase(sent)

    # — Dernier paragraphe —
    if current_has_content:
        flush_paragraph()
    else:
        flush_title_only()

    return paragraphs


# ── Comptage aligné sur le script 4 ──────────────────────────────────────────
def _count_phrases(paragraphs: list[Paragraph]) -> int:
    """
    Compte le nombre de phrases équivalent au script 4, en s'appuyant sur
    la même logique de déduplication globale que le script de phrases plates.

    - section_title non vide  → +1 si le titre n'a pas déjà été vu comme phrase.
    - chaque phrase de contenu → +1 si elle n'a pas déjà été vue.
    """
    total = 0
    seen: set[str] = set()

    for p in paragraphs:
        title = p.section_title.strip()
        if title:
            fp = fingerprint(title)
            if fp not in seen:
                seen.add(fp)
                total += 1

        for phrase in p.phrases:
            fp = fingerprint(phrase["phrase"])
            if fp not in seen:
                seen.add(fp)
                total += 1

    return total


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("=" * 80)
    print("DECOUPAGE MARKDOWN -> PARAGRAPHES -> PHRASES")
    print("(tokeniseur identique au script 4 — comptage aligné)")
    print("=" * 80)

    _load_models()

    root_path = Path(MARKDOWN_ROOT)
    if not root_path.exists():
        print(f"[ERREUR] Dossier introuvable : {MARKDOWN_ROOT}")
        return

    md_files = sorted(root_path.rglob("*.md"))
    if not md_files:
        print(f"[ERREUR] Aucun .md dans {MARKDOWN_ROOT}")
        return

    print(f"  {len(md_files)} fichier(s) .md trouvé(s)\n")

    total_phrases_global = 0
    fichiers_ok          = 0
    fichiers_ko          = 0

    for i, md_path in enumerate(md_files, 1):
        try:
            md_text = md_path.read_text(encoding="utf-8")
            lang    = _detect_lang(md_text)
            folder  = (
                str(md_path.parent.relative_to(root_path))
                if md_path.parent != root_path
                else "."
            )

            paragraphs = parse_markdown(md_text, md_path.name, folder, lang)

            out_dir  = Path(SEGMENTS_OUTPUT_DIR) / md_path.parent.relative_to(root_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{md_path.stem}_paragraphs.json"
            out_path.write_text(
                json.dumps([asdict(p) for p in paragraphs], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            nb_phrases = _count_phrases(paragraphs)
            total_phrases_global += nb_phrases
            fichiers_ok          += 1

            _safe_print(f"[{i}/{len(md_files)}] {md_path.name} → {nb_phrases} phrases")

        except Exception as e:
                _safe_print(f"[{i}/{len(md_files)}] ERREUR sur {md_path.name}: {e}")
                fichiers_ko += 1
    print("\n" + "=" * 80)
    print("RESUME PARAGRAPHES")
    print("=" * 80)
    print(f"  Fichiers traites     : {fichiers_ok}/{len(md_files)}")
    print(f"  Fichiers erreurs     : {fichiers_ko}")
    print(f"  TOTAL PHRASES GLOBAL : {total_phrases_global}")
    print(f"  Sortie               : {SEGMENTS_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        import traceback
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()