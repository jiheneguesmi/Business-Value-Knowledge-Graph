"""
Script : Découpage optimisé des documents Markdown en segments
          avec découpage en phrases via spaCy

Fonctionnalités :
  - Suppression des balises HTML (<span>, <div>, etc.)
  - Fusion des petits paragraphes consécutifs sous le même titre
  - Détection et déduplication des segments répétés
  - Nettoyage complet avant classification
  - Traitement des titres seuls : contexte vide pour éviter redondance
  - Découpage de chaque paragraphe en phrases via spaCy (fr_core_news_sm)
  - Structure JSON : paragraph = {"phrase_1": "...", "phrase_2": "...", ...}

Installation :
  pip install spacy
  python -m spacy download fr_core_news_sm
"""

import os
import re
import json
import hashlib
import spacy
from pathlib import Path
from dataclasses import dataclass, asdict

# ── Configuration ──────────────────────────────────────────────────────────────
MARKDOWN_ROOT       = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_per_client\Docs"
SEGMENTS_OUTPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments-phrases"

MIN_CHARS       = 60    # seuil minimum de caractères pour qu'un segment soit retenu
MERGE_THRESHOLD = 120   # si un paragraphe fait moins de N chars, on le fusionne avec le suivant
MIN_TITLE_CHARS = 30    # seuil minimum pour qu'un titre seul devienne un segment
MIN_SENTENCE_CHARS = 10 # seuil minimum pour qu'une phrase soit retenue

os.makedirs(SEGMENTS_OUTPUT_DIR, exist_ok=True)

# ── Chargement du modèle spaCy (une seule fois) ────────────────────────────────
print("Chargement du modèle spaCy...")
try:
    nlp = spacy.load("fr_core_news_sm")
    nlp.max_length = 2_000_000
    print("Modèle spaCy chargé avec succès.\n")
except OSError:
    raise SystemExit(
        "[ERREUR] Modèle spaCy introuvable.\n"
        "Installez-le avec : python -m spacy download fr_core_news_sm"
    )


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    section_title:   str
    paragraph:       dict   # {"phrase_1": "...", "phrase_2": "..."}
    paragraph_index: int
    source_file:     str
    source_folder:   str


# ── Nettoyage ──────────────────────────────────────────────────────────────────

# Patterns compilés une seule fois
_RE_HTML_TAG      = re.compile(r'<[^>]+>')
_RE_HTML_ENTITY   = re.compile(r'&[a-z]+;|&#\d+;')
_RE_MD_IMAGE      = re.compile(r'!\[.*?\]\(.*?\)')
_RE_MD_LINK       = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
_RE_MD_BOLD_IT    = re.compile(r'\*{1,3}(.+?)\*{1,3}')
_RE_MD_BACKTICK   = re.compile(r'`(.+?)`')
_RE_MD_HEADING    = re.compile(r'^#{1,6}\s+')
_RE_MULTI_SPACES  = re.compile(r'[ \t]{2,}')
_RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
_RE_SPAN_ID       = re.compile(r'<span\s+id="[^"]*"[^>]*>.*?</span>', re.IGNORECASE)
_RE_HORIZONTAL    = re.compile(r'^[-=_\*]{3,}$')


def clean_line(line: str) -> str:
    """Nettoie une ligne Markdown : supprime HTML, balises, formatage."""
    line = _RE_SPAN_ID.sub('', line)
    line = _RE_HTML_TAG.sub('', line)
    line = _RE_HTML_ENTITY.sub(' ', line)
    line = _RE_MD_IMAGE.sub('', line)
    line = _RE_MD_LINK.sub(r'\1', line)
    line = _RE_MD_BOLD_IT.sub(r'\1', line)
    line = _RE_MD_BACKTICK.sub(r'\1', line)
    line = _RE_MD_HEADING.sub('', line)
    line = _RE_MULTI_SPACES.sub(' ', line)
    return line.strip()


def is_table_line(line: str) -> bool:
    """Détecte une ligne de tableau Markdown."""
    stripped = line.strip()
    if stripped.startswith('|'):
        return True
    if re.match(r'^[\|\-\:\s]+$', stripped) and '-' in stripped:
        return True
    return False


def is_structural_only(text: str) -> bool:
    """
    Retourne True si le texte ne contient que des éléments structurels
    sans valeur sémantique.
    """
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return True
    if re.match(r'^[\d\s\.\-\:\|\/\\]+$', stripped):
        return True
    if _RE_HORIZONTAL.match(stripped):
        return True
    return False


def fingerprint(text: str) -> str:
    """Empreinte MD5 du texte normalisé pour déduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# ── Découpage en phrases via spaCy ─────────────────────────────────────────────

def split_into_sentences(text: str) -> dict:
    """
    Découpe un texte en phrases via spaCy.
    Retourne un dict {"phrase_1": "...", "phrase_2": "...", ...}.
    Les phrases trop courtes (< MIN_SENTENCE_CHARS) sont ignorées.
    Si aucune phrase valide n'est détectée, retourne le texte entier en phrase_1.
    """
    doc = nlp(text)
    phrases = [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip() and len(sent.text.strip()) >= MIN_SENTENCE_CHARS
    ]

    # Fallback : si spaCy ne détecte rien, garder le texte entier
    if not phrases:
        phrases = [text.strip()]

    return {f"phrase_{i + 1}": phrase for i, phrase in enumerate(phrases)}


# ── Parser principal ───────────────────────────────────────────────────────────

def parse_markdown_to_segments(
    markdown_text: str,
    source_file: str,
    source_folder: str,
) -> list[Segment]:
    """
    Découpe le Markdown en segments propres.
    Chaque segment contient un paragraphe découpé en phrases via spaCy :
      paragraph = {"phrase_1": "...", "phrase_2": "...", ...}
    """
    lines = markdown_text.split("\n")

    # ── Passe 1 : collecter les blocs bruts ───────────────────────────────────
    sections: list[tuple[str, str]] = []   # (titre_section, texte_paragraphe)
    current_title = "Introduction"
    buffer: list[str] = []
    in_code_block = False
    in_table = False
    last_title = None

    def flush(title: str, buf: list[str]) -> None:
        raw = "\n".join(buf).strip()
        cleaned = clean_line(raw) if '\n' not in raw else "\n".join(
            cl for l in buf if (cl := clean_line(l))
        )
        if cleaned and not is_structural_only(cleaned):
            sections.append((title, cleaned))

    def flush_title_as_segment(title: str) -> None:
        """
        Crée un segment à partir d'un titre seul.
        section_title = "" (vide), paragraph traité comme texte normal.
        """
        if title and len(title) >= MIN_TITLE_CHARS:
            sections.append(("", title))

    for line in lines:
        stripped = line.strip()

        # Blocs de code
        if stripped.startswith('```'):
            if in_code_block:
                in_code_block = False
            else:
                flush(current_title, buffer)
                buffer = []
                in_code_block = True
            continue
        if in_code_block:
            continue

        # Tableaux
        if is_table_line(stripped):
            if not in_table:
                flush(current_title, buffer)
                buffer = []
                in_table = True
            continue
        else:
            in_table = False

        # Images standalone
        if stripped.startswith('!['):
            continue

        # Lignes horizontales
        if _RE_HORIZONTAL.match(stripped):
            continue

        # Titres
        heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if heading_match:
            flush(current_title, buffer)
            buffer = []

            raw_title = heading_match.group(2).strip()
            cleaned_title = clean_line(raw_title)

            if last_title is not None:
                flush_title_as_segment(last_title)

            current_title = cleaned_title if cleaned_title else current_title
            last_title = current_title
            continue

        # Ligne vide → séparateur de paragraphe
        if stripped == "":
            flush(current_title, buffer)
            buffer = []
            last_title = None
            continue

        # Ligne normale (contenu)
        cleaned = clean_line(line)
        if cleaned:
            buffer.append(cleaned)
            last_title = None

    # Flush final du buffer
    flush(current_title, buffer)

    if last_title is not None:
        flush_title_as_segment(last_title)

    # ── Passe 2 : fusion des paragraphes trop courts ──────────────────────────
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(sections):
        title, text = sections[i]
        while (
            len(text) < MERGE_THRESHOLD
            and i + 1 < len(sections)
            and sections[i + 1][0] == title
        ):
            i += 1
            text = text + " " + sections[i][1]
        merged.append((title, text.strip()))
        i += 1

    # ── Passe 3 : déduplication + découpage en phrases ────────────────────────
    seen: set[str] = set()
    segments: list[Segment] = []
    para_idx = 0

    for title, text in merged:
        fp = fingerprint(text)
        if fp in seen:
            continue
        seen.add(fp)
        if is_structural_only(text):
            continue

        # Découpage en phrases via spaCy → dict {"phrase_1": ..., "phrase_2": ...}
        paragraph_dict = split_into_sentences(text)

        segments.append(Segment(
            section_title=title,
            paragraph=paragraph_dict,
            paragraph_index=para_idx,
            source_file=source_file,
            source_folder=source_folder,
        ))
        para_idx += 1

    return segments


# ── Découverte des fichiers ────────────────────────────────────────────────────

def find_all_markdown_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("*.md"))


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 80)
    print("DÉCOUPAGE MARKDOWN OPTIMISÉ — PHRASES VIA SPACY")
    print("=" * 80)

    root_path = Path(MARKDOWN_ROOT)
    if not root_path.exists():
        print(f"[ERREUR] Dossier introuvable : {MARKDOWN_ROOT}")
        return

    md_files = find_all_markdown_files(root_path)
    if not md_files:
        print(f"[ERREUR] Aucun .md dans {MARKDOWN_ROOT}")
        return

    print(f"  {len(md_files)} fichier(s) .md trouvé(s)\n")

    total_segments  = 0
    total_phrases   = 0
    fichiers_ok     = 0
    fichiers_ko     = 0

    for i, md_path in enumerate(md_files, 1):
        try:
            rel_path = md_path.relative_to(root_path)
            print(f"[{i}/{len(md_files)}] {rel_path}", end=" ... ", flush=True)

            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()

            source_folder = str(md_path.parent.relative_to(root_path)) \
                if md_path.parent != root_path else "."

            segments = parse_markdown_to_segments(md_text, md_path.name, source_folder)

            # Compter le nombre total de phrases
            nb_phrases = sum(len(s.paragraph) for s in segments)

            # Sauvegarde
            out_dir = Path(SEGMENTS_OUTPUT_DIR) / md_path.parent.relative_to(root_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{md_path.stem}_segments.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)

            print(f"{len(segments)} segments  |  {nb_phrases} phrases")
            total_segments += len(segments)
            total_phrases  += nb_phrases
            fichiers_ok += 1

        except Exception as e:
            print(f"ERREUR : {e}")
            fichiers_ko += 1

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"  Fichiers traités  : {fichiers_ok}/{len(md_files)}")
    print(f"  Fichiers erreurs  : {fichiers_ko}")
    print(f"  Total segments    : {total_segments}")
    print(f"  Total phrases     : {total_phrases}")
    print(f"  Sortie            : {SEGMENTS_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        print(f"\nErreur fatale : {e}")
        import traceback
        traceback.print_exc()