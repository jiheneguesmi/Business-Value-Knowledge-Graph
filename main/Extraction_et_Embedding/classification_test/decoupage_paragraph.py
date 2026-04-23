"""
Script : Découpage optimisé des documents Markdown en segments
Corrections :
  - Suppression des balises HTML (<span>, <div>, etc.)
  - Fusion des petits paragraphes consécutifs sous le même titre
  - Détection et déduplication des segments répétés
  - Nettoyage complet avant classification
  - Traitement des titres seuls : contexte vide pour éviter redondance
"""

import os
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

# ── Configuration ──────────────────────────────────────────────────────────────
MARKDOWN_ROOT       = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_per_client\Docs"
SEGMENTS_OUTPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments"

MIN_CHARS       = 60    # seuil minimum de caractères pour qu'un segment soit retenu
MERGE_THRESHOLD = 120   # si un paragraphe fait moins de N chars, on le fusionne avec le suivant
MIN_TITLE_CHARS = 30    # seuil minimum pour qu'un titre seul devienne un segment

os.makedirs(SEGMENTS_OUTPUT_DIR, exist_ok=True)


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    section_title:   str
    paragraph:       str
    paragraph_index: int
    source_file:     str
    source_folder:   str


# ── Nettoyage ──────────────────────────────────────────────────────────────────

# Patterns compilés une seule fois
_RE_HTML_TAG      = re.compile(r'<[^>]+>')                        # <span id="...">, <div>, etc.
_RE_HTML_ENTITY   = re.compile(r'&[a-z]+;|&#\d+;')               # &nbsp; &#160; etc.
_RE_MD_IMAGE      = re.compile(r'!\[.*?\]\(.*?\)')                # ![alt](url)
_RE_MD_LINK       = re.compile(r'\[([^\]]+)\]\([^\)]+\)')         # [texte](url) → texte
_RE_MD_BOLD_IT    = re.compile(r'\*{1,3}(.+?)\*{1,3}')           # **gras** *italique*
_RE_MD_BACKTICK   = re.compile(r'`(.+?)`')                        # `code`
_RE_MD_HEADING    = re.compile(r'^#{1,6}\s+')                     # ## Titre
_RE_MULTI_SPACES  = re.compile(r'[ \t]{2,}')                      # espaces multiples
_RE_MULTI_NEWLINE = re.compile(r'\n{3,}')                         # sauts de ligne excessifs
_RE_SPAN_ID       = re.compile(r'<span\s+id="[^"]*"[^>]*>.*?</span>', re.IGNORECASE)
_RE_HORIZONTAL    = re.compile(r'^[-=_\*]{3,}$')                  # --- ou === ou ***


def clean_line(line: str) -> str:
    """Nettoie une ligne Markdown : supprime HTML, balises, formatage."""
    # 1. Supprimer les spans avec id (pagination marker)
    line = _RE_SPAN_ID.sub('', line)
    # 2. Supprimer toutes les balises HTML restantes
    line = _RE_HTML_TAG.sub('', line)
    # 3. Entités HTML
    line = _RE_HTML_ENTITY.sub(' ', line)
    # 4. Images Markdown
    line = _RE_MD_IMAGE.sub('', line)
    # 5. Liens → garder le texte
    line = _RE_MD_LINK.sub(r'\1', line)
    # 6. Gras / italique → garder le texte
    line = _RE_MD_BOLD_IT.sub(r'\1', line)
    # 7. Backticks inline
    line = _RE_MD_BACKTICK.sub(r'\1', line)
    # 8. Titres Markdown (# ## ###)
    line = _RE_MD_HEADING.sub('', line)
    # 9. Espaces multiples
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
    sans valeur sémantique (numéros de page, titres de phase seuls, etc.)
    """
    stripped = text.strip()
    # Trop court
    if len(stripped) < MIN_CHARS:
        return True
    # Uniquement des chiffres, ponctuation, tirets
    if re.match(r'^[\d\s\.\-\:\|\/\\]+$', stripped):
        return True
    # Ligne horizontale
    if _RE_HORIZONTAL.match(stripped):
        return True
    return False


def fingerprint(text: str) -> str:
    """Empreinte MD5 du texte normalisé pour déduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# ── Parser principal ───────────────────────────────────────────────────────────

def parse_markdown_to_segments(
    markdown_text: str,
    source_file: str,
    source_folder: str,
) -> list[Segment]:
    """
    Découpe le Markdown en segments propres.
    - Ignore tableaux, images, blocs de code
    - Fusionne les paragraphes trop courts sous le même titre
    - Déduplique les segments identiques
    - Supprime toutes les balises HTML (spans, divs, etc.)
    - Les titres seuls (sans paragraphe) sont traités comme segments
      OPTION B : section_title = "", paragraph = titre (évite la redondance)
    """
    lines = markdown_text.split("\n")

    # ── Passe 1 : collecter les blocs bruts ──
    sections: list[tuple[str, str]] = []   # (titre_section, texte_paragraphe)
    current_title = "Introduction"
    buffer: list[str] = []
    in_code_block = False
    in_table = False
    last_title = None      # Pour détecter les titres consécutifs

    def flush(title: str, buf: list[str]) -> None:
        raw = "\n".join(buf).strip()
        cleaned = clean_line(raw) if '\n' not in raw else "\n".join(
            cl for l in buf if (cl := clean_line(l))
        )
        if cleaned and not is_structural_only(cleaned):
            sections.append((title, cleaned))

    def flush_title_as_segment(title: str) -> None:
        """
        OPTION B : Crée un segment à partir d'un titre seul.
        section_title = "" (vide)
        paragraph = titre
        """
        if title and len(title) >= MIN_TITLE_CHARS:
            sections.append(("", title))  # ← contexte vide, contenu = titre

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
            # Vider le buffer précédent
            flush(current_title, buffer)
            buffer = []
            
            # Nettoyer le titre
            raw_title = heading_match.group(2).strip()
            cleaned_title = clean_line(raw_title)
            
            # Si on avait un titre précédent sans contenu, le traiter comme segment seul
            if last_title is not None:
                flush_title_as_segment(last_title)
            
            # Mettre à jour le titre courant
            current_title = cleaned_title if cleaned_title else current_title
            last_title = current_title
            continue

        # Ligne vide → séparateur de paragraphe
        if stripped == "":
            flush(current_title, buffer)
            buffer = []
            # Réinitialiser last_title car on a un contenu
            last_title = None
            continue

        # Ligne normale (contenu)
        cleaned = clean_line(line)
        if cleaned:
            buffer.append(cleaned)
            # Réinitialiser last_title car on a du contenu
            last_title = None

    # Flush final du buffer
    flush(current_title, buffer)
    
    # Si le dernier élément était un titre sans contenu, le traiter
    if last_title is not None:
        flush_title_as_segment(last_title)

    # ── Passe 2 : fusion des paragraphes trop courts ────────────────────────
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(sections):
        title, text = sections[i]
        # Si trop court ET même titre que le suivant → fusionner
        while (
            len(text) < MERGE_THRESHOLD
            and i + 1 < len(sections)
            and sections[i + 1][0] == title
        ):
            i += 1
            text = text + " " + sections[i][1]
        merged.append((title, text.strip()))
        i += 1

    # ── Passe 3 : déduplication par empreinte ───────────────────────────────
    seen: set[str] = set()
    segments: list[Segment] = []
    idx = 0
    for title, text in merged:
        fp = fingerprint(text)
        if fp in seen:
            continue
        seen.add(fp)
        if not is_structural_only(text):
            segments.append(Segment(
                section_title=title,
                paragraph=text,
                paragraph_index=idx,
                source_file=source_file,
                source_folder=source_folder,
            ))
            idx += 1

    return segments


# ── Découverte des fichiers ────────────────────────────────────────────────────

def find_all_markdown_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("*.md"))


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 80)
    print("DÉCOUPAGE MARKDOWN OPTIMISÉ")
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
    total_dupliques = 0
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

            # Sauvegarde
            out_dir = Path(SEGMENTS_OUTPUT_DIR) / md_path.parent.relative_to(root_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{md_path.stem}_segments.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)

            print(f"{len(segments)} segments")
            total_segments += len(segments)
            fichiers_ok += 1

        except Exception as e:
            print(f"ERREUR : {e}")
            fichiers_ko += 1

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"  Fichiers traités : {fichiers_ok}/{len(md_files)}")
    print(f"  Fichiers erreurs : {fichiers_ko}")
    print(f"  Total segments   : {total_segments}")
    print(f"  Sortie           : {SEGMENTS_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        print(f"\nErreur fatale : {e}")
        import traceback
        traceback.print_exc()