"""
Script 1 : Nettoyage des fichiers Markdown
───────────────────────────────────────────
Entrée  : fichiers .md bruts (issus de conversion PDF)
Sortie  : fichiers .md nettoyés (même nom, dossier différent)

Ce qui est supprimé :
  - Images : ![](...) — si elles sont seules sur une ligne
  - Balises HTML (<span>, <div>, etc.) et entités (&nbsp; etc.)
  - Lignes horizontales (--- === ***)
  - Lignes vides consécutives (max 1 ligne vide entre blocs)

Ce qui est transformé :
  - Liens [texte](url)        → texte
  - Gras/italique **texte**   → texte
  - Backticks `code`          → code
  - Backslash escapes \*      → *
  - Espaces multiples         → espace simple

Ce qui est conservé intact :
  - Titres          : # ## ### etc.
  - Bullets         : - * +
  - Listes numérotées : 1. 2.
  - Texte normal
  - Lignes vides    : 1 seule entre blocs
  - Blocs de code   : ``` ... ```
"""

import os
import re
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
MARKDOWN_ROOT = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extraction\docs_test"
CLEAN_OUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\decoupage\clean_markdown"

os.makedirs(CLEAN_OUT_DIR, exist_ok=True)

# ── Patterns compilés ──────────────────────────────────────────────────────────
_RE_SPAN_ID     = re.compile(r'<span\s+id="[^"]*"[^>]*>.*?</span>', re.IGNORECASE)
_RE_HTML_TAG    = re.compile(r'<[^>]+>')
_RE_HTML_ENT    = re.compile(r'&[a-z]+;|&#\d+;')
_RE_IMAGE       = re.compile(r'!\[.*?\]\(.*?\)')
_RE_LINK        = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
_RE_BOLD_IT     = re.compile(r'\*{1,3}(.+?)\*{1,3}')
_RE_BACKTICK    = re.compile(r'`(.+?)`')
_RE_BACKSLASH   = re.compile(r'\\(.)')
_RE_MULTI_SPC   = re.compile(r'[ \t]{2,}')
_RE_HORIZONTAL  = re.compile(r'^[-=_\*]{3,}\s*$')
_RE_MULTI_BLANK = re.compile(r'\n{3,}')


def clean_line(line: str):
    """
    Nettoie une ligne markdown.
    Retourne :
      None  → ligne à supprimer entièrement (image seule, ligne horizontale)
      ''    → ligne vide (ligne devenue vide après nettoyage)
      str   → ligne nettoyée
    """
    stripped = line.strip()

    # Supprimer les lignes horizontales
    if _RE_HORIZONTAL.match(stripped):
        return None

    # Supprimer les lignes qui ne contiennent que des images
    if stripped and not _RE_IMAGE.sub('', stripped).strip():
        return None

    # ── Nettoyage du contenu ───────────────────────────────────────────────
    line = _RE_SPAN_ID.sub('', line)
    line = _RE_HTML_TAG.sub('', line)
    line = _RE_HTML_ENT.sub(' ', line)
    line = _RE_IMAGE.sub('', line)        # images inline dans du texte
    line = _RE_LINK.sub(r'\1', line)
    line = _RE_BOLD_IT.sub(r'\1', line)
    line = _RE_BACKTICK.sub(r'\1', line)
    line = _RE_BACKSLASH.sub(r'\1', line)
    line = _RE_MULTI_SPC.sub(' ', line)

    return line.strip()


def clean_markdown(md_text: str) -> str:
    """
    Nettoie un texte markdown complet.
    Retourne le markdown nettoyé.
    """
    lines = md_text.split('\n')
    out_lines = []
    in_code_block = False

    for line in lines:
        # Préserver les blocs de code tels quels
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue

        result = clean_line(line)

        if result is None:
            # Ligne supprimée → on n'ajoute rien
            continue

        out_lines.append(result)

    # Rejoindre et réduire les lignes vides consécutives à 1 maximum
    cleaned = '\n'.join(out_lines)
    cleaned = _RE_MULTI_BLANK.sub('\n\n', cleaned)

    return cleaned.strip()


# ── Découverte et traitement des fichiers ─────────────────────────────────────

def find_markdown_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob('*.md'))


def run():
    print('=' * 80)
    print('SCRIPT 1 — NETTOYAGE MARKDOWN')
    print('=' * 80)

    root_path = Path(MARKDOWN_ROOT)
    if not root_path.exists():
        print(f'[ERREUR] Dossier introuvable : {MARKDOWN_ROOT}')
        return

    md_files = find_markdown_files(root_path)
    if not md_files:
        print(f'[ERREUR] Aucun .md dans {MARKDOWN_ROOT}')
        return

    print(f'  {len(md_files)} fichier(s) .md trouvé(s)\n')

    fichiers_ok = 0
    fichiers_ko = 0

    for i, md_path in enumerate(md_files, 1):
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()

            rel_path = md_path.relative_to(root_path)
            cleaned  = clean_markdown(md_text)

            # Sauvegarde avec la même arborescence dans CLEAN_OUT_DIR
            out_dir = Path(CLEAN_OUT_DIR) / md_path.parent.relative_to(root_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / md_path.name   # même nom de fichier .md

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            n_before = md_text.count('\n')
            n_after  = cleaned.count('\n')
            print(f'[{i}/{len(md_files)}] {rel_path}')
            print(f'    {n_before} lignes brutes → {n_after} lignes nettoyées')
            print(f'    → {out_path}')

            fichiers_ok += 1

        except Exception as e:
            print(f'[{i}/{len(md_files)}] ERREUR : {e}')
            import traceback; traceback.print_exc()
            fichiers_ko += 1

    print('\n' + '=' * 80)
    print('RÉSUMÉ')
    print('=' * 80)
    print(f'  Fichiers traités : {fichiers_ok}/{len(md_files)}')
    print(f'  Fichiers erreurs : {fichiers_ko}')
    print(f'  Sortie           : {CLEAN_OUT_DIR}')


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print('\nInterruption utilisateur')
    except Exception as e:
        import traceback
        print(f'\nErreur fatale : {e}')
        traceback.print_exc()