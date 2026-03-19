
"""
 Extraction PDF → Texte Structuré
Approche :
  1. Tableaux  → extract_tables() + gestion cellules multi-lignes
  2. Colonnes  → DBSCAN clustering sur axe X
  3. Images    → légendes uniquement
"""

import pdfplumber
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sklearn.cluster import DBSCAN

try:
    import spacy
    nlp = spacy.load("fr_core_news_sm")
except:
    nlp = None

# ═══════════════════════════════════════════
# TABLEAUX
# ═══════════════════════════════════════════

def extract_tables_as_sentences(page) -> List[Dict]:
    """
    Extrait tableaux et fusionne cellules multi-lignes.
    
    Logique de fusion:
      - Si ligne suivante a MOINS de cellules remplies que la ligne courante
        ET au moins une cellule → probable continuation
      - Fusionne cellule par cellule dans les colonnes correspondantes
    """
    settings = {
        'vertical_strategy': 'lines',
        'horizontal_strategy': 'lines',
        'snap_tolerance': 3,
        'join_tolerance': 3,
    }
    
    tables = page.extract_tables(table_settings=settings)
    results = []

    if not tables:
        return results

    for table in tables:
        if not table or len(table) < 1:
            continue

        # Nettoyer tableau
        cleaned_table = []
        for row in table:
            cells = []
            for cell in row:
                if cell is None:
                    cells.append(None)
                else:
                    cleaned = str(cell).replace('\n', ' ').replace('\r', ' ')
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    cells.append(cleaned if cleaned else None)
            cleaned_table.append(cells)
        
        if not cleaned_table:
            continue
        
        # Déterminer nombre de colonnes max (ligne la plus remplie)
        n_cols = max(sum(1 for c in row if c) for row in cleaned_table)
        
        # Fusionner lignes
        merged_rows = []
        i = 0
        
        while i < len(cleaned_table):
            current_row = list(cleaned_table[i])
            
            # Regarder si ligne suivante est continuation
            while i + 1 < len(cleaned_table):
                next_row = cleaned_table[i + 1]
                next_filled = sum(1 for c in next_row if c)
                
                # Continuation si ligne suivante a MOINS de cellules que n_cols
                is_continuation = (0 < next_filled < n_cols)
                
                if is_continuation:
                    # Fusionner cellule par cellule
                    for col_idx in range(min(len(current_row), len(next_row))):
                        if next_row[col_idx]:
                            if current_row[col_idx]:
                                current_row[col_idx] += ' ' + next_row[col_idx]
                            else:
                                current_row[col_idx] = next_row[col_idx]
                    i += 1
                else:
                    break
            
            # Extraire cellules non-None
            final_cells = [c for c in current_row if c]
            
            if final_cells:
                merged_rows.append(final_cells)
            
            i += 1

        # Convertir en phrases
        for cells in merged_rows:
            if len(cells) == 1:
                sentence = cells[0]
            elif len(cells) == 2:
                sentence = f"{cells[0]}: {cells[1]}"
            else:
                sentence = ' | '.join(cells)

            if len(sentence) > 5:
                results.append({
                    'text': sentence,
                    'page': None,
                    'x': 0.0,
                    'y': 0.0,
                    'type': 'table',
                    'has_image': False,
                    'from_table': True
                })

    return results


def get_table_bboxes(page) -> List[tuple]:
    """Retourne bounding boxes des tableaux détectés."""
    bboxes = []
    if hasattr(page, 'find_tables'):
        for tbl in page.find_tables():
            bboxes.append(tbl.bbox)
    return bboxes


def word_in_table(word: Dict, bboxes: List[tuple]) -> bool:
    """Vérifie si mot est dans zone tableau."""
    for (x0, top, x1, bottom) in bboxes:
        if (x0 - 5 <= word['x0'] <= x1 + 5 and 
            top - 5 <= word['top'] <= bottom + 5):
            return True
    return False


# ═══════════════════════════════════════════
# COLONNES (DBSCAN)
# ═══════════════════════════════════════════

def detect_columns_dbscan(words: List[Dict]) -> List[List[Dict]]:
    """
    Détecte colonnes via DBSCAN clustering sur positions X.
    
    IMPORTANT: Retourne colonnes triées GAUCHE→DROITE,
    chaque colonne triée HAUT→BAS (lecture verticale).
    """
    if not words or len(words) < 5:
        return [sorted(words, key=lambda w: (w['top'], w['x0']))]

    # Calculer centres X
    centers_x = np.array([[(w['x0'] + w['x1']) / 2] for w in words])

    # eps adaptatif (5% largeur page, min 15px)
    page_width = centers_x.max() - centers_x.min()
    eps = max(page_width * 0.05, 15)

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=3).fit(centers_x)
    labels = clustering.labels_

    n_cols = len(set(labels)) - (1 if -1 in labels else 0)

    # Mono-colonne
    if n_cols <= 1:
        col = sorted(words, key=lambda w: (w['top'], w['x0']))
        return [col]

    # Regrouper par cluster
    columns = {}
    for word, label in zip(words, labels):
        if label == -1:
            # Bruit → affecter au cluster le plus proche
            word_cx = (word['x0'] + word['x1']) / 2
            min_dist = float('inf')
            best_label = 0
            for lbl in set(labels):
                if lbl == -1:
                    continue
                cluster_words = [w for w, l in zip(words, labels) if l == lbl]
                cluster_cx = np.mean([(w['x0'] + w['x1']) / 2 for w in cluster_words])
                dist = abs(word_cx - cluster_cx)
                if dist < min_dist:
                    min_dist = dist
                    best_label = lbl
            label = best_label

        if label not in columns:
            columns[label] = []
        columns[label].append(word)

    # CRITIQUE: Trier colonnes GAUCHE→DROITE par position X moyenne
    sorted_cols = []
    for label in sorted(columns.keys(), key=lambda l: np.mean([(w['x0'] + w['x1'])/2 for w in columns[l]])):
        # IMPORTANT: Trier mots dans colonne HAUT→BAS (top d'abord)
        col = sorted(columns[label], key=lambda w: (w['top'], w['x0']))
        sorted_cols.append(col)

    return sorted_cols


def words_to_lines(words: List[Dict], y_tolerance: float = 5.0) -> List[List[Dict]]:
    """Regroupe mots en lignes selon position verticale."""
    if not words:
        return []

    lines = []
    current_line = [words[0]]
    ref_top = words[0]['top']

    for word in words[1:]:
        if abs(word['top'] - ref_top) <= y_tolerance:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda w: w['x0']))
            current_line = [word]
            ref_top = word['top']

    if current_line:
        lines.append(sorted(current_line, key=lambda w: w['x0']))

    return lines


# ═══════════════════════════════════════════
# IMAGES / LÉGENDES
# ═══════════════════════════════════════════

CAPTION_PATTERNS = re.compile(
    r'(figure|fig\.|schéma|schema|image|source|légende|legende|tableau|photo)',
    re.IGNORECASE
)


def extract_image_captions(page) -> List[Dict]:
    """Extrait légendes d'images uniquement."""
    if not page.images:
        return []

    all_words = page.extract_words()
    captions = []

    for img in page.images:
        img_x0, img_top = img['x0'], img['top']
        img_x1, img_bottom = img['x1'], img['bottom']
        margin_x = (img_x1 - img_x0) * 0.15
        margin_y = 30

        nearby = []
        for w in all_words:
            in_x = (img_x0 - margin_x) <= w['x0'] <= (img_x1 + margin_x)
            below = img_bottom <= w['top'] <= img_bottom + margin_y
            above = img_top - margin_y <= w['bottom'] <= img_top
            if in_x and (below or above):
                nearby.append(w)

        if not nearby:
            continue

        nearby.sort(key=lambda w: (w['top'], w['x0']))
        caption_text = ' '.join(w['text'] for w in nearby).strip()

        if CAPTION_PATTERNS.search(caption_text):
            captions.append({
                'text': caption_text,
                'page': None,
                'x': img_x0,
                'y': img_bottom,
                'type': 'caption',
                'has_image': True,
                'from_table': False
            })

    return captions


# ═══════════════════════════════════════════
# ASSEMBLAGE PAR PAGE
# ═══════════════════════════════════════════

def extract_page_content(page, page_num: int) -> List[Dict]:
    """
    Pipeline par page:
      1. Tableaux (extraits séparément, zones exclues du texte)
      2. Légendes images
      3. Texte normal → colonnes DBSCAN → lignes
    """
    results = []

    # ── Tableaux ──
    table_bboxes = get_table_bboxes(page)
    for item in extract_tables_as_sentences(page):
        item['page'] = page_num
        results.append(item)

    # ── Légendes ──
    for caption in extract_image_captions(page):
        caption['page'] = page_num
        results.append(caption)

    # ── Texte normal ──
    words = page.extract_words(x_tolerance=3, y_tolerance=0)

    # Exclure mots dans zones tableaux
    if table_bboxes:
        words = [w for w in words if not word_in_table(w, table_bboxes)]

    if not words:
        return results

    # Détection colonnes DBSCAN
    columns = detect_columns_dbscan(words)
    
    print(f"[DEBUG] Page {page_num}: {len(columns)} colonnes détectées, {len(words)} mots total")
    for i, col in enumerate(columns):
        print(f"  Col {i+1}: {len(col)} mots")

    # Traiter chaque colonne INDÉPENDAMMENT (lecture verticale pure)
    for column in columns:
        # Chaque colonne est déjà triée top→bottom par DBSCAN
        # Ne PAS regrouper en lignes horizontales
        # Joindre tous les mots de la colonne en UNE phrase
        text = ' '.join(w['text'] for w in column).strip()
        
        if len(text) >= 3:
            is_bullet = text.startswith(('•', '-', '›', '▪', '–', '◦'))
            results.append({
                'text': text,
                'page': page_num,
                'x': column[0]['x0'],
                'y': column[0]['top'],
                'type': 'bullet' if is_bullet else 'text',
                'has_image': False,
                'from_table': False
            })

    return results


# ═══════════════════════════════════════════
# POST-TRAITEMENT
# ═══════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def segment_sentences(text: str) -> List[str]:
    if nlp:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def merge_short_lines(records: List[Dict]) -> List[Dict]:
    """
    Fusionne lignes courtes avec précédente si:
      - même page, même type, même colonne (x ±40px)
      - ligne précédente sans ponctuation finale
      - ligne courante commence minuscule OU <6 mots
    """
    if not records:
        return []

    merged = [records[0]]

    for rec in records[1:]:
        prev = merged[-1]

        # Ne fusionner que texte/bullet
        if prev['type'] not in ('text', 'bullet') or rec['type'] not in ('text', 'bullet'):
            merged.append(rec)
            continue

        # Même page
        if prev['page'] != rec['page']:
            merged.append(rec)
            continue

        # Même colonne
        same_col = abs(prev['x'] - rec['x']) < 40

        # Pas de ponctuation finale
        ends_open = not prev['sentence'].rstrip().endswith(('.', '!', '?', ':', ';'))

        # Commence minuscule ou court
        starts_lower = rec['sentence'] and rec['sentence'][0].islower()
        is_short = len(rec['sentence'].split()) < 6

        if same_col and ends_open and (starts_lower or is_short):
            prev['sentence'] += ' ' + rec['sentence']
        else:
            merged.append(rec)

    return merged


# ═══════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════

def extract_sentences_from_pdf(pdf_path: str) -> pd.DataFrame:
    print(f"Extraction de {pdf_path}...")
    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for block in extract_page_content(page, page_num):
                text = clean_text(block['text'])
                if len(text) < 10:
                    continue

                # Tableaux → pas de segmentation
                if block['type'] == 'table':
                    records.append({
                        'sentence': text,
                        'page': block['page'],
                        'x': block['x'],
                        'y': block['y'],
                        'type': block['type'],
                        'has_image': block['has_image'],
                        'from_table': block['from_table'],
                        'doc_name': Path(pdf_path).stem
                    })
                else:
                    # Segmentation phrases
                    for sentence in segment_sentences(text):
                        if len(sentence) > 15:
                            records.append({
                                'sentence': sentence,
                                'page': block['page'],
                                'x': block['x'],
                                'y': block['y'],
                                'type': block['type'],
                                'has_image': block['has_image'],
                                'from_table': block['from_table'],
                                'doc_name': Path(pdf_path).stem
                            })

    # Fusion lignes courtes
    records = merge_short_lines(records)

    df = pd.DataFrame(records)
    print(f"  ✓ {len(df)} phrases extraites")
    return df


def extract_from_folder(folder_path: str, output_csv: str) -> pd.DataFrame:
    folder = Path(folder_path)
    pdf_files = list(folder.glob('*.pdf'))
    print(f"Trouvé {len(pdf_files)} fichiers PDF")

    all_dfs = []
    for pdf_file in pdf_files:
        try:
            all_dfs.append(extract_sentences_from_pdf(str(pdf_file)))
        except Exception as e:
            print(f"  ❌ Erreur {pdf_file.name}: {e}")

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ {len(df_combined)} phrases sauvegardées dans {output_csv}")
    return df_combined


if __name__ == "__main__":
    PDF_FOLDER = "exemples brochures commerciales"
    OUTPUT_CSV = "extracted_sentences.csv"

    df = extract_from_folder(PDF_FOLDER, OUTPUT_CSV)

    print("\nAperçu:")
    print(df.head(10).to_string())
    print(f"\nStatistiques:")
    print(f"  Documents : {df['doc_name'].nunique()}")
    print(f"  Pages     : {df['page'].max()}")
    print(f"  Phrases   : {len(df)}")
    print(f"\nTypes:")
    print(df['type'].value_counts())
    print(f"\nImages    : {df['has_image'].sum()}")
    print(f"Tableaux  : {df['from_table'].sum()}")