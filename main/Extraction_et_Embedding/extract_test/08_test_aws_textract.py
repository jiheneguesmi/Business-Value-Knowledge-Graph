"""
Test AWS Textract - Extraction texte + tableaux + formulaires
pip install boto3
Prérequis : AWS credentials configurées (aws configure) OU variables d'env
            AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
            https://aws.amazon.com/textract/
Pricing : 0.0015$/page (texte) + 0.015$/page (tableaux)
"""

import os
import time
import json
from pathlib import Path

import boto3
import pandas as pd

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/aws_textract"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION AWS - Option A : profil AWS CLI (recommandé)
# ============================================================
AWS_REGION          = "eu-west-1"      # adapter à votre région
AWS_PROFILE         = None             # None = credentials par défaut
# Option B : credentials directes (déconseillé en prod)
AWS_ACCESS_KEY_ID   = None             # ou "AKIA..."
AWS_SECRET_KEY      = None             # ou "xxxx"
# ============================================================


def get_textract_client():
    if AWS_ACCESS_KEY_ID and AWS_SECRET_KEY:
        return boto3.client(
            "textract",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    return session.client("textract", region_name=AWS_REGION)


def blocks_to_text(blocks: list) -> str:
    """Reconstruit le texte ordonné depuis les blocs Textract."""
    lines = [b["Text"] for b in blocks if b["BlockType"] == "LINE" and "Text" in b]
    return "\n".join(lines)


def extract_tables_from_blocks(blocks: list) -> list:
    """Reconstruit les tableaux depuis la hiérarchie de blocs Textract."""
    block_map = {b["Id"]: b for b in blocks}
    tables = []

    for block in blocks:
        if block["BlockType"] != "TABLE":
            continue

        cells = {}
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cell_id in rel["Ids"]:
                    cell = block_map.get(cell_id)
                    if cell and cell["BlockType"] == "CELL":
                        row = cell.get("RowIndex", 1)
                        col = cell.get("ColumnIndex", 1)
                        row_span = cell.get("RowSpan", 1)
                        col_span = cell.get("ColumnSpan", 1)

                        # Contenu de la cellule
                        cell_text = ""
                        for c_rel in cell.get("Relationships", []):
                            if c_rel["Type"] == "CHILD":
                                for word_id in c_rel["Ids"]:
                                    word = block_map.get(word_id)
                                    if word and word["BlockType"] == "WORD":
                                        cell_text += word.get("Text", "") + " "

                        cells[(row, col)] = {
                            "text": cell_text.strip(),
                            "row_span": row_span,
                            "col_span": col_span,
                        }

        if cells:
            max_row = max(k[0] for k in cells)
            max_col = max(k[1] for k in cells)
            grid = [[""] * max_col for _ in range(max_row)]
            for (r, c), v in cells.items():
                grid[r - 1][c - 1] = v["text"]

            df = pd.DataFrame(grid)
            tables.append({
                "page": block.get("Page", None),
                "shape": [max_row, max_col],
                "apercu": df.head(3).to_dict(orient="records"),
                "dataframe": df,
            })

    return tables


def extract_with_textract(pdf_path: str, client) -> dict:
    start = time.time()
    result = {
        "outil": "aws_textract",
        "fichier": Path(pdf_path).name,
        "texte_complet": "",
        "tableaux": [],
        "nb_blocs": 0,
        "nb_tableaux": 0,
        "nb_chars": 0,
        "temps_secondes": 0,
        "erreur": None,
    }

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # AnalyzeDocument supporte 1 page max en synchrone
        # Pour PDFs multi-pages → StartDocumentAnalysis (async via S3)
        # Ici on traite page par page via PyMuPDF pour rester synchrone
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        all_text = []
        all_tables = []

        for page_num, page in enumerate(doc):
            # Rasteriser la page en image PNG
            mat = fitz.Matrix(2.0, 2.0)  # zoom x2 pour meilleure qualité OCR
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            response = client.analyze_document(
                Document={"Bytes": img_bytes},
                FeatureTypes=["TABLES", "FORMS"],
            )

            blocks = response.get("Blocks", [])
            result["nb_blocs"] += len(blocks)

            page_text = blocks_to_text(blocks)
            all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

            page_tables = extract_tables_from_blocks(blocks)
            for t in page_tables:
                t["page"] = page_num + 1
                all_tables.append(t)

        result["texte_complet"] = "\n\n".join(all_text)
        result["nb_chars"] = len(result["texte_complet"])
        result["nb_tableaux"] = len(all_tables)
        result["tableaux"] = [
            {k: v for k, v in t.items() if k != "dataframe"}
            for t in all_tables
        ]

        stem = Path(pdf_path).stem

        # Texte brut
        with open(os.path.join(OUTPUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(result["texte_complet"])

        # Tableaux Excel
        if all_tables:
            xl_path = os.path.join(OUTPUT_DIR, f"{stem}_tableaux.xlsx")
            with pd.ExcelWriter(xl_path) as writer:
                for i, t in enumerate(all_tables):
                    sheet = f"Table_p{t['page']}_{i+1}"[:31]
                    t["dataframe"].to_excel(writer, sheet_name=sheet, index=False, header=False)

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"AWS Textract — {len(pdfs)} fichier(s)\n")
    print("  Note : traitement synchrone page par page via rastérisation.\n")

    client = get_textract_client()
    summary = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"  Traitement : {pdf_path.name} ...", end=" ", flush=True)
        res = extract_with_textract(str(pdf_path), client)

        meta_path = os.path.join(OUTPUT_DIR, f"{stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            meta = {k: v for k, v in res.items() if k not in ("texte_complet", "tableaux")}
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = f"OK — {res['nb_tableaux']} tableaux, {res['nb_chars']} chars, {res['temps_secondes']}s"
        if res["erreur"]:
            status = f"ERREUR : {res['erreur']}"
        print(status)

        summary.append({
            "fichier": res["fichier"],
            "nb_tableaux": res["nb_tableaux"],
            "nb_chars": res["nb_chars"],
            "temps_s": res["temps_secondes"],
            "erreur": res["erreur"],
        })

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()