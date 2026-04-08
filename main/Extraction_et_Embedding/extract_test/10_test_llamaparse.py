"""
Test LlamaParse (LlamaIndex) - PDF → Markdown optimisé LLM
pip install llama-parse
Prérequis : clé API LlamaIndex
            https://cloud.llamaindex.ai → API Keys
Free tier : 1000 pages/jour
"""

import os
import time
import json
from pathlib import Path

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/llamaparse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CREDENTIALS - à renseigner
# ============================================================
LLAMA_API_KEY = "llx-KIrzVWwAaWEMOFw9Q4GobU3SjChMEy2CXjrEBZFpXcVjusT8"
# ============================================================

# ── Paramètres LlamaParse ────────────────────────────────────
# result_type : "markdown" (recommandé) ou "text"
# premium_mode : False = mode fast (défaut), True = GPT-4o backend (meilleur, 2x plus cher)
# parsing_instruction : prompt système pour guider l'extraction
RESULT_TYPE  = "markdown"
PREMIUM_MODE = False
LANGUAGE     = "fr"   # langue principale de vos PDFs

PARSING_INSTRUCTION = """
Extraire intégralement le contenu de ce document commercial en respectant :
- La hiérarchie des titres (# ## ###)
- Les tableaux en format Markdown complet avec toutes les colonnes et lignes
- Les listes à puces et numérotées
- Les prix, références produits et données chiffrées sans approximation
- L'ordre de lecture naturel du document
Ne pas paraphraser, extraire le texte tel quel.
"""
# ────────────────────────────────────────────────────────────


def extract_with_llamaparse_sync(pdf_path: str) -> dict:
    """Mode synchrone : un fichier à la fois."""
    from llama_parse import LlamaParse

    start = time.time()
    result = {
        "outil": "llamaparse",
        "fichier": Path(pdf_path).name,
        "markdown": "",
        "nb_pages": 0,
        "nb_chars": 0,
        "mode": "premium" if PREMIUM_MODE else "fast",
        "temps_secondes": 0,
        "erreur": None,
    }

    try:
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            result_type=RESULT_TYPE,
            premium_mode=PREMIUM_MODE,
            parsing_instruction=PARSING_INSTRUCTION,
            language=LANGUAGE,
            verbose=False,
        )

        documents = parser.load_data(pdf_path)

        # Concaténation de toutes les pages
        pages_text = []
        for i, doc in enumerate(documents):
            pages_text.append(doc.text)

        result["markdown"] = "\n\n---\n\n".join(pages_text)
        result["nb_pages"] = len(documents)
        result["nb_chars"] = len(result["markdown"])

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def extract_tables_from_markdown(markdown: str) -> list:
    """Extrait les tableaux Markdown pour analyse."""
    import re
    tables = []
    lines = markdown.split("\n")
    in_table = False
    current_table = []

    for line in lines:
        stripped = line.strip()
        if "|" in stripped and stripped.startswith("|"):
            in_table = True
            current_table.append(stripped)
        else:
            if in_table and current_table:
                tables.append("\n".join(current_table))
                current_table = []
            in_table = False

    if current_table:
        tables.append("\n".join(current_table))

    return tables


def markdown_table_to_df(md_table: str):
    """Convertit un tableau Markdown en DataFrame pandas."""
    import pandas as pd
    import io
    lines = [l for l in md_table.strip().split("\n") if not set(l.replace("|", "").replace("-", "").replace(" ", "")) == set()]
    if len(lines) < 2:
        return None
    try:
        rows = []
        for line in lines:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    except Exception:
        return None


def run():
    import pandas as pd

    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"LlamaParse — {len(pdfs)} fichier(s) | mode={'premium' if PREMIUM_MODE else 'fast'}\n")
    summary = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"  Traitement : {pdf_path.name} ...", end=" ", flush=True)
        res = extract_with_llamaparse_sync(str(pdf_path))

        # Markdown complet
        md_path = os.path.join(OUTPUT_DIR, f"{stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(res["markdown"])

        # Extraction et sauvegarde des tableaux
        if res["markdown"]:
            md_tables = extract_tables_from_markdown(res["markdown"])
            res["nb_tableaux"] = len(md_tables)

            if md_tables:
                xl_path = os.path.join(OUTPUT_DIR, f"{stem}_tableaux.xlsx")
                with pd.ExcelWriter(xl_path) as writer:
                    for i, md_t in enumerate(md_tables):
                        df = markdown_table_to_df(md_t)
                        if df is not None and not df.empty:
                            df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)

                # JSON tableaux bruts Markdown
                tables_json_path = os.path.join(OUTPUT_DIR, f"{stem}_tableaux_md.json")
                with open(tables_json_path, "w", encoding="utf-8") as f:
                    json.dump(md_tables, f, ensure_ascii=False, indent=2)
        else:
            res["nb_tableaux"] = 0

        # Métadonnées
        meta_path = os.path.join(OUTPUT_DIR, f"{stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            meta = {k: v for k, v in res.items() if k != "markdown"}
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = (
            f"OK — {res['nb_pages']}p, {res.get('nb_tableaux', 0)} tableaux, "
            f"{res['nb_chars']} chars, {res['temps_secondes']}s"
        )
        if res["erreur"]:
            status = f"ERREUR : {res['erreur']}"
        print(status)

        summary.append({
            "fichier": res["fichier"],
            "nb_pages": res["nb_pages"],
            "nb_tableaux": res.get("nb_tableaux", 0),
            "nb_chars": res["nb_chars"],
            "temps_s": res["temps_secondes"],
            "erreur": res["erreur"],
        })

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats dans : {OUTPUT_DIR}/")
    print("Chaque PDF → .md (Markdown complet), _tableaux.xlsx, _tableaux_md.json")


if __name__ == "__main__":
    run()