"""
Test docling (IBM Research) - Pipeline ML complet
pip install docling
Note : premier lancement télécharge les modèles (~500 MB)
       GPU recommandé, CPU possible (~2-5s/page)
"""

import os
import time
import json
from pathlib import Path

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/docling"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_with_docling(pdf_path: str, converter) -> dict:
    start = time.time()
    result = {
        "outil": "docling",
        "fichier": Path(pdf_path).name,
        "markdown": "",
        "texte_brut": "",
        "tableaux": [],
        "nb_pages": 0,
        "nb_chars_md": 0,
        "nb_tableaux": 0,
        "temps_secondes": 0,
        "erreur": None,
    }
    try:
        conv_result = converter.convert(pdf_path)
        doc = conv_result.document

        # Export Markdown complet (texte + tableaux en Markdown)
        markdown = doc.export_to_markdown()
        result["markdown"] = markdown
        result["nb_chars_md"] = len(markdown)

        # Export texte brut sans formatage
        result["texte_brut"] = doc.export_to_text()

        # Extraction des tableaux structurés
        tables_data = []
        for i, table in enumerate(doc.tables):
            try:
                df = table.export_to_dataframe()
                tables_data.append({
                    "index": i,
                    "shape": list(df.shape),
                    "colonnes": list(df.columns.astype(str)),
                    "apercu": df.head(3).fillna("").astype(str).to_dict(orient="records"),
                    "markdown": table.export_to_markdown(),
                })
            except Exception as te:
                tables_data.append({"index": i, "erreur": str(te)})

        result["tableaux"] = tables_data
        result["nb_tableaux"] = len(tables_data)

        # Nombre de pages
        result["nb_pages"] = len(doc.pages) if hasattr(doc, "pages") else 0

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"docling — {len(pdfs)} fichier(s) trouvé(s)\n")

    # Initialisation du converter une seule fois (charge les modèles)
    print("  Initialisation de docling (peut prendre 30-60s au premier lancement) ...")
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False          # pas d'OCR pour PDFs natifs
    pipeline_options.do_table_structure = True   # activer détection tableaux
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    print("  Docling prêt.\n")

    summary = []
    import pandas as pd

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"  Traitement : {pdf_path.name} ...", end=" ", flush=True)
        res = extract_with_docling(str(pdf_path), converter)

        # Markdown complet
        md_path = os.path.join(OUTPUT_DIR, f"{stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(res["markdown"])

        # Texte brut
        txt_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(res["texte_brut"])

        # Tableaux en Excel
        if res["nb_tableaux"] > 0:
            xl_path = os.path.join(OUTPUT_DIR, f"{stem}_tableaux.xlsx")
            try:
                conv_result = converter.convert(str(pdf_path))
                doc = conv_result.document
                with pd.ExcelWriter(xl_path) as writer:
                    for i, table in enumerate(doc.tables):
                        try:
                            df = table.export_to_dataframe()
                            df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
                        except Exception:
                            pass
            except Exception:
                pass

        # Métadonnées JSON
        meta_path = os.path.join(OUTPUT_DIR, f"{stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            meta = {k: v for k, v in res.items() if k not in ("markdown", "texte_brut", "tableaux")}
            meta["tableaux_apercu"] = res["tableaux"]
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = f"OK — {res['nb_pages']}p, {res['nb_tableaux']} tableaux, {res['nb_chars_md']} chars, {res['temps_secondes']}s"
        if res["erreur"]:
            status = f"ERREUR : {res['erreur']}"
        print(status)

        summary.append({
            "fichier": res["fichier"],
            "nb_pages": res["nb_pages"],
            "nb_tableaux": res["nb_tableaux"],
            "nb_chars_md": res["nb_chars_md"],
            "temps_s": res["temps_secondes"],
            "erreur": res["erreur"],
        })

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}/")
    print("Fichiers disponibles par PDF : .md (Markdown), .txt (texte brut), _tableaux.xlsx")


if __name__ == "__main__":
    run()