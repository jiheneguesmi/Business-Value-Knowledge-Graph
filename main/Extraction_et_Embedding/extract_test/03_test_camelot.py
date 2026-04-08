"""
Test camelot - Extraction de tableaux (Lattice + Stream)
pip install camelot-py[cv]
Prérequis : Ghostscript installé sur le système
"""

import os
import time
import json
from pathlib import Path

import camelot
import pandas as pd

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/camelot"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_tables_camelot(pdf_path: str, flavor: str = "lattice") -> dict:
    """
    flavor : "lattice" pour tableaux avec bordures visibles
             "stream"  pour tableaux sans bordures (whitespace)
    """
    start = time.time()
    result = {
        "outil": f"camelot-{flavor}",
        "fichier": Path(pdf_path).name,
        "flavor": flavor,
        "nb_tableaux": 0,
        "tableaux": [],
        "temps_secondes": 0,
        "erreur": None,
    }
    try:
        tables = camelot.read_pdf(
            pdf_path,
            flavor=flavor,
            pages="all",
            # Paramètres lattice
            **({"line_scale": 40} if flavor == "lattice" else {}),
            # Paramètres stream
            **({"edge_tol": 50, "row_tol": 10} if flavor == "stream" else {}),
        )
        result["nb_tableaux"] = len(tables)

        for i, table in enumerate(tables):
            df = table.df
            result["tableaux"].append({
                "index": i,
                "page": table.page,
                "accuracy": round(table.accuracy, 2),
                "whitespace": round(table.whitespace, 2),
                "shape": list(df.shape),  # [rows, cols]
                "apercu": df.head(3).to_dict(orient="records"),
            })

        # Export CSV par tableau
        stem = Path(pdf_path).stem
        csv_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(csv_dir, exist_ok=True)
        tables.export(os.path.join(csv_dir, f"{stem}_{flavor}.csv"), f="csv", compress=False)

        # Export Excel toutes les tables
        if len(tables) > 0:
            xl_path = os.path.join(OUTPUT_DIR, f"{stem}_{flavor}.xlsx")
            with pd.ExcelWriter(xl_path) as writer:
                for i, table in enumerate(tables):
                    sheet = f"Table_p{table.page}_{i+1}"[:31]
                    table.df.to_excel(writer, sheet_name=sheet, index=False)

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"camelot — {len(pdfs)} fichier(s) trouvé(s)\n")
    summary = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"  {pdf_path.name}")

        # Test les deux modes
        for flavor in ["lattice", "stream"]:
            print(f"    [{flavor}] ...", end=" ", flush=True)
            res = extract_tables_camelot(str(pdf_path), flavor=flavor)

            json_path = os.path.join(OUTPUT_DIR, f"{stem}_{flavor}_meta.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            status = f"OK — {res['nb_tableaux']} tableaux, {res['temps_secondes']}s"
            if res["erreur"]:
                status = f"ERREUR : {res['erreur']}"
            print(status)

            summary.append({
                "fichier": res["fichier"],
                "flavor": flavor,
                "nb_tableaux": res["nb_tableaux"],
                "temps_s": res["temps_secondes"],
                "erreur": res["erreur"],
            })

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()