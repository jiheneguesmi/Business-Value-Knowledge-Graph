"""
Test tabula-py - Extraction de tableaux via tabula-java
pip install tabula-py
Prérequis : Java 8+ installé (java -version doit fonctionner)
"""

import os
import time
import json
from pathlib import Path

import tabula
import pandas as pd

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/tabula"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_tables_tabula(pdf_path: str, lattice: bool = False) -> dict:
    """
    lattice=True  : tableaux avec bordures (lignes visibles)
    lattice=False : tableaux sans bordures (whitespace / stream)
    """
    mode = "lattice" if lattice else "stream"
    start = time.time()
    result = {
        "outil": f"tabula-{mode}",
        "fichier": Path(pdf_path).name,
        "mode": mode,
        "nb_tableaux": 0,
        "tableaux": [],
        "temps_secondes": 0,
        "erreur": None,
    }
    try:
        dfs = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            lattice=lattice,
            stream=not lattice,
            guess=True,          # détection automatique des zones
            pandas_options={"header": 0},
            encoding="utf-8",
            silent=True,
        )

        result["nb_tableaux"] = len(dfs)

        for i, df in enumerate(dfs):
            if df is None or df.empty:
                continue
            result["tableaux"].append({
                "index": i,
                "shape": list(df.shape),
                "colonnes": list(df.columns.astype(str)),
                "apercu": df.head(3).fillna("").astype(str).to_dict(orient="records"),
            })

        # Export Excel
        stem = Path(pdf_path).stem
        if len(dfs) > 0:
            xl_path = os.path.join(OUTPUT_DIR, f"{stem}_{mode}.xlsx")
            with pd.ExcelWriter(xl_path) as writer:
                for i, df in enumerate(dfs):
                    if df is not None and not df.empty:
                        sheet = f"Table_{i+1}"
                        df.to_excel(writer, sheet_name=sheet, index=False)

        # Export JSON brut des DataFrames
        json_data_path = os.path.join(OUTPUT_DIR, f"{stem}_{mode}_tables.json")
        tables_json = []
        for i, df in enumerate(dfs):
            if df is not None and not df.empty:
                tables_json.append({
                    "index": i,
                    "data": df.fillna("").astype(str).to_dict(orient="records"),
                })
        with open(json_data_path, "w", encoding="utf-8") as f:
            json.dump(tables_json, f, ensure_ascii=False, indent=2)

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"tabula-py — {len(pdfs)} fichier(s) trouvé(s)\n")
    summary = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"  {pdf_path.name}")

        for lattice in [False, True]:
            mode = "lattice" if lattice else "stream"
            print(f"    [{mode}] ...", end=" ", flush=True)
            res = extract_tables_tabula(str(pdf_path), lattice=lattice)

            json_path = os.path.join(OUTPUT_DIR, f"{stem}_{mode}_meta.json")
            with open(json_path, "w", encoding="utf-8") as f:
                meta = {k: v for k, v in res.items() if k != "tableaux"}
                json.dump(meta, f, ensure_ascii=False, indent=2)

            status = f"OK — {res['nb_tableaux']} tableaux, {res['temps_secondes']}s"
            if res["erreur"]:
                status = f"ERREUR : {res['erreur']}"
            print(status)

            summary.append({
                "fichier": res["fichier"],
                "mode": mode,
                "nb_tableaux": res["nb_tableaux"],
                "temps_s": res["temps_secondes"],
                "erreur": res["erreur"],
            })

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()