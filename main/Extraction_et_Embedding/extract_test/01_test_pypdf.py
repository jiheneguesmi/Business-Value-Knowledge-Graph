"""
Test PyPDF2 - Extraction de texte basique (version corrigée)
"""

import os
import time
import json
from pathlib import Path
from pypdf import PdfReader

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/pypdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_with_pypdf(pdf_path: str) -> dict:
    start = time.time()
    result = {
        "outil": "pypdf",
        "fichier": Path(pdf_path).name,
        "pages": [],
        "texte_complet": "",
        "nb_pages": 0,
        "nb_chars_total": 0,
        "temps_secondes": 0,
        "erreur": None,
    }
    try:
        reader = PdfReader(pdf_path)
        result["nb_pages"] = len(reader.pages)
        all_text = []

        # Version corrigée : sans extraction_mode
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""  # Suppression de extraction_mode
            result["pages"].append({
                "page": i + 1,
                "texte": text,
                "nb_chars": len(text),
            })
            all_text.append(text)

        result["texte_complet"] = "\n\n--- PAGE SUIVANTE ---\n\n".join(all_text)
        result["nb_chars_total"] = len(result["texte_complet"])

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"PyPDF2 — {len(pdfs)} fichier(s) trouvé(s)\n")
    summary = []

    for pdf_path in pdfs:
        print(f"  Traitement : {pdf_path.name} ...", end=" ", flush=True)
        res = extract_with_pypdf(str(pdf_path))

        # Sauvegarde texte brut
        stem = pdf_path.stem
        txt_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(res["texte_complet"])

        # Sauvegarde JSON métadonnées
        json_path = os.path.join(OUTPUT_DIR, f"{stem}_meta.json")
        with open(json_path, "w", encoding="utf-8") as f:
            meta = {k: v for k, v in res.items() if k != "pages"}
            json.dump(meta, f, ensure_ascii=False, indent=2)

        status = f"OK — {res['nb_pages']}p, {res.get('nb_chars_total',0)} chars, {res['temps_secondes']}s"
        if res["erreur"]:
            status = f"ERREUR : {res['erreur']}"
        print(status)

        summary.append({
            "fichier": res["fichier"],
            "nb_pages": res["nb_pages"],
            "nb_chars": res.get("nb_chars_total", 0),
            "temps_s": res["temps_secondes"],
            "erreur": res["erreur"],
        })

    # Résumé global
    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}/")
    print(f"Résumé global : {summary_path}")


if __name__ == "__main__":
    run()