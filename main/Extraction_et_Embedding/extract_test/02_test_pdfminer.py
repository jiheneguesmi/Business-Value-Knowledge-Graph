"""
Test pdfminer.six - Extraction layout bas niveau
"""

import os
import time
import json
from pathlib import Path
from io import StringIO

from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTAnno, LTChar
from pdfminer.pdfpage import PDFPage

PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/pdfminer"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_with_pdfminer(pdf_path: str) -> dict:
    start = time.time()
    result = {
        "outil": "pdfminer.six",
        "fichier": Path(pdf_path).name,
        "pages": [],
        "texte_complet": "",
        "nb_pages": 0,
        "temps_secondes": 0,
        "erreur": None,
    }

    # LAParams : paramètres de reconstruction de layout
    laparams = LAParams(
        line_margin=0.5,      # marge entre lignes pour fusion
        word_margin=0.1,      # marge entre mots
        char_margin=2.0,      # marge entre caractères
        boxes_flow=0.5,       # pondération vertical/horizontal
        detect_vertical=False,
    )

    try:
        # Méthode 1 : extraction haut niveau (rapide)
        texte_global = extract_text(pdf_path, laparams=laparams)
        result["texte_complet"] = texte_global
        result["nb_chars_total"] = len(texte_global)

        # Méthode 2 : extraction page par page avec layout détaillé
        all_pages = []
        page_num = 0
        for page_layout in extract_pages(pdf_path, laparams=laparams):
            page_num += 1
            page_text_blocks = []
            textboxes = []

            for element in page_layout:
                if isinstance(element, LTTextBox):
                    box_text = element.get_text().strip()
                    if box_text:
                        textboxes.append({
                            "bbox": [round(x, 2) for x in element.bbox],
                            "texte": box_text,
                        })
                        page_text_blocks.append(box_text)

            all_pages.append({
                "page": page_num,
                "texte": "\n".join(page_text_blocks),
                "nb_blocs": len(textboxes),
                "nb_chars": sum(len(b["texte"]) for b in textboxes),
                "blocs_detail": textboxes,
            })

        result["pages"] = all_pages
        result["nb_pages"] = page_num

    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé dans : {PDF_DIR}")
        return

    print(f"pdfminer.six — {len(pdfs)} fichier(s) trouvé(s)\n")
    summary = []

    for pdf_path in pdfs:
        print(f"  Traitement : {pdf_path.name} ...", end=" ", flush=True)
        res = extract_with_pdfminer(str(pdf_path))

        stem = pdf_path.stem

        # Texte brut
        with open(os.path.join(OUTPUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(res["texte_complet"])

        # JSON avec blocs et positions
        with open(os.path.join(OUTPUT_DIR, f"{stem}_layout.json"), "w", encoding="utf-8") as f:
            json.dump(res["pages"], f, ensure_ascii=False, indent=2)

        # Métadonnées
        with open(os.path.join(OUTPUT_DIR, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
            meta = {k: v for k, v in res.items() if k not in ("pages", "texte_complet")}
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

    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()