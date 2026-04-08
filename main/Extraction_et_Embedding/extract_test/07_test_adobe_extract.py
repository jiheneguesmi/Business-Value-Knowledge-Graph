"""
Test Adobe PDF Extract API - Version avec export Markdown
"""

import os
import time
import json
import zipfile
from pathlib import Path

# Configuration
PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/adobe_extract"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Credentials (à sécuriser !)
ADOBE_CLIENT_ID = "2a4daf9e094942dfb4e0a7d265662b64"
ADOBE_CLIENT_SECRET = "p8e-IsHSb9_jE1XmwLzjdYkeNFTgneYLFxf7"


def convert_to_markdown(adobe_data: dict) -> str:
    """
    Convertit les données Adobe en Markdown structuré.
    """
    markdown_lines = []
    elements = adobe_data.get("elements", [])
    
    current_page = 1
    in_table = False
    
    for el in elements:
        # Récupérer les infos
        path = el.get("Path", "")
        text = el.get("Text", "")
        page = el.get("Page", 1)
        
        # Séparateur de pages
        if page != current_page:
            markdown_lines.append(f"\n---\n## Page {page}\n")
            current_page = page
            in_table = False
        
        # Détecter les tableaux
        if "Table" in path:
            if not in_table:
                markdown_lines.append("\n### Tableau\n")
                in_table = True
            
            # Pour les cellules de tableau
            if "TD" in path:  # Table Data cell
                # Format simple pour tableau
                markdown_lines.append(f"| {text} |")
            elif "TR" in path:  # Table Row
                markdown_lines.append("\n")
        else:
            # C'est du texte normal
            if in_table and "Table" not in path:
                in_table = False
                markdown_lines.append("\n")
            
            # Nettoyer le texte et l'ajouter
            if text and text.strip():
                # Détecter les titres (basé sur la taille de police)
                font_size = el.get("TextSize", 12)
                if font_size and font_size >= 18:  # Titre
                    markdown_lines.append(f"\n## {text.strip()}\n")
                elif font_size and font_size >= 14:  # Sous-titre
                    markdown_lines.append(f"\n### {text.strip()}\n")
                else:  # Paragraphe normal
                    markdown_lines.append(f"{text.strip()} ")
    
    return "\n".join(markdown_lines)


def extract_with_adobe(pdf_path: str) -> dict:
    """Extrait le contenu avec Adobe et génère du Markdown."""
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

    start = time.time()
    result = {
        "outil": "adobe_pdf_extract",
        "fichier": Path(pdf_path).name,
        "markdown": "",
        "texte_complet": "",
        "nb_tableaux": 0,
        "nb_elements": 0,
        "nb_chars": 0,
        "temps_secondes": 0,
        "erreur": None,
    }

    try:
        # Connexion
        credentials = ServicePrincipalCredentials(
            client_id=ADOBE_CLIENT_ID,
            client_secret=ADOBE_CLIENT_SECRET,
        )
        pdf_services = PDFServices(credentials=credentials)

        # Upload du PDF
        with open(pdf_path, "rb") as f:
            input_asset = pdf_services.upload(
                input_stream=f,
                mime_type=PDFServicesMediaType.PDF,
            )

        # Paramètres d'extraction
        params = ExtractPDFParams(
            elements_to_extract=[
                ExtractElementType.TEXT,
                ExtractElementType.TABLES,
            ],
        )

        # Exécution du job
        print("    Job Adobe...", end=" ", flush=True)
        job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=params)
        location = pdf_services.submit(job)
        job_result = pdf_services.get_job_result(location, ExtractPDFResult)
        print("✅", flush=True)

        # Récupération des résultats
        result_asset = job_result.get_result().get_resource()
        stream_asset = pdf_services.get_content(result_asset)

        stem = Path(pdf_path).stem
        zip_path = os.path.join(OUTPUT_DIR, f"{stem}_result.zip")
        extract_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(extract_dir, exist_ok=True)

        # Sauvegarde du ZIP
        with open(zip_path, "wb") as f:
            f.write(stream_asset.get_input_stream())

        # Décompression
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Lecture du JSON structuré
        json_result_path = os.path.join(extract_dir, "structuredData.json")
        if os.path.exists(json_result_path):
            with open(json_result_path, "r", encoding="utf-8") as f:
                adobe_data = json.load(f)

            # Statistiques
            elements = adobe_data.get("elements", [])
            result["nb_elements"] = len(elements)
            
            # Extraire le texte complet
            text_parts = [el.get("Text", "") for el in elements if el.get("Text")]
            result["texte_complet"] = "\n".join(text_parts)
            result["nb_chars"] = len(result["texte_complet"])
            
            # Compter les tableaux
            result["nb_tableaux"] = len([e for e in elements if "Table" in e.get("Path", "")])
            
            # CONVERSION EN MARKDOWN
            result["markdown"] = convert_to_markdown(adobe_data)
            
            # Sauvegarde Markdown
            md_path = os.path.join(OUTPUT_DIR, f"{stem}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
            
            # Sauvegarde texte brut (fallback)
            txt_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["texte_complet"])
            
            print(f"    📝 Markdown: {md_path}")
            print(f"    📄 Texte brut: {txt_path}")

    except Exception as e:
        result["erreur"] = str(e)
        print(f"    ❌ Erreur: {e}")

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé")
        return

    print(f"Adobe PDF Extract API — {len(pdfs)} fichier(s)\n")
    
    summary = []
    success = 0

    for i, pdf_path in enumerate(pdfs, 1):
        stem = pdf_path.stem
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        
        res = extract_with_adobe(str(pdf_path))

        # Sauvegarde des métadonnées
        meta_path = os.path.join(OUTPUT_DIR, f"{stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            meta = {
                "fichier": res["fichier"],
                "outil": res["outil"],
                "nb_elements": res["nb_elements"],
                "nb_tableaux": res["nb_tableaux"],
                "nb_chars": res["nb_chars"],
                "temps_secondes": res["temps_secondes"],
                "erreur": res["erreur"],
            }
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if res["erreur"]:
            print(f"  ❌ ERREUR: {res['erreur'][:100]}")
        else:
            print(f"  ✅ OK — {res['nb_elements']} éléments, {res['nb_tableaux']} tableaux, {res['nb_chars']:,} caractères, {res['temps_secondes']}s")
            success += 1

        summary.append({
            "fichier": res["fichier"],
            "nb_elements": res["nb_elements"],
            "nb_tableaux": res["nb_tableaux"],
            "nb_chars": res.get("nb_chars", 0),
            "temps_s": res["temps_secondes"],
            "erreur": res["erreur"],
        })

    # Résumé global
    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"✅ Succès: {success}/{len(pdfs)} fichiers")
    print(f"📁 Résultats dans: {OUTPUT_DIR}/")
    print("\nFichiers générés:")
    print("  - *.md : Markdown formaté")
    print("  - *.txt : Texte brut")
    print("  - *_meta.json : Métadonnées")
    print("  - *_result.zip : Réponse brute Adobe")
    print(f"  - <nom_pdf>/structuredData.json : Données détaillées")


if __name__ == "__main__":
    run()