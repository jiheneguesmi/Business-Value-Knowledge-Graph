"""
Reconstruction du Markdown avec tableaux intégrés
À partir des fichiers .md et des tableaux Excel déjà extraits par Adobe PDF Extract
"""

import os
import pandas as pd
from pathlib import Path

# Configuration - ADAPTEZ CES CHEMINS SI NÉCESSAIRE
BASE_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_test\resultats_benchmark\adobe_extract"
OUTPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_test\resultats_benchmark\adobe_markdown_fixed"

# Créer le dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)


def excel_to_markdown_table(excel_path: str) -> str:
    """
    Convertit un fichier Excel en tableau Markdown.
    """
    try:
        # Lire le fichier Excel
        df = pd.read_excel(excel_path, header=0)
        
        if df.empty:
            return "*Tableau vide*"
        
        # Convertir en markdown
        markdown_lines = []
        
        # En-tête
        header = "| " + " | ".join(str(col) for col in df.columns) + " |"
        markdown_lines.append(header)
        
        # Séparateur
        separator = "|" + "|".join(["---" for _ in df.columns]) + "|"
        markdown_lines.append(separator)
        
        # Données (limiter à 20 lignes pour lisibilité)
        for _, row in df.head(20).iterrows():
            data_row = "| " + " | ".join(str(val) for val in row.values) + " |"
            markdown_lines.append(data_row)
        
        # Indiquer s'il y a plus de lignes
        if len(df) > 20:
            markdown_lines.append(f"\n*... et {len(df) - 20} lignes supplémentaires*")
        
        return "\n".join(markdown_lines)
        
    except Exception as e:
        return f"*Erreur lors du chargement du tableau: {e}*"


def find_all_tables(pdf_folder: Path) -> list:
    """
    Trouve tous les fichiers Excel dans le dossier tables.
    """
    tables_dir = pdf_folder / "tables"
    if not tables_dir.exists():
        return []
    
    # Récupérer tous les fichiers Excel
    excel_files = list(tables_dir.glob("*.xlsx")) + list(tables_dir.glob("*.xls"))
    
    # Trier par nom
    excel_files.sort()
    
    return excel_files


def fix_markdown_with_tables(pdf_name: str) -> bool:
    """
    Corrige le Markdown en intégrant les tableaux Excel.
    """
    # Chemins des fichiers
    md_file = Path(BASE_DIR) / f"{pdf_name}.md"
    pdf_folder = Path(BASE_DIR) / pdf_name
    
    # Vérifier si le fichier Markdown existe
    if not md_file.exists():
        print(f"  ❌ Fichier Markdown non trouvé: {md_file}")
        return False
    
    # Lire le contenu du Markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouver tous les tableaux Excel
    excel_files = find_all_tables(pdf_folder)
    
    if not excel_files:
        print(f"  ⚠️ Aucun tableau Excel trouvé")
        # Copier simplement le fichier MD original
        output_file = Path(OUTPUT_DIR) / f"{pdf_name}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    print(f"    📊 {len(excel_files)} tableau(x) Excel trouvé(s)")
    
    # Construire le nouveau Markdown avec les tableaux intégrés
    new_markdown_lines = []
    lines = content.split('\n')
    
    i = 0
    table_index = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Détecter l'en-tête "### Tableau"
        if "### Tableau" in line or "### TABLEAU" in line or "Tableau" in line and line.strip().startswith('#'):
            # Ajouter la ligne d'en-tête
            new_markdown_lines.append(line)
            i += 1
            
            # Sauter les lignes vides éventuelles après l'en-tête
            while i < len(lines) and lines[i].strip() == "":
                new_markdown_lines.append(lines[i])
                i += 1
            
            # Ajouter le tableau correspondant
            if table_index < len(excel_files):
                excel_file = excel_files[table_index]
                print(f"      Intégration tableau {table_index + 1}: {excel_file.name}")
                
                new_markdown_lines.append("")
                markdown_table = excel_to_markdown_table(str(excel_file))
                new_markdown_lines.append(markdown_table)
                new_markdown_lines.append("")
                table_index += 1
            else:
                # Plus de tableaux que d'en-têtes
                new_markdown_lines.append("")
                new_markdown_lines.append("*Tableau non disponible*")
                new_markdown_lines.append("")
        else:
            new_markdown_lines.append(line)
            i += 1
    
    # S'il reste des tableaux non utilisés, les ajouter à la fin
    if table_index < len(excel_files):
        new_markdown_lines.append("\n## Tableaux supplémentaires\n")
        for remaining_idx in range(table_index, len(excel_files)):
            excel_file = excel_files[remaining_idx]
            new_markdown_lines.append(f"\n### Tableau {remaining_idx + 1}\n")
            markdown_table = excel_to_markdown_table(str(excel_file))
            new_markdown_lines.append(markdown_table)
            new_markdown_lines.append("")
    
    # Sauvegarder le nouveau Markdown
    output_file = Path(OUTPUT_DIR) / f"{pdf_name}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_markdown_lines))
    
    return True


def fix_markdown_from_scratch(pdf_name: str) -> bool:
    """
    Version alternative: reconstruit le Markdown depuis le JSON et les tableaux.
    """
    json_file = Path(BASE_DIR) / pdf_name / "structuredData.json"
    pdf_folder = Path(BASE_DIR) / pdf_name
    
    if not json_file.exists():
        return False
    
    with open(json_file, 'r', encoding='utf-8') as f:
        import json
        adobe_data = json.load(f)
    
    elements = adobe_data.get("elements", [])
    excel_files = find_all_tables(pdf_folder)
    
    print(f"    Reconstruction depuis JSON...", end=" ", flush=True)
    
    # Construire le Markdown
    markdown_lines = []
    current_page = 1
    table_index = 0
    
    for el in elements:
        path = el.get("Path", "")
        text = el.get("Text", "")
        page = el.get("Page", 1)
        
        # Séparateur de pages
        if page != current_page:
            markdown_lines.append(f"\n---\n## Page {page}\n")
            current_page = page
        
        # Détecter les tableaux
        if "Table" in path and "TD" not in path:
            if table_index < len(excel_files):
                markdown_lines.append(f"\n### Tableau {table_index + 1}\n")
                markdown_lines.append(excel_to_markdown_table(str(excel_files[table_index])))
                markdown_lines.append("")
                table_index += 1
        
        elif text and text.strip() and "Table" not in path:
            # Texte normal
            markdown_lines.append(f"{text.strip()} ")
    
    # Sauvegarder
    output_file = Path(OUTPUT_DIR) / f"{pdf_name}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
    
    print(f"✅ {len(excel_files)} tableaux intégrés")
    return True


def process_all_pdfs():
    """
    Traite tous les PDFs du dossier.
    """
    print("="*60)
    print("RECONSTRUCTION DES MARKDOWN AVEC TABLEAUX EXCEL")
    print("="*60)
    print(f"📁 Dossier source: {BASE_DIR}")
    print(f"📁 Dossier sortie: {OUTPUT_DIR}\n")
    
    # Trouver tous les fichiers .md
    md_files = list(Path(BASE_DIR).glob("*.md"))
    
    if not md_files:
        print("❌ Aucun fichier .md trouvé")
        return
    
    # Extraire les noms des PDFs (sans extension)
    pdf_names = [f.stem for f in md_files]
    pdf_names.sort()
    
    print(f"📄 {len(pdf_names)} fichier(s) à traiter\n")
    
    success_count = 0
    error_count = 0
    no_tables_count = 0
    
    for i, pdf_name in enumerate(pdf_names, 1):
        print(f"[{i}/{len(pdf_names)}] {pdf_name}")
        
        # Vérifier s'il y a des tableaux
        pdf_folder = Path(BASE_DIR) / pdf_name
        tables_dir = pdf_folder / "tables"
        
        if tables_dir.exists():
            excel_files = list(tables_dir.glob("*.xlsx")) + list(tables_dir.glob("*.xls"))
            print(f"    📁 Dossier tables trouvé: {len(excel_files)} fichier(s) Excel")
        else:
            print(f"    ⚠️ Pas de dossier tables")
        
        # Essayer la méthode 1: depuis JSON (plus riche)
        if fix_markdown_from_scratch(pdf_name):
            success_count += 1
        else:
            # Méthode 2: depuis MD existant
            if fix_markdown_with_tables(pdf_name):
                success_count += 1
            else:
                print(f"  ❌ Échec pour {pdf_name}")
                error_count += 1
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"✅ Succès: {success_count}/{len(pdf_names)}")
    print(f"❌ Échecs: {error_count}/{len(pdf_names)}")
    print(f"📁 Résultats dans: {OUTPUT_DIR}/")
    
    # Statistiques sur les tableaux
    total_tables = 0
    for pdf_name in pdf_names:
        tables_dir = Path(BASE_DIR) / pdf_name / "tables"
        if tables_dir.exists():
            total_tables += len(list(tables_dir.glob("*.xlsx")))
    
    if total_tables > 0:
        print(f"📊 Total tableaux Excel trouvés: {total_tables}")
    
    # Lister les fichiers générés
    md_files = list(Path(OUTPUT_DIR).glob("*.md"))
    if md_files:
        print(f"\n📄 Fichiers Markdown générés ({len(md_files)}):")
        for f in md_files[:10]:
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} octets)")
        if len(md_files) > 10:
            print(f"  ... et {len(md_files)-10} autre(s)")


def test_single_pdf(pdf_name: str):
    """
    Test sur un seul PDF.
    """
    print(f"Test sur: {pdf_name}")
    print("-" * 40)
    
    # Vérifier l'existence
    md_file = Path(BASE_DIR) / f"{pdf_name}.md"
    tables_dir = Path(BASE_DIR) / pdf_name / "tables"
    
    if not md_file.exists():
        print(f"❌ Fichier MD non trouvé: {md_file}")
        return
    
    print(f"✅ MD trouvé: {md_file}")
    
    if tables_dir.exists():
        excel_files = list(tables_dir.glob("*.xlsx")) + list(tables_dir.glob("*.xls"))
        print(f"✅ Dossier tables trouvé: {len(excel_files)} fichiers Excel")
        for f in excel_files:
            print(f"   - {f.name}")
    else:
        print(f"⚠️ Dossier tables non trouvé: {tables_dir}")
    
    print("\nConversion...")
    
    if fix_markdown_from_scratch(pdf_name):
        print("✅ Succès")
    elif fix_markdown_with_tables(pdf_name):
        print("✅ Succès (méthode alternative)")
    else:
        print("❌ Échec")


if __name__ == "__main__":
    import sys
    
    # Permettre de tester un seul PDF en argument
    if len(sys.argv) > 1:
        test_single_pdf(sys.argv[1])
    else:
        process_all_pdfs()