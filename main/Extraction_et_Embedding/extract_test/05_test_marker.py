"""
Test marker - PDF vers Markdown avec la bonne API
Version corrigée pour la sérialisation JSON
"""

import os
import time
import json
from pathlib import Path

# Configuration des chemins
PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\Exemples Brochures Commerciales PDF"
OUTPUT_DIR = "resultats_benchmark/marker"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize_for_json(obj):
    """Nettoie les objets pour les rendre sérialisables en JSON."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    try:
        # Essayer de convertir en string pour les objets complexes
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'Image' in class_name or 'PIL' in str(type(obj)):
                return str(obj) if str(obj) else f"<{class_name}>"
        return str(obj) if obj != obj else None
    except:
        return str(obj)


def load_marker_models():
    """Charge les modèles marker."""
    try:
        from marker.models import create_model_dict
        print("  ✅ Modules marker importés")
        print("  Chargement des modèles...")
        models = create_model_dict()
        print("  ✅ Modèles chargés")
        return models
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None


def extract_with_marker(pdf_path: str, models: dict) -> dict:
    """Extrait le contenu d'un PDF avec marker."""
    result = {
        "outil": "marker",
        "fichier": Path(pdf_path).name,
        "markdown": "",
        "nb_pages": 0,
        "nb_chars": 0,
        "metadata": {},
        "temps_secondes": 0,
        "erreur": None,
    }
    
    start = time.time()
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
        
        # Créer un convertisseur PDF
        converter = PdfConverter(models)
        
        # Convertir le PDF
        rendered = converter(pdf_path)
        
        # Extraire le texte markdown
        markdown, _, out_meta = text_from_rendered(rendered)
        
        result["markdown"] = markdown
        result["nb_chars"] = len(markdown)
        result["nb_pages"] = out_meta.get("pages", 0)
        
        # Nettoyer les métadonnées pour JSON
        cleaned_meta = {}
        for key, value in out_meta.items():
            if key not in ['images', 'debug_data']:  # Exclure les champs problématiques
                cleaned_meta[key] = sanitize_for_json(value)
        result["metadata"] = cleaned_meta
        
    except ImportError as e:
        result["erreur"] = f"Erreur d'import: {e}"
    except Exception as e:
        result["erreur"] = str(e)
    
    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def run():
    print("=== Test marker-pdf ===\n")
    
    # Vérifier le dossier PDF
    if not os.path.exists(PDF_DIR):
        print(f"[ERREUR] Dossier PDF introuvable : {PDF_DIR}")
        return
        
    pdfs = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[ERREUR] Aucun PDF trouvé")
        return

    print(f"Trouvé {len(pdfs)} fichier(s) PDF\n")
    
    print("Chargement des modèles...")
    models = load_marker_models()
    
    if models is None:
        print("\n❌ Échec du chargement des modèles")
        return
    
    print("\n" + "="*60)
    print("Traitement de tous les PDFs...")
    print("="*60)
    
    summary = []
    success_count = 0
    error_count = 0
    
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        
        res = extract_with_marker(str(pdf_path), models)
        
        # Sauvegarder le Markdown
        if res.get("markdown"):
            md_path = os.path.join(OUTPUT_DIR, f"{pdf_path.stem}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(res["markdown"])
            print(f"  ✅ {res['nb_chars']:,} caractères en {res['temps_secondes']:.1f}s")
            success_count += 1
        else:
            print(f"  ❌ {res.get('erreur', 'Erreur inconnue')[:100]}")
            error_count += 1
        
        # Sauvegarder les métadonnées (nettoyées pour JSON)
        meta_path = os.path.join(OUTPUT_DIR, f"{pdf_path.stem}_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                meta = {
                    "fichier": res["fichier"],
                    "outil": res["outil"],
                    "nb_pages": res.get("nb_pages", 0),
                    "nb_chars": res.get("nb_chars", 0),
                    "temps_secondes": res.get("temps_secondes", 0),
                    "erreur": res.get("erreur"),
                    "metadata": res.get("metadata", {})
                }
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠️  Erreur sauvegarde JSON: {e}")
            # Sauvegarde alternative
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"Erreur JSON: {e}\n")
                f.write(f"Fichier: {res['fichier']}\n")
                f.write(f"Pages: {res.get('nb_pages', 0)}\n")
                f.write(f"Caractères: {res.get('nb_chars', 0)}\n")
        
        summary.append({
            "fichier": res["fichier"],
            "nb_pages": res.get("nb_pages", 0),
            "nb_chars": res.get("nb_chars", 0),
            "temps_s": res.get("temps_secondes", 0),
            "erreur": res.get("erreur"),
        })
    
    # Sauvegarder le résumé global
    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Résumé sauvegardé: {summary_path}")
    except Exception as e:
        print(f"\n⚠️  Erreur sauvegarde résumé: {e}")
    
    # Statistiques
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"✅ Succès: {success_count}/{len(pdfs)} fichiers")
    print(f"❌ Échecs: {error_count}/{len(pdfs)} fichiers")
    
    if success_count > 0:
        total_chars = sum(s.get("nb_chars", 0) for s in summary)
        total_time = sum(s.get("temps_s", 0) for s in summary)
        print(f"📊 Total caractères extraits: {total_chars:,}")
        print(f"⏱️  Temps total: {total_time:.1f}s")
        print(f"📁 Résultats sauvegardés dans: {OUTPUT_DIR}/")
    
    # Afficher les erreurs si nécessaire
    if error_count > 0:
        print(f"\n⚠️  Fichiers en erreur:")
        for err in [s for s in summary if s.get("erreur")]:
            print(f"  - {err['fichier']}: {err['erreur'][:80]}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()