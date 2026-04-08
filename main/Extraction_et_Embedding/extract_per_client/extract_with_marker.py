"""
Test marker - PDF vers Markdown avec organisation par sous-dossier
Chaque PDF est placé dans son propre sous-dossier avec ses résultats
AVEC NETTOYAGE DE LA MÉMOIRE GPU ENTRE CHAQUE TRAITEMENT
"""

import os
import time
import json
import shutil
import gc
from pathlib import Path

# Configuration des chemins
PDF_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\extract_per_client\Docs"

# Limite Windows MAX_PATH
WIN_MAX_PATH = 260


def clear_gpu_memory():
    """
    Nettoie la mémoire GPU entre les traitements pour éviter les fuites mémoire.
    Appeler cette fonction après chaque extraction.
    """
    try:
        import torch
        
        # Vider le cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Afficher l'état de la mémoire (optionnel - désactivable)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            if allocated > 0.1:  # N'afficher que si significatif
                print(f"       GPU: {allocated:.2f}GB utilisé / {reserved:.2f}GB réservé")
        
        # Forcer le garbage collector Python
        gc.collect()
        
    except ImportError:
        pass  # torch non installé (pas de GPU)
    except Exception as e:
        print(f"       Erreur nettoyage mémoire: {e}")


def get_gpu_memory_info():
    """Retourne les informations de mémoire GPU disponibles."""
    try:
        import torch
        if torch.cuda.is_available():
            free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            free_gb = free / 1024**3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return free_gb, total_gb
    except:
        pass
    return None, None


def safe_stem(pdf_path: Path, suffix: str = "") -> str:
    """
    Retourne un nom de fichier tronqué pour que le chemin complet
    (dossier_parent + séparateur + stem + suffix) reste sous WIN_MAX_PATH.
    suffix inclut le point : ex. ".md", "_meta.json"
    """
    parent_len = len(str(pdf_path.parent)) + 1  # +1 pour le séparateur
    available = WIN_MAX_PATH - parent_len - len(suffix)
    available = max(10, available)
    return pdf_path.stem[:available].rstrip()


def safe_open(path: Path, mode: str, encoding: str = "utf-8"):
    """Ouvre un fichier en s'assurant que le dossier parent existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode, encoding=encoding)


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
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'Image' in class_name or 'PIL' in str(type(obj)):
                return str(obj) if str(obj) else f"<{class_name}>"
        return str(obj) if obj != obj else None
    except Exception:
        return str(obj)


def load_marker_models():
    """Charge les modèles marker."""
    try:
        from marker.models import create_model_dict
        print("   Modules marker importés")
        print("  Chargement des modèles...")
        models = create_model_dict()
        print("   Modèles chargés")
        return models
    except Exception as e:
        print(f"   Erreur: {e}")
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

        converter = PdfConverter(models)
        rendered = converter(pdf_path)
        markdown, _, out_meta = text_from_rendered(rendered)

        result["markdown"] = markdown
        result["nb_chars"] = len(markdown)
        result["nb_pages"] = out_meta.get("pages", 0)

        cleaned_meta = {}
        for key, value in out_meta.items():
            if key not in ['images', 'debug_data']:
                cleaned_meta[key] = sanitize_for_json(value)
        result["metadata"] = cleaned_meta

    except ImportError as e:
        result["erreur"] = f"Erreur d'import: {e}"
    except Exception as e:
        result["erreur"] = str(e)

    result["temps_secondes"] = round(time.time() - start, 3)
    return result


def find_all_pdfs(root_dir: Path) -> list:
    """Parcourt toute l'arborescence récursivement et retourne tous les PDFs."""
    if not root_dir.exists():
        return []
    return list(root_dir.rglob("*.pdf"))


def compute_folder_name(pdf_path: Path) -> str:
    """
    Calcule le nom du sous-dossier cible pour un PDF,
    tronqué pour respecter MAX_PATH Windows.
    50 de marge = longueur max d'un nom de fichier de sortie (_meta.json = 10 + stem)
    """
    parent_len = len(str(pdf_path.parent)) + 1
    available  = WIN_MAX_PATH - parent_len - 50
    available  = max(10, available)
    return pdf_path.stem[:available].rstrip()


def is_already_organized(pdf_path: Path) -> bool:
    """Le PDF est organisé si son dossier parent porte le nom attendu (tronqué)."""
    return pdf_path.parent.name == compute_folder_name(pdf_path)


def organize_pdf_in_subfolder(pdf_path: Path) -> Path:
    """
    Déplace le PDF dans son propre sous-dossier (créé si nécessaire).
    Nom de dossier tronqué pour respecter MAX_PATH Windows.
    Retourne le nouveau chemin du PDF.
    """
    if not pdf_path.exists():
        print(f"       PDF introuvable : {pdf_path.name}")
        return pdf_path

    folder_name   = compute_folder_name(pdf_path)
    target_folder = pdf_path.parent / folder_name

    # Déjà dans le bon dossier
    if pdf_path.parent.name == folder_name:
        return pdf_path

    target_folder.mkdir(parents=True, exist_ok=True)
    new_pdf_path = target_folder / pdf_path.name

    # Destination déjà peuplée
    if new_pdf_path.exists():
        print(f"       Déjà présent : {folder_name}/{pdf_path.name}")
        if pdf_path.exists() and pdf_path != new_pdf_path:
            try:
                pdf_path.unlink()
            except Exception:
                pass
        return new_pdf_path

    try:
        shutil.move(str(pdf_path), str(new_pdf_path))
        print(f"       Déplacé : {pdf_path.name} → {folder_name}/")
    except FileNotFoundError:
        return new_pdf_path if new_pdf_path.exists() else pdf_path
    except Exception as e:
        print(f"       Erreur déplacement : {e}")
        return pdf_path

    return new_pdf_path


def save_results(pdf_path: Path, res: dict) -> tuple[Path, Path]:
    """
    Sauvegarde le Markdown et le JSON de métadonnées.
    Calcule des noms de fichiers tronqués si nécessaire.
    Retourne (md_path, meta_path).
    """
    output_dir = pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{pdf_path.stem}.md"
    meta_path = output_dir / f"{pdf_path.stem}_meta.json"

    # Sauvegarde Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(res["markdown"])

    # Sauvegarde JSON métadonnées
    meta = {
        "fichier":         res["fichier"],
        "outil":           res["outil"],
        "nb_pages":        res.get("nb_pages", 0),
        "nb_chars":        res.get("nb_chars", 0),
        "temps_secondes":  res.get("temps_secondes", 0),
        "erreur":          res.get("erreur"),
        "metadata":        res.get("metadata", {}),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return md_path, meta_path


def run():
    print("=" * 60)
    print("Test marker-pdf - Organisation par sous-dossier par PDF")
    print("=" * 60)
    
    # Afficher l'état GPU au démarrage
    free_gb, total_gb = get_gpu_memory_info()
    if free_gb is not None:
        print(f"  GPU détecté : {total_gb:.0f} GB total, {free_gb:.1f} GB libre")
    else:
        print("  GPU non détecté (utilisation CPU)")
    print()

    root_path = Path(PDF_DIR)
    if not root_path.exists():
        print(f"[ERREUR] Dossier introuvable : {PDF_DIR}")
        return

    print(f" Recherche récursive de tous les PDFs dans : {PDF_DIR}")
    all_pdfs = find_all_pdfs(root_path)

    if not all_pdfs:
        print("[ERREUR] Aucun PDF trouvé")
        return

    print(f" {len(all_pdfs)} fichier(s) PDF trouvé(s)\n")

    # ── Étape 1 : organisation de tous les PDFs ────────────────────────────
    print(" Organisation de tous les PDFs dans leurs sous-dossiers...")
    organized_pdfs = []
    for pdf in all_pdfs:
        new_path = organize_pdf_in_subfolder(pdf)
        organized_pdfs.append(new_path)
    print(f"   {len(organized_pdfs)} PDFs organisés\n")

    # Dédoublonnage
    organized_pdfs = list({str(p): p for p in organized_pdfs}.values())

    # ── Étape 2 : traitement de TOUS les PDFs ──────────────────────────────
    print("Chargement des modèles...")
    models = load_marker_models()
    if models is None:
        print("\n Échec du chargement des modèles")
        return

    print("\n" + "=" * 60)
    print("Traitement de TOUS les PDFs (sans vérification .md)...")
    print("=" * 60)

    summary       = []
    success_count = 0
    error_count   = 0

    for i, pdf_path in enumerate(organized_pdfs, 1):
        try:
            rel_path = pdf_path.relative_to(root_path)
        except ValueError:
            rel_path = pdf_path.name

        # Afficher la mémoire GPU avant traitement
        free_gb, _ = get_gpu_memory_info()
        mem_status = f" [GPU: {free_gb:.1f}GB libre]" if free_gb is not None else ""
        
        print(f"\n[{i}/{len(organized_pdfs)}] {rel_path}{mem_status}")

        res = extract_with_marker(str(pdf_path), models)

        if res.get("markdown"):
            try:
                md_path, meta_path = save_results(pdf_path, res)
                print(f"       {res['nb_chars']:,} caractères en {res['temps_secondes']:.1f}s")
                print(f"       Sauvegardé dans : {md_path.parent}")
                success_count += 1
            except Exception as e:
                print(f"       Erreur sauvegarde : {e}")
                error_count += 1
        else:
            print(f"       {res.get('erreur', 'Erreur inconnue')[:120]}")
            error_count += 1

        #  NETTOYAGE DE LA MÉMOIRE GPU APRÈS CHAQUE TRAITEMENT
        clear_gpu_memory()

        summary.append({
            "dossier_client": pdf_path.parent.parent.name if pdf_path.parent.parent != root_path else "",
            "dossier_pdf":    pdf_path.parent.name,
            "fichier":        pdf_path.name,
            "nb_pages":       res.get("nb_pages", 0),
            "nb_chars":       res.get("nb_chars", 0),
            "temps_s":        res.get("temps_secondes", 0),
            "erreur":         res.get("erreur"),
        })

    # ── Résumé global ─────────────────────────────────────────────────────
    summary_path = root_path / "_summary_marker_extraction.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n Résumé sauvegardé : {summary_path}")
    except Exception as e:
        print(f"\n Erreur sauvegarde résumé : {e}")

    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"   Total PDFs trouvés   : {len(all_pdfs)}")
    print(f"   PDFs traités         : {success_count + error_count}")
    print(f"    Succès            : {success_count}")
    print(f"    Erreurs           : {error_count}")

    if success_count > 0:
        total_chars = sum(s.get("nb_chars", 0) for s in summary)
        total_time  = sum(s.get("temps_s",   0) for s in summary)
        print(f"\n Statistiques d'extraction :")
        print(f"    Total caractères : {total_chars:,}")
        print(f"     Temps total      : {total_time:.1f}s")

    if error_count > 0:
        print(f"\n Fichiers en erreur ({error_count}) :")
        for err in [s for s in summary if s.get("erreur")]:
            print(f"  - {err['dossier_client']}/{err['dossier_pdf']}/{err['fichier']}")
            print(f"    {err['erreur'][:100]}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n\n Erreur fatale : {e}")
        import traceback
        traceback.print_exc()