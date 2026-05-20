"""
Script : Comparaison d'ensembles de modèles par substitution
Adapté au workflow v3 (vote majoritaire sur 9 questions oui/non)

Question centrale : Si je supprime certains modèles, la catégorie finale change-t-elle ?

Workflow :
  1. Charge les JSON *_classification.json produits par multi_llm_paragraph_v3.py
  2. Pour chaque ensemble alternatif défini → recalcule le vote sur les 9 questions
     en gardant uniquement les prédictions des modèles de cet ensemble
  3. Compare la catégorie finale obtenue vs la référence (Ensemble A)
  4. Produit métriques + Excel
"""

import json
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENTS_INPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\segments\1km"
OUTPUT_DIR = r"C:\Users\Jihene\Downloads\Business-Value-Knowledge-Graph\main\Extraction_et_Embedding\classification_test\Comparaison_multi_llm"

CATEGORIES     = ["ROI", "Notoriété", "Obligation", "Description"]
PRIORITY_ORDER = ["ROI", "Obligation", "Notoriété", "Description"]
QUESTION_KEYS  = ["roi_1","roi_2","roi_3","not_1","not_2","not_3","obl_1","obl_2","obl_3"]

# ── Ensembles à comparer ───────────────────────────────────────────────────────
ENSEMBLE_A = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "mistralai/mistral-nemo",
    "qwen/qwen3-8b",
    "deepseek/deepseek-r1-0528",
]

# Modèles alternatifs (pour les autres substitutions)
ALTERNATIVE_MODELS = {
    "phi4":       "microsoft/phi-4",
    "llama8b":    "meta-llama/llama-3.1-8b-instruct",
    "gemma9b":    "google/gemma-3-9b-it",
    "mistral7b":  "mistralai/mistral-7b-instruct",
    "qwen14b":    "qwen/qwen2.5-14b-instruct",
}

ENSEMBLES = {
    "A_reference": ENSEMBLE_A,

    # Substitution 1 modèle : remplace qwen par phi4
    "B_sans_qwen": [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it",
        "mistralai/mistral-nemo",
        ALTERNATIVE_MODELS["phi4"],
        "deepseek/deepseek-r1-0528",
    ],

    # Substitution 1 modèle : remplace mistral par mistral7b
    "C_sans_mistral_nemo": [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it",
        ALTERNATIVE_MODELS["mistral7b"],
        "qwen/qwen3-8b",
        "deepseek/deepseek-r1-0528",
    ],

    # Substitution 1 modèle : remplace deepseek par llama8b
    "D_sans_deepseek": [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it",
        "mistralai/mistral-nemo",
        "qwen/qwen3-8b",
        ALTERNATIVE_MODELS["llama8b"],
    ],

    # Substitution 2 modèles : remplace qwen + mistral
    "E_sans_qwen_mistral": [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it",
        ALTERNATIVE_MODELS["llama8b"],
        ALTERNATIVE_MODELS["phi4"],
        "deepseek/deepseek-r1-0528",
    ],

    # ================================================================
    # F_3_substitutions : SUPPRESSION de 3 modèles (SANS remplacement)
    # On garde seulement 2 modèles : llama-3.3-70b-instruct et gemma-3-27b-it
    # ================================================================
    "F_3_suppressions": [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemma-3-27b-it",
        # Les 3 modèles suivants sont SUPPRIMÉS (pas de remplacement)
    ],
}


# ── Logique de vote ───────────────────────────────────────────────────────────
def aggregate_responses_from_subset(preds_subset: list[dict]) -> dict:
    valid = [p for p in preds_subset if not p.get("erreur")]
    n = len(valid)
    if n == 0:
        return {k: "non" for k in QUESTION_KEYS}
    return {
        k: ("oui" if sum(1 for p in valid if p["reponses_questions"].get(k) == "oui") > n / 2 else "non")
        for k in QUESTION_KEYS
    }


def compute_scores(reponses: dict) -> tuple[int, int, int]:
    roi  = sum(1 for k in ["roi_1","roi_2","roi_3"] if reponses.get(k) == "oui")
    not_ = sum(1 for k in ["not_1","not_2","not_3"] if reponses.get(k) == "oui")
    obl  = sum(1 for k in ["obl_1","obl_2","obl_3"] if reponses.get(k) == "oui")
    return roi, not_, obl


def determine_label(roi: int, not_: int, obl: int) -> str:
    if roi == 0 and not_ == 0 and obl == 0:
        return "Description"
    scores = {"ROI": roi, "Obligation": obl, "Notoriété": not_}
    max_score = max(scores.values())
    tied = [c for c, s in scores.items() if s == max_score]
    for priority in PRIORITY_ORDER:
        if priority in tied:
            return priority
    return tied[0]


# ── Chargement des JSON ───────────────────────────────────────────────────────
def load_all_classification_jsons(segments_root: str) -> list[dict]:
    root = Path(segments_root)
    files = sorted(root.rglob("*_segments_classification.json"))

    if not files:
        raise FileNotFoundError(
            f"Aucun *_segments_classification.json trouvé dans {segments_root}\n"
            "Lance d'abord multi_llm_paragraph_v3.py pour générer les résultats."
        )

    all_segments = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            item["_source_json"] = str(f)
        all_segments.extend(data)
        print(f"  Chargé : {f.relative_to(root)}  ({len(data)} segments)")

    print(f"\n  Total : {len(all_segments)} segments dans {len(files)} fichier(s)\n")
    return all_segments


# ── Recalcul pour un ensemble donné ───────────────────────────────────────────
def recalculate_for_ensemble(segments: list[dict], ensemble_models: list[str]) -> list[dict]:
    results = []
    for seg in segments:
        all_preds = seg.get("predictions_par_modele", [])

        # Filtre : garde seulement les modèles de l'ensemble courant
        subset = [p for p in all_preds if p["model_name"] in ensemble_models]

        if not subset:
            # Aucun modèle de l'ensemble trouvé → pas de vote possible
            # On garde tous les modèles (fallback) mais on indique le problème
            subset = all_preds
            n_actifs = 0
        else:
            n_actifs = len(subset)

        agg_reponses = aggregate_responses_from_subset(subset)
        roi_s, not_s, obl_s = compute_scores(agg_reponses)
        categorie = determine_label(roi_s, not_s, obl_s)

        results.append({
            "paragraph_index":  seg["paragraph_index"],
            "source_file":      seg["source_file"],
            "source_folder":    seg.get("source_folder", ""),
            "paragraph":        seg["paragraph"][:200],
            "categorie":        categorie,
            "scores_roi":       roi_s,
            "scores_not":       not_s,
            "scores_obl":       obl_s,
            "reponses":         agg_reponses,
            "n_modeles_actifs": n_actifs,
        })
    return results


# ── Dataclass résultat de comparaison ─────────────────────────────────────────
@dataclass
class ComparisonResult:
    nom_ensemble_ref: str
    nom_ensemble_alt: str
    modeles_retires:  list
    modeles_ajoutes:  list
    n_segments:       int
    n_concordants:    int
    taux_concordance: float
    concordance_par_classe:  dict = field(default_factory=dict)
    divergence_par_question: dict = field(default_factory=dict)
    segments_divergents:     list = field(default_factory=list)


# ── Comparaison ref vs alt ────────────────────────────────────────────────────
def compare_ensembles(
    ref_name: str, ref_results: list[dict],
    alt_name: str, alt_results: list[dict],
    ref_models: list[str], alt_models: list[str],
    segments_orig: list[dict],
) -> ComparisonResult:

    concordants = 0
    classe_stats = {c: {"ref": 0, "concordant": 0} for c in CATEGORIES}
    divergents = []
    question_divergence = {k: 0 for k in QUESTION_KEYS}

    for r, a, orig in zip(ref_results, alt_results, segments_orig):
        ref_cat = r["categorie"]
        alt_cat = a["categorie"]
        classe_stats[ref_cat]["ref"] += 1

        if ref_cat == alt_cat:
            concordants += 1
            classe_stats[ref_cat]["concordant"] += 1
        else:
            all_preds = orig.get("predictions_par_modele", [])
            ref_subset = [p for p in all_preds if p["model_name"] in ref_models]
            alt_subset = [p for p in all_preds if p["model_name"] in alt_models]
            if not ref_subset:
                ref_subset = all_preds
            if not alt_subset:
                alt_subset = all_preds

            ref_agg = aggregate_responses_from_subset(ref_subset)
            alt_agg = aggregate_responses_from_subset(alt_subset)

            questions_changees = [k for k in QUESTION_KEYS if ref_agg.get(k) != alt_agg.get(k)]
            for qk in questions_changees:
                question_divergence[qk] += 1

            divergents.append({
                "paragraph_index":    r["paragraph_index"],
                "source_file":        r["source_file"],
                "paragraph":          r["paragraph"][:200],
                "ref_categorie":      ref_cat,
                "alt_categorie":      alt_cat,
                "ref_scores":         f"ROI:{r['scores_roi']} NOT:{r['scores_not']} OBL:{r['scores_obl']}",
                "alt_scores":         f"ROI:{a['scores_roi']} NOT:{a['scores_not']} OBL:{a['scores_obl']}",
                "questions_changees": " | ".join(questions_changees),
                "n_modeles_actifs":   a.get("n_modeles_actifs", len(alt_models)),
            })

    n = len(ref_results)
    concordance_par_classe = {
        c: round(s["concordant"] / s["ref"] * 100, 1) if s["ref"] > 0 else None
        for c, s in classe_stats.items()
    }

    n_div = len(divergents)
    divergence_par_question = {
        k: round(question_divergence[k] / n_div * 100, 1) if n_div > 0 else 0.0
        for k in QUESTION_KEYS
    }

    retires = [m for m in ref_models if m not in alt_models]
    ajoutes = [m for m in alt_models if m not in ref_models]

    return ComparisonResult(
        nom_ensemble_ref=ref_name,
        nom_ensemble_alt=alt_name,
        modeles_retires=retires,
        modeles_ajoutes=ajoutes,
        n_segments=n,
        n_concordants=concordants,
        taux_concordance=round(concordants / n * 100, 2) if n else 0,
        concordance_par_classe=concordance_par_classe,
        divergence_par_question=divergence_par_question,
        segments_divergents=divergents,
    )


# ── Matrice d'influence ───────────────────────────────────────────────────────
def compute_influence_matrix(all_comparisons: list[ComparisonResult]) -> pd.DataFrame:
    influence = defaultdict(list)
    for comp in all_comparisons:
        divergence = 100 - comp.taux_concordance
        for m in comp.modeles_retires:
            influence[m].append(divergence)

    rows = []
    for model, divs in influence.items():
        moy = sum(divs) / len(divs)
        rows.append({
            "Modèle":           model.split("/")[-1],
            "Modèle_complet":   model,
            "Nb_comparaisons":  len(divs),
            "Divergence_moy_%": round(moy, 2),
            "Divergence_max_%": round(max(divs), 2),
            "Influence":        "FORT" if moy > 15 else ("MOYEN" if moy > 5 else "FAIBLE"),
            "Interprétation":   (
                "Ce modèle influence fortement la classification — à garder en priorité"
                if moy > 15 else
                "Ce modèle a un impact modéré"
                if moy > 5 else
                "Ce modèle est interchangeable sans impact majeur"
            ),
        })
    return pd.DataFrame(rows).sort_values("Divergence_moy_%", ascending=False) if rows else pd.DataFrame()


# ── Export Excel ───────────────────────────────────────────────────────────────
def export_comparison_excel(
    all_comparisons: list[ComparisonResult],
    df_influence: pd.DataFrame,
    segments_orig: list[dict],
    ref_results_map: dict,
    out_path: Path,
):
    # Vue globale
    global_rows = []
    for c in all_comparisons:
        row = {
            "Ensemble_Ref":     c.nom_ensemble_ref,
            "Ensemble_Alt":     c.nom_ensemble_alt,
            "Modèles_Retirés":  " | ".join(m.split("/")[-1] for m in c.modeles_retires),
            "Modèles_Ajoutés":  " | ".join(m.split("/")[-1] for m in c.modeles_ajoutes),
            "N_Segments":       c.n_segments,
            "N_Concordants":    c.n_concordants,
            "N_Divergents":     c.n_segments - c.n_concordants,
            "Taux_Concordance": c.taux_concordance,
            "Taux_Divergence":  round(100 - c.taux_concordance, 2),
        }
        for cat, val in c.concordance_par_classe.items():
            row[f"Concord_{cat}_%"] = val
        global_rows.append(row)
    df_global = pd.DataFrame(global_rows)

    # Analyse par question
    question_labels = {
        "roi_1": "ROI-1 : Gain financier",
        "roi_2": "ROI-2 : Amélioration fonctionnelle",
        "roi_3": "ROI-3 : Impact performance org.",
        "not_1": "NOT-1 : Bien-être usager",
        "not_2": "NOT-2 : Label/image positive",
        "not_3": "NOT-3 : Impact perçu positif",
        "obl_1": "OBL-1 : Norme/loi/réglementation",
        "obl_2": "OBL-2 : Sécurité/prévention risques",
        "obl_3": "OBL-3 : Eviter danger/sanction",
    }
    q_rows = []
    for qk in QUESTION_KEYS:
        vals = [c.divergence_par_question.get(qk, 0) for c in all_comparisons]
        q_rows.append({
            "Question":          qk,
            "Label":             question_labels[qk],
            "Divergence_moy_%":  round(sum(vals) / len(vals), 2) if vals else 0,
            "Divergence_max_%":  round(max(vals), 2) if vals else 0,
            "Sensibilité":       "ÉLEVÉE" if (sum(vals)/len(vals) if vals else 0) > 30
                                 else "MOYENNE" if (sum(vals)/len(vals) if vals else 0) > 10
                                 else "FAIBLE",
        })
    df_questions = pd.DataFrame(q_rows).sort_values("Divergence_moy_%", ascending=False)

    # Segments divergents
    div_rows = []
    for c in all_comparisons:
        for d in c.segments_divergents:
            div_rows.append({
                "Ensemble_Alt":       c.nom_ensemble_alt,
                "Modèles_Retirés":    " | ".join(m.split("/")[-1] for m in c.modeles_retires),
                "Source_File":        d["source_file"],
                "Paragraph_Index":    d["paragraph_index"],
                "Paragraph":          d["paragraph"],
                "Catégorie_Ref":      d["ref_categorie"],
                "Catégorie_Alt":      d["alt_categorie"],
                "Scores_Ref":         d["ref_scores"],
                "Scores_Alt":         d["alt_scores"],
                "Questions_Changées": d["questions_changees"],
                "N_Modèles_Actifs":   d.get("n_modeles_actifs", "?"),
            })
    df_divergents = pd.DataFrame(div_rows) if div_rows else pd.DataFrame()

    # Segments instables (changent dans ≥2 ensembles)
    instability = defaultdict(list)
    for c in all_comparisons:
        for d in c.segments_divergents:
            key = (d["source_file"], d["paragraph_index"])
            instability[key].append({
                "ensemble":   c.nom_ensemble_alt,
                "retires":    " | ".join(m.split("/")[-1] for m in c.modeles_retires),
                "classe_alt": d["alt_categorie"],
            })

    instable_rows = []
    for (sf, pi), changes in instability.items():
        if len(changes) >= 2:
            seg_match = next(
                (s for s in segments_orig
                 if s["source_file"] == sf and s["paragraph_index"] == pi), None
            )
            ref_cat = ref_results_map.get((sf, pi), "?")
            classes_vues = list(set(c["classe_alt"] for c in changes))
            instable_rows.append({
                "Source_File":       sf,
                "Paragraph_Index":   pi,
                "Paragraph":         seg_match["paragraph"][:250] if seg_match else "",
                "Catégorie_Ref":     ref_cat,
                "N_Changements":     len(changes),
                "Catégories_Alt":    " | ".join(classes_vues),
                "Ensembles_Alt":     " | ".join(c["ensemble"] for c in changes),
                "Modèles_Retirés":   " | ".join(c["retires"] for c in changes),
                "Diagnostic":        (
                    "Segment AMBIGU — catégorie réelle incertaine, révision manuelle conseillée"
                    if len(classes_vues) > 1
                    else f"Catégorie {ref_cat} robuste mais sensible au choix des modèles"
                ),
            })
    df_instables = pd.DataFrame(instable_rows)
    if not df_instables.empty:
        df_instables = df_instables.sort_values("N_Changements", ascending=False)

    # Segments stables
    all_divergent_keys = set()
    for c in all_comparisons:
        for d in c.segments_divergents:
            all_divergent_keys.add((d["source_file"], d["paragraph_index"]))

    stable_rows = []
    seen = set()
    for seg in segments_orig:
        key = (seg["source_file"], seg["paragraph_index"])
        if key not in all_divergent_keys and key not in seen:
            seen.add(key)
            ref_cat = ref_results_map.get(key, "?")
            stable_rows.append({
                "Source_File":     seg["source_file"],
                "Paragraph_Index": seg["paragraph_index"],
                "Paragraph":       seg["paragraph"][:250],
                "Catégorie":       ref_cat,
                "Fiabilité":       "TRÈS HAUTE — stable dans tous les ensembles testés",
            })
    df_stables = pd.DataFrame(stable_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_global.to_excel(writer,    sheet_name="Vue_Globale",           index=False)
        df_questions.to_excel(writer, sheet_name="Sensibilité_Questions", index=False)
        df_influence.to_excel(writer, sheet_name="Influence_Modèles",     index=False)
        df_divergents.to_excel(writer,sheet_name="Segments_Divergents",   index=False)
        df_instables.to_excel(writer, sheet_name="Segments_Instables",    index=False)
        df_stables.to_excel(writer,   sheet_name="Segments_Stables",      index=False)

    print(f"   Excel sauvegardé → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(ensembles: dict = None):
    if ensembles is None:
        ensembles = ENSEMBLES

    print("=" * 80)
    print("COMPARAISON D'ENSEMBLES — Question : le choix des modèles change-t-il la catégorie ?")
    print("=" * 80)
    print(f"  {len(ensembles)} ensembles définis\n")

    print("  Chargement des résultats de classification...")
    segments = load_all_classification_jsons(SEGMENTS_INPUT_DIR)

    print("  Recalcul des votes par ensemble...")
    ensemble_results = {}
    for ens_name, ens_models in ensembles.items():
        short_models = [m.split("/")[-1] for m in ens_models] if ens_models else ["(aucun modèle)"]
        print(f"    → {ens_name} : {short_models}")
        ensemble_results[ens_name] = recalculate_for_ensemble(segments, ens_models)

    ref_name = "A_reference"
    ref_results = ensemble_results[ref_name]
    ref_models = ensembles[ref_name]

    ref_results_map = {
        (r["source_file"], r["paragraph_index"]): r["categorie"]
        for r in ref_results
    }

    print("\n  Calcul des métriques de comparaison...")
    all_comparisons = []
    for ens_name, ens_results in ensemble_results.items():
        if ens_name == ref_name:
            continue
        comp = compare_ensembles(
            ref_name=ref_name, ref_results=ref_results,
            alt_name=ens_name, alt_results=ens_results,
            ref_models=ref_models, alt_models=ensembles[ens_name],
            segments_orig=segments,
        )
        all_comparisons.append(comp)
        print(f"    {ens_name:30s} : concordance={comp.taux_concordance:5.1f}% | "
              f"divergents={len(comp.segments_divergents):3d}/{comp.n_segments}")

    df_influence = compute_influence_matrix(all_comparisons)
    print("\n  Influence des modèles (impact sur la catégorie finale si retiré) :")
    if not df_influence.empty:
        print(df_influence[["Modèle","Divergence_moy_%","Influence","Interprétation"]].to_string(index=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"comparaison_modeles_{ts}.xlsx"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    export_comparison_excel(
        all_comparisons=all_comparisons,
        df_influence=df_influence,
        segments_orig=segments,
        ref_results_map=ref_results_map,
        out_path=out_path,
    )

    n_total = len(segments)
    all_div_keys = set()
    for c in all_comparisons:
        for d in c.segments_divergents:
            all_div_keys.add((d["source_file"], d["paragraph_index"]))
    n_toujours_stable = n_total - len(all_div_keys)

    print("\n" + "=" * 80)
    print("RÉPONSE À LA QUESTION CENTRALE")
    print("=" * 80)
    print(f"\n  Sur {n_total} segments analysés :")
    print(f"  ✅ {n_toujours_stable} segments ({round(n_toujours_stable/n_total*100,1)}%) "
          f"→ catégorie STABLE peu importe les modèles")
    print(f"  ⚠️  {len(all_div_keys)} segments ({round(len(all_div_keys)/n_total*100,1)}%) "
          f"→ catégorie CHANGE selon les modèles utilisés")

    print("\n  Détail par ensemble :")
    for comp in all_comparisons:
        retires = " | ".join(m.split("/")[-1] for m in comp.modeles_retires)
        print(f"\n  [{comp.nom_ensemble_alt}]  Retiré(s): {retires}")
        print(f"    Concordance globale : {comp.taux_concordance}%")
        for cat, val in comp.concordance_par_classe.items():
            if val is not None:
                bar = "█" * int(val // 10) + "░" * (10 - int(val // 10))
                print(f"    {cat:<12} {bar} {val}%")

    print(f"\n  Excel → {out_path}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        import traceback
        print(f"\nErreur fatale : {e}")
        traceback.print_exc()