"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       ANALYSE COMPARATIVE — VERSION CONTEXTE vs VERSION SANS CONTEXTE       ║
║       Organisée par axe | Basée sur tenseurs NPY + Excels                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

FORMAT DES TENSEURS (identique dans les deux versions) :
  shape = (n_clients, max_phrases, n_models, n_questions)
  dtype = int8   (1 = "oui", 0 = "non")
  axes  = [client, phrase, model, question]

Chaque tenseur est accompagné d'un fichier _meta.json qui contient :
  - clients, models, questions, questions_text
  - max_phrases_per_client, n_models, n_questions

STRUCTURE DES EXCELS (feuilles communes aux deux versions) :
  Toutes_Phrases       → 1 ligne par phrase, colonnes AGG_{qk}, ACCORD_{qk}_%
  Synthese_Par_Fichier → agrégat par fichier source
  Stats_Par_Model      → tokens, coûts, erreurs par modèle
  Accord_Questions_Global → taux d'accord inter-modèles par question
  Distribution_Categories → comptage ROI/Notoriete/Obligation/Description
  Stats_Globales       → coûts totaux, durées
  Accord_Par_Phrase    → taux d'accord par phrase × question

DÉPENDANCES :
  pip install numpy pandas scipy openpyxl matplotlib seaborn statsmodels
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from statsmodels.stats.inter_rater import cohens_kappa
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — adaptez ces chemins à votre environnement
# ══════════════════════════════════════════════════════════════════════════════

# ── Version avec contexte (v2 paragraphes) ────────────────────────────────────
CTX_TENSOR_PATH = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_paragraphes\tensor_complet_20260520_114800.npy"
CTX_META_PATH   = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_paragraphes\tensor_complet_20260520_114800_meta.json"
CTX_EXCEL_PATH  = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_paragraphes\recapitulatif_complet_20260520_114800.xlsx"

# ── Version sans contexte (v5 phrases) ────────────────────────────────────────
NOC_TENSOR_PATH = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_phrases\tensor_complet_20260517_074344.npy" 
NOC_META_PATH   = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_phrases\tensor_complet_20260517_074344_meta.json"
NOC_EXCEL_PATH  = r"F:\Jihene\business_value_classification\classification_test\Classification\resultats_classification_phrases\recapitulatif_complet_20260517_074344.xlsx"

# ── Dossier de sortie pour les résultats ──────────────────────────────────────
OUTPUT_DIR = r"F:\Jihene\business_value_classification\classification_test\analyse_comparative\résultats_analyse_comparative"

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Charge les deux paires (tenseur + méta + excel).
    Retourne un dict avec clés 'ctx' et 'noc'.
    """
    print("=" * 70)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 70)

    def _load_one(tensor_path, meta_path, excel_path, label):
        T    = np.load(tensor_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        xl   = pd.read_excel(excel_path, sheet_name=None)
        print(f"\n  [{label}]")
        print(f"    Tensor shape : {T.shape}  — (clients, phrases, modèles, questions)")
        print(f"    Clients      : {meta['n_clients']}")
        print(f"    Modèles      : {meta['n_models']}  → {meta['models']}")
        print(f"    Questions    : {meta['n_questions']}  → {meta['questions']}")
        print(f"    Excel sheets : {list(xl.keys())}")
        return {"tensor": T, "meta": meta, "excel": xl}

    ctx = _load_one(CTX_TENSOR_PATH, CTX_META_PATH, CTX_EXCEL_PATH, "AVEC CONTEXTE")
    noc = _load_one(NOC_TENSOR_PATH, NOC_META_PATH, NOC_EXCEL_PATH, "SANS CONTEXTE")

    # ── Alignement des phrases communes ──────────────────────────────────────
    # On travaille sur les phrases présentes dans les deux versions(sauf les phrasezs qui n'appartiennent à aucun paragraphe donc sans contexte)
    df_ctx = ctx["excel"]["Toutes_Phrases"].copy()
    df_noc = noc["excel"]["Toutes_Phrases"].copy()

    # Clé de jointure : (source_file, phrase_index)
    # La version paragraphes inclut les titres (phrase_index == -1) — on les garde
    df_ctx["_key"] = df_ctx["Source_File"].astype(str) + "||" + df_ctx["Phrase_Index"].astype(str)
    df_noc["_key"] = df_noc["Source_File"].astype(str) + "||" + df_noc["Phrase_Index"].astype(str)

    common_keys = set(df_ctx["_key"]) & set(df_noc["_key"])
    print(f"\n  Phrases communes aux deux versions : {len(common_keys)}")
    print(f"  Phrases uniquement dans CTX       : {len(df_ctx) - len(common_keys)}")
    print(f"  Phrases uniquement dans NOC       : {len(df_noc) - len(common_keys)}")

    df_ctx_common = df_ctx[df_ctx["_key"].isin(common_keys)].set_index("_key")
    df_noc_common = df_noc[df_noc["_key"].isin(common_keys)].set_index("_key")

    ctx["df"]        = df_ctx_common
    noc["df"]        = df_noc_common
    ctx["df_full"]   = df_ctx
    noc["df_full"]   = df_noc

    return ctx, noc, sorted(common_keys)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_KEYS_DEFAULT = [
    "roi_1", "roi_2", "roi_3",
    "notoriete_1", "notoriete_2", "notoriete_3",
    "obl_1", "obl_2", "obl_3",
]
LABEL_FAMILIES = {
    "ROI":       ["roi_1", "roi_2", "roi_3"],
    "NOT":       ["notoriete_1", "notoriete_2", "notoriete_3"],
    "OBL":       ["obl_1", "obl_2", "obl_3"],
}
CATEGORIES = ["ROI", "Notoriete", "Obligation", "Description"]

def get_question_keys(meta):
    return meta.get("questions", QUESTION_KEYS_DEFAULT)

def save_excel(dfs: dict, path: str):
    """Sauvegarde un dict {sheet_name: DataFrame} dans un fichier Excel."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"  → Sauvegardé : {path}")

def save_fig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure : {path}")

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 1 — ACCORD GLOBAL ENTRE LES DEUX VERSIONS
#  Métriques : taux d'accord brut, Cohen's Kappa par question et par famille
# ══════════════════════════════════════════════════════════════════════════════

def axe1_accord_global(ctx, noc, common_keys):
    """
    Compare les réponses agrégées (majoritaires) des deux versions
    pour chaque phrase commune × chaque question.

    Colonnes utilisées dans Toutes_Phrases :
      AGG_{qk}  → réponse agrégée après vote majoritaire ("oui" / "non")
    """
    print("\n" + "=" * 70)
    print("AXE 1 — ACCORD GLOBAL ENTRE LES DEUX VERSIONS")
    print("=" * 70)

    qkeys  = get_question_keys(ctx["meta"])
    df_ctx = ctx["df"]
    df_noc = noc["df"]

    results = []
    vectors_ctx = {}
    vectors_noc = {}

    for qk in qkeys:
        col = f"AGG_{qk}"
        if col not in df_ctx.columns or col not in df_noc.columns:
            print(f"  [AVERT] colonne {col} absente — question ignorée")
            continue

        v_ctx = (df_ctx.loc[common_keys, col] == "oui").astype(int).values
        v_noc = (df_noc.loc[common_keys, col] == "oui").astype(int).values
        vectors_ctx[qk] = v_ctx
        vectors_noc[qk] = v_noc

        n     = len(v_ctx)
        agree = int(np.sum(v_ctx == v_noc))
        taux_brut = agree / n if n else 0

        # Cohen's Kappa via tableau de contingence 2×2
        # tn=00, fp=01, fn=10, tp=11
        tn = int(np.sum((v_ctx == 0) & (v_noc == 0)))
        fp = int(np.sum((v_ctx == 0) & (v_noc == 1)))
        fn = int(np.sum((v_ctx == 1) & (v_noc == 0)))
        tp = int(np.sum((v_ctx == 1) & (v_noc == 1)))
        table = np.array([[tn, fp], [fn, tp]])

        try:
            kappa_res = cohens_kappa(table, return_results=False)
            kappa_val = float(kappa_res)
        except Exception:
            # Calcul manuel si statsmodels échoue
            po = (tn + tp) / n if n else 0
            p1 = ((tn + fn) / n) * ((tn + fp) / n) + \
                 ((fp + tp) / n) * ((fn + tp) / n)
            kappa_val = (po - p1) / (1 - p1) if (1 - p1) > 0 else 0.0

        # Famille
        famille = "ROI" if qk.startswith("roi") else \
                  "NOT" if qk.startswith("not") else "OBL"

        results.append({
            "Question_Key": qk,
            "Famille":      famille,
            "N_Phrases":    n,
            "N_Accord":     agree,
            "N_Desaccord":  n - agree,
            "Taux_Accord_%": round(taux_brut * 100, 2),
            "Cohen_Kappa":  round(kappa_val, 4),
            "Kappa_Interpretation": _kappa_label(kappa_val),
            "TN_(ctx=non,noc=non)": tn,
            "FP_(ctx=non,noc=oui)": fp,
            "FN_(ctx=oui,noc=non)": fn,
            "TP_(ctx=oui,noc=oui)": tp,
            "Pct_oui_CTX_%": round(v_ctx.mean() * 100, 2),
            "Pct_oui_NOC_%": round(v_noc.mean() * 100, 2),
        })

    df_res = pd.DataFrame(results)

    # Résumé par famille
    famille_summary = []
    for fam, fam_keys in LABEL_FAMILIES.items():
        sub = df_res[df_res["Question_Key"].isin(fam_keys)]
        if len(sub) == 0:
            continue
        famille_summary.append({
            "Famille":              fam,
            "Taux_Accord_Moy_%":   round(sub["Taux_Accord_%"].mean(), 2),
            "Kappa_Moy":           round(sub["Cohen_Kappa"].mean(), 4),
            "Kappa_Min":           round(sub["Cohen_Kappa"].min(), 4),
            "Kappa_Max":           round(sub["Cohen_Kappa"].max(), 4),
            "Kappa_Interpretation":_kappa_label(sub["Cohen_Kappa"].mean()),
        })
    df_fam = pd.DataFrame(famille_summary)

    # Kappa global (toutes questions)
    all_v_ctx = np.concatenate([vectors_ctx[qk] for qk in vectors_ctx])
    all_v_noc = np.concatenate([vectors_noc[qk] for qk in vectors_noc])
    n_all = len(all_v_ctx)
    tp_g = int(np.sum((all_v_ctx == 1) & (all_v_noc == 1)))
    tn_g = int(np.sum((all_v_ctx == 0) & (all_v_noc == 0)))
    fp_g = int(np.sum((all_v_ctx == 0) & (all_v_noc == 1)))
    fn_g = int(np.sum((all_v_ctx == 1) & (all_v_noc == 0)))
    table_g = np.array([[tn_g, fp_g], [fn_g, tp_g]])
    try:
        kappa_global = float(cohens_kappa(table_g, return_results=False))
    except Exception:
        po = (tn_g + tp_g) / n_all
        p1 = ((tn_g + fn_g) / n_all) * ((tn_g + fp_g) / n_all) + \
             ((fp_g + tp_g) / n_all) * ((fn_g + tp_g) / n_all)
        kappa_global = (po - p1) / (1 - p1) if (1 - p1) > 0 else 0.0

    df_global = pd.DataFrame([{
        "N_Phrases_Communes":    len(common_keys),
        "N_Observations_Total":  n_all,
        "Taux_Accord_Global_%":  round(np.mean(all_v_ctx == all_v_noc) * 100, 2),
        "Cohen_Kappa_Global":    round(kappa_global, 4),
        "Kappa_Interpretation":  _kappa_label(kappa_global),
    }])

    # Sauvegarde
    save_excel({
        "Accord_Par_Question": df_res,
        "Accord_Par_Famille":  df_fam,
        "Accord_Global":       df_global,
    }, str(out / "axe1_accord_global.xlsx"))

    # Figure — heatmap Kappa par question
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    kappa_vals = df_res.set_index("Question_Key")["Cohen_Kappa"]
    colors = [_kappa_color(v) for v in kappa_vals.values]
    axes[0].barh(kappa_vals.index, kappa_vals.values, color=colors)
    axes[0].axvline(0.8, color="green",  ls="--", lw=1, label="Quasi-parfait (0.8)")
    axes[0].axvline(0.6, color="orange", ls="--", lw=1, label="Bon (0.6)")
    axes[0].axvline(0.4, color="red",    ls="--", lw=1, label="Modéré (0.4)")
    axes[0].set_xlabel("Cohen's Kappa")
    axes[0].set_title("Kappa par question")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(-0.1, 1.05)

    taux_vals = df_res.set_index("Question_Key")["Taux_Accord_%"]
    axes[1].barh(taux_vals.index, taux_vals.values, color="#5b8dd9")
    axes[1].axvline(80, color="orange", ls="--", lw=1, label="Seuil 80%")
    axes[1].set_xlabel("Taux d'accord brut (%)")
    axes[1].set_title("Taux d'accord brut par question")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 105)
    plt.tight_layout()
    save_fig(fig, str(out / "axe1_kappa_par_question.png"))

    # Affichage console
    print(f"\n  Kappa global (toutes questions) : {kappa_global:.4f}  "
          f"→ {_kappa_label(kappa_global)}")
    print(f"\n  Kappa par question :")
    print(df_res[["Question_Key", "Famille", "Taux_Accord_%",
                  "Cohen_Kappa", "Kappa_Interpretation"]].to_string(index=False))
    print(f"\n  Par famille :")
    print(df_fam.to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 1                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Kappa > 0.8  → accord quasi-parfait : le contexte n'apporte pas      │
  │    de différence significative pour cette question. Les deux versions    │
  │    sont interchangeables.                                               │
  │  • Kappa 0.6–0.8 → bon accord : légères divergences, probablement sur  │
  │    des phrases courtes ou ambiguës. Le contexte aide ponctuellement.    │
  │  • Kappa 0.4–0.6 → accord modéré : le contexte change la réponse       │
  │    sur une fraction non négligeable des phrases. Creuser l'Axe 2.       │
  │  • Kappa < 0.4  → désaccord substantiel : le contexte influence         │
  │    fortement le jugement du LLM. La question est sensible au contexte. │
  │                                                                          │
  │  • Si une famille (ROI / NOT / OBL) a un Kappa systématiquement plus   │
  │    bas, cela signifie que ses questions sont intrinsèquement plus       │
  │    dépendantes du contexte sémantique du paragraphe.                    │
  │                                                                          │
  │  • Comparez Pct_oui_CTX vs Pct_oui_NOC : si CTX donne plus de "oui",  │
  │    le contexte induit une sur-attribution de bénéfice. Si NOC donne     │
  │    plus de "oui", la phrase isolée est sur-interprétée sans contexte.   │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_res, vectors_ctx, vectors_noc


def _kappa_label(k):
    if k >= 0.8:  return "Quasi-parfait"
    if k >= 0.6:  return "Bon"
    if k >= 0.4:  return "Modéré"
    if k >= 0.2:  return "Faible"
    return "Très faible / Pas d'accord"

def _kappa_color(k):
    if k >= 0.8:  return "#2ecc71"
    if k >= 0.6:  return "#f39c12"
    if k >= 0.4:  return "#e67e22"
    return "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 2 — EFFET DU CONTEXTE SUR LES DÉCISIONS (FLIPS)
#  Métriques : taux de flip, direction du flip, asymétrie oui→non vs non→oui
# ══════════════════════════════════════════════════════════════════════════════

def axe2_effet_contexte(ctx, noc, common_keys, vectors_ctx, vectors_noc):
    """
    Un 'flip' = la réponse change entre les deux versions pour une même phrase.
    Direction :
      ctx=oui / noc=non → "oui→non" : le contexte AJOUTE la réponse positive
      ctx=non / noc=oui → "non→oui" : le contexte RETIRE la réponse positive
    """
    print("\n" + "=" * 70)
    print("AXE 2 — EFFET DU CONTEXTE SUR LES DÉCISIONS (FLIPS)")
    print("=" * 70)

    qkeys   = list(vectors_ctx.keys())
    df_ctx  = ctx["df"]
    df_noc  = noc["df"]

    flip_rows = []
    phrase_flip_counts = {k: 0 for k in common_keys}

    for qk in qkeys:
        v_ctx = vectors_ctx[qk]
        v_noc = vectors_noc[qk]
        n = len(v_ctx)

        flipped      = v_ctx != v_noc
        n_flip       = int(flipped.sum())
        n_oui_vers_non = int(((v_ctx == 1) & (v_noc == 0)).sum())  # ctx=oui, noc=non
        n_non_vers_oui = int(((v_ctx == 0) & (v_noc == 1)).sum())  # ctx=non, noc=oui

        famille = "ROI" if qk.startswith("roi") else \
                  "NOT" if qk.startswith("not") else "OBL"

        flip_rows.append({
            "Question_Key":         qk,
            "Famille":              famille,
            "N_Phrases":            n,
            "N_Flips":              n_flip,
            "Taux_Flip_%":          round(n_flip / n * 100, 2) if n else 0,
            "N_CTX_oui_NOC_non":    n_oui_vers_non,
            "N_CTX_non_NOC_oui":    n_non_vers_oui,
            "Pct_flip_ctx_plus_%":  round(n_oui_vers_non / n_flip * 100, 1) if n_flip else 0,
            "Pct_flip_noc_plus_%":  round(n_non_vers_oui / n_flip * 100, 1) if n_flip else 0,
            "Direction_Dominante":  "CTX plus permissif" if n_oui_vers_non > n_non_vers_oui
                                    else "NOC plus permissif" if n_non_vers_oui > n_oui_vers_non
                                    else "Symétrique",
        })

        # Comptage flip par phrase
        for i, key in enumerate(common_keys):
            if flipped[i]:
                phrase_flip_counts[key] += 1

    df_flips = pd.DataFrame(flip_rows)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # Résumé par famille
    fam_rows = []
    for fam, fam_keys in LABEL_FAMILIES.items():
        sub = df_flips[df_flips["Question_Key"].isin(fam_keys)]
        if len(sub) == 0:
            continue
        fam_rows.append({
            "Famille":             fam,
            "Taux_Flip_Moy_%":     round(sub["Taux_Flip_%"].mean(), 2),
            "Taux_Flip_Max_%":     round(sub["Taux_Flip_%"].max(), 2),
            "N_Total_Flips":       int(sub["N_Flips"].sum()),
            "N_CTX_plus_oui":      int(sub["N_CTX_oui_NOC_non"].sum()),
            "N_NOC_plus_oui":      int(sub["N_CTX_non_NOC_oui"].sum()),
            "Direction_Globale":   "CTX plus permissif"
                                   if sub["N_CTX_oui_NOC_non"].sum() > sub["N_CTX_non_NOC_oui"].sum()
                                   else "NOC plus permissif",
        })
    df_fam_flips = pd.DataFrame(fam_rows)

    # Distribution du nombre de flips par phrase
    flip_dist = pd.Series(phrase_flip_counts).value_counts().sort_index()
    df_flip_dist = pd.DataFrame({
        "N_questions_flippees": flip_dist.index,
        "N_phrases":            flip_dist.values,
        "Pct_%":                (flip_dist.values / len(common_keys) * 100).round(2),
    })

    # Phrases avec le plus de flips
    top_flip_phrases = pd.DataFrame([
        {
            "Key":          k,
            "Source_File":  df_ctx.loc[k, "Source_File"] if k in df_ctx.index else "",
            "Phrase_Index": df_ctx.loc[k, "Phrase_Index"] if k in df_ctx.index else "",
            "Phrase_Text":  (df_ctx.loc[k, "Phrase_Text"] if "Phrase_Text" in df_ctx.columns
                             else df_ctx.loc[k, "Phrase"] if "Phrase" in df_ctx.columns else "")[:120]
                             if k in df_ctx.index else "",
            "N_Flips":      phrase_flip_counts[k],
            "Label_CTX":    df_ctx.loc[k, "Category"] if k in df_ctx.index else "",
            "Label_NOC":    df_noc.loc[k, "Category"] if k in df_noc.index else "",
        }
        for k in common_keys
    ]).sort_values("N_Flips", ascending=False).head(30)

    # Sauvegarde
    save_excel({
        "Flips_Par_Question":    df_flips,
        "Flips_Par_Famille":     df_fam_flips,
        "Distribution_N_Flips":  df_flip_dist,
        "Top_Phrases_Flippees":  top_flip_phrases,
    }, str(out / "axe2_effet_contexte.xlsx"))

    # Figure — taux de flip par question + direction
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_plot = df_flips.set_index("Question_Key")
    bar_colors = ["#e74c3c" if d == "CTX plus permissif" else "#3498db"
                  for d in df_plot["Direction_Dominante"]]
    axes[0].barh(df_plot.index, df_plot["Taux_Flip_%"], color=bar_colors)
    axes[0].set_xlabel("Taux de flip (%)")
    axes[0].set_title("Taux de flip par question\n(rouge=CTX+, bleu=NOC+)")
    axes[0].axvline(10, color="gray", ls="--", lw=1, label="Seuil 10%")
    axes[0].legend(fontsize=8)

    # Barres empilées ctx vs noc pour les flips
    df_flips_plot = df_flips[df_flips["N_Flips"] > 0]
    if len(df_flips_plot):
        x = range(len(df_flips_plot))
        axes[1].bar(x, df_flips_plot["N_CTX_oui_NOC_non"],
                    label="CTX=oui / NOC=non (contexte +)", color="#e74c3c", alpha=0.8)
        axes[1].bar(x, df_flips_plot["N_CTX_non_NOC_oui"],
                    bottom=df_flips_plot["N_CTX_oui_NOC_non"],
                    label="CTX=non / NOC=oui (contexte −)", color="#3498db", alpha=0.8)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(df_flips_plot["Question_Key"].tolist(),
                                rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Nombre de flips")
        axes[1].set_title("Direction des flips par question")
        axes[1].legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, str(out / "axe2_flips_par_question.png"))

    # Distribution des flips par phrase
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_flip_dist["N_questions_flippees"], df_flip_dist["N_phrases"], color="#9b59b6")
    ax.set_xlabel("Nombre de questions flippées par phrase")
    ax.set_ylabel("Nombre de phrases")
    ax.set_title("Distribution du nombre de flips par phrase")
    plt.tight_layout()
    save_fig(fig2, str(out / "axe2_distribution_flips.png"))

    print(f"\n  Taux de flip global (toutes questions) : "
          f"{df_flips['Taux_Flip_%'].mean():.2f}%  (moy. par question)")
    print(f"\n  Par question :")
    print(df_flips[["Question_Key", "Famille", "Taux_Flip_%",
                     "Direction_Dominante"]].to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 2                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Taux de flip < 5%  → le contexte est quasi-neutre pour cette         │
  │    question : les deux versions arrivent à la même conclusion.           │
  │  • Taux de flip 5–15% → impact modéré du contexte. Regarder la          │
  │    direction : si CTX donne plus de "oui", le contexte "charge" le       │
  │    jugement du LLM et peut induire des faux positifs.                   │
  │  • Taux de flip > 15% → impact fort. Cette question est intrinsèquement │
  │    dépendante du contexte sémantique. Sans contexte, le LLM ne peut pas │
  │    correctement l'évaluer.                                               │
  │                                                                          │
  │  • Direction "CTX plus permissif" (CTX=oui, NOC=non) :                  │
  │    → Le paragraphe "contamine" l'évaluation de la phrase. Le système    │
  │      prompt "Le paragraphe sert UNIQUEMENT à désambiguïser" n'est       │
  │      peut-être pas parfaitement respecté par tous les modèles.          │
  │                                                                          │
  │  • Direction "NOC plus permissif" (CTX=non, NOC=oui) :                  │
  │    → La phrase hors contexte est sur-interprétée. Le modèle comble       │
  │      l'ambiguïté en faveur d'un "oui". Le contexte joue son rôle        │
  │      désambiguïsant correctement.                                        │
  │                                                                          │
  │  • Phrases avec 5+ flips → phrases très ambiguës : elles méritent une  │
  │    annotation manuelle pour valider quelle version est la meilleure.     │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_flips


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 3 — ACCORD INTER-MODÈLES PAR VERSION
#  Métriques : taux d'accord interne à chaque version, comparaison CTX vs NOC
#  Source : tenseur (n_clients, n_phrases, n_models, n_questions)
# ══════════════════════════════════════════════════════════════════════════════

def axe3_accord_intermodeles(ctx, noc):
    """
    Pour chaque version, calcule le taux d'accord entre les 5 modèles
    (unanimité) par question et par famille.
    Compare ensuite si une version obtient un meilleur consensus interne.

    Source : tenseur NPY shape = (clients, phrases, models, questions)
    """
    print("\n" + "=" * 70)
    print("AXE 3 — ACCORD INTER-MODÈLES PAR VERSION")
    print("=" * 70)

    def _compute_intermodel_stats(tensor, meta, label):
        """
        tensor : (C, P, M, Q)
        Accord = unanimité (tous à 1 ou tous à 0)
        """
        C, P, M, Q = tensor.shape
        qkeys = meta["questions"]

        rows = []
        for qi, qk in enumerate(qkeys):
            slice_q = tensor[:, :, :, qi]  # (C, P, M)

            # Masque de phrases réelles (au moins 1 modèle a répondu)
            has_data = slice_q.sum(axis=2) + (M - slice_q.sum(axis=2)) > 0
            flat_votes = slice_q.reshape(-1, M)  # (C*P, M)

            votes_oui   = flat_votes.sum(axis=1)     # par phrase
            unanimous_oui = (votes_oui == M).sum()
            unanimous_non = (votes_oui == 0).sum()
            n_total       = len(flat_votes)
            n_accord      = int(unanimous_oui + unanimous_non)

            # Entropie moyenne du vote (0 = accord parfait, 1 = désaccord max)
            # H = -p*log2(p) - (1-p)*log2(1-p)
            p = votes_oui / M
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = np.where(
                    (p > 0) & (p < 1),
                    -(p * np.log2(p) + (1 - p) * np.log2(1 - p)),
                    0.0
                )

            famille = "ROI" if qk.startswith("roi") else \
                      "NOT" if qk.startswith("not") else "OBL"

            rows.append({
                "Question_Key":        qk,
                "Famille":             famille,
                "Version":             label,
                "N_Observations":      n_total,
                "N_Unanime_oui":       int(unanimous_oui),
                "N_Unanime_non":       int(unanimous_non),
                "N_Accord_Total":      n_accord,
                "Taux_Accord_%":       round(n_accord / n_total * 100, 2) if n_total else 0,
                "Entropie_Moy":        round(float(entropy.mean()), 4),
                "Pct_oui_Moy_%":       round(float(p.mean()) * 100, 2),
                "Pct_Majorite_3+_%":   round(float((votes_oui >= 3).mean()) * 100, 2),
            })

        return pd.DataFrame(rows)

    df_ctx_acc = _compute_intermodel_stats(ctx["tensor"], ctx["meta"], "CTX")
    df_noc_acc = _compute_intermodel_stats(noc["tensor"], noc["meta"], "NOC")

    # Comparaison côte à côte
    df_merge = pd.merge(
        df_ctx_acc[["Question_Key", "Famille", "Taux_Accord_%", "Entropie_Moy",
                    "Pct_oui_Moy_%"]].rename(
            columns={"Taux_Accord_%": "Accord_CTX_%",
                     "Entropie_Moy":  "Entropie_CTX",
                     "Pct_oui_Moy_%": "PctOui_CTX_%"}),
        df_noc_acc[["Question_Key", "Taux_Accord_%", "Entropie_Moy",
                    "Pct_oui_Moy_%"]].rename(
            columns={"Taux_Accord_%": "Accord_NOC_%",
                     "Entropie_Moy":  "Entropie_NOC",
                     "Pct_oui_Moy_%": "PctOui_NOC_%"}),
        on="Question_Key",
    )
    df_merge["Delta_Accord_%"]  = (df_merge["Accord_CTX_%"] - df_merge["Accord_NOC_%"]).round(2)
    df_merge["Delta_Entropie"]  = (df_merge["Entropie_CTX"] - df_merge["Entropie_NOC"]).round(4)
    df_merge["Meilleur_Accord"] = df_merge.apply(
        lambda r: "CTX" if r["Accord_CTX_%"] > r["Accord_NOC_%"]
                  else "NOC" if r["Accord_NOC_%"] > r["Accord_CTX_%"]
                  else "Égal", axis=1
    )

    # Par famille
    fam_rows = []
    for fam, fam_keys in LABEL_FAMILIES.items():
        sub = df_merge[df_merge["Question_Key"].isin(fam_keys)]
        if len(sub) == 0:
            continue
        fam_rows.append({
            "Famille":            fam,
            "Accord_CTX_Moy_%":   round(sub["Accord_CTX_%"].mean(), 2),
            "Accord_NOC_Moy_%":   round(sub["Accord_NOC_%"].mean(), 2),
            "Delta_Moy_%":        round(sub["Delta_Accord_%"].mean(), 2),
            "Entropie_CTX_Moy":   round(sub["Entropie_CTX"].mean(), 4),
            "Entropie_NOC_Moy":   round(sub["Entropie_NOC"].mean(), 4),
            "Meilleur_Consensus": "CTX" if sub["Delta_Accord_%"].mean() > 0
                                  else "NOC" if sub["Delta_Accord_%"].mean() < 0
                                  else "Égal",
        })
    df_fam_acc = pd.DataFrame(fam_rows)

    save_excel({
        "Accord_CTX":         df_ctx_acc,
        "Accord_NOC":         df_noc_acc,
        "Comparaison":        df_merge,
        "Par_Famille":        df_fam_acc,
    }, str(out / "axe3_accord_intermodeles.xlsx"))

    # Figure — comparaison accord CTX vs NOC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df_merge))
    w = 0.35
    axes[0].bar(x - w/2, df_merge["Accord_CTX_%"], w, label="Avec contexte",  color="#2ecc71", alpha=0.85)
    axes[0].bar(x + w/2, df_merge["Accord_NOC_%"], w, label="Sans contexte",  color="#3498db", alpha=0.85)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df_merge["Question_Key"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Taux d'accord inter-modèles (%)")
    axes[0].set_title("Accord inter-modèles : CTX vs NOC")
    axes[0].legend()
    axes[0].set_ylim(0, 105)

    axes[1].bar(x, df_merge["Delta_Accord_%"],
                color=["#2ecc71" if v > 0 else "#e74c3c" for v in df_merge["Delta_Accord_%"]])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df_merge["Question_Key"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Delta accord (CTX − NOC) %")
    axes[1].set_title("Delta accord : vert=CTX meilleur, rouge=NOC meilleur")
    plt.tight_layout()
    save_fig(fig, str(out / "axe3_accord_intermodeles.png"))

    print(f"\n  Comparaison accord inter-modèles CTX vs NOC :")
    print(df_merge[["Question_Key", "Famille",
                    "Accord_CTX_%", "Accord_NOC_%",
                    "Delta_Accord_%", "Meilleur_Accord"]].to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 3                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Delta_Accord positif (CTX > NOC) → avec contexte, les 5 modèles      │
  │    s'accordent mieux entre eux. Le contexte réduit l'ambiguïté et       │
  │    harmonise les jugements. C'est le signe que le contexte est utile.   │
  │                                                                          │
  │  • Delta_Accord négatif (NOC > CTX) → le contexte introduit de la       │
  │    divergence entre modèles : certains l'utilisent différemment.        │
  │    Cela peut indiquer que des modèles violent la règle "le paragraphe   │
  │    sert uniquement à désambiguïser".                                    │
  │                                                                          │
  │  • Entropie_CTX < Entropie_NOC → les votes sont plus concentrés         │
  │    (moins dispersés) avec contexte → meilleure cohérence d'ensemble.    │
  │                                                                          │
  │  • Si une famille (ex : NOT) a systématiquement Delta négatif →         │
  │    les questions de Notoriété sont plus sujettes à l'effet "pollution    │
  │    contextuelle" : le ton positif du paragraphe gonfle les réponses.    │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_merge


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 4 — ANALYSE PAR QUESTION (Q1–Q9) : HEATMAP & SENSIBILITÉ
#  Métriques : heatmap accord × question, questions les plus sensibles
# ══════════════════════════════════════════════════════════════════════════════

def axe4_par_question(ctx, noc, common_keys, vectors_ctx, vectors_noc, df_flips):
    """
    Heatmap combinée : pour chaque question, visualise ensemble
    - le Kappa entre les deux versions
    - le taux de flip
    - le taux d'accord inter-modèles CTX et NOC
    """
    print("\n" + "=" * 70)
    print("AXE 4 — ANALYSE PAR QUESTION (Q1–Q9)")
    print("=" * 70)

    qkeys = list(vectors_ctx.keys())

    # Accord inter-modèles depuis les colonnes ACCORD_{qk}_% de l'Excel
    df_ctx_full = ctx["df"]
    df_noc_full = noc["df"]

    q_rows = []
    for qk in qkeys:
        v_ctx = vectors_ctx[qk]
        v_noc = vectors_noc[qk]
        n     = len(v_ctx)

        # Kappa
        tp = int(np.sum((v_ctx == 1) & (v_noc == 1)))
        tn = int(np.sum((v_ctx == 0) & (v_noc == 0)))
        fp = int(np.sum((v_ctx == 0) & (v_noc == 1)))
        fn = int(np.sum((v_ctx == 1) & (v_noc == 0)))
        table = np.array([[tn, fp], [fn, tp]])
        try:
            kappa = float(cohens_kappa(table, return_results=False))
        except Exception:
            po = (tn + tp) / n if n else 0
            p1 = ((tn + fn) / n) * ((tn + fp) / n) + \
                 ((fp + tp) / n) * ((fn + tp) / n)
            kappa = (po - p1) / (1 - p1) if (1 - p1) > 0 else 0.0

        # Flip rate
        flip_row = df_flips[df_flips["Question_Key"] == qk]
        flip_rate = float(flip_row["Taux_Flip_%"].values[0]) if len(flip_row) else 0

        # Accord inter-modèles dans l'Excel (colonne ACCORD_{qk}_%)
        acc_col = f"ACCORD_{qk}_%"
        acc_ctx = float(df_ctx_full[acc_col].mean()) if acc_col in df_ctx_full.columns else 0
        acc_noc = float(df_noc_full[acc_col].mean()) if acc_col in df_noc_full.columns else 0

        famille = "ROI" if qk.startswith("roi") else \
                  "NOT" if qk.startswith("not") else "OBL"

        q_rows.append({
            "Question_Key":       qk,
            "Famille":            famille,
            "Kappa_CTX_vs_NOC":   round(kappa, 4),
            "Taux_Flip_%":        round(flip_rate, 2),
            "Accord_IM_CTX_%":    round(acc_ctx, 2),
            "Accord_IM_NOC_%":    round(acc_noc, 2),
            "Score_Sensibilite":  round((100 - kappa * 100 + flip_rate) / 2, 2),
            "Rang_Sensibilite":   0,  # sera calculé après
        })

    df_q = pd.DataFrame(q_rows)
    df_q["Rang_Sensibilite"] = df_q["Score_Sensibilite"].rank(ascending=False).astype(int)
    df_q = df_q.sort_values("Rang_Sensibilite")

    # Matrice pour heatmap
    heatmap_data = df_q.set_index("Question_Key")[
        ["Kappa_CTX_vs_NOC", "Taux_Flip_%", "Accord_IM_CTX_%", "Accord_IM_NOC_%"]
    ]

    save_excel({
        "Analyse_Par_Question": df_q,
        "Heatmap_Data":         heatmap_data.reset_index(),
    }, str(out / "axe4_par_question.xlsx"))

    # Figure — heatmap 4 métriques × 9 questions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap normalisée
    norm_data = heatmap_data.copy()
    norm_data["Kappa_CTX_vs_NOC"] = norm_data["Kappa_CTX_vs_NOC"] * 100
    sns.heatmap(
        norm_data, ax=axes[0], annot=True, fmt=".1f",
        cmap="RdYlGn", vmin=0, vmax=100,
        linewidths=0.5, cbar_kws={"label": "Score (%)"}
    )
    axes[0].set_title("Heatmap des métriques par question\n(Kappa×100 / Flip% / AccordIM%)")
    axes[0].set_xlabel("")

    # Radar-like bar groupé
    x    = np.arange(len(df_q))
    w    = 0.2
    axes[1].bar(x - w*1.5, df_q["Kappa_CTX_vs_NOC"] * 100, w, label="Kappa×100", color="#9b59b6", alpha=0.8)
    axes[1].bar(x - w*0.5, df_q["Taux_Flip_%"],             w, label="Flip%",     color="#e74c3c", alpha=0.8)
    axes[1].bar(x + w*0.5, df_q["Accord_IM_CTX_%"],         w, label="AccIM-CTX", color="#2ecc71", alpha=0.8)
    axes[1].bar(x + w*1.5, df_q["Accord_IM_NOC_%"],         w, label="AccIM-NOC", color="#3498db", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df_q["Question_Key"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("4 métriques par question")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, str(out / "axe4_heatmap_questions.png"))

    print(f"\n  Questions classées par sensibilité (plus sensible = rang 1) :")
    print(df_q[["Rang_Sensibilite", "Question_Key", "Famille",
                "Kappa_CTX_vs_NOC", "Taux_Flip_%",
                "Score_Sensibilite"]].to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 4                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Score_Sensibilite élevé = la question est fortement impactée par le  │
  │    contexte. À l'inverse un score faible signifie que les deux versions │
  │    convergent naturellement sur cette question.                          │
  │                                                                          │
  │  • Questions de rang 1–3 : ce sont les "questions critiques". Leur      │
  │    résultat dépend fortement du mode de classification. Il faut décider │
  │    laquelle des deux versions est la plus fiable en les analysant        │
  │    manuellement sur un échantillon.                                      │
  │                                                                          │
  │  • Si toutes les questions d'une même famille (ex : OBL) sont en bas    │
  │    du classement de sensibilité → les questions réglementaires sont      │
  │    stables et ne nécessitent pas de contexte. Utiliser la version NOC   │
  │    (moins coûteuse) pour ces questions est justifié.                    │
  │                                                                          │
  │  • Accord_IM_CTX % < Accord_IM_NOC % pour une question → le contexte  │
  │    divise les modèles : certains l'intègrent, d'autres l'ignorent.      │
  │    Cela fragilise la fiabilité du vote majoritaire CTX.                  │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_q


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 5 — ANALYSE PAR CARACTÉRISTIQUES DE PHRASE
#  Métriques : longueur, position, présence de pronoms, flip rate associé
# ══════════════════════════════════════════════════════════════════════════════

def axe5_caracteristiques_phrases(ctx, noc, common_keys, vectors_ctx, vectors_noc):
    """
    Enrichit chaque phrase commune avec :
    - longueur en mots
    - position dans le paragraphe
    - présence de pronoms anaphoriques
    Puis croise ces features avec le taux de flip.
    """
    print("\n" + "=" * 70)
    print("AXE 5 — ANALYSE PAR CARACTÉRISTIQUES DE PHRASE")
    print("=" * 70)

    import re as re_mod

    PRONOMS = {"il", "elle", "ils", "elles", "ce", "cela", "ça", "ceci",
               "cette", "ces", "leur", "leurs", "y", "en", "le", "la",
               "les", "lui", "eux", "son", "sa", "ses"}

    df_ctx = ctx["df"]
    df_noc = noc["df"]

    qkeys = list(vectors_ctx.keys())

    # Calculer le nombre de flips par phrase
    flip_per_phrase = {}
    for k in common_keys:
        idx = list(common_keys).index(k)
        n_flips = sum(
            1 for qk in qkeys
            if vectors_ctx[qk][idx] != vectors_noc[qk][idx]
        )
        flip_per_phrase[k] = n_flips

    rows = []
    for key in common_keys:
        if key not in df_ctx.index:
            continue
        row_ctx = df_ctx.loc[key]

        # Texte de la phrase
        phrase_text = ""
        for col in ["Phrase_Text", "Phrase"]:
            if col in df_ctx.columns and pd.notna(row_ctx.get(col, "")):
                phrase_text = str(row_ctx[col])
                break

        words       = phrase_text.lower().split()
        n_words     = len(words)
        has_pronoun = any(w.strip(".,;:!?") in PRONOMS for w in words)

        # Longueur catégorie
        if n_words <= 8:        len_cat = "Très courte (≤8)"
        elif n_words <= 15:     len_cat = "Courte (9–15)"
        elif n_words <= 25:     len_cat = "Moyenne (16–25)"
        else:                   len_cat = "Longue (>25)"

        phrase_idx = int(row_ctx.get("Phrase_Index", 0))
        is_title   = phrase_idx == -1

        rows.append({
            "Key":          key,
            "Phrase_Text":  phrase_text[:100],
            "N_Words":      n_words,
            "Length_Cat":   len_cat,
            "Phrase_Index": phrase_idx,
            "Is_Title":     is_title,
            "Has_Pronoun":  has_pronoun,
            "Label_CTX":    row_ctx.get("Category", ""),
            "Label_NOC":    df_noc.loc[key, "Category"] if key in df_noc.index else "",
            "N_Flips":      flip_per_phrase.get(key, 0),
            "Has_Flips":    flip_per_phrase.get(key, 0) > 0,
            "Label_Changed":row_ctx.get("Category", "") != (
                df_noc.loc[key, "Category"] if key in df_noc.index else ""),
        })

    df_feat = pd.DataFrame(rows)

    # ── Analyse 1 : Flip rate par longueur de phrase ──────────────────────────
    len_analysis = df_feat.groupby("Length_Cat").agg(
        N_Phrases    = ("N_Flips", "count"),
        Flip_Moy     = ("N_Flips", "mean"),
        Pct_Flipped  = ("Has_Flips", "mean"),
        N_FlipTotal  = ("N_Flips", "sum"),
    ).reset_index()
    len_analysis["Pct_Flipped_%"] = (len_analysis["Pct_Flipped"] * 100).round(2)
    len_analysis["Flip_Moy"]      = len_analysis["Flip_Moy"].round(3)
    len_analysis = len_analysis.drop(columns=["Pct_Flipped"])

    # ── Analyse 2 : Flip rate avec / sans pronoms ────────────────────────────
    pronoun_analysis = df_feat.groupby("Has_Pronoun").agg(
        N_Phrases    = ("N_Flips", "count"),
        Flip_Moy     = ("N_Flips", "mean"),
        Pct_Flipped  = ("Has_Flips", "mean"),
    ).reset_index()
    pronoun_analysis["Pct_Flipped_%"] = (pronoun_analysis["Pct_Flipped"] * 100).round(2)
    pronoun_analysis["Flip_Moy"]      = pronoun_analysis["Flip_Moy"].round(3)
    pronoun_analysis = pronoun_analysis.drop(columns=["Pct_Flipped"])
    pronoun_analysis["Has_Pronoun"]   = pronoun_analysis["Has_Pronoun"].map(
        {True: "Avec pronoms", False: "Sans pronoms"})

    # ── Analyse 3 : Titres vs phrases ────────────────────────────────────────
    title_analysis = df_feat.groupby("Is_Title").agg(
        N       = ("N_Flips", "count"),
        Flip_Moy= ("N_Flips", "mean"),
        Pct_Flip= ("Has_Flips", "mean"),
    ).reset_index()
    title_analysis["Pct_Flip_%"] = (title_analysis["Pct_Flip"] * 100).round(2)
    title_analysis["Is_Title"]   = title_analysis["Is_Title"].map(
        {True: "Titre de section", False: "Phrase de corps"})

    # ── Analyse 4 : Changement de label (catégorie) ───────────────────────────
    label_change = df_feat.groupby(["Label_CTX", "Label_NOC"]).size().reset_index(name="N")
    label_change = label_change.sort_values("N", ascending=False)

    # Test statistique : chi2 longueur × flip
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df_feat["Length_Cat"], df_feat["Has_Flips"])
    chi2, p_chi2, dof, expected = chi2_contingency(ct)

    # Test Mann-Whitney : n_mots avec flip vs sans flip
    flipped_words   = df_feat[df_feat["Has_Flips"]]["N_Words"].dropna()
    no_flip_words   = df_feat[~df_feat["Has_Flips"]]["N_Words"].dropna()
    if len(flipped_words) > 0 and len(no_flip_words) > 0:
        mw_stat, mw_p = scipy_stats.mannwhitneyu(
            flipped_words, no_flip_words, alternative="two-sided")
    else:
        mw_stat, mw_p = 0, 1

    df_stats_tests = pd.DataFrame([{
        "Test":       "Chi2 (longueur × flip)",
        "Statistique":round(chi2, 4),
        "p_value":    round(p_chi2, 4),
        "Conclusion": "Longueur liée au flip" if p_chi2 < 0.05 else "Pas d'effet significatif",
    }, {
        "Test":       "Mann-Whitney (n_mots flippées vs non-flippées)",
        "Statistique":round(mw_stat, 4),
        "p_value":    round(mw_p, 4),
        "Conclusion": "Différence significative de longueur" if mw_p < 0.05
                      else "Pas de différence significative",
    }])

    save_excel({
        "Phrases_Avec_Features":  df_feat,
        "Par_Longueur":           len_analysis,
        "Par_Pronoms":            pronoun_analysis,
        "Titres_vs_Phrases":      title_analysis,
        "Changements_Label":      label_change,
        "Tests_Statistiques":     df_stats_tests,
    }, str(out / "axe5_caracteristiques_phrases.xlsx"))

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Flip par longueur
    order_len = ["Très courte (≤8)", "Courte (9–15)", "Moyenne (16–25)", "Longue (>25)"]
    len_plot  = len_analysis.set_index("Length_Cat").reindex(order_len).reset_index()
    axes[0,0].bar(len_plot["Length_Cat"], len_plot["Pct_Flipped_%"], color="#e67e22", alpha=0.8)
    axes[0,0].set_title("% phrases flippées par longueur")
    axes[0,0].set_ylabel("% phrases avec au moins 1 flip")
    axes[0,0].set_xticklabels(len_plot["Length_Cat"].tolist(), rotation=15, ha="right", fontsize=9)

    # Flip avec/sans pronoms
    axes[0,1].bar(pronoun_analysis["Has_Pronoun"], pronoun_analysis["Pct_Flipped_%"],
                  color=["#9b59b6", "#3498db"], alpha=0.8)
    axes[0,1].set_title("% phrases flippées : avec vs sans pronoms")
    axes[0,1].set_ylabel("% phrases avec au moins 1 flip")

    # Scatter longueur vs n_flips
    axes[1,0].scatter(df_feat["N_Words"], df_feat["N_Flips"],
                      alpha=0.3, s=15, color="#2c3e50")
    z = np.polyfit(df_feat["N_Words"].fillna(0), df_feat["N_Flips"].fillna(0), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_feat["N_Words"].min(), df_feat["N_Words"].max(), 100)
    axes[1,0].plot(x_line, p(x_line), "r--", lw=1.5, label=f"Tendance")
    axes[1,0].set_xlabel("Longueur (mots)")
    axes[1,0].set_ylabel("Nombre de flips")
    axes[1,0].set_title("Longueur vs nombre de flips")
    axes[1,0].legend()

    # Matrice changements de labels (heatmap)
    pivot_lc = label_change.pivot_table(
        index="Label_CTX", columns="Label_NOC", values="N", fill_value=0)
    sns.heatmap(pivot_lc, ax=axes[1,1], annot=True, fmt="d",
                cmap="Blues", linewidths=0.5)
    axes[1,1].set_title("Matrice de changement de label\n(lignes=CTX, colonnes=NOC)")
    plt.tight_layout()
    save_fig(fig, str(out / "axe5_caracteristiques.png"))

    print(f"\n  Flip rate par longueur de phrase :")
    print(len_analysis.to_string(index=False))
    print(f"\n  Flip rate avec/sans pronoms :")
    print(pronoun_analysis.to_string(index=False))
    print(f"\n  Tests statistiques :")
    print(df_stats_tests.to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 5                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Si les phrases courtes (≤8 mots) ont un Pct_Flipped élevé →          │
  │    elles sont intrinsèquement ambiguës sans contexte. La version CTX    │
  │    est plus fiable pour ce type de phrases.                              │
  │                                                                          │
  │  • Si les phrases longues (>25 mots) ont aussi un Pct_Flipped élevé →   │
  │    elles contiennent des informations que le contexte précise ou         │
  │    contredit. Le modèle peut être confus avec trop de tokens en input.  │
  │                                                                          │
  │  • Pct_Flipped plus élevé "Avec pronoms" → le contexte remplit bien     │
  │    son rôle de résolution d'anaphore. La version CTX est préférable     │
  │    pour les phrases contenant "il", "cela", "cette solution"…           │
  │                                                                          │
  │  • Matrice de changement de label : regardez les cases hors-diagonale.  │
  │    Ex: CTX=ROI / NOC=Description → le contexte enrichit la phrase et   │
  │    lui permet d'être reconnue comme ROI. Sans contexte, elle est        │
  │    banalisée en Description.                                             │
  │                                                                          │
  │  • Test Chi2 p < 0.05 → la longueur de phrase est un prédicteur         │
  │    statistiquement significatif du flip. Le contexte a un effet          │
  │    hétérogène selon la longueur.                                         │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_feat


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 6 — COÛTS, STABILITÉ ET SCALABILITÉ
#  Métriques : tokens, coût par phrase, ratio CTX/NOC, accord par modèle
# ══════════════════════════════════════════════════════════════════════════════

def axe6_couts_scalabilite(ctx, noc):
    """
    Compare les coûts et l'utilisation des tokens entre les deux versions.
    Source : feuilles Stats_Par_Model et Stats_Globales des Excels.
    """
    print("\n" + "=" * 70)
    print("AXE 6 — COÛTS, STABILITÉ ET SCALABILITÉ")
    print("=" * 70)

    def _get_sheet(xl, name, fallback=None):
        for k in xl.keys():
            if k.lower().replace(" ", "_") == name.lower().replace(" ", "_"):
                return xl[k]
        if fallback:
            for k in xl.keys():
                if fallback.lower() in k.lower():
                    return xl[k]
        return pd.DataFrame()

    df_model_ctx = _get_sheet(ctx["excel"], "Stats_Par_Model",  "Model")
    df_model_noc = _get_sheet(noc["excel"], "Stats_Par_Model",  "Model")
    df_glob_ctx  = _get_sheet(ctx["excel"], "Stats_Globales",   "Global")
    df_glob_noc  = _get_sheet(noc["excel"], "Stats_Globales",   "Global")
    df_acc_ctx   = _get_sheet(ctx["excel"], "Accord_Questions_Global", "Accord")
    df_acc_noc   = _get_sheet(noc["excel"], "Accord_Questions_Global", "Accord")

    # ── Coûts globaux ─────────────────────────────────────────────────────────
    def _extract_global(df_g):
        if df_g.empty:
            return {}
        row = df_g.iloc[0]
        return {
            "Total_EUR":          row.get("Total_Cout_EUR", row.get("Cout_Total_EUR", 0)),
            "Cout_Phrase_EUR":    row.get("Cout_Moyen_EUR_Phrase", 0),
            "N_Phrases":          row.get("Total_Phrases", 0),
            "Duree_min":          row.get("Duree_Totale_min", 0),
            "Duree_s_Phrase":     row.get("Duree_Moyenne_sec_Phrase", 0),
        }

    g_ctx = _extract_global(df_glob_ctx)
    g_noc = _extract_global(df_glob_noc)

    ratio_cout  = g_ctx.get("Cout_Phrase_EUR", 1) / g_noc.get("Cout_Phrase_EUR", 1) \
                  if g_noc.get("Cout_Phrase_EUR", 0) > 0 else 0
    ratio_duree = g_ctx.get("Duree_s_Phrase", 1) / g_noc.get("Duree_s_Phrase", 1) \
                  if g_noc.get("Duree_s_Phrase", 0) > 0 else 0

    df_cout_comp = pd.DataFrame([{
        "Métrique":                    "Coût total (EUR)",
        "Valeur_CTX":                  round(g_ctx.get("Total_EUR", 0), 4),
        "Valeur_NOC":                  round(g_noc.get("Total_EUR", 0), 4),
        "Ratio_CTX_sur_NOC":           round(ratio_cout, 3),
        "Surcoût_%":                   round((ratio_cout - 1) * 100, 1),
    }, {
        "Métrique":                    "Coût moyen par phrase (EUR)",
        "Valeur_CTX":                  round(g_ctx.get("Cout_Phrase_EUR", 0), 6),
        "Valeur_NOC":                  round(g_noc.get("Cout_Phrase_EUR", 0), 6),
        "Ratio_CTX_sur_NOC":           round(ratio_cout, 3),
        "Surcoût_%":                   round((ratio_cout - 1) * 100, 1),
    }, {
        "Métrique":                    "Durée moyenne par phrase (s)",
        "Valeur_CTX":                  round(g_ctx.get("Duree_s_Phrase", 0), 2),
        "Valeur_NOC":                  round(g_noc.get("Duree_s_Phrase", 0), 2),
        "Ratio_CTX_sur_NOC":           round(ratio_duree, 3),
        "Surcoût_%":                   round((ratio_duree - 1) * 100, 1),
    }])

    # ── Tokens par modèle ─────────────────────────────────────────────────────
    def _merge_model_stats(df_ctx_m, df_noc_m):
        if df_ctx_m.empty or df_noc_m.empty:
            return pd.DataFrame()
        col_m = "Modele"
        df_ctx_m = df_ctx_m.copy()
        df_noc_m = df_noc_m.copy()
        df_ctx_m.columns = [f"{c}_CTX" if c != col_m else c for c in df_ctx_m.columns]
        df_noc_m.columns = [f"{c}_NOC" if c != col_m else c for c in df_noc_m.columns]
        merged = pd.merge(df_ctx_m, df_noc_m, on=col_m, how="outer")
        if "Tokens_Input_CTX" in merged.columns and "Tokens_Input_NOC" in merged.columns:
            merged["Ratio_Tokens_Input"] = (
                merged["Tokens_Input_CTX"] / merged["Tokens_Input_NOC"].replace(0, np.nan)
            ).round(2)
        return merged

    df_model_comp = _merge_model_stats(df_model_ctx, df_model_noc)

    # ── Accord inter-modèles par question dans chaque version ─────────────────
    def _compare_accord(df_a_ctx, df_a_noc):
        if df_a_ctx.empty or df_a_noc.empty:
            return pd.DataFrame()
        qk_col   = "Question_Key"
        acc_col  = "Taux_Accord_%"
        df_merge = pd.merge(
            df_a_ctx[[qk_col, acc_col]].rename(columns={acc_col: "Accord_CTX_%"}),
            df_a_noc[[qk_col, acc_col]].rename(columns={acc_col: "Accord_NOC_%"}),
            on=qk_col
        )
        df_merge["Delta_%"] = (df_merge["Accord_CTX_%"] - df_merge["Accord_NOC_%"]).round(2)
        return df_merge

    df_accord_comp = _compare_accord(df_acc_ctx, df_acc_noc)

    # Distribution des labels CTX vs NOC
    df_ctx_full = ctx["df_full"]
    df_noc_full = noc["df_full"]
    dist_ctx = df_ctx_full["Category"].value_counts().rename("CTX")
    dist_noc = df_noc_full["Category"].value_counts().rename("NOC")
    df_dist  = pd.concat([dist_ctx, dist_noc], axis=1).fillna(0).astype(int).reset_index()
    df_dist.columns = ["Category", "CTX", "NOC"]
    df_dist["Delta"] = df_dist["CTX"] - df_dist["NOC"]
    n_ctx = len(df_ctx_full)
    n_noc = len(df_noc_full)
    df_dist["Pct_CTX_%"] = (df_dist["CTX"] / n_ctx * 100).round(2) if n_ctx else 0
    df_dist["Pct_NOC_%"] = (df_dist["NOC"] / n_noc * 100).round(2) if n_noc else 0

    save_excel({
        "Comparaison_Couts":    df_cout_comp,
        "Comparaison_Modeles":  df_model_comp,
        "Accord_Par_Question":  df_accord_comp,
        "Distribution_Labels":  df_dist,
    }, str(out / "axe6_couts_scalabilite.xlsx"))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Coûts
    versions     = ["Avec contexte (CTX)", "Sans contexte (NOC)"]
    cout_vals    = [g_ctx.get("Total_EUR", 0), g_noc.get("Total_EUR", 0)]
    bar_c = axes[0].bar(versions, cout_vals, color=["#2ecc71", "#3498db"], alpha=0.85)
    axes[0].set_ylabel("Coût total (EUR)")
    axes[0].set_title(f"Coût total par version\n(ratio CTX/NOC = {ratio_cout:.2f}x)")
    for bar, val in zip(bar_c, cout_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"€{val:.4f}", ha="center", va="bottom", fontsize=10)

    # Tokens input par modèle
    if not df_model_comp.empty and "Tokens_Input_CTX" in df_model_comp.columns:
        x = np.arange(len(df_model_comp))
        w = 0.35
        axes[1].bar(x - w/2, df_model_comp.get("Tokens_Input_CTX", 0), w,
                    label="CTX", color="#2ecc71", alpha=0.85)
        axes[1].bar(x + w/2, df_model_comp.get("Tokens_Input_NOC", 0), w,
                    label="NOC", color="#3498db", alpha=0.85)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(df_model_comp["Modele"].tolist(),
                                rotation=30, ha="right", fontsize=8)
        axes[1].set_ylabel("Tokens input")
        axes[1].set_title("Tokens input par modèle")
        axes[1].legend()

    # Distribution des labels
    x  = np.arange(len(df_dist))
    w  = 0.35
    axes[2].bar(x - w/2, df_dist["CTX"], w, label="CTX", color="#2ecc71", alpha=0.85)
    axes[2].bar(x + w/2, df_dist["NOC"], w, label="NOC", color="#3498db", alpha=0.85)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(df_dist["Category"].tolist(), rotation=15, ha="right")
    axes[2].set_ylabel("Nombre de phrases")
    axes[2].set_title("Distribution des labels CTX vs NOC")
    axes[2].legend()
    plt.tight_layout()
    save_fig(fig, str(out / "axe6_couts_distribution.png"))

    print(f"\n  Comparaison des coûts :")
    print(df_cout_comp.to_string(index=False))
    if not df_accord_comp.empty:
        print(f"\n  Accord inter-modèles CTX vs NOC (par question) :")
        print(df_accord_comp.to_string(index=False))
    print(f"\n  Distribution des labels :")
    print(df_dist.to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 6                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Ratio_CTX_sur_NOC du coût : si > 1.5 (CTX coûte 50% de plus que     │
  │    NOC), le surcoût doit être justifié par un gain de qualité mesuré    │
  │    aux axes 1–4. Sinon, la version NOC est préférable économiquement.  │
  │                                                                          │
  │  • Ratio_Tokens_Input par modèle : les modèles avec un ratio élevé      │
  │    consomment beaucoup plus de tokens avec le contexte → ils "lisent"   │
  │    le paragraphe. Les modèles avec un ratio proche de 1 ignorent peut-  │
  │    être le contexte malgré le prompt.                                    │
  │                                                                          │
  │  • Distribution des labels : si CTX produit plus de "ROI" ou "OBL"     │
  │    que NOC → le contexte enrichit la détection. Si CTX produit plus de  │
  │    "Description" → il introduit de la prudence excessive.               │
  │                                                                          │
  │  • Si l'accord inter-modèles (Accord_CTX % > Accord_NOC %) →           │
  │    le contexte harmonise les 5 modèles → meilleure robustesse du vote. │
  │    C'est un argument fort en faveur de la version CTX malgré le coût.  │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_cout_comp


# ══════════════════════════════════════════════════════════════════════════════
#  AXE 7 — ANALYSE PAR MODÈLE (COMPORTEMENT INDIVIDUEL)
#  Métriques : accord modèle-vs-consensus, influence par version
# ══════════════════════════════════════════════════════════════════════════════

def axe7_par_modele(ctx, noc, common_keys):
    """
    Pour chaque modèle, compare ses réponses individuelles au consensus
    (vote majoritaire) dans chaque version.
    Détecte les modèles qui se comportent différemment avec/sans contexte.

    Source : colonnes {short_model}__{qk} dans Toutes_Phrases Excel
    """
    print("\n" + "=" * 70)
    print("AXE 7 — COMPORTEMENT INDIVIDUEL DES MODÈLES")
    print("=" * 70)

    df_ctx = ctx["df"]
    df_noc = noc["df"]
    qkeys  = get_question_keys(ctx["meta"])
    models = ctx["meta"].get("models", [])

    def _model_vs_consensus(df, models, qkeys, label):
        """Pour chaque modèle, calcule l'écart au consensus par question."""
        rows = []
        for model in models:
            for qk in qkeys:
                ind_col  = f"{model}__{qk}"
                agg_col  = f"AGG_{qk}"
                if ind_col not in df.columns or agg_col not in df.columns:
                    continue
                v_ind = (df[ind_col] == "oui").astype(int)
                v_agg = (df[agg_col] == "oui").astype(int)
                n     = len(v_ind)

                accord_consensus = int((v_ind == v_agg).sum())
                diverge_oui_vs_non = int(((v_ind == 1) & (v_agg == 0)).sum())
                diverge_non_vs_oui = int(((v_ind == 0) & (v_agg == 1)).sum())

                famille = "ROI" if qk.startswith("roi") else \
                          "NOT" if qk.startswith("not") else "OBL"
                rows.append({
                    "Modele":         model,
                    "Question_Key":   qk,
                    "Famille":        famille,
                    "Version":        label,
                    "N":              n,
                    "Accord_Cns_%":   round(accord_consensus / n * 100, 2) if n else 0,
                    "Diverge_+_%":    round(diverge_oui_vs_non / n * 100, 2) if n else 0,
                    "Diverge_-_%":    round(diverge_non_vs_oui / n * 100, 2) if n else 0,
                    "Biais_Net":      round((diverge_oui_vs_non - diverge_non_vs_oui) / n * 100, 2) if n else 0,
                })
        return pd.DataFrame(rows)

    df_ctx_mod = _model_vs_consensus(df_ctx, models, qkeys, "CTX")
    df_noc_mod = _model_vs_consensus(df_noc, models, qkeys, "NOC")

    # Résumé par modèle (toutes questions)
    def _model_summary(df_mod):
        return df_mod.groupby("Modele").agg(
            Accord_Cns_Moy_pct   = ("Accord_Cns_%", "mean"),
            Biais_Net_Moy        = ("Biais_Net", "mean"),
            Diverge_Plus_Moy_pct = ("Diverge_+_%", "mean"),
            Diverge_Moins_Moy_pct= ("Diverge_-_%", "mean"),
        ).round(3).reset_index()

    sum_ctx = _model_summary(df_ctx_mod)
    sum_ctx["Version"] = "CTX"
    sum_noc = _model_summary(df_noc_mod)
    sum_noc["Version"] = "NOC"
    df_model_summary = pd.concat([sum_ctx, sum_noc], ignore_index=True)

    # Comparaison CTX vs NOC par modèle
    df_comp_mod = pd.merge(
        sum_ctx[["Modele", "Accord_Cns_Moy_pct", "Biais_Net_Moy"]].rename(
            columns={"Accord_Cns_Moy_pct": "Accord_CTX", "Biais_Net_Moy": "Biais_CTX"}),
        sum_noc[["Modele", "Accord_Cns_Moy_pct", "Biais_Net_Moy"]].rename(
            columns={"Accord_Cns_Moy_pct": "Accord_NOC", "Biais_Net_Moy": "Biais_NOC"}),
        on="Modele"
    )
    df_comp_mod["Delta_Accord"] = (df_comp_mod["Accord_CTX"] - df_comp_mod["Accord_NOC"]).round(3)
    df_comp_mod["Delta_Biais"]  = (df_comp_mod["Biais_CTX"] - df_comp_mod["Biais_NOC"]).round(3)

    save_excel({
        "Detail_CTX":      df_ctx_mod,
        "Detail_NOC":      df_noc_mod,
        "Synthese_Modeles":df_model_summary,
        "Comparaison":     df_comp_mod,
    }, str(out / "axe7_par_modele.xlsx"))

    # Figure
    if len(df_comp_mod) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(df_comp_mod))
        w = 0.35
        axes[0].bar(x - w/2, df_comp_mod["Accord_CTX"], w,
                    label="CTX", color="#2ecc71", alpha=0.85)
        axes[0].bar(x + w/2, df_comp_mod["Accord_NOC"], w,
                    label="NOC", color="#3498db", alpha=0.85)
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(df_comp_mod["Modele"].tolist(),
                                rotation=30, ha="right", fontsize=9)
        axes[0].set_ylabel("Accord avec consensus moyen (%)")
        axes[0].set_title("Accord modèle-consensus : CTX vs NOC")
        axes[0].legend()

        axes[1].bar(x - w/2, df_comp_mod["Biais_CTX"], w,
                    label="Biais CTX", color="#e74c3c", alpha=0.8)
        axes[1].bar(x + w/2, df_comp_mod["Biais_NOC"], w,
                    label="Biais NOC", color="#f39c12", alpha=0.8)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(df_comp_mod["Modele"].tolist(),
                                rotation=30, ha="right", fontsize=9)
        axes[1].set_ylabel("Biais net moyen (% oui − % non vs consensus)")
        axes[1].set_title("Biais individuel CTX vs NOC\n(+ = sur-classe, − = sous-classe)")
        axes[1].legend()
        plt.tight_layout()
        save_fig(fig, str(out / "axe7_modeles.png"))

    print(f"\n  Comparaison comportement modèles CTX vs NOC :")
    print(df_comp_mod.to_string(index=False))

    # ── INTERPRÉTATIONS ───────────────────────────────────────────────────────
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INTERPRÉTATIONS — AXE 7                                                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  • Delta_Accord fort et positif (CTX > NOC) pour un modèle → ce modèle │
  │    bénéficie le plus du contexte : il converge mieux vers le consensus  │
  │    quand il a le paragraphe. C'est un "bon utilisateur du contexte".    │
  │                                                                          │
  │  • Biais_CTX positif élevé → ce modèle dit "oui" plus souvent que le  │
  │    consensus dans la version contexte. Il est "contaminé" par le        │
  │    paragraphe et génère des faux positifs. À surveiller.                │
  │                                                                          │
  │  • Biais_NOC négatif → ce modèle dit "non" plus souvent que le          │
  │    consensus sans contexte. Il est trop prudent sans information         │
  │    supplémentaire.                                                       │
  │                                                                          │
  │  • Si un modèle a Accord_CTX ≈ Accord_NOC → il ignore le contexte.     │
  │    Son comportement est identique dans les deux versions. Le surcoût    │
  │    du contexte ne lui apporte rien.                                      │
  └─────────────────────────────────────────────────────────────────────────┘
    """)

    return df_comp_mod


# ══════════════════════════════════════════════════════════════════════════════
#  RAPPORT DE SYNTHÈSE FINAL
# ══════════════════════════════════════════════════════════════════════════════

def synthese_finale(df_axe1, df_axe2, df_axe3, df_axe4, df_axe6):
    """
    Produit un tableau de bord de synthèse avec recommandation finale.
    """
    print("\n" + "=" * 70)
    print("SYNTHÈSE FINALE — RECOMMANDATION")
    print("=" * 70)

    rows = []

    # Axe 1 — Accord global
    kappa_moy = df_axe1["Cohen_Kappa"].mean() if "Cohen_Kappa" in df_axe1.columns else 0
    rows.append({
        "Axe":          "Axe 1 — Accord global",
        "Métrique_Clé": f"Kappa moyen = {kappa_moy:.3f}",
        "Signal":       _kappa_label(kappa_moy),
        "Verdict":      "Versions très similaires" if kappa_moy > 0.8
                        else "Divergences notables" if kappa_moy < 0.6
                        else "Divergences modérées",
    })

    # Axe 2 — Flips
    flip_moy = df_axe2["Taux_Flip_%"].mean() if "Taux_Flip_%" in df_axe2.columns else 0
    dir_dom  = df_axe2["Direction_Dominante"].mode()[0] \
               if "Direction_Dominante" in df_axe2.columns else "?"
    rows.append({
        "Axe":          "Axe 2 — Effet contexte",
        "Métrique_Clé": f"Flip moyen = {flip_moy:.1f}%",
        "Signal":       "Impact fort" if flip_moy > 15 else
                        "Impact modéré" if flip_moy > 5 else "Impact faible",
        "Verdict":      f"Direction dominante : {dir_dom}",
    })

    # Axe 3 — Accord inter-modèles
    if "Delta_Accord_%" in df_axe3.columns:
        delta_acc = df_axe3["Delta_Accord_%"].mean()
        rows.append({
            "Axe":          "Axe 3 — Consensus modèles",
            "Métrique_Clé": f"Delta accord CTX−NOC = {delta_acc:.2f}%",
            "Signal":       "CTX = meilleur consensus" if delta_acc > 2
                            else "NOC = meilleur consensus" if delta_acc < -2
                            else "Consensus équivalent",
            "Verdict":      "Le contexte harmonise les modèles" if delta_acc > 2
                            else "Le contexte divise les modèles" if delta_acc < -2
                            else "Pas d'impact sur la cohérence ensemble",
        })

    # Axe 4 — Questions sensibles
    if "Score_Sensibilite" in df_axe4.columns:
        most_sensitive = df_axe4.iloc[0]["Question_Key"] if len(df_axe4) > 0 else "?"
        rows.append({
            "Axe":          "Axe 4 — Questions sensibles",
            "Métrique_Clé": f"Question la plus sensible : {most_sensitive}",
            "Signal":       "Hétérogénéité entre questions",
            "Verdict":      "Certaines questions nécessitent contexte, d'autres non",
        })

    # Axe 6 — Coûts
    if "Ratio_CTX_sur_NOC" in df_axe6.columns:
        ratio = df_axe6[df_axe6["Métrique"] == "Coût moyen par phrase (EUR)"][
            "Ratio_CTX_sur_NOC"].values
        ratio = float(ratio[0]) if len(ratio) > 0 else 1.0
        rows.append({
            "Axe":          "Axe 6 — Coûts",
            "Métrique_Clé": f"Ratio coût CTX/NOC = {ratio:.2f}x",
            "Signal":       "Surcoût fort (>50%)" if ratio > 1.5
                            else "Surcoût modéré" if ratio > 1.15
                            else "Surcoût faible (<15%)",
            "Verdict":      "Justification économique requise" if ratio > 1.5
                            else "Surcoût acceptable si gain de qualité prouvé",
        })

    df_synth = pd.DataFrame(rows)

    # Recommandation globale
    score_ctx = 0
    if kappa_moy < 0.7 and flip_moy > 10:
        score_ctx += 2   # les versions divergent ET il y a des flips → contexte compte
    if "Delta_Accord_%" in df_axe3.columns and df_axe3["Delta_Accord_%"].mean() > 2:
        score_ctx += 1   # CTX = meilleur consensus
    if ratio <= 1.3:
        score_ctx += 1   # surcoût faible

    if score_ctx >= 3:
        reco = "RECOMMANDÉ : Version avec contexte (CTX) — meilleure cohérence inter-modèles et impact mesurable du contexte justifié par le surcoût."
    elif score_ctx == 2:
        reco = "HYBRIDE : Utiliser CTX pour les questions sensibles (top rang Axe 4) et NOC pour les questions stables — optimiser le coût."
    else:
        reco = "SUFFISANT : Version sans contexte (NOC) — les deux versions convergent, le contexte n'apporte pas de gain significatif pour ce corpus."

    df_reco = pd.DataFrame([{
        "Score_CTX":     score_ctx,
        "Recommandation":reco,
    }])

    save_excel({
        "Synthese":        df_synth,
        "Recommandation":  df_reco,
    }, str(out / "synthese_finale.xlsx"))

    print(f"\n  Tableau de bord :")
    print(df_synth.to_string(index=False))
    print(f"\n  ╔{'═'*65}╗")
    print(f"  ║  RECOMMANDATION FINALE (score CTX = {score_ctx}/4){' '*22}║")
    print(f"  ╠{'═'*65}╣")
    for line in reco.split(" — "):
        print(f"  ║  {line:<63}║")
    print(f"  ╚{'═'*65}╝\n")

    return df_synth


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  ANALYSE COMPARATIVE CTX vs NOC — Démarrage")
    print("█" * 70)

    ctx, noc, common_keys = load_data()

    df_axe1, vectors_ctx, vectors_noc = axe1_accord_global(ctx, noc, common_keys)
    df_axe2   = axe2_effet_contexte(ctx, noc, common_keys, vectors_ctx, vectors_noc)
    df_axe3   = axe3_accord_intermodeles(ctx, noc)
    df_axe4   = axe4_par_question(ctx, noc, common_keys, vectors_ctx, vectors_noc, df_axe2)
    _         = axe5_caracteristiques_phrases(ctx, noc, common_keys, vectors_ctx, vectors_noc)
    df_axe6   = axe6_couts_scalabilite(ctx, noc)
    _         = axe7_par_modele(ctx, noc, common_keys)
    synthese_finale(df_axe1, df_axe2, df_axe3, df_axe4, df_axe6)

    print("\n" + "█" * 70)
    print(f"  Tous les résultats exportés dans : {OUTPUT_DIR}")
    print("█" * 70)