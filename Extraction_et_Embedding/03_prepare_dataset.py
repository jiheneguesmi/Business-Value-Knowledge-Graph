#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 3: Préparation Dataset pour Fine-Tuning
Combine UNHAS + Synthetic Data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_unhas_data(excel_path: str) -> pd.DataFrame:
    """
    Charge données UNHAS annotées
    
    Returns:
        DataFrame: [sentence, roi, notoriete, obligation]
    """
    print(f"Chargement UNHAS: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')
    
    # Sélectionner colonnes pertinentes
    df_clean = df[['Sentences', 'roi', 'notoriete', 'obligation']].copy()
    df_clean.columns = ['sentence', 'roi', 'notoriete', 'obligation']
    
    # Convertir scores 0-3 en binaire 0/1
    df_clean['roi'] = (df_clean['roi'] > 0).astype(int)
    df_clean['notoriete'] = (df_clean['notoriete'] > 0).astype(int)
    df_clean['obligation'] = (df_clean['obligation'] > 0).astype(int)
    
    # Filtrer phrases vides
    df_clean = df_clean[df_clean['sentence'].str.len() > 10].reset_index(drop=True)
    
    df_clean['source'] = 'UNHAS'
    
    print(f"  ✓ {len(df_clean)} phrases UNHAS")
    return df_clean

def load_synthetic_data(synthetic_folder: str, max_per_category: int = 500) -> pd.DataFrame:
    """
    Charge données synthétiques GPT-3.5
    
    Args:
        synthetic_folder: Dossier racine synthetic data
        max_per_category: Nombre max phrases par catégorie
    
    Returns:
        DataFrame: [sentence, roi, notoriete, obligation, source]
    """
    print(f"Chargement Synthetic Data: {synthetic_folder}")
    
    base_path = Path(synthetic_folder)
    categories = {
        'Retour sur investissement': {'roi': 1, 'notoriete': 0, 'obligation': 0},
        'Notoriété': {'roi': 0, 'notoriete': 1, 'obligation': 0},
        'Obligation': {'roi': 0, 'notoriete': 0, 'obligation': 1}
    }
    
    all_sentences = []
    
    for cat_name, labels in categories.items():
        cat_path = base_path / cat_name
        
        if not cat_path.exists():
            print(f"  ⚠️  Catégorie non trouvée: {cat_name}")
            continue
        
        # Lister sous-catégories
        sub_cats = [d for d in cat_path.iterdir() if d.is_dir()]
        
        sentences_cat = []
        
        for sub_cat in sub_cats:
            fr_path = sub_cat / 'fr'
            if not fr_path.exists():
                continue
            
            # Lire fichiers .txt
            txt_files = list(fr_path.glob('*.txt'))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        sentence = f.read().strip()
                        
                        if len(sentence) > 15:
                            sentences_cat.append({
                                'sentence': sentence,
                                'roi': labels['roi'],
                                'notoriete': labels['notoriete'],
                                'obligation': labels['obligation'],
                                'source': 'Synthetic',
                                'sub_category': sub_cat.name
                            })
                except Exception as e:
                    continue
        
        # Échantillonner si trop de données
        if len(sentences_cat) > max_per_category:
            sentences_cat = np.random.choice(sentences_cat, max_per_category, replace=False).tolist()
        
        all_sentences.extend(sentences_cat)
        print(f"  ✓ {len(sentences_cat)} phrases {cat_name}")
    
    df = pd.DataFrame(all_sentences)
    print(f"  ✓ Total Synthetic: {len(df)} phrases")
    
    return df

def create_train_val_test_split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """
    Split stratifié train/val/test
    
    Args:
        df: DataFrame avec colonnes roi, notoriete, obligation
        test_size: Proportion test
        val_size: Proportion validation
    
    Returns:
        train_df, val_df, test_df
    """
    # Créer label composite pour stratification
    df['label_combo'] = df['roi'].astype(str) + '_' + df['notoriete'].astype(str) + '_' + df['obligation'].astype(str)
    
    # Split train + (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df['label_combo'],
        random_state=42
    )
    
    # Split val + test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df['label_combo'],
        random_state=42
    )
    
    # Supprimer colonne temporaire
    train_df = train_df.drop(columns=['label_combo'])
    val_df = val_df.drop(columns=['label_combo'])
    test_df = test_df.drop(columns=['label_combo'])
    
    print(f"\nSplit effectué:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def analyze_dataset(df: pd.DataFrame, name: str = "Dataset"):
    """Affiche statistiques du dataset"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Nombre total: {len(df)}")
    print(f"\nDistribution par source:")
    print(df['source'].value_counts())
    print(f"\nDistribution par label:")
    print(f"  ROI:        {df['roi'].sum()} ({df['roi'].mean()*100:.1f}%)")
    print(f"  Notoriété:  {df['notoriete'].sum()} ({df['notoriete'].mean()*100:.1f}%)")
    print(f"  Obligation: {df['obligation'].sum()} ({df['obligation'].mean()*100:.1f}%)")
    
    # Multi-label
    multi_label = ((df['roi'] + df['notoriete'] + df['obligation']) > 1).sum()
    print(f"\nPhrases multi-label: {multi_label} ({multi_label/len(df)*100:.1f}%)")

def main(unhas_path: str, synthetic_folder: str, output_dir: str = "training_data"):
    """
    Pipeline complet: Préparation dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Charger UNHAS
    df_unhas = load_unhas_data(unhas_path)
    
    # Charger Synthetic
    df_synthetic = load_synthetic_data(synthetic_folder, max_per_category=500)
    
    # Combiner
    df_combined = pd.concat([df_unhas, df_synthetic], ignore_index=True)
    
    # Analyser
    analyze_dataset(df_combined, "Dataset Combiné")
    
    # Split
    train_df, val_df, test_df = create_train_val_test_split(df_combined)
    
    # Analyser chaque split
    analyze_dataset(train_df, "Train Set")
    analyze_dataset(val_df, "Validation Set")
    analyze_dataset(test_df, "Test Set")
    
    # Sauvegarder
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    print(f"\n Datasets sauvegardés dans {output_dir}/")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Configuration
    UNHAS_PATH = "C:/Users/USER/Documents/PFE/Extraction_et_Embedding/Exemple Coloration Documents/extraction_valeur_2023 UNHAS Annual Report.xlsm"
    SYNTHETIC_FOLDER = "C:/Users/USER/Documents/PFE/Comparer_Données_synthétiques/Dataset GPT-3.5 Synthetic"
    OUTPUT_DIR = "training_data"
    
    train_df, val_df, test_df = main(UNHAS_PATH, SYNTHETIC_FOLDER, OUTPUT_DIR)