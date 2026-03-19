#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2: Génération Embeddings Sémantiques
Sentence-BERT pour similarité
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

def load_model(model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
    """
    Charge modèle Sentence-BERT
    
    Args:
        model_name: Nom du modèle Hugging Face
    
    Returns:
        SentenceTransformer
    """
    print(f"Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  ✓ Modèle chargé (dimension: {model.get_sentence_embedding_dimension()})")
    return model

def generate_embeddings(sentences: list, model: SentenceTransformer, batch_size: int = 32):
    """
    Génère embeddings pour liste de phrases
    
    Args:
        sentences: Liste de phrases
        model: Modèle Sentence-BERT
        batch_size: Taille batch
    
    Returns:
        np.array: Embeddings (n_sentences, embedding_dim)
    """
    print(f"Génération embeddings pour {len(sentences)} phrases...")
    
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normaliser pour cosine similarity
    )
    
    print(f"  ✓ Embeddings générés: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings: np.array, output_path: str):
    """Sauvegarde embeddings"""
    np.save(output_path, embeddings)
    print(f"  ✓ Embeddings sauvegardés: {output_path}")

def save_model_cache(model: SentenceTransformer, cache_path: str):
    """Sauvegarde modèle pour réutilisation"""
    with open(cache_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Modèle sauvegardé: {cache_path}")

def main(input_csv: str, output_embeddings: str, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
    """
    Pipeline complet: CSV → Embeddings
    
    Args:
        input_csv: Fichier CSV avec colonne 'sentence'
        output_embeddings: Fichier .npy de sortie
        model_name: Nom modèle Sentence-BERT
    """
    # Charger données
    print(f"Chargement données depuis {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"  ✓ {len(df)} phrases chargées")
    
    # Charger modèle
    model = load_model(model_name)
    
    # Générer embeddings
    sentences = df['sentence'].tolist()
    embeddings = generate_embeddings(sentences, model, batch_size=32)
    
    # Sauvegarder
    save_embeddings(embeddings, output_embeddings)
    
    # Sauvegarder avec métadonnées
    output_dir = Path(output_embeddings).parent
    df['embedding_index'] = range(len(df))
    df.to_csv(output_dir / 'sentences_with_embeddings.csv', index=False)
    
    print(f"\n✅ Pipeline terminé")
    print(f"   Embeddings: {output_embeddings}")
    print(f"   Métadonnées: {output_dir / 'sentences_with_embeddings.csv'}")
    
    return embeddings, df

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "extracted_sentences.csv"
    OUTPUT_EMBEDDINGS = "embeddings_sbert.npy"
    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    
    # Alternative plus légère:
    # MODEL_NAME = "distiluse-base-multilingual-cased-v2"
    
    embeddings, df = main(INPUT_CSV, OUTPUT_EMBEDDINGS, MODEL_NAME)
    
    # Statistiques
    print(f"\nStatistiques embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")