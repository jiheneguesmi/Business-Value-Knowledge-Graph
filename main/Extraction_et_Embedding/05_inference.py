#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 5: Inférence - Prédiction sur Nouvelles Phrases
Utilise modèle fine-tuné
"""

import torch
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from typing import List, Dict

class ClassificationModel:
    def __init__(self, model_path='camembert_multilabel_final', device=None):
        """
        Charge modèle fine-tuné
        
        Args:
            model_path: Chemin vers modèle sauvegardé
            device: 'cuda' ou 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Chargement modèle depuis {model_path}...")
        self.tokenizer = CamembertTokenizer.from_pretrained(model_path)
        self.model = CamembertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✓ Modèle chargé sur {self.device}")
        
        self.labels = ['roi', 'notoriete', 'obligation']
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Prédit labels pour liste de phrases
        
        Args:
            texts: Liste de phrases
            threshold: Seuil de décision (0-1)
        
        Returns:
            List[Dict]: [{roi, notoriete, obligation, roi_prob, ...}, ...]
        """
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Prédire
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Sigmoid pour multi-label
                probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                
                # Binariser
                preds = (probs > threshold).astype(int)
                
                result = {
                    'sentence': text,
                    'roi': int(preds[0]),
                    'notoriete': int(preds[1]),
                    'obligation': int(preds[2]),
                    'roi_prob': float(probs[0]),
                    'notoriete_prob': float(probs[1]),
                    'obligation_prob': float(probs[2])
                }
                
                results.append(result)
        
        return results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, threshold: float = 0.5) -> pd.DataFrame:
        """
        Prédiction batch optimisée
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Prédire
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Sigmoid
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            # Ajouter résultats
            for j, text in enumerate(batch_texts):
                all_results.append({
                    'sentence': text,
                    'roi': int(preds[j, 0]),
                    'notoriete': int(preds[j, 1]),
                    'obligation': int(preds[j, 2]),
                    'roi_prob': float(probs[j, 0]),
                    'notoriete_prob': float(probs[j, 1]),
                    'obligation_prob': float(probs[j, 2])
                })
        
        return pd.DataFrame(all_results)

def predict_on_csv(model: ClassificationModel, input_csv: str, output_csv: str):
    """
    Applique prédictions sur fichier CSV
    
    Args:
        model: Modèle de classification
        input_csv: Fichier CSV avec colonne 'sentence'
        output_csv: Fichier CSV de sortie
    """
    print(f"\nPrédiction sur {input_csv}...")
    
    # Charger
    df = pd.read_csv(input_csv)
    print(f"  {len(df)} phrases à classifier")
    
    # Prédire
    sentences = df['sentence'].tolist()
    results_df = model.predict_batch(sentences, batch_size=32, threshold=0.5)
    
    # Combiner avec données originales
    df_combined = pd.concat([df.reset_index(drop=True), results_df.drop(columns=['sentence'])], axis=1)
    
    # Sauvegarder
    df_combined.to_csv(output_csv, index=False)
    print(f"  ✓ Résultats sauvegardés: {output_csv}")
    
    # Statistiques
    print(f"\nStatistiques prédictions:")
    print(f"  ROI:        {results_df['roi'].sum()} ({results_df['roi'].mean()*100:.1f}%)")
    print(f"  Notoriété:  {results_df['notoriete'].sum()} ({results_df['notoriete'].mean()*100:.1f}%)")
    print(f"  Obligation: {results_df['obligation'].sum()} ({results_df['obligation'].mean()*100:.1f}%)")
    
    multi = ((results_df['roi'] + results_df['notoriete'] + results_df['obligation']) > 1).sum()
    print(f"  Multi-label: {multi} ({multi/len(results_df)*100:.1f}%)")
    
    return df_combined

def demo_predictions():
    """
    Démonstration sur phrases d'exemple
    """
    model = ClassificationModel()
    
    examples = [
        "Cette solution permet de réduire les coûts opérationnels de 30%.",
        "Le projet améliore significativement l'image de marque de l'entreprise.",
        "La conformité aux normes ISO 27001 est garantie par ce système.",
        "L'automatisation des processus génère des gains de temps importants tout en respectant les réglementations.",
        "Notre initiative renforce l'attractivité territoriale et favorise le développement durable."
    ]
    
    print("\n" + "="*80)
    print("DÉMONSTRATION - PRÉDICTIONS")
    print("="*80)
    
    results = model.predict(examples, threshold=0.5)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['sentence']}")
        print(f"   ROI:        {result['roi']} (prob: {result['roi_prob']:.3f})")
        print(f"   Notoriété:  {result['notoriete']} (prob: {result['notoriete_prob']:.3f})")
        print(f"   Obligation: {result['obligation']} (prob: {result['obligation_prob']:.3f})")

if __name__ == "__main__":
    # Démo
    demo_predictions()
    
    # Prédiction sur fichier
    # model = ClassificationModel()
    # df_results = predict_on_csv(
    #     model,
    #     input_csv='extracted_sentences.csv',
    #     output_csv='sentences_classified.csv'
    # )