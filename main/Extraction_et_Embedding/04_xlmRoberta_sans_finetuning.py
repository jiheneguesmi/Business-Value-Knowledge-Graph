#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test XLM-RoBERTa SANS Fine-Tuning (Zero-Shot Classification)
"""

import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from tqdm import tqdm

# Charger données test
test_df = pd.read_csv('training_data/test.csv')

# Pipeline zero-shot classification
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=0 if torch.cuda.is_available() else -1
)

# Labels candidats
candidate_labels = [
    "retour sur investissement",  # ROI
    "notoriété et image de marque",  # Notoriété
    "conformité et obligation légale"  # Obligation
]

# Prédictions
predictions = []
for sentence in tqdm(test_df['sentence'], desc="Prédiction"):
    result = classifier(sentence, candidate_labels, multi_label=True)
    
    # Seuil 0.5
    pred = [1 if score > 0.5 else 0 for score in result['scores']]
    predictions.append(pred)

predictions = torch.tensor(predictions).numpy()
labels = test_df[['roi', 'notoriete', 'obligation']].values

# Métriques
print("\n" + "="*60)
print("XLM-RoBERTa Zero-Shot (SANS Fine-Tuning)")
print("="*60)
print(f"F1 Macro:     {f1_score(labels, predictions, average='macro'):.4f}")
print(f"F1 Micro:     {f1_score(labels, predictions, average='micro'):.4f}")
print(f"Accuracy:     {accuracy_score(labels, predictions):.4f}")
print(f"Hamming Loss: {hamming_loss(labels, predictions):.4f}")

f1_per_class = f1_score(labels, predictions, average=None)
print(f"\nF1 par classe:")
print(f"  ROI:        {f1_per_class[0]:.4f}")
print(f"  Notoriété:  {f1_per_class[1]:.4f}")
print(f"  Obligation: {f1_per_class[2]:.4f}")