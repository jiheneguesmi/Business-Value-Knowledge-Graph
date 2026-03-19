#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 4 - VERSION 1: Fine-Tuning CamemBERT Multi-Label
Avec Early Stopping + Learning Rate Scheduler
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from tqdm import tqdm
import json

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def load_data(train_path, val_path, test_path):
    """Charge datasets"""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def prepare_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=16):
    """Crée DataLoaders"""
    train_texts = train_df['sentence'].values
    train_labels = train_df[['roi', 'notoriete', 'obligation']].values
    
    val_texts = val_df['sentence'].values
    val_labels = val_df[['roi', 'notoriete', 'obligation']].values
    
    test_texts = test_df['sentence'].values
    test_labels = test_df[['roi', 'notoriete', 'obligation']].values
    
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)
    test_dataset = MultiLabelDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Entraîne une epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Évalue le modèle"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            # Prédictions (sigmoid)
            logits = outputs.logits
            preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Binariser prédictions (seuil 0.5)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    # Métriques
    metrics = {
        'loss': total_loss / len(dataloader),
        'hamming_loss': hamming_loss(all_labels, all_preds_binary),
        'f1_macro': f1_score(all_labels, all_preds_binary, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds_binary, average='micro'),
        'f1_weighted': f1_score(all_labels, all_preds_binary, average='weighted'),
        'accuracy': accuracy_score(all_labels, all_preds_binary)
    }
    
    # F1 par classe
    f1_per_class = f1_score(all_labels, all_preds_binary, average=None)
    metrics['f1_roi'] = f1_per_class[0]
    metrics['f1_notoriete'] = f1_per_class[1]
    metrics['f1_obligation'] = f1_per_class[2]
    
    return metrics, all_preds, all_labels

def train_model(model, train_loader, val_loader, epochs=20, lr=2e-5, device='cuda', 
                patience=5, lr_scheduler_patience=2):
    """
    Pipeline d'entraînement avec Early Stopping + LR Scheduler
    
    Args:
        patience: Nombre d'epochs sans amélioration avant arrêt
        lr_scheduler_patience: Nombre d'epochs avant réduction LR
    """
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    # Warmup scheduler (linéaire)
    total_steps = len(train_loader) * epochs
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # LR Scheduler (réduit LR quand plateau)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',              # Maximiser F1
        factor=0.5,              # Réduire LR de 50%
        patience=lr_scheduler_patience,
        verbose=True,
        min_lr=1e-7
    )
    
    best_f1 = 0
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, warmup_scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        print(f"\nValidation Metrics:")
        print(f"  Loss:          {val_metrics['loss']:.4f}")
        print(f"  Hamming Loss:  {val_metrics['hamming_loss']:.4f}")
        print(f"  F1 Macro:      {val_metrics['f1_macro']:.4f}")
        print(f"  F1 Micro:      {val_metrics['f1_micro']:.4f}")
        print(f"  F1 Weighted:   {val_metrics['f1_weighted']:.4f}")
        print(f"  Accuracy:      {val_metrics['accuracy']:.4f}")
        print(f"\n  F1 par classe:")
        print(f"    ROI:         {val_metrics['f1_roi']:.4f}")
        print(f"    Notoriété:   {val_metrics['f1_notoriete']:.4f}")
        print(f"    Obligation:  {val_metrics['f1_obligation']:.4f}")
        
        # Update plateau scheduler
        plateau_scheduler.step(val_metrics['f1_macro'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Learning Rate: {current_lr:.2e}")
        
        # Sauvegarder meilleur modèle
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ Meilleur modèle sauvegardé (F1 Macro: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠ Pas d'amélioration ({epochs_without_improvement}/{patience})")
        
        # Early Stopping
        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"⏹️  EARLY STOPPING à epoch {epoch + 1}")
            print(f"{'='*60}")
            print(f"   Pas d'amélioration depuis {patience} epochs")
            print(f"   Meilleur F1 Macro: {best_f1:.4f}")
            break
        
        # Historique
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            **val_metrics
        })
    
    return history

def main(train_path='training_data/train.csv',
         val_path='training_data/val.csv',
         test_path='training_data/test.csv',
         epochs=20,
         batch_size=16,
         lr=2e-5,
         patience=5,
         lr_scheduler_patience=2):
    """
    Pipeline complet avec Early Stopping + LR Scheduler
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger données
    print("\nChargement des données...")
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # Tokenizer
    print("\nChargement tokenizer CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # DataLoaders
    print("\nPréparation DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size
    )
    
    # Modèle
    print("\nChargement modèle CamemBERT...")
    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels=3,
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Entraînement
    print("\n" + "="*60)
    print("DÉBUT FINE-TUNING (avec Early Stopping + LR Scheduler)")
    print("="*60)
    print(f"Patience Early Stopping: {patience} epochs")
    print(f"Patience LR Scheduler:   {lr_scheduler_patience} epochs")
    
    history = train_model(
        model, train_loader, val_loader, 
        epochs, lr, device, 
        patience, lr_scheduler_patience
    )
    
    # Charger meilleur modèle
    print("\n" + "="*60)
    print("ÉVALUATION SUR TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"\nTest Metrics:")
    print(f"  Loss:          {test_metrics['loss']:.4f}")
    print(f"  Hamming Loss:  {test_metrics['hamming_loss']:.4f}")
    print(f"  F1 Macro:      {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Micro:      {test_metrics['f1_micro']:.4f}")
    print(f"  F1 Weighted:   {test_metrics['f1_weighted']:.4f}")
    print(f"  Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"\n  F1 par classe:")
    print(f"    ROI:         {test_metrics['f1_roi']:.4f}")
    print(f"    Notoriété:   {test_metrics['f1_notoriete']:.4f}")
    print(f"    Obligation:  {test_metrics['f1_obligation']:.4f}")
    
    # Sauvegarder résultats
    with open('training_history_v1.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open('test_metrics_v1.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Sauvegarder modèle final
    model.save_pretrained('camembert_multilabel_v1')
    tokenizer.save_pretrained('camembert_multilabel_v1')
    
    print(f"\n✅ Entraînement terminé")
    print(f"   Modèle: best_model.pt")
    print(f"   Historique: training_history_v1.json")
    print(f"   Métriques test: test_metrics_v1.json")
    print(f"   Modèle complet: camembert_multilabel_v1/")

if __name__ == "__main__":
    main(
        train_path='training_data/train.csv',
        val_path='training_data/val.csv',
        test_path='training_data/test.csv',
        epochs=20,
        batch_size=16,
        lr=2e-5,
        patience=5,              # Early stopping après 5 epochs
        lr_scheduler_patience=2  # Réduire LR après 2 epochs
    )