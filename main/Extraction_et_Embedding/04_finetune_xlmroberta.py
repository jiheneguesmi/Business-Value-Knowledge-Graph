#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 4 - VERSION ULTRA LIGHT: Fine-Tuning XLM-RoBERTa Multi-Label
Optimisé pour GPU 4GB avec toutes les réductions possibles
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from tqdm import tqdm
import json
import gc
import os

# Libérer mémoire GPU au maximum
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
gc.collect()

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
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

def prepare_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=2):
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=8, scaler=None):
    """Entraîne une epoch avec gradient accumulation, FP16 et gestion mémoire optimisée"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    from torch.cuda.amp import autocast
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed Precision FP16 (économise 50% mémoire)
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / accumulation_steps
        
        # Backward avec scaler pour FP16
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # Libérer immédiatement après backward
        del outputs
        
        # Mise à jour des poids après accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Libérer mémoire GPU
            del input_ids, attention_mask, labels
            if (i + 1) % (accumulation_steps * 4) == 0:
                torch.cuda.empty_cache()
        else:
            del input_ids, attention_mask, labels
        
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Dernière mise à jour si nécessaire
    if (i + 1) % accumulation_steps != 0:
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
    
    torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Évalue le modèle avec gestion mémoire"""
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
            
            logits = outputs.logits
            preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            del input_ids, attention_mask, labels, outputs
            torch.cuda.empty_cache()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'hamming_loss': hamming_loss(all_labels, all_preds_binary),
        'f1_macro': f1_score(all_labels, all_preds_binary, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds_binary, average='micro'),
        'f1_weighted': f1_score(all_labels, all_preds_binary, average='weighted'),
        'accuracy': accuracy_score(all_labels, all_preds_binary)
    }
    
    f1_per_class = f1_score(all_labels, all_preds_binary, average=None)
    metrics['f1_roi'] = f1_per_class[0]
    metrics['f1_notoriete'] = f1_per_class[1]
    metrics['f1_obligation'] = f1_per_class[2]
    
    return metrics, all_preds, all_labels

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-5, device='cuda',
                patience=3, lr_scheduler_patience=1):
    """Pipeline d'entraînement ultra optimisé avec FP16"""
    
    # Activer gradient checkpointing (économise 30-40% mémoire)
    model.gradient_checkpointing_enable()
    
    # Mixed Precision Scaler (FP16)
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    print("✓ Mixed Precision (FP16) activée")
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    total_steps = len(train_loader) * epochs
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
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
        
        train_loss = train_epoch(model, train_loader, optimizer, warmup_scheduler, device, scaler=scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        print(f"\nValidation Metrics:")
        print(f"  Loss:          {val_metrics['loss']:.4f}")
        print(f"  F1 Macro:      {val_metrics['f1_macro']:.4f}")
        
        plateau_scheduler.step(val_metrics['f1_macro'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")
        
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model_xlmr_light.pt')
            print(f"  ✓ Meilleur modèle sauvegardé (F1 Macro: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠ Pas d'amélioration ({epochs_without_improvement}/{patience})")
        
        if epochs_without_improvement >= patience:
            print(f"\n⏹️  EARLY STOPPING à epoch {epoch + 1}")
            break
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            **val_metrics
        })
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return history

def main():
    """Pipeline ultra optimisé pour GPU 4GB"""
    
    device = torch.device('cuda')
    print(f"Device: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\nChargement des données...")
    train_df, val_df, test_df = load_data(
        'training_data/train.csv',
        'training_data/val.csv',
        'training_data/test.csv'
    )
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    print("\nChargement tokenizer XLM-RoBERTa...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    print("\nPréparation DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=2
    )
    
    print("\nChargement modèle XLM-RoBERTa...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=3,
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    print(f"Mémoire après chargement: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*60)
    print("DÉBUT FINE-TUNING (Mode ULTRA LIGHT)")
    print("="*60)
    print(f"Batch size: 2")
    print(f"Max length: 32")
    print(f"Gradient accumulation: 8")
    print(f"Gradient checkpointing: Activé")
    print(f"Mixed Precision (FP16): Activé")
    print(f"Learning rate: 1e-5")
    
    history = train_model(
        model, train_loader, val_loader, 
        epochs=10,
        lr=1e-5,
        device=device,
        patience=3,
        lr_scheduler_patience=1
    )
    
    print("\n" + "="*60)
    print("ÉVALUATION SUR TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model_xlmr_light.pt'))
    test_metrics, _, _ = evaluate(model, test_loader, device)
    
    print(f"\nTest F1 Macro: {test_metrics['f1_macro']:.4f}")
    
    with open('training_history_xlmr.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    model.save_pretrained('xlmroberta_light')
    tokenizer.save_pretrained('xlmroberta_light')
    
    print(f"\n✅ Entraînement terminé")

if __name__ == "__main__":
    main()