#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 4 - VERSION 2: Fine-Tuning CamemBERT Multi-Label
Avec Augmentation de Données (NLP Libraries: nlpaug + transformers)
+ Early Stopping + LR Scheduler
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
import random

# ═══════════════════════════════════════════════════════════════
# AUGMENTATION DE DONNÉES - NLP LIBRARIES
# ═══════════════════════════════════════════════════════════════

try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    NLPAUG_AVAILABLE = True
except ImportError:
    print("⚠️  nlpaug non installé. Installation: pip install nlpaug")
    NLPAUG_AVAILABLE = False

class DataAugmenter:
    """
    Augmentation de données avec plusieurs techniques NLP
    """
    
    def __init__(self, methods=['synonym', 'contextual', 'back_translation']):
        """
        Args:
            methods: Liste des méthodes d'augmentation
                - 'synonym': Remplacement synonymes (WordNet)
                - 'contextual': Substitution contextuelle (CamemBERT)
                - 'back_translation': Traduction FR→EN→FR
                - 'random_swap': Échange positions mots
                - 'random_deletion': Suppression aléatoire mots
        """
        self.methods = methods
        self.augmenters = {}
        
        if not NLPAUG_AVAILABLE:
            print("❌ nlpaug requis. Installation:")
            print("   pip install nlpaug transformers")
            return
        
        print("Initialisation augmenteurs...")
        
        # 1. Synonym Augmenter (WordNet multilingue)
        if 'synonym' in methods:
            try:
                self.augmenters['synonym'] = naw.SynonymAug(
                    aug_src='wordnet',
                    lang='fra',
                    aug_min=1,
                    aug_max=3,
                    aug_p=0.3
                )
                print("  ✓ Synonym Augmenter (WordNet FR)")
            except Exception as e:
                print(f"  ⚠️  WordNet FR non disponible: {e}")
        
        # 2. Contextual Word Embeddings (CamemBERT)
        if 'contextual' in methods:
            try:
                self.augmenters['contextual'] = naw.ContextualWordEmbsAug(
                    model_path='camembert-base',
                    action="substitute",
                    aug_min=1,
                    aug_max=2,
                    aug_p=0.3,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print("  ✓ Contextual Augmenter (CamemBERT)")
            except Exception as e:
                print(f"  ⚠️  Contextual aug erreur: {e}")
        
        # 3. Back Translation (FR→EN→FR)
        if 'back_translation' in methods:
            try:
                self.augmenters['back_translation'] = nas.BackTranslationAug(
                    from_model_name='Helsinki-NLP/opus-mt-fr-en',
                    to_model_name='Helsinki-NLP/opus-mt-en-fr',
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print("  ✓ Back Translation Augmenter (FR↔EN)")
            except Exception as e:
                print(f"  ⚠️  Back translation erreur: {e}")
        
        # 4. Random Swap (échange positions)
        if 'random_swap' in methods:
            self.augmenters['random_swap'] = naw.RandomWordAug(
                action="swap",
                aug_min=1,
                aug_max=2
            )
            print("  ✓ Random Swap Augmenter")
        
        # 5. Random Deletion
        if 'random_deletion' in methods:
            self.augmenters['random_deletion'] = naw.RandomWordAug(
                action="delete",
                aug_p=0.1
            )
            print("  ✓ Random Deletion Augmenter")
    
    def augment(self, text: str, method: str = None) -> str:
        """
        Augmente une phrase avec une méthode spécifique
        
        Args:
            text: Phrase originale
            method: Méthode à utiliser (si None, choix aléatoire)
        
        Returns:
            Phrase augmentée
        """
        if not self.augmenters:
            return text
        
        # Choisir méthode aléatoirement si non spécifiée
        if method is None:
            method = random.choice(list(self.augmenters.keys()))
        
        if method not in self.augmenters:
            return text
        
        try:
            augmenter = self.augmenters[method]
            augmented = augmenter.augment(text)
            
            # nlpaug retourne parfois liste
            if isinstance(augmented, list):
                augmented = augmented[0] if augmented else text
            
            # Vérifier que augmentation a changé le texte
            if augmented and augmented != text:
                return augmented
            else:
                return text
        
        except Exception as e:
            # Si erreur, retourner texte original
            return text

def augment_dataset(df: pd.DataFrame, augmenter: DataAugmenter, 
                   augmentation_factor: float = 0.5) -> pd.DataFrame:
    """
    Augmente dataset avec techniques NLP
    
    Args:
        df: DataFrame avec colonnes [sentence, roi, notoriete, obligation]
        augmenter: Instance DataAugmenter
        augmentation_factor: Pourcentage d'augmentation
    
    Returns:
        DataFrame augmenté
    """
    n_augment = int(len(df) * augmentation_factor)
    
    print(f"\n{'='*60}")
    print("AUGMENTATION DE DONNÉES (NLP Libraries)")
    print(f"{'='*60}")
    print(f"  Original:  {len(df)} phrases")
    print(f"  Facteur:   {augmentation_factor} (+{int(augmentation_factor*100)}%)")
    print(f"  À générer: {n_augment} phrases")
    
    if not augmenter.augmenters:
        print("  ❌ Aucun augmenteur disponible")
        return df
    
    # Sélectionner phrases à augmenter
    indices_to_augment = random.sample(range(len(df)), n_augment)
    
    augmented_rows = []
    methods_used = {method: 0 for method in augmenter.augmenters.keys()}
    
    for idx in tqdm(indices_to_augment, desc="Augmentation"):
        row = df.iloc[idx]
        original_sentence = row['sentence']
        
        # Choisir méthode aléatoire
        method = random.choice(list(augmenter.augmenters.keys()))
        
        # Augmenter
        augmented_sentence = augmenter.augment(original_sentence, method)
        
        # Vérifier que augmentation a fonctionné
        if augmented_sentence != original_sentence and len(augmented_sentence) > 15:
            augmented_rows.append({
                'sentence': augmented_sentence,
                'roi': row['roi'],
                'notoriete': row['notoriete'],
                'obligation': row['obligation'],
                'source': row.get('source', 'unknown'),
                'augmented': True,
                'aug_method': method
            })
            methods_used[method] += 1
    
    # Combiner
    df_original = df.copy()
    df_original['augmented'] = False
    df_original['aug_method'] = None
    
    df_augmented = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    
    # Mélanger
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n  ✓ Augmentation réussie:")
    print(f"    Générées: {len(augmented_rows)} phrases")
    print(f"    Total:    {len(df_combined)} phrases")
    print(f"\n  Méthodes utilisées:")
    for method, count in methods_used.items():
        if count > 0:
            print(f"    {method:20s}: {count:4d} phrases")
    
    return df_combined

# ═══════════════════════════════════════════════════════════════
# DATASET ET DATALOADERS
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# FONCTIONS D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

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
            
            logits = outputs.logits
            preds = torch.sigmoid(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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

def train_model(model, train_loader, val_loader, epochs=20, lr=2e-5, device='cuda', 
                patience=5, lr_scheduler_patience=2):
    """Pipeline d'entraînement avec Early Stopping + LR Scheduler"""
    
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
        
        train_loss = train_epoch(model, train_loader, optimizer, warmup_scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
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
        
        plateau_scheduler.step(val_metrics['f1_macro'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Learning Rate: {current_lr:.2e}")
        
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model_v2.pt')
            print(f"  ✓ Meilleur modèle sauvegardé (F1 Macro: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠ Pas d'amélioration ({epochs_without_improvement}/{patience})")
        
        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"⏹️  EARLY STOPPING à epoch {epoch + 1}")
            print(f"{'='*60}")
            print(f"   Pas d'amélioration depuis {patience} epochs")
            print(f"   Meilleur F1 Macro: {best_f1:.4f}")
            break
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            **val_metrics
        })
    
    return history

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main(train_path='training_data/train.csv',
         val_path='training_data/val.csv',
         test_path='training_data/test.csv',
         epochs=20,
         batch_size=16,
         lr=2e-5,
         patience=5,
         lr_scheduler_patience=2,
         augmentation_factor=0.5,
         augmentation_methods=['synonym', 'contextual']):
    """
    Pipeline complet avec Augmentation NLP + Early Stopping + LR Scheduler
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger données
    print("\nChargement des données...")
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)
    print(f"  Train original: {len(train_df)}")
    print(f"  Val:            {len(val_df)}")
    print(f"  Test:           {len(test_df)}")
    
    # AUGMENTATION DE DONNÉES (train uniquement)
    augmenter = DataAugmenter(methods=augmentation_methods)
    train_df = augment_dataset(train_df, augmenter, augmentation_factor)
    
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
    print("DÉBUT FINE-TUNING (Augmentation NLP + Early Stop + LR Scheduler)")
    print("="*60)
    print(f"Méthodes augmentation: {augmentation_methods}")
    print(f"Augmentation Factor:   {augmentation_factor}")
    print(f"Patience Early Stop:   {patience} epochs")
    print(f"Patience LR Scheduler: {lr_scheduler_patience} epochs")
    
    history = train_model(
        model, train_loader, val_loader, 
        epochs, lr, device, 
        patience, lr_scheduler_patience
    )
    
    # Évaluation test
    print("\n" + "="*60)
    print("ÉVALUATION SUR TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model_v2.pt'))
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
    
    # Sauvegarder
    with open('training_history_v2.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open('test_metrics_v2.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    model.save_pretrained('camembert_multilabel_v2')
    tokenizer.save_pretrained('camembert_multilabel_v2')
    
    print(f"\n✅ Entraînement terminé")
    print(f"   Modèle: best_model_v2.pt")
    print(f"   Historique: training_history_v2.json")
    print(f"   Métriques test: test_metrics_v2.json")
    print(f"   Modèle complet: camembert_multilabel_v2/")

if __name__ == "__main__":
    main(
        train_path='training_data/train.csv',
        val_path='training_data/val.csv',
        test_path='training_data/test.csv',
        epochs=20,
        batch_size=16,
        lr=2e-5,
        patience=5,
        lr_scheduler_patience=2,
        augmentation_factor=0.5,
        augmentation_methods=['synonym', 'contextual']  # Modifier ici
    )