"""
Training script for ReviewGPT multi-task classification model V2.
Uses BERT-base-multilingual-cased with two classification heads for:
1. Star rating prediction (1-5)
2. Needs reply prediction (binary)

V2 Improvements:
- Upgraded to BERT-base (110M params vs DistilBERT 66M params)
- Increased max_length to 256 (from 128)
- Deeper classification heads (3 layers vs 2)
- Mean pooling instead of [CLS] token only
- Class weights for balanced training
- Gradient accumulation for larger effective batch size
- Mixed precision training (FP16)
- Per-class metrics for better analysis
- More epochs with longer patience
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, precision_recall_fscore_support,
                              roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, get_linear_schedule_with_warmup)


class ReviewDataset(Dataset):
    """PyTorch Dataset for app reviews."""

    def __init__(self, texts, stars, needs_reply, tokenizer, max_length=256):
        self.texts = texts
        self.stars = stars
        self.needs_reply = needs_reply
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stars': torch.tensor(self.stars[idx] - 1, dtype=torch.long),  # 0-indexed
            'needs_reply': torch.tensor(self.needs_reply[idx], dtype=torch.float)
        }


class ReviewClassifier(nn.Module):
    """Multi-task classifier with two heads on BERT-base-multilingual-cased.

    V2 Improvements:
    - Deeper classification heads (3 layers instead of 2)
    - Mean pooling instead of just [CLS] token
    - Reduced dropout (0.2 vs 0.3) for more capacity
    """

    def __init__(self, model_name="bert-base-multilingual-cased", dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        # Star rating head (5-class classification) - DEEPER
        self.star_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 5)
        )

        # Needs reply head (binary classification) - DEEPER
        self.reply_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1)
        )

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # V2: Use MEAN POOLING instead of just [CLS] token
        # Mean pooling over all tokens (excluding padding)
        token_embeddings = outputs.last_hidden_state

        # Mask out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Get predictions from both heads
        stars_logits = self.star_head(pooled)
        reply_logits = self.reply_head(pooled)

        return stars_logits, reply_logits


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler,
                stars_criterion, reply_criterion, accumulation_steps=2,
                stars_weight=1.0, reply_weight=1.0):
    """Train for one epoch with gradient accumulation and mixed precision.

    V2 Improvements:
    - Gradient accumulation (effective batch size = batch_size * accumulation_steps)
    - Mixed precision training (FP16) for faster training
    - Class-weighted loss functions passed as parameters
    """
    model.train()

    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stars = batch['stars'].to(device)
        needs_reply = batch['needs_reply'].to(device)

        # V2: Mixed precision training
        with autocast():
            # Forward pass
            stars_logits, reply_logits = model(input_ids, attention_mask)

            # Calculate losses
            stars_loss = stars_criterion(stars_logits, stars)
            reply_loss = reply_criterion(reply_logits.squeeze(), needs_reply)

            # Combined loss
            loss = stars_weight * stars_loss + reply_weight * reply_loss

            # V2: Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # V2: Gradient accumulation - only update every N steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model with per-class metrics.

    V2 Improvement: Added per-class accuracy for star ratings
    """
    model.eval()

    all_stars_preds = []
    all_stars_labels = []
    all_reply_preds = []
    all_reply_probs = []
    all_reply_labels = []

    stars_criterion = nn.CrossEntropyLoss()
    reply_criterion = nn.BCEWithLogitsLoss()

    total_stars_loss = 0
    total_reply_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            stars = batch['stars'].to(device)
            needs_reply = batch['needs_reply'].to(device)

            stars_logits, reply_logits = model(input_ids, attention_mask)

            # Calculate losses
            stars_loss = stars_criterion(stars_logits, stars)
            reply_loss = reply_criterion(reply_logits.squeeze(), needs_reply)

            total_stars_loss += stars_loss.item()
            total_reply_loss += reply_loss.item()

            # Get predictions
            stars_preds = torch.argmax(stars_logits, dim=1)
            reply_probs = torch.sigmoid(reply_logits.squeeze())
            reply_preds = (reply_probs > 0.5).long()

            all_stars_preds.extend(stars_preds.cpu().numpy())
            all_stars_labels.extend(stars.cpu().numpy())
            all_reply_preds.extend(reply_preds.cpu().numpy())
            all_reply_probs.extend(reply_probs.cpu().numpy())
            all_reply_labels.extend(needs_reply.cpu().numpy())

    # Calculate metrics
    stars_preds_original = np.array(all_stars_preds) + 1  # Convert back to 1-5
    stars_labels_original = np.array(all_stars_labels) + 1

    # V2: Calculate per-class accuracy
    per_class_accuracy = {}
    for star_rating in range(1, 6):
        mask = stars_labels_original == star_rating
        if mask.sum() > 0:
            per_class_accuracy[star_rating] = accuracy_score(
                stars_labels_original[mask],
                stars_preds_original[mask]
            )

    metrics = {
        'stars': {
            'accuracy': accuracy_score(stars_labels_original, stars_preds_original),
            'mae': mean_absolute_error(stars_labels_original, stars_preds_original),
            'loss': total_stars_loss / len(dataloader),
            'per_class_accuracy': per_class_accuracy  # V2: New metric
        },
        'needs_reply': {
            'accuracy': accuracy_score(all_reply_labels, all_reply_preds),
            'precision': precision_recall_fscore_support(all_reply_labels, all_reply_preds, average='binary')[0],
            'recall': precision_recall_fscore_support(all_reply_labels, all_reply_preds, average='binary')[1],
            'f1': f1_score(all_reply_labels, all_reply_preds, average='binary'),
            'roc_auc': roc_auc_score(all_reply_labels, all_reply_probs),
            'loss': total_reply_loss / len(dataloader)
        }
    }

    return metrics, confusion_matrix(stars_labels_original, stars_preds_original)


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create artifacts directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data from pre-split files
    print("=" * 80)
    print("REVIEWGPT V2 TRAINING")
    print("=" * 80)
    print("\nV2 Improvements:")
    print("  - BERT-base-multilingual-cased (110M params)")
    print("  - max_length: 256 tokens")
    print("  - Deeper classification heads (3 layers)")
    print("  - Mean pooling")
    print("  - Class weights for balanced training")
    print("  - Gradient accumulation (effective batch = 32)")
    print("  - Mixed precision training (FP16)")
    print("  - Per-class metrics")
    print("=" * 80)

    print("\nLoading data from pre-split files...")
    train_df = pd.read_csv('reviews-train.csv')
    val_df = pd.read_csv('reviews-validate.csv')
    test_df = pd.read_csv('reviews-test.csv')

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Verify data integrity
    print(f"\nTrain set stars distribution:")
    print(train_df['stars'].value_counts().sort_index())
    print(f"\nValidate set stars distribution:")
    print(val_df['stars'].value_counts().sort_index())
    print(f"\nTest set stars distribution:")
    print(test_df['stars'].value_counts().sort_index())

    # V2: Compute class weights for balanced training
    print("\n" + "=" * 80)
    print("Computing class weights for balanced training...")
    print("=" * 80)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['stars']),
        y=train_df['stars']
    )
    # Convert to 0-indexed for model (stars are 1-5, model uses 0-4)
    class_weights_tensor = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights_tensor}")
    print(f"  1 star: {class_weights[0]:.3f}")
    print(f"  2 star: {class_weights[1]:.3f}")
    print(f"  3 star: {class_weights[2]:.3f}")
    print(f"  4 star: {class_weights[3]:.3f}")
    print(f"  5 star: {class_weights[4]:.3f}")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = ReviewDataset(
        train_df['review_text'].values,
        train_df['stars'].values,
        train_df['needs_reply'].values,
        tokenizer,
        max_length=args.max_length
    )

    val_dataset = ReviewDataset(
        val_df['review_text'].values,
        val_df['stars'].values,
        val_df['needs_reply'].values,
        tokenizer,
        max_length=args.max_length
    )

    test_dataset = ReviewDataset(
        test_df['review_text'].values,
        test_df['stars'].values,
        test_df['needs_reply'].values,
        tokenizer,
        max_length=args.max_length
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    print(f"\nInitializing model: {args.model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ReviewClassifier(args.model_name, dropout=args.dropout)
    model.to(device)

    # V2: Initialize loss functions with class weights
    stars_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    reply_criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # V2: Adjust total steps for gradient accumulation
    total_steps = (len(train_loader) // args.accumulation_steps) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # V2: Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Training loop
    best_val_f1 = 0
    patience_counter = 0

    print("\nStarting training...")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")

    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # Train with V2 improvements
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            stars_criterion, reply_criterion,
            accumulation_steps=args.accumulation_steps
        )
        print(f"Average training loss: {train_loss:.4f}")

        # Validate
        val_metrics, _ = evaluate(model, val_loader, device)

        print(f"\nValidation Metrics:")
        print(f"  Stars - Accuracy: {val_metrics['stars']['accuracy']:.4f}, MAE: {val_metrics['stars']['mae']:.4f}")

        # V2: Print per-class accuracy
        print(f"  Per-class accuracy:")
        for star, acc in val_metrics['stars']['per_class_accuracy'].items():
            print(f"    {star} star: {acc:.4f}")

        print(f"  Needs Reply - F1: {val_metrics['needs_reply']['f1']:.4f}, ROC-AUC: {val_metrics['needs_reply']['roc_auc']:.4f}")

        # Early stopping based on needs_reply F1 score
        current_f1 = val_metrics['needs_reply']['f1']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_counter = 0

            # Save best model
            print(f"New best F1 score: {best_val_f1:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir, 'model.pt'))

            # Save tokenizer
            tokenizer.save_pretrained(args.output_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    # Load best model and evaluate on test set
    print(f"\n{'='*80}")
    print("Evaluating best model on test set...")
    print(f"{'='*80}")

    checkpoint = torch.load(os.path.join(args.output_dir, 'model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, conf_matrix = evaluate(model, test_loader, device)

    print(f"\nTest Metrics:")
    print(f"\nStar Rating:")
    print(f"  Accuracy: {test_metrics['stars']['accuracy']:.4f}")
    print(f"  MAE: {test_metrics['stars']['mae']:.4f}")

    # V2: Print per-class accuracy
    print(f"\n  Per-class accuracy:")
    for star, acc in test_metrics['stars']['per_class_accuracy'].items():
        print(f"    {star} star: {acc:.4f}")

    print(f"\nNeeds Reply:")
    print(f"  Accuracy: {test_metrics['needs_reply']['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['needs_reply']['precision']:.4f}")
    print(f"  Recall: {test_metrics['needs_reply']['recall']:.4f}")
    print(f"  F1: {test_metrics['needs_reply']['f1']:.4f}")
    print(f"  ROC-AUC: {test_metrics['needs_reply']['roc_auc']:.4f}")

    print(f"\nConfusion Matrix (Stars):")
    print(conf_matrix)

    # Save metrics
    # V2: Handle per_class_accuracy separately as it's a dict
    stars_metrics = {k: (float(v) if not isinstance(v, dict) else v)
                     for k, v in test_metrics['stars'].items()}
    # Convert per_class_accuracy values to float
    if 'per_class_accuracy' in stars_metrics:
        stars_metrics['per_class_accuracy'] = {
            int(k): float(v) for k, v in stars_metrics['per_class_accuracy'].items()
        }

    metrics_output = {
        'test_metrics': {
            'stars': stars_metrics,
            'needs_reply': {k: float(v) for k, v in test_metrics['needs_reply'].items()}
        },
        'confusion_matrix': conf_matrix.tolist(),
        'best_epoch': int(checkpoint['epoch']),
        'hyperparameters': vars(args)
    }

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nTraining complete! Model and metrics saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ReviewGPT V2 classifier with BERT-base',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output_dir', type=str, default='artifacts',
                        help='Directory to save model and metrics')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased',
                        help='Pretrained model name (V2 uses BERT-base)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (V2: 256 vs V1: 128)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs (V2: 10 vs V1: 5)')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate (V2: 3e-5 vs V1: 2e-5)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (V2: 0.2 vs V1: 0.3)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (V2: 5 vs V1: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
