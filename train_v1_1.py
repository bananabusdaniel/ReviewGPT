"""
Training script for ReviewGPT V1.1.
Uses DistilBERT-base-multilingual-cased with two classification heads:
1. Star rating prediction (1-5) with ORDINAL-AWARE LOSS
2. Needs reply prediction (binary)

V1.1 adds distance-based penalty to star rating loss.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, precision_recall_fscore_support,
                              roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, get_linear_schedule_with_warmup)


class ReviewDataset(Dataset):
    """PyTorch Dataset for app reviews."""

    def __init__(self, texts, stars, needs_reply, tokenizer, max_length=128):
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
            'stars': torch.tensor(self.stars[idx] - 1, dtype=torch.long),
            'needs_reply': torch.tensor(self.needs_reply[idx], dtype=torch.float)
        }


class ReviewClassifier(nn.Module):
    """Simple multi-task classifier with two heads on DistilBERT."""

    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        # Star rating head (5-class classification)
        self.star_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 5)
        )

        # Needs reply head (binary classification)
        self.reply_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs ([CLS] token)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # Use [CLS] token

        # Get predictions from both heads
        stars_logits = self.star_head(pooled)
        reply_logits = self.reply_head(pooled)

        return stars_logits, reply_logits


class OrdinalCrossEntropyLoss(nn.Module):
    """
    CrossEntropy loss with distance-based penalty for ordinal data.

    Penalizes predictions based on how far they are from the true label:
    - Predicting 1-star when actual is 5-star: heavily penalized (distance=4)
    - Predicting 4-star when actual is 5-star: lightly penalized (distance=1)
    - Predicting 5-star when actual is 5-star: no penalty (distance=0)
    """

    def __init__(self, weight_factor=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.weight_factor = weight_factor

    def forward(self, logits, targets):
        # Standard cross-entropy loss (unreduced)
        base_loss = self.ce_loss(logits, targets)

        # Get predicted classes
        preds = torch.argmax(logits, dim=1)

        # Calculate absolute distance between prediction and target
        # Distance ranges from 0 (correct) to 4 (maximum error, e.g., 1 vs 5)
        distance = torch.abs(preds - targets).float()

        # Weight the loss by (1 + weight_factor * distance)
        # weight_factor=1.0 means:
        #   - distance=0: weight=1.0 (correct prediction)
        #   - distance=1: weight=2.0 (off by 1 star)
        #   - distance=2: weight=3.0 (off by 2 stars)
        #   - distance=3: weight=4.0 (off by 3 stars)
        #   - distance=4: weight=5.0 (worst case: 1 vs 5)
        weight = 1.0 + self.weight_factor * distance

        weighted_loss = base_loss * weight

        return weighted_loss.mean()


def train_epoch(model, dataloader, optimizer, scheduler, device,
                stars_criterion, reply_criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stars = batch['stars'].to(device)
        needs_reply = batch['needs_reply'].to(device)

        # Forward pass
        stars_logits, reply_logits = model(input_ids, attention_mask)

        # Calculate losses
        stars_loss = stars_criterion(stars_logits, stars)
        reply_loss = reply_criterion(reply_logits.squeeze(-1), needs_reply)
        loss = stars_loss + reply_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
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
            reply_loss = reply_criterion(reply_logits.squeeze(-1), needs_reply)

            total_stars_loss += stars_loss.item()
            total_reply_loss += reply_loss.item()

            # Get predictions
            stars_preds = torch.argmax(stars_logits, dim=1)
            reply_probs = torch.sigmoid(reply_logits.squeeze(-1))
            reply_preds = (reply_probs > 0.5).long()

            all_stars_preds.extend(stars_preds.cpu().numpy())
            all_stars_labels.extend(stars.cpu().numpy())
            all_reply_preds.extend(reply_preds.cpu().numpy())
            all_reply_probs.extend(reply_probs.cpu().numpy())
            all_reply_labels.extend(needs_reply.cpu().numpy())

    # Calculate metrics
    stars_preds_original = np.array(all_stars_preds) + 1  # Convert back to 1-5
    stars_labels_original = np.array(all_stars_labels) + 1

    metrics = {
        'stars': {
            'accuracy': accuracy_score(stars_labels_original, stars_preds_original),
            'mae': mean_absolute_error(stars_labels_original, stars_preds_original),
            'loss': total_stars_loss / len(dataloader)
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
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("REVIEWGPT V1.1 TRAINING")
    print("=" * 80)
    print("\nV1.1 (Ordinal-Aware Loss):")
    print("  - DistilBERT-base-multilingual-cased (66M params)")
    print("  - Max length: 128 tokens")
    print("  - Simple 2-layer heads")
    print("  - [CLS] token pooling")
    print(f"  - ORDINAL CROSS-ENTROPY (weight_factor={args.ordinal_weight})")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('reviews-train.csv')
    val_df = pd.read_csv('reviews-validate.csv')
    test_df = pd.read_csv('reviews-test.csv')
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Initialize tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
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

    # Loss functions
    stars_criterion = OrdinalCrossEntropyLoss(weight_factor=args.ordinal_weight)
    reply_criterion = nn.BCEWithLogitsLoss()

    print(f"\nLoss functions:")
    print(f"  Stars: OrdinalCrossEntropyLoss (weight_factor={args.ordinal_weight})")
    print(f"  Reply: BCEWithLogitsLoss")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            stars_criterion, reply_criterion
        )
        print(f"Average training loss: {train_loss:.4f}")

        # Validate
        val_metrics, _ = evaluate(model, val_loader, device)

        print(f"\nValidation Metrics:")
        print(f"  Stars - Accuracy: {val_metrics['stars']['accuracy']:.4f}, MAE: {val_metrics['stars']['mae']:.4f}")
        print(f"  Needs Reply - F1: {val_metrics['needs_reply']['f1']:.4f}, ROC-AUC: {val_metrics['needs_reply']['roc_auc']:.4f}")

        # Early stopping based on F1
        current_f1 = val_metrics['needs_reply']['f1']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_counter = 0

            print(f"New best F1: {best_val_f1:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir, 'model_v1_1.pt'))

            tokenizer.save_pretrained(args.output_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Evaluating best model on test set...")
    print(f"{'='*80}")

    checkpoint = torch.load(os.path.join(args.output_dir, 'model_v1_1.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, conf_matrix = evaluate(model, test_loader, device)

    print(f"\nTest Metrics:")
    print(f"\nStar Rating:")
    print(f"  Accuracy: {test_metrics['stars']['accuracy']:.4f}")
    print(f"  MAE: {test_metrics['stars']['mae']:.4f}")

    print(f"\nNeeds Reply:")
    print(f"  Accuracy: {test_metrics['needs_reply']['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['needs_reply']['precision']:.4f}")
    print(f"  Recall: {test_metrics['needs_reply']['recall']:.4f}")
    print(f"  F1: {test_metrics['needs_reply']['f1']:.4f}")
    print(f"  ROC-AUC: {test_metrics['needs_reply']['roc_auc']:.4f}")

    print(f"\nConfusion Matrix (Stars):")
    print(conf_matrix)

    # Save metrics
    metrics_output = {
        'version': 'V1.1',
        'ordinal_weight_factor': args.ordinal_weight,
        'test_metrics': {
            'stars': {k: float(v) for k, v in test_metrics['stars'].items()},
            'needs_reply': {k: float(v) for k, v in test_metrics['needs_reply'].items()}
        },
        'confusion_matrix': conf_matrix.tolist(),
        'best_epoch': int(checkpoint['epoch']),
        'hyperparameters': vars(args)
    }

    with open(os.path.join(args.output_dir, 'metrics_v1_1.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nTraining complete! Model and metrics saved to {args.output_dir}")
    print(f"  - model_v1_1.pt")
    print(f"  - metrics_v1_1.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ReviewGPT V1.1 classifier with ordinal-aware loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output_dir', type=str, default='artifacts',
                        help='Directory to save model and metrics')
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased',
                        help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--ordinal_weight', type=float, default=1.0,
                        help='Weight factor for ordinal distance penalty (default: 1.0)')

    args = parser.parse_args()
    main(args)
