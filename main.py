"""
CLI inference script for ReviewGPT V2.
Loads trained model and predicts star rating and needs_reply for review text.

Compatible with both V1 (DistilBERT) and V2 (BERT-base) models.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import model from train.py
from train import ReviewClassifier


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model
    model = ReviewClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def predict(text, model, tokenizer, device, max_length=256):
    """
    Predict star rating and needs_reply for a single review.

    Args:
        text: Review text
        model: Trained ReviewClassifier
        tokenizer: Tokenizer
        device: torch device
        max_length: Maximum sequence length (V2 default: 256)

    Returns:
        Dictionary with predictions and confidence scores
    """
    # Tokenize input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get predictions
    with torch.no_grad():
        stars_logits, reply_logits = model(input_ids, attention_mask)

        # Star rating predictions
        stars_probs = F.softmax(stars_logits, dim=1).squeeze().cpu().numpy()
        predicted_stars = int(stars_probs.argmax()) + 1  # Convert to 1-5

        # Needs reply prediction
        reply_prob = torch.sigmoid(reply_logits).squeeze().cpu().item()
        needs_reply = reply_prob > 0.5

    return {
        "stars": predicted_stars,
        "needs_reply": bool(needs_reply),
        "confidence": {
            "stars": [float(p) for p in stars_probs],
            "needs_reply": float(reply_prob)
        }
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load tokenizer
    # Try to load from checkpoint directory first, fallback to V2 default
    tokenizer_path = os.path.dirname(args.checkpoint)
    if not os.path.exists(tokenizer_path) or not os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json')):
        tokenizer_path = 'bert-base-multilingual-cased'  # V2 default
        print(f"No tokenizer found in checkpoint directory, using default: {tokenizer_path}")

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load model
    model = load_model(args.checkpoint, device)

    # Get review text
    if args.text:
        review_text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            review_text = f.read().strip()
    else:
        # Interactive mode
        print("Enter review text (Ctrl+D or Ctrl+Z when done):")
        review_text = sys.stdin.read().strip()

    if not review_text:
        print("Error: No review text provided")
        sys.exit(1)

    # Make prediction
    print(f"\n{'='*80}")
    print("Review:")
    print(f"{'='*80}")
    print(review_text)
    print(f"{'='*80}\n")

    # V2: Use max_length=256 by default (can be overridden if needed)
    result = predict(review_text, model, tokenizer, device, max_length=256)

    # Output results
    if args.output_format == 'json':
        print(json.dumps(result, indent=2))
    else:
        print("Predictions:")
        print(f"  Stars: {result['stars']}/5")
        print(f"  Needs Reply: {'Yes' if result['needs_reply'] else 'No'}")
        print(f"\nConfidence Scores:")
        print(f"  Star Distribution:")
        for i, conf in enumerate(result['confidence']['stars'], 1):
            bar = '█' * int(conf * 50)
            print(f"    {i} star: {conf:.3f} {bar}")
        print(f"  Needs Reply: {result['confidence']['needs_reply']:.3f}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ReviewGPT V2: Predict star rating and reply necessity for app reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from command line
  python main.py --checkpoint artifacts/model.pt --text "האפליקציה קורסת כל הזמן"

  # Predict from file
  python main.py --checkpoint artifacts/model.pt --file review.txt

  # Interactive mode
  python main.py --checkpoint artifacts/model.pt

  # Output as JSON
  python main.py --checkpoint artifacts/model.pt --text "Great app!" --output-format json

Note: V2 models use BERT-base-multilingual-cased with max_length=256
      V1 models use DistilBERT with max_length=128
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--text', type=str,
                        help='Review text to classify')
    parser.add_argument('--file', type=str,
                        help='Path to file containing review text')
    parser.add_argument('--output', type=str,
                        help='Path to save output JSON')
    parser.add_argument('--output-format', type=str, choices=['json', 'human'],
                        default='human', help='Output format (default: human)')

    args = parser.parse_args()

    if args.text and args.file:
        print("Error: Cannot specify both --text and --file")
        sys.exit(1)

    main(args)
