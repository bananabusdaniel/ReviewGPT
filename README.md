# ReviewGPT V1

AI-powered app review analyzer that predicts star ratings and reply necessity using DistilBERT.

## What It Does

Analyzes Google Play reviews to predict:
1. **Star rating** (1-5 stars)
2. **Needs reply** (yes/no)

## Performance

- Star Accuracy: **51.4%**
- Needs Reply F1: **85%** (production-ready!)
- Training Time: **~9 minutes** on T4 GPU
- Model Size: **265 MB**

## Quick Start

### 1. Setup

```bash
pip install transformers torch scikit-learn pandas numpy tqdm
```

### 2. Train

```bash
python train.py
```

Done. Model saves to `artifacts/model.pt`.

### 3. Use It

```bash
python main.py --checkpoint artifacts/model.pt --text "App crashes constantly"
```

Output:
```
Predicted star rating: 1
Needs reply: Yes
```

## Architecture

**Model**: DistilBERT-base-multilingual-cased (66M params)
- Simple 2-layer classification heads
- [CLS] token pooling
- Early stopping on validation F1

**Training**:
- 5 epochs (early stopping patience=3)
- Batch size: 16
- Learning rate: 2e-5
- Dropout: 0.3
- Optimizer: AdamW

## Files

```
ReviewGPT/
├── train.py              # Training script (V1)
├── main.py               # CLI inference
├── preprocess_data.py    # Data preprocessing
├── reviews-train.csv     # Training data (80%)
├── reviews-validate.csv  # Validation (10%)
├── reviews-test.csv      # Test data (10%)
└── artifacts/
    ├── model.pt          # Trained model (~265 MB)
    └── metrics.json      # Performance metrics
```

## How It Works

1. **Preprocessing**: Heuristics label which reviews need replies
2. **Training**: DistilBERT learns patterns from 9,137 reviews
3. **Inference**: Model predicts stars + reply necessity

The reply detector uses ML (not heuristics!) with 85% F1 - production-ready.

## CLI Options

```bash
python train.py \
  --output_dir artifacts \
  --batch_size 16 \
  --num_epochs 5 \
  --learning_rate 2e-5 \
  --max_length 128
```

## Google Colab

**Block 1 - Setup:**
```python
import os, shutil, subprocess
from google.colab import drive

drive.mount('/content/drive')
os.chdir('/content')
if os.path.exists('ReviewGPT'): shutil.rmtree('ReviewGPT')
subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/ReviewGPT.git'], check=True)
os.chdir('ReviewGPT')

for f in ['reviews-train.csv', 'reviews-validate.csv', 'reviews-test.csv']:
    shutil.copy2(f'/content/drive/MyDrive/ReviewGPT_Data/{f}', f)

!pip install -q transformers torch scikit-learn pandas numpy tqdm
```

**Block 2 - Train:**
```python
!python train.py
```

## Performance Breakdown

### Star Rating
- 1 star: 66% accuracy
- 2 star: 61% accuracy
- 3 star: 32% accuracy (hardest!)
- 4 star: 58% accuracy
- 5 star: 64% accuracy

### Needs Reply
- Precision: 82%
- Recall: 87%
- F1: 85%
- ROC-AUC: 83%


## License

MIT

## Summary

Smart system that reads app reviews and predicts star ratings + reply necessity. Uses DistilBERT fine-tuned on 9K reviews. Reply detection: 85% accurate (production-ready). Star prediction: 51% accurate. Training: 9 minutes on free Google Colab.
