# ReviewGPT

A multi-task NLP classification system using PyTorch and DistilBERT that analyzes Google Play app reviews to predict:
1. **Star rating** (1-5 stars)
2. **Reply necessity** (binary: needs reply or not)

## Features

- 🤖 **Multi-task Learning**: Single DistilBERT encoder with two classification heads
- 🌍 **Multilingual Support**: Uses `distilbert-base-multilingual-cased` for multiple languages
- 📊 **Comprehensive Metrics**: Accuracy, MAE, F1, Precision, Recall, ROC-AUC
- 🔍 **Exploratory Analysis**: Jupyter notebook with EDA and visualization
- 💻 **CLI Interface**: Easy-to-use command-line tool for predictions
- 🚀 **REST API** (Optional): FastAPI endpoint for production deployment

## Project Structure

```
ReviewGPT/
├── reviews.csv                 # Original dataset
├── reviews-train.csv           # Training set (80%)
├── reviews-validate.csv        # Validation set (10%)
├── reviews-test.csv            # Test set (10%)
├── preprocess_data.py         # Data preprocessing script
├── train.py                   # Training script
├── main.py                    # CLI inference script
├── api.py                     # REST API (optional)
├── requirements.txt           # Python dependencies
├── artifacts/                 # Saved models and metrics
│   ├── model.pt              # Trained model checkpoint
│   ├── metrics.json          # Evaluation metrics
│   └── tokenizer files       # Saved tokenizer
└── notebooks/
    └── dev.ipynb             # EDA and analysis notebook
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository (or navigate to the project directory):

```bash
cd ReviewGPT
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### GPU Support (Optional)

For faster training with GPU:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### 1. Data Preprocessing

First, preprocess the raw reviews data:

```bash
python preprocess_data.py
```

This will:
- Load `reviews.csv`
- Create `needs_reply` labels based on heuristics
- Clean and filter the data
- Split into three stratified datasets:
  - `reviews-train.csv` (80%)
  - `reviews-validate.csv` (10%)
  - `reviews-test.csv` (10%)
- Ensure equal distribution of star ratings across all splits

### 2. Training

Train the model with default parameters:

```bash
python train.py
```

#### Training Options

```bash
python train.py \
  --output_dir artifacts \
  --model_name distilbert-base-multilingual-cased \
  --batch_size 16 \
  --num_epochs 5 \
  --learning_rate 2e-5 \
  --max_length 128 \
  --patience 3
```

#### Key Parameters

- `--output_dir`: Directory to save model and metrics (default: `artifacts`)
- `--batch_size`: Batch size for training (default: 16)
- `--num_epochs`: Maximum number of epochs (default: 5)
- `--learning_rate`: Learning rate for AdamW optimizer (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 128)
- `--patience`: Early stopping patience (default: 3)

The training script will:
- Load pre-split datasets (reviews-train.csv, reviews-validate.csv, reviews-test.csv)
- Train the model with AdamW optimizer
- Use early stopping based on validation F1 score
- Save the best model to `artifacts/model.pt`
- Save metrics to `artifacts/metrics.json`

### 3. Inference

#### CLI - Single Review

```bash
python main.py --checkpoint artifacts/model.pt --text "האפליקציה קורסת כל הזמן"
```

#### CLI - From File

```bash
echo "This app is terrible and keeps crashing!" > review.txt
python main.py --checkpoint artifacts/model.pt --file review.txt
```

#### CLI - Interactive Mode

```bash
python main.py --checkpoint artifacts/model.pt
# Then type or paste your review text
```

#### JSON Output

```bash
python main.py --checkpoint artifacts/model.pt \
  --text "Great app!" \
  --output-format json
```

Example output:

```json
{
  "stars": 2,
  "needs_reply": true,
  "confidence": {
    "stars": [0.01, 0.85, 0.08, 0.04, 0.02],
    "needs_reply": 0.91
  }
}
```

### 4. REST API (Optional)

Start the API server:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "האפליקציה קורסת כל הזמן"}'
```

**API Documentation:**

Visit `http://localhost:8000/docs` for interactive API documentation.

### 5. Exploratory Analysis

Open the Jupyter notebook for data exploration and visualization:

```bash
jupyter notebook notebooks/dev.ipynb
```

The notebook includes:
- Data distribution analysis
- Star rating vs. needs_reply correlation
- Review length analysis
- Model performance visualization
- Confusion matrix
- Error analysis

## Model Architecture

```
DistilBERT (Multilingual)
    │
    ├─── [CLS] Token Representation
         │
         ├─── Star Rating Head (5-class)
         │    └─── Dropout → Linear(768→384) → ReLU → Dropout → Linear(384→5)
         │
         └─── Needs Reply Head (binary)
              └─── Dropout → Linear(768→384) → ReLU → Dropout → Linear(384→1)
```

## Evaluation Metrics

### Star Rating Prediction
- **Accuracy**: Overall classification accuracy
- **MAE (Mean Absolute Error)**: Average distance from true rating
- **Confusion Matrix**: Per-class performance

### Needs Reply Prediction
- **Accuracy**: Binary classification accuracy
- **Precision**: Ratio of correct "needs reply" predictions
- **Recall**: Ratio of actual "needs reply" cases identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Example Results

After training, you should see metrics similar to:

```
Star Rating:
  Accuracy: 0.7234
  MAE: 0.4521

Needs Reply:
  Accuracy: 0.8567
  Precision: 0.8234
  Recall: 0.8912
  F1: 0.8561
  ROC-AUC: 0.9123
```

## Dataset

The model is trained on Google Play app reviews with the following fields:
- `review_text`: The text content of the review
- `stars`: Star rating (1-5)
- `needs_reply`: Binary label indicating if the review needs a response

The `needs_reply` label is generated using heuristics:
- Reviews with 1-2 stars typically need a reply
- Reviews that already received a reply needed one
- 5-star reviews typically don't need a reply
- 3-4 star reviews are evaluated based on community engagement

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors during training:

```bash
python train.py --batch_size 8 --max_length 64
```

### Slow Training

- Use a GPU if available (automatically detected)
- Reduce `max_length` to 64 or 96
- Reduce `batch_size`

### ImportError

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Model

You can use a different pretrained model:

```bash
python train.py --model_name bert-base-multilingual-cased
```

Supported models:
- `distilbert-base-multilingual-cased` (default, faster)
- `bert-base-multilingual-cased` (more accurate)
- `xlm-roberta-base` (best multilingual support)

### Hyperparameter Tuning

Experiment with different hyperparameters:

```bash
python train.py \
  --learning_rate 3e-5 \
  --batch_size 32 \
  --dropout 0.2 \
  --num_epochs 10 \
  --patience 5
```

## Development

### Running Tests

```bash
# Test preprocessing
python preprocess_data.py

# Test inference
python main.py --checkpoint artifacts/model.pt --text "Test review"
```

### Code Quality

The code follows best practices:
- Modular design with reusable components
- Type hints and docstrings
- Error handling
- Progress bars for long operations
- Configurable via command-line arguments

## Citation

If you use this code, please cite:

```
ReviewGPT: Multi-task Classification for App Reviews
Built with PyTorch and Hugging Face Transformers
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional evaluation metrics
- More sophisticated `needs_reply` labeling
- Multi-language support improvements
- Model ensemble methods
- Production deployment guides

## Contact

For questions or issues, please open a GitHub issue or contact the development team.

---

**Built with ❤️ using PyTorch and Hugging Face Transformers**
