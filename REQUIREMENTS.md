Objective

Build an NLP classification system using PyTorch and DistilBERT that analyzes Google Play app reviews (from a CSV file) and predicts:

The star rating (1–5) of the review.

Whether the review requires a reply or action (binary classification).

This is a supervised learning task with two classification heads on top of a shared DistilBERT encoder.

Scope of Work
1. Input

A CSV file named reviews.csv, containing at least:

review_text — the text of the review.

stars — integer label (1–5).

needs_reply — binary label (0 or 1).

2. Output

Trained model weights (PyTorch .pt or .bin file).

CLI and/or REST API to run predictions.

Evaluation metrics on a held-out validation/test set.

Example output for a single review:

{
  "stars": 2,
  "needs_reply": true,
  "confidence": {
    "stars": [0.01, 0.85, 0.08, 0.04, 0.02],
    "needs_reply": 0.91
  }
}

System Components
1. Data Processing

Load and clean the dataset (pandas).

Tokenize text with DistilBERT tokenizer (AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')).

Split into train, validation, and test (e.g., 80/10/10).

2. Model Architecture

Use DistilBERT (base multilingual cased) as the backbone encoder.

Add two classification heads:

Head A: 5-class softmax for star ratings.

Head B: binary sigmoid classifier for needs_reply.

Example (PyTorch-style):

class ReviewClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.star_head = nn.Linear(hidden, 5)
        self.reply_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        stars = self.star_head(pooled)
        reply = self.reply_head(pooled)
        return stars, reply

3. Training

Framework: PyTorch (not TensorFlow).

Optimizer: AdamW.

Scheduler: linear warmup/decay.

Loss function:

total_loss = CE_loss(stars_pred, stars_label) + BCE_loss(reply_pred, reply_label)


Use early stopping on validation F1 or accuracy.

4. Evaluation

Metrics:

Star rating: Accuracy, MAE, and confusion matrix.

Needs reply: Precision, Recall, F1-score, ROC-AUC.

Store metrics in artifacts/metrics.json.

5. Scripts
train.py

Reads CSV.

Trains the model.

Logs losses and metrics.

Saves weights and tokenizer.

main.py

Loads trained model.

Accepts a review via CLI or file.

Outputs predictions in JSON format.

Example:

python main.py --checkpoint artifacts/model.pt --text "האפליקציה קורסת כל הזמן"

Optional:

api.py — FastAPI or Flask REST route for prediction.

Deliverables

✅ train.py — PyTorch training script

✅ main.py — CLI inference script

✅ notebooks/dev.ipynb — exploratory notebook with EDA, model selection, plots

✅ requirements.txt — dependencies (PyTorch, Transformers, Pandas, Scikit-learn, etc.)

✅ README.md — setup, training, and inference instructions

✅ artifacts/ folder — model, tokenizer, metrics

⬜ (Optional bonus) api.py — REST endpoint

Evaluation Criteria
Area	Details
Model quality	Accuracy, F1, and evidence of learning
Code structure	Modular, clean, reusable
Documentation	Clear instructions, readable notebook
Efficiency	Uses batching, GPU if available
Innovation	Any bonus tasks or creative improvements
Allowed Libraries

torch, torch.nn, torch.utils.data

transformers (Hugging Face)

pandas, numpy, sklearn, matplotlib, seaborn

Optional: fastapi / flask, uvicorn

Bonus (Optional)

Add a REST API endpoint (/predict) that returns JSON results.

Add Markdown analysis cells showing confusion matrix and examples of misclassification.

Add visualization of training/validation curves.

Add requirements.txt and optional Dockerfile for portability.