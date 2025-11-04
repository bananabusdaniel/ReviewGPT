"""
Simple REST API for ReviewGPT.
Single endpoint: POST /predict
"""

import os
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from train import ReviewClassifier

app = FastAPI(title="ReviewGPT API", version="1.0.0")

# Global model variables
model = None
tokenizer = None
device = None


class PredictRequest(BaseModel):
    """Request: review text to analyze."""
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "App crashes every time I open it"}
        }


class PredictResponse(BaseModel):
    """Response: predictions with confidence scores."""
    stars: int
    needs_reply: bool
    confidence: dict

    class Config:
        json_schema_extra = {
            "example": {
                "stars": 1,
                "needs_reply": True,
                "confidence": {
                    "stars": [0.75, 0.15, 0.05, 0.03, 0.02],
                    "needs_reply": 0.92
                }
            }
        }


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, tokenizer, device

    checkpoint_path = "artifacts/model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Model not found at {checkpoint_path}")
        print("API will start but /predict will fail until model is loaded.")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_path = os.path.dirname(checkpoint_path)
    if not os.path.exists(os.path.join(tokenizer_path, 'tokenizer_config.json')):
        tokenizer_path = 'distilbert-base-multilingual-cased'

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ReviewClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully!")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict star rating and reply necessity for a review.

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "Great app but crashes sometimes"}'
    ```
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize
        encoding = tokenizer(
            request.text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            stars_logits, reply_logits = model(input_ids, attention_mask)

            # Star rating
            stars_probs = F.softmax(stars_logits, dim=1).squeeze().cpu().numpy()
            predicted_stars = int(stars_probs.argmax()) + 1  # 1-5

            # Needs reply
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReviewGPT API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')

    args = parser.parse_args()

    print(f"\nStarting API at http://{args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/docs\n")

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
