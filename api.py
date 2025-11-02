"""
REST API for ReviewGPT using FastAPI.
Provides endpoints for health checks and review classification.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from train import ReviewClassifier

# Initialize FastAPI app
app = FastAPI(
    title="ReviewGPT API",
    description="Multi-task classification API for app reviews",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


class ReviewRequest(BaseModel):
    """Request model for review prediction."""
    text: str = Field(..., description="Review text to classify", min_length=1)
    max_length: Optional[int] = Field(128, description="Maximum sequence length for tokenization")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "האפליקציה קורסת כל הזמן",
                "max_length": 128
            }
        }


class ReviewResponse(BaseModel):
    """Response model for review prediction."""
    stars: int = Field(..., description="Predicted star rating (1-5)")
    needs_reply: bool = Field(..., description="Whether the review needs a reply")
    confidence: dict = Field(..., description="Confidence scores for predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "stars": 2,
                "needs_reply": True,
                "confidence": {
                    "stars": [0.01, 0.85, 0.08, 0.04, 0.02],
                    "needs_reply": 0.91
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


def load_model_and_tokenizer(checkpoint_path: str = "artifacts/model.pt"):
    """
    Load the trained model and tokenizer.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global model, tokenizer, device

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Set device
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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ReviewClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model and tokenizer loaded successfully!")
    return model, tokenizer, device


def predict_review(text: str, max_length: int = 128):
    """
    Predict star rating and needs_reply for a review.

    Args:
        text: Review text
        max_length: Maximum sequence length

    Returns:
        Dictionary with predictions and confidence scores
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model_and_tokenizer first.")

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


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model_and_tokenizer()
        print("API server ready!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded.")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ReviewGPT API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and model information
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device is not None else "unknown"
    }


@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
async def predict(request: ReviewRequest):
    """
    Predict star rating and reply necessity for a review.

    Args:
        request: Review text and optional parameters

    Returns:
        Predictions with confidence scores

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        result = predict_review(request.text, request.max_length)
        return ReviewResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Get API statistics (if metrics file exists).

    Returns:
        Model performance metrics
    """
    import json
    metrics_path = "artifacts/metrics.json"

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Train the model first."
        )

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metrics: {str(e)}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReviewGPT API Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--checkpoint', type=str, default='artifacts/model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')

    args = parser.parse_args()

    # Pre-load model if checkpoint exists
    if os.path.exists(args.checkpoint):
        print("Pre-loading model...")
        load_model_and_tokenizer(args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("API will start but predictions will fail until model is loaded.")

    # Run server
    print(f"\nStarting API server at http://{args.host}:{args.port}")
    print(f"Documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
