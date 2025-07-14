#!/usr/bin/env python3
"""
ToxiGuard French Tweets - FastAPI Application

This module provides a REST API for French tweet toxicity detection.
It loads the trained ML model and provides endpoints for real-time
toxicity classification with comprehensive error handling and logging.

Author: ToxiGuard Team
License: MIT
"""

# Standard library imports
import os
import pickle
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Third-party imports
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

# Model configuration
MODEL_PATH = './model/'
MODEL_FILENAME = 'finalized_tfidf.sav'
VECTORIZER_FILENAME = 'vectorizer_tfidf.sav'
LABEL_ENCODER_FILENAME = 'label_encoder.sav'

# API configuration
API_TITLE = "ToxiGuard French Tweets API"
API_DESCRIPTION = """
🛡️ **ToxiGuard French Tweets API**

A machine learning-powered API for detecting toxic content in French tweets.

## Features
- 🔍 Real-time toxicity detection
- 🇫🇷 Optimized for French language content  
- 🧠 SVM classifier with TF-IDF vectorization
- 📊 Confidence scores and detailed predictions
- 🚀 High-performance FastAPI backend

## Usage
Send text to the `/predict/{message}` endpoint to get toxicity classification.
The API returns both the prediction and confidence levels.
"""
API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "ToxiGuard Team",
    "email": "contact@toxiguard.com",
    "url": "https://github.com/yourusername/ToxiGuard-French-Tweets"
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToxicityPrediction(BaseModel):
    """
    Response model for toxicity prediction results.
    """
    text: str = Field(..., description="The analyzed text")
    toxicity: str = Field(..., description="Toxicity classification (Toxique/Non-Toxique)")
    confidence: float = Field(..., description="Prediction confidence (0.0 to 1.0)", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Bonjour, comment allez-vous aujourd'hui?",
                "toxicity": "Non-Toxique",
                "confidence": 0.95,
                "timestamp": "2023-12-01T10:30:00"
            }
        }


class HealthCheck(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# GLOBAL VARIABLES FOR MODEL ARTIFACTS
# =============================================================================

# Global variables to store loaded models
ml_model: Optional[SVC] = None
vectorizer: Optional[TfidfVectorizer] = None
label_encoder: Optional[LabelEncoder] = None


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_model_artifact(filepath: str, artifact_name: str) -> Optional[Any]:
    """
    Load a pickled model artifact with error handling.
    
    Args:
        filepath (str): Path to the pickle file
        artifact_name (str): Name of the artifact for logging
        
    Returns:
        Optional[Any]: Loaded artifact or None if failed
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"❌ {artifact_name} file not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            artifact = pickle.load(f)
            
        logger.info(f"✅ {artifact_name} loaded successfully from {filepath}")
        return artifact
        
    except Exception as e:
        logger.error(f"❌ Error loading {artifact_name}: {e}")
        return None


def load_all_models() -> bool:
    """
    Load all required model artifacts.
    
    Returns:
        bool: True if all models loaded successfully, False otherwise
    """
    global ml_model, vectorizer, label_encoder
    
    logger.info("🔄 Loading model artifacts...")
    
    # Load the trained SVM model
    model_path = os.path.join(MODEL_PATH, MODEL_FILENAME)
    ml_model = load_model_artifact(model_path, "SVM Model")
    
    # Load the TF-IDF vectorizer
    vectorizer_path = os.path.join(MODEL_PATH, VECTORIZER_FILENAME)
    vectorizer = load_model_artifact(vectorizer_path, "TF-IDF Vectorizer")
    
    # Load the label encoder (if exists)
    encoder_path = os.path.join(MODEL_PATH, LABEL_ENCODER_FILENAME)
    label_encoder = load_model_artifact(encoder_path, "Label Encoder")
    
    # Check if essential models are loaded
    if ml_model is None or vectorizer is None:
        logger.error("❌ Failed to load essential model artifacts")
        return False
    
    logger.info("🎉 All model artifacts loaded successfully!")
    return True


def get_model_info() -> Dict[str, Any]:
    """
    Get information about loaded models.
    
    Returns:
        Dict[str, Any]: Model information dictionary
    """
    info = {
        "model_loaded": ml_model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "label_encoder_loaded": label_encoder is not None,
        "vocabulary_size": len(vectorizer.vocabulary_) if vectorizer else 0,
        "model_type": type(ml_model).__name__ if ml_model else None
    }
    return info


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing for prediction.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning (can be enhanced with helpers.clean_text if needed)
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text


def predict_toxicity(text: str) -> ToxicityPrediction:
    """
    Predict toxicity for a given text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        ToxicityPrediction: Prediction result with confidence
        
    Raises:
        HTTPException: If prediction fails or models not loaded
    """
    # Check if models are loaded
    if ml_model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Please check server configuration."
        )
    
    # Validate input
    if not text or not isinstance(text, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text must be a non-empty string"
        )
    
    # Preprocess text
    processed_text = preprocess_text(text)
    if not processed_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is empty after preprocessing"
        )
    
    try:
        # Vectorize the text using the loaded TF-IDF vectorizer
        # Create a new vectorizer with the same vocabulary for consistent feature space
        prediction_vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            vocabulary=vectorizer.vocabulary_
        )
        
        # Transform the text to feature vector
        text_vector = prediction_vectorizer.fit_transform([processed_text])
        
        # Get prediction and confidence
        prediction = ml_model.predict(text_vector)[0]
        
        # Get prediction probabilities for confidence score
        if hasattr(ml_model, 'predict_proba'):
            probabilities = ml_model.predict_proba(text_vector)[0]
            confidence = float(max(probabilities))
        elif hasattr(ml_model, 'decision_function'):
            decision_scores = ml_model.decision_function(text_vector)[0]
            # Convert decision function to probability-like score
            confidence = float(1 / (1 + np.exp(-abs(decision_scores))))
        else:
            confidence = 0.8  # Default confidence for models without probability
        
        # Map prediction to label
        if label_encoder is not None:
            toxicity_label = label_encoder.inverse_transform([prediction])[0]
        else:
            # Fallback mapping if label encoder not available
            label_map = {0: 'Non-Toxique', 1: 'Toxique'}
            toxicity_label = label_map.get(prediction, 'Non-Toxique')
        
        return ToxicityPrediction(
            text=text,
            toxicity=toxicity_label,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Load models on application startup.
    """
    logger.info("🚀 Starting ToxiGuard API...")
    
    if not load_all_models():
        logger.error("❌ Failed to load models on startup")
        # You might want to exit here in production
    else:
        logger.info("✅ API startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown.
    """
    logger.info("🔚 Shutting down ToxiGuard API...")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "🛡️ ToxiGuard French Tweets API",
        "description": "Machine learning-powered toxicity detection for French content",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    model_info = get_model_info()
    
    return HealthCheck(
        status="healthy" if model_info["model_loaded"] else "unhealthy",
        model_loaded=model_info["model_loaded"],
        version=API_VERSION
    )


@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """
    Get detailed information about loaded models.
    """
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    return get_model_info()


@app.get("/predict/{message}", response_model=ToxicityPrediction)
async def predict_toxicity_endpoint(message: str):
    """
    Predict toxicity for a given text message.
    
    Args:
        message (str): The text to analyze for toxicity
        
    Returns:
        ToxicityPrediction: Toxicity classification with confidence score
    """
    # URL decode the message (FastAPI handles this automatically)
    return predict_toxicity(message)


@app.post("/predict", response_model=ToxicityPrediction)
async def predict_toxicity_post(request: Dict[str, str]):
    """
    POST endpoint for toxicity prediction.
    
    Args:
        request (Dict[str, str]): JSON with 'text' field
        
    Returns:
        ToxicityPrediction: Toxicity classification result
    """
    if "text" not in request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request must contain 'text' field"
        )
    
    return predict_toxicity(request["text"])


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler for unexpected errors.
    """
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
