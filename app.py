# ======================================================
# MEMORY-SAFE FASTAPI BACKEND
# ======================================================
# Serves 6 lightweight ML models with SEQUENTIAL loading
# Designed for 512MB RAM deployment (Render Free Tier)
# ======================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re
import gc
import json
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from model_loader import SafeModelLoader

# ======================================================
# GLOBAL VARIABLES
# ======================================================
model_loader = None
stop_words = None
lemmatizer = None

# ======================================================
# STARTUP/SHUTDOWN EVENTS
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown"""
    global model_loader, stop_words, lemmatizer
    
    print("\n" + "="*70)
    print("STARTING MEMORY-SAFE NEWS CLASSIFICATION API")
    print("="*70)
    
    # Download NLTK data
    print("\n[1/3] Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Initialize preprocessing tools
    print("[2/3] Initializing preprocessing...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Initialize model loader (does NOT load models yet)
    print("[3/3] Initializing model loader...")
    model_loader = SafeModelLoader()
    # Load vectorizer once (stays in memory)
    model_loader.load_vectorizer()
    model_loader.load_metadata()
    
    print("\n✓ API Ready!")
    print("✓ Models will be loaded ONE-BY-ONE on demand")
    print("✓ Memory-safe mode active\n")
    print("="*70)
    
    yield
    
    # Cleanup on shutdown
    print("\n[Shutdown] Cleaning up...")
    del model_loader, stop_words, lemmatizer
    gc.collect()

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(
    title="Memory-Safe News Classification API",
    description="Sequential model inference for low-RAM deployment",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# REQUEST/RESPONSE MODELS
# ======================================================
class PredictionRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    model: str
    prediction: str
    confidence: float
    status: str = "completed"

# ======================================================
# TEXT PREPROCESSING
# ======================================================
def preprocess_text(text: str) -> str:
    """
    Clean and normalize user input text.
    Same preprocessing used during training.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(tokens)

# ======================================================
# API ENDPOINTS
# ======================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Memory-Safe News Classification API",
        "status": "running",
        "models": 6,
        "deployment": "Render Free Tier (512MB RAM)"
    }

@app.get("/models")
async def get_models():
    """Get list of available models with metadata"""
    try:
        model_info = model_loader.get_model_info()
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get list of news categories"""
    try:
        metadata = model_loader.load_metadata()
        return {
            "categories": metadata['target_names'],
            "total": len(metadata['target_names'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Classify news text using ALL models sequentially.
    Models are loaded ONE-BY-ONE to save memory.
    
    Returns predictions as JSON array (not streaming).
    """
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text must be at least 10 characters"
            )
        
        # Preprocess text
        cleaned_text = preprocess_text(request.text)
        
        if not cleaned_text or cleaned_text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Text contains no valid content after preprocessing"
            )
        
        # Get predictions from all models sequentially
        results = []
        for prediction in model_loader.predict_all_sequential(cleaned_text):
            results.append(prediction)
        
        # Find best model (highest confidence)
        best_prediction = max(results, key=lambda x: x.get('confidence', 0))
        
        # Return all results
        return {
            "input_text": request.text,
            "cleaned_text": cleaned_text,
            "predictions": results,
            "best_model": best_prediction['model'],
            "best_prediction": best_prediction['prediction'],
            "confidence": best_prediction['confidence']
        }
    except Exception as e:
            "best_prediction": best_prediction['prediction'],
            "total_models": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-stream")
async def predict_stream(request: PredictionRequest):
    """
    Stream predictions as models complete (one-by-one).
    Useful for showing incremental progress in UI.
    """
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text must be at least 10 characters"
            )
        
        # Preprocess text
        cleaned_text = preprocess_text(request.text)
        
        if not cleaned_text or cleaned_text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Text contains no valid content after preprocessing"
            )
        
        # Generator function for streaming
        async def prediction_generator():
            for prediction in model_loader.predict_all_sequential(cleaned_text):
                # Yield as Server-Sent Events (SSE) format
                yield f"data: {json.dumps(prediction)}\n\n"
        
        return StreamingResponse(
            prediction_generator(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during streaming prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
