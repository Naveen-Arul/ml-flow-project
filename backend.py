from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import SafeModelLoader
import uvicorn
import re
import nltk
import mlflow
import mlflow.sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI(title="News Classification API (MLflow Integrated)", description="API for classifying news articles using MLflow stored models")

# Set MLflow tracking URI (optional, default is local)
# mlflow.set_tracking_uri("http://localhost:5000")

# Initialize model loader
loader = SafeModelLoader(models_dir="models")
loader.load_vectorizer()
metadata = loader.load_metadata()

# Preprocessing components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens 
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

class PredictionRequest(BaseModel):
    text: str
    model_name: str = "Logistic Regression"

class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
    prediction_index: int

@app.get("/")
def read_root():
    return {"message": "News Classification API is running", "available_models": list(loader.model_files.keys())}

@app.post("/predict")
def predict_all(request: PredictionRequest):
    """Classify news text using ALL models sequentially"""
    # Preprocess
    cleaned_text = preprocess_text(request.text)
    if not cleaned_text:
         raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")
         
    # Vectorize
    vec = loader.load_vectorizer().transform([cleaned_text])
    
    # Predict with all models
    try:
        results_list = []
        for model_name in loader.model_files.keys():
            res = loader.predict_with_model(model_name, vec)
            results_list.append(res)
            
        # Find best model
        best_prediction = max(results_list, key=lambda x: x.get('confidence', 0))
        
        response_data = {
            "predictions": results_list,
            "best_model": best_prediction['model'],
            "best_prediction": best_prediction['prediction'],
            "confidence": best_prediction['confidence']
        }
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_mlflow", response_model=PredictionResponse)
def predict_mlflow(request: PredictionRequest):
    """Predict using a model loaded from MLflow Model Registry"""
    model_name_clean = f"News_Classifier_{request.model_name.replace(' ', '_')}"
    model_uri = f"models:/{model_name_clean}/latest"
    
    try:
        # Preprocess
        cleaned_text = preprocess_text(request.text)
        vec = loader.load_vectorizer().transform([cleaned_text])
        
        # Load from MLflow
        model = mlflow.sklearn.load_model(model_uri)
        prediction_idx = model.predict(vec)[0]
        
        # Get target names from local metadata (since registry only stores model)
        metadata = loader.load_metadata()
        predicted_class = metadata['target_names'][prediction_idx]
        
        return {
            "model": f"MLflow: {request.model_name}",
            "prediction": predicted_class,
            "confidence": 1.0, # Simple return for MLflow load
            "prediction_index": int(prediction_idx)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow Error: {str(e)}")

@app.get("/models")
def get_models():
    return {"models": list(loader.model_files.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
