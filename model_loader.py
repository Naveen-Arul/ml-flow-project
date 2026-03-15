# ======================================================
# MEMORY-SAFE MODEL LOADER
# ======================================================
# Loads ONE model at a time, makes prediction, then unloads
# Ensures only ONE ML model exists in memory at any point
# ======================================================

import joblib
import os
import gc
from typing import Dict, Any

class SafeModelLoader:
    """
    Memory-safe model loader for low-RAM environments (512MB).
    Loads models sequentially, predicts, and immediately unloads.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.vectorizer = None
        self.metadata = None
        self.model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Linear SVM": "linear_svm.pkl",
            "Multinomial Naive Bayes": "multinomial_naive_bayes.pkl",
            "Complement Naive Bayes": "complement_naive_bayes.pkl",
            "Bernoulli Naive Bayes": "bernoulli_naive_bayes.pkl",
            "SGD Classifier": "sgd_classifier.pkl"
        }
    
    def load_vectorizer(self):
        """Load TF-IDF vectorizer once and keep in memory"""
        if self.vectorizer is None:
            vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"✓ Loaded TF-IDF vectorizer")
        return self.vectorizer
    
    def load_metadata(self):
        """Load metadata (target names, model info)"""
        if self.metadata is None:
            metadata_path = os.path.join(self.models_dir, "metadata.pkl")
            self.metadata = joblib.load(metadata_path)
        return self.metadata
    
    def predict_with_model(self, model_name: str, text_vector) -> Dict[str, Any]:
        """
        Load ONE model, make prediction, unload immediately.
        
        Args:
            model_name: Name of the model
            text_vector: Vectorized text
            
        Returns:
            Dictionary with prediction results
        """
        model_file = self.model_files.get(model_name)
        if not model_file:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = os.path.join(self.models_dir, model_file)
        
        # Load model
        model = joblib.load(model_path)
        
        # Make prediction
        prediction_idx = model.predict(text_vector)[0]
        
        # Get confidence/probability if available
        try:
            probabilities = model.predict_proba(text_vector)[0]
            confidence = float(probabilities[prediction_idx])
        except AttributeError:
            # SVM doesn't have predict_proba
            decision = model.decision_function(text_vector)[0]
            # Normalize to 0-1 range
            exp_decision = np.exp(decision - np.max(decision))
            confidence = float(exp_decision[prediction_idx] / exp_decision.sum())
        
        # Get class name
        metadata = self.load_metadata()
        predicted_class = metadata['target_names'][prediction_idx]
        
        # Prepare result
        result = {
            "model": model_name,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "prediction_index": int(prediction_idx)
        }
        
        # CRITICAL: Immediately unload model
        del model
        gc.collect()
        
        return result
    
    def predict_all_sequential(self, text: str):
        """
        Predict using ALL models sequentially (one-by-one).
        Yields results as each model completes.
        
        Args:
            text: Raw text to classify
            
        Yields:
            Dictionary with model prediction results
        """
        # Load vectorizer (stays in memory)
        vectorizer = self.load_vectorizer()
        
        # Vectorize text once
        text_vector = vectorizer.transform([text])
        
        # Predict with each model sequentially
        for model_name in self.model_files.keys():
            try:
                result = self.predict_with_model(model_name, text_vector)
                yield result
            except Exception as e:
                # If a model fails, return error for that model
                yield {
                    "model": model_name,
                    "error": str(e),
                    "prediction": None,
                    "confidence": 0.0
                }
        
        # Clean up text vector
        del text_vector
        gc.collect()
    
    def get_model_info(self):
        """Get list of available models with metadata"""
        metadata = self.load_metadata()
        return {
            "models": [
                {
                    "name": model_info['name'],
                    "accuracy": model_info['accuracy'],
                    "f1_score": model_info['f1_score']
                }
                for model_info in metadata['models']
            ],
            "total_models": len(self.model_files),
            "categories": metadata['target_names']
        }


# Import numpy for confidence calculation
import numpy as np
