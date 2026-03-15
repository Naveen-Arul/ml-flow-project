import streamlit as st
import requests
import json

import os

st.set_page_config(page_title="News Classifier", page_icon="📰")

st.title("📰 News Article Classifier")
st.markdown("Classify news articles into 20 different categories using machine learning models.")

# Backend URL (Use environment variable for production, fallback to local)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar for model selection
st.sidebar.header("Configuration")
# (No model selection - using all models by default)
st.sidebar.info("Using all 6 available models sequentially for prediction.")

# Text input
placeholder_text = """Example News Content (Messy Input WITH ALL NLP CHALLENGES):

--- Tech & Web (URLs, HTML Tags) ---
Check out the <bold>latest news</bold> at https://techcrunch.com/2026/ai about Apple's new M5 chip!

--- Sports (Contractions, Numbers, Punctuation) ---
I can't believe the Lakers scoring 120 points! It's a #huge victory... (3/15/26)

--- Business (Stopwords, Mixed Case, Symbols) ---
The CEO of NVIDIA $NVDA announced that the Earnings are Up By 20% this quarter despite the global inflation.

--- Random Noise ---
!!!!! BREAKING NEWS !!!!! >>>>> 
"""
user_input = st.text_area("Enter News Content:", height=300, placeholder=placeholder_text)

# Checkbox outside the button to avoid state issues
use_mlflow = st.sidebar.checkbox("Use MLflow Registry Model", value=False)

if st.button("Classify"):
    if not user_input.strip():
        # Fallback to placeholder if nothing is typed
        input_to_process = placeholder_text
    else:
        input_to_process = user_input.strip()
    
    with st.spinner("Classifying..."):
        try:
            # Force local /predict to get ALL model results
            payload = {"text": input_to_process}
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                
            if response.status_code == 200:
                result = response.json()
                st.success("Analysis Complete!")
                
                # Check for BOTH possible formats (multi-model and single-model)
                if "predictions" in result:
                    # MULTI-MODEL FORMAT
                    st.subheader("📊 Comparison of All Models")
                    
                    preds_data = []
                    for p in result["predictions"]:
                        preds_data.append({
                            "Model Name": p.get("model", "Unknown"),
                            "Predicted Category": p.get("prediction", "Unknown"),
                            "Confidence Score": f"{p.get('confidence', 0)*100:.2f}%"
                        })
                    
                    preds_data = sorted(preds_data, key=lambda x: float(x["Confidence Score"].strip('%')), reverse=True)
                    st.table(preds_data)
                    
                    st.markdown("---")
                    st.subheader("🥇 Best Prediction")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Category", result.get("best_prediction", "N/A"))
                    with col2:
                        st.metric("Winning Model", result.get("best_model", "N/A"))
                
                elif "prediction" in result:
                    # SINGLE-MODEL FORMAT (Fallback)
                    st.warning("Backend is currently in single-model mode. Showing single results.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category", result.get("prediction", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0)*100:.2f}%")
                    st.info(f"Model: {result.get('model', 'Unknown')}")
                else:
                    st.error("Unexpected response format from backend.")
                    st.write(result)
            else:
                st.error(f"Error from API: {response.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")

st.markdown("---")
st.markdown("### How it works")
st.write("1. The backend uses a custom `SafeModelLoader` to save memory by loading only one model at a time.")
st.write("2. Models are trained on the 20 Newsgroups dataset.")
st.write("3. You can track model performance and artifacts using MLflow.")
