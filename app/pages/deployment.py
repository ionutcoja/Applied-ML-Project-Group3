import streamlit as st
import joblib
import pandas as pd
import os
import sys
from project_code.features.text_cleaning import parse_words_dataset
from project_code.features.text_embeddings import embedding_words


# Ensure the project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.title("Model Deployment")

# 1. Select model type
model_type = st.selectbox("Select model type", ["logreg", "advanced"])

# 2. Load model from local file based on selection
model_path = f"{model_type}_model.joblib"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found in this directory.")
    model = None
else:
    model = joblib.load(model_path)
    st.success(f"{model_type.capitalize()} model loaded from {model_path}!")

# 3. Input data for prediction
st.header("Input Data Point")
words_input = st.text_input("Enter words (space-separated, e.g. 'hello world')")
lids_input = st.text_input("Enter corresponding LIDs (space-separated, e.g. 'lang1 lang1')")

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            # Convert string input to Python lists
            words = words_input.strip().split()
            lids = lids_input.strip().split()
            
            lid_map = {"English": "lang1", "Spanish": "lang2"}
            lids = [lid_map.get(tag.lower(), tag) for tag in lids]
            
            if len(words) != len(lids):
                st.error("The number of words must match the number of LID tags.")
            else:
                df = pd.DataFrame([{
                    "words": words,
                    "lid": lids
                }])
                
                parse_words_dataset(df)
                result = embedding_words(df)
                X = result
                
                pred = model.predict(X)
                label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
                st.success(f"Prediction: {label_mapping[int(pred[0])]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
