import streamlit as st
import joblib
import pandas as pd
import os


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
words = st.text_input("Words (as a Python list, e.g. ['hello', 'world'])")
lid = st.text_input("LID (as a Python list, e.g. ['lang1', 'lang1'])")

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            # Convert string input to Python lists
            words_list = eval(words)
            lid_list = eval(lid)
            df = pd.DataFrame([{
                "words": words_list,
                "lid": lid_list
            }])
            # You may need to preprocess df here, depending on your pipeline
            pred = model.predict(df)
            st.success(f"Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")