from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    words: str  # stringified list: "['hello', 'world']" --> change to list of strings
    lid: str    # stringified list: "['lang1', 'lang1']" --> same

# Load trained model once
model = joblib.load("logreg_model.joblib")

@app.get("/")
def root():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Create a dataframe from input
        df = pd.DataFrame([{
            "joined_text": "",      # will be overwritten by parse_words_dataset
            "sa": "neutral",        # dummy label
            "words": data.words,
            "lid": data.lid
        }])

        # Preprocess and embed
        parse_words_dataset(df, df, df)
        # X, _, _ = embedding_words(df, df, df)
        
        result = embedding_words(df, df, df)
        print("embedding_words result:", result)
        X, _, _ = result

        # Predict
        pred = model.predict(X)
        label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
        return {"prediction": label_mapping[int(pred[0])]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
