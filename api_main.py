from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
import os

app = FastAPI()


# Define input schema
class InputData(BaseModel):
    words: str  # stringified list: "['hello', 'world']"
    lid: str  # stringified list: "['lang1', 'lang1']"


# Load trained model once
model = joblib.load("advanced_model.joblib")


@app.get("/")
def root():
    return {"message": "Model API is running!"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Create a dataframe from input
        df = pd.DataFrame([{
            "joined_text": "",  # will be overwritten by parse_words_dataset
            "sa": "neutral",  # dummy label
            "words": data.words,
            "lid": data.lid
        }])

        # Preprocess and embed
        parse_words_dataset(df)
        # X, _, _ = embedding_words(df, df, df)

        result = embedding_words(df)
        print("embedding_words result:", result)
        X = result

        # Predict
        print(os.path.isfile("advanced_model.joblib"))
        pred = model.predict(X)
        label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
        return {"prediction": label_mapping[int(pred[0])]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))