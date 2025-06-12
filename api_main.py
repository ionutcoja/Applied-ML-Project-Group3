from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words

app = FastAPI()

class InputData(BaseModel):
    """
    Input data model for prediction endpoint.

    Attributes:
        words (str): Space-separated words (e.g., "sad mundo")
        lid (str): Space-separated language tags (e.g., "English Spanish")
    """
    words: str
    lid: str

model = joblib.load("advanced_model.joblib")


@app.get("/")
def root():
    """
    Root endpoint to check if the API is running.

    Returns:
        dict: A message indicating the API status.
    """
    return {"message": "Model API is running!"}


@app.post("/predict")
def predict(data: InputData):
    """
    Predicts sentiment based on input words and language identifiers.

    Args:
        data (InputData): Input data containing stringified lists of words and language IDs.
    Returns:
        dict: The predicted sentiment label.
    Raises:
        HTTPException: If prediction fails due to invalid input or processing error.
    """

    try:
        # Split input strings by space
        words = data.words.strip().split()
        lids = data.lid.strip().split()

        if len(words) != len(lids):
            raise ValueError("The number of words and language tags must match.")

        # Change the lids tags to lang1 and lang2
        lids = ['lang1' if l == 'English' else 'lang2' if l == 'Spanish' else l for l in lids
        ]

        df = pd.DataFrame([{
            "words": str(words),
            "lid": str(lids)
        }])

        parse_words_dataset(df)
        result = embedding_words(df)
        X = result

        pred = model.predict(X)
        label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
        return {"prediction": label_mapping[int(pred[0])]}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed. Please ensure 'words' and 'lid' are valid stringified lists and contain matching lengths. Error: {str(e)}"
        )

