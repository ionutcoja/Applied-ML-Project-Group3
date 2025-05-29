from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words

app = FastAPI()

class InputData(BaseModel):
    words: str  #this is a stringified list: "['sad', 'mundo']"
    lid: str  #this is a stringified list: "['lang1', 'lang2']"

model = joblib.load("advanced_model.joblib")


@app.get("/")
def root():
    return {"message": "Model API is running!"}


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([{
            "joined_text": "",
            "sa": "neutral",
            "words": data.words,
            "lid": data.lid
        }])

        parse_words_dataset(df)
        result = embedding_words(df)
        X = result

        pred = model.predict(X)
        label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
        return {"prediction": label_mapping[int(pred[0])]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
