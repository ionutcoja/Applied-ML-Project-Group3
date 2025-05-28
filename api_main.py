from fastapi import FastAPI
from pydantic import BaseModel
from project_name.pipeline import Pipeline
import pandas as pd

import joblib
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World!!"}


class InputData(BaseModel):
    joined_text: str
    

@app.post("/predict")
async def predict(data: InputData):
    # Example: replace with your actual pipeline/model
    # prediction = Pipeline().predict([[data.feature1, data.feature2]])
    prediction = Pipeline.execute()  # dummy value
    return {"prediction": prediction}
