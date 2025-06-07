# README

## Setup Instructions

> **Note:** All commands below are intended to be run in your IDE terminal.

### 1. Install Required Libraries

To install all necessary dependencies, run:

```
pip install -r requirements.txt
```

### 2. Run the File for Training the Model

Before starting the API, you must first run the script that preprocesses the features, trains the model, and saves the trained model's parameters into a joblib file. Let it finish before proceeding.

```
python3 train_model.py
```
### 3. Start the API

Afterward, in a separate terminal, run:

```
uvicorn api_main:app --reload
```
### 4. Access the API

Once the API is running, open your browser and go to:

```
http://127.0.0.1:8000/docs
```
### 5. Send a Prediction Request

You can test the API using a curl command. In the same terminal where you ran main.py, run:

```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"words": "[\"triste\", \"world\"]", "lid": "[\"Spanish\", \"English\"]"}'
```

The API can also be tested through the app itself, using input of the format:

```
{
  "words": "[\"triste\", \"child\", \"esta\", \"crying\"]", 
  "lid": "[\"Spanish\", \"English\", \"Spanish\", \"English\"]"
}
```

### 6. Inspect the performance of the models

You can compare the performance of the two models by running the evaluate_models.py file, which prints the accuracy and other metrics for both the baseline and the advanced model. You can run the file using the following command: 

```
python3 evaluate_models.py
```

### Troubleshooting

If you encounter an error about a missing library, install it manually:

```
pip install library_name
```

### Notes

The script main.py trains two models:

  - A baseline Logistic Regression model.

  - An advanced model (XGBoost).

The API uses the advanced model by default. If you want to change which model the API uses, update the file passed to joblib.load on line 17 of api_main.py from 'advanced_model.joblib' to 'logreg_model.joblib':

```
model = joblib.load("advanced_model.joblib")
```


