
# Language Identification Project – README

This guide explains how to set up and run the project step-by-step. All commands should be run in your IDE terminal.

---

## 1. Install Required Libraries

Before running the project, install all necessary Python libraries using:

```bash
pip install -r requirements.txt
```

If you encounter a missing library error later, you can install it manually like this:

```bash
pip install library_name
```

---

## 2. Run the Training Script

You need to run this script first. It trains two models and saves them for the API to use later.

```bash
python3 main.py
```

Make sure this finishes before continuing. It creates:
- A baseline Logistic Regression model
- An advanced XGBoost model

---

## 3. Start the API

Open a new terminal window (do not close the one running `main.py`) and run:

```bash
uvicorn api.main:app --reload
```

This starts the FastAPI server locally.

---

## 4. Open the API in a Browser

After starting the API, open the following link in your browser:

```
http://127.0.0.1:8000
```

This will open the FastAPI Swagger UI where you can test the endpoints.

---

## 5. Send a Test POST Request

In the same terminal where you ran `main.py`, use this command to send a test request to the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"words": ["triste", "world"], "lid": ["lang2", "lang1"]}'
```

You can modify the `words` and `lid` values to test different input combinations.

---

## 6. Note on the Model Used by the API

By default, the API serves predictions using the advanced XGBoost model.

If you want the API to use a different model (e.g., the baseline Logistic Regression), open the file:

```
api/main.py
```

Then go to line 17 and change this line:

```python
model = joblib.load("advanced_model.joblib")
```

Replace `"advanced_model.joblib"` with `"baseline_model.joblib"` to switch models.

---
