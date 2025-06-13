# README

IMPORTANT NOTES:
- All code was run in Python 3.9
- The two necessary datasets are in the repository, under the 'data' folder. They can also be found here: training - https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset?select=sa_spaeng_train.csv; validation (used for testing) - https://www.kaggle.com/datasets/thedevastator/unlock-universal-language-with-the-lince-dataset?select=sa_spaeng_validation.csv

## Description of the task

The task involves sentiment classification of sentences from a bilingual dataset of tweets in English and Spanish. Each sentence may include a mix of both languages, along with tags, emojis, and other special characters (links, etc.). The goal is to assign one of three sentiment labels (`positive`, `negative`, or `neutral`) to each sentence.

## Preprocessing

Before generating embeddings or performing classification, the tweet data is preprocessed to clean and normalize text:

- **Parsing**: Tweets are originally stored as stringified lists. These are parsed to extract words and their corresponding language labels.
- **Noise Removal**: Mentions (`@user`), retweet markers (`RT`), and URLs are removed.
- **Accent Normalization**: Accents are stripped from characters to ensure consistency (e.g., *niño* → *nino*).
- **Casing**: Words are lowercased.
- **Text Reconstruction**: A `joined_text` column is added by concatenating the cleaned words, ready for embedding and modeling.

> **IMPORTANT** Emojis, hashtags, and other non-standard tokens are retained as their meaning might alter the assigned label.

## Embedders

This project uses two separate models from Hugging Face to embed English and Spanish words:

- **Spanish Text**: To embed Spanish (`lid` = `lang2`) words we use **DCCucuchile BERT** model (dccuchile/bert-base-spanish-wwm-cased).
- **English and Other Text** For all other cases (including English words), we fall back to a multilingual model **LaBSE** (sentence-transformers/LaBSE).

The code checks language distribution in each text and routes it to the appropriate model.

## Description of the models

This project uses two models trained on a multi-lingual sentiment analysis task:
- **Baseline model**: A wrapper around the Logistic Regression classifier from Sklearn  
- **Advanced model**: An cnn.Sequential` Deep Neural Network from PyTorch Trained using gradient descent, which has:
  - Linear layers  
  - ReLU activation  
  - Normalization layers  
  - Dropout layers  

The optimal hyperparameters for the two models were found by measuring the performance of the models with different hyperparameter settings using cross-validation. Cross-validation on the two models can be performed by executing the cross_validation.py file

## Setup Instructions

> **Notes:** All commands below are intended to be run in your IDE terminal.

### 1. Install Required Libraries

To install all necessary dependencies, run:

```
pip install -r requirements.txt
```

### 2. Run the File for Training the Model

Before starting the API, you must first run the script that preprocesses the features, trains the two models (Logistic Regression and DNN), and saves the trained models' parameters into joblib files. Let it finish before proceeding.

```
python3 train_model.py
```

### 3. Inspect the performance of the models

You can compare the performance of the two models by running the evaluate_models.py file, which loads the models, evaluates them, prints the accuracy, F1, and other metrics for both the baseline and the advanced model. The code also outputs a statistical comparison between the two models using bootstraping. You can run the file using the following command: 

```
python3 evaluate_models.py
```

### 4. Start the API

Afterward, in a separate terminal, run:

```
uvicorn api_main:app --reload
```

### 5. Access the API

Once the API is running, open your browser and go to:

```
http://127.0.0.1:8000/docs
```
### 6. Send a Prediction Request

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

### Troubleshooting

If you encounter an error about a missing library, install it manually:

```
pip install library_name
```

### Notes

The API uses the advanced model by default. If you want to change which model the API uses, update the file passed to joblib.load on line 17 of api_main.py from 'advanced_model.joblib' to 'logreg_model.joblib':

```
model = joblib.load("advanced_model.joblib")
```


