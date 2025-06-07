import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
from project_name.models.xgb_model import XGBoostClassifier
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
import joblib


def split_data(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into training and testing
    """

    dataset_train, dataset_test = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset['sa']
    )

    return dataset_train, dataset_test


def preprocess_features(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    parse_words_dataset(dataset)
    X = embedding_words(dataset)

    # encode the sentiment labels as integers
    y = dataset['sa']
    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    y = y.map(label_mapping)
    y = np.asarray(y).astype(np.int32)

    return X, y


def train(model, X_train, y_train, X_val, y_val) -> None:
    """
    Trains the model using the training dataset after compacting the
    input vectors.
    """
    # only pass validation data to the advanced model
    val_data = (X_val, y_val) if isinstance(model, XGBoostClassifier) else None
    model.fit(X_train, y_train, val_data)


def main():
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
    dataset_val = pd.read_csv('project_name/data/sa_spaeng_validation.csv')

    # Replace 'ling1' with 'English' and 'ling2' with 'Spanish' in the 'lid' column
    dataset_train['lid'] = dataset_train['lid'].apply(lambda labels: ['English' if l == 'lang1' else 'Spanish' if l == 'lang2' else l for l in labels])
    dataset_val['lid'] = dataset_val['lid'].apply(lambda labels: ['English' if l == 'lang1' else 'Spanish' if l == 'lang2' else l for l in labels])

    baseline_model = LogisticRegressionClassifier()
    advanced_model = XGBoostClassifier()

    dataset_train, _ = split_data(dataset_train)

    X_train, y_train = preprocess_features(dataset_train)
    X_val, y_val = preprocess_features(dataset_val)

    train(baseline_model, X_train, y_train, X_val, y_val)
    train(advanced_model, X_train, y_train, X_val, y_val)

    joblib.dump(baseline_model, "logreg_model.joblib")
    joblib.dump(advanced_model, "advanced_model.joblib")

if __name__ == "__main__":
    main()
