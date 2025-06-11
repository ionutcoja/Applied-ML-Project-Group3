import pandas as pd
import numpy as np
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
from project_name.models.DNN import DNNClassifier
import joblib


def preprocess_features(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    parse_words_dataset(dataset)
    X = embedding_words(dataset)

    # encode the sentiment labels as integers
    y = dataset['sa']
    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    y = y.map(label_mapping)
    y = np.asarray(y).astype(np.int32)

    return X, y


def train(model, X_train, y_train) -> None:
    """
    Trains the model using the training dataset after compacting the
    input vectors.
    """
    # only pass validation data to the advanced model
    val_data = None
    model.fit(X_train, y_train, val_data)


def main():
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')

    baseline_model = LogisticRegressionClassifier()

    X_train, y_train = preprocess_features(dataset_train)

    # number of features for DNN
    input_dim = X_train.shape[1]  
    advanced_model = DNNClassifier(input_dim=input_dim)

    train(baseline_model, X_train, y_train)
    train(advanced_model, X_train, y_train)

    joblib.dump(baseline_model, "logreg_model.joblib")
    joblib.dump(advanced_model, "advanced_model.joblib")
    

if __name__ == "__main__":
    main()
