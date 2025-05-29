
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
from project_name.models.xgb_model import XGBoostClassifier
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
import joblib


def split_data(dataset: pd.DataFrame) -> None:
    """
    Splits a dataset into training and testing
    """

    dataset_train, dataset_test = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset['sa']  # Keeps class distribution
    )

    return dataset_train, dataset_test


def preprocess_features(dataset_train: pd.DataFrame, dataset_val: pd.DataFrame, dataset_test: pd.DataFrame) -> None:
    parse_words_dataset(dataset_train, dataset_val, dataset_test)
    X_train, X_val, X_test = embedding_words(dataset_train, dataset_val, dataset_test)

    y_train = dataset_train['sa']
    y_val   = dataset_val['sa']
    y_test  = dataset_test['sa']

    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    y_train = y_train.map(label_mapping)
    y_val   = y_val.map(label_mapping)
    y_test  = y_test.map(label_mapping)

    y_train = np.asarray(y_train).astype(np.int32)
    y_val   = np.asarray(y_val).astype(np.int32)
    y_test  = np.asarray(y_test).astype(np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train(model, X_train, y_train, X_val, y_val):
    """
    Trains the model using the training dataset after compacting the
    input vectors.
    """
    val_data = (X_val, y_val) if isinstance(model, XGBoostClassifier) else None

    model.fit(X_train, y_train, val_data)
    return model #possibly working


def predict(model, X_test):
    """
    Predicts the labels for the test dataset using the trained model.
    """

    predictions = model.predict(X_test)

    return predictions


def evaluate(model, X_test, y_test) -> None:
    """
    Evaluates the model using the test dataset, calculating metrics
    for model performance.
    """

    metrics_results = model.evaluate(X_test, y_test)
    return metrics_results


def main():
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
    dataset_val = pd.read_csv('project_name/data/sa_spaeng_validation.csv')

    baseline_model = LogisticRegressionClassifier()
    advanced_model = XGBoostClassifier()

    dataset_train, dataset_test = split_data(dataset=dataset_train)
    X_train, y_train, X_val, y_val, X_test, y_test =  preprocess_features(dataset_train, dataset_val, dataset_test)

    trained_baseline_model = train(baseline_model, X_train, y_train, X_val, y_val)
    trained_advanced_model = train(advanced_model, X_train, y_train, X_val, y_val)

    joblib.dump(trained_baseline_model, "logreg_model.joblib")  # Save the model
    joblib.dump(trained_advanced_model, "neural_network_model.joblib")  # Save the model

    metrics_results_training = evaluate(baseline_model, X_train, y_train)
    print("Baseline Model Metrics Training:", metrics_results_training)

    metrics_results_test = evaluate(baseline_model, X_test, y_test)
    print("Baseline Model Metrics Test:", metrics_results_test)

    metrics_results_training = evaluate(advanced_model, X_train, y_train)
    print("Advanced Model Metrics Training:", metrics_results_training)

    metrics_results_test = evaluate(advanced_model, X_test, y_test)
    print("Advanced Model Metrics Test:", metrics_results_test)


if __name__ == "__main__":
    main()


'''
from project_name.pipeline import Pipeline
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
from project_name.models.tf_keras_sequential_model import KerasSequentialClassifier

import pandas as pd
import joblib  # For saving sklearn models

# Load datasets
dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
dataset_val = pd.read_csv('project_name/data/sa_spaeng_validation.csv')

# Train Logistic Regression model
pipeline_log_reg = Pipeline(dataset_train, dataset_val, model=LogisticRegressionClassifier())
logreg_results = pipeline_log_reg.execute()
joblib.dump(pipeline_log_reg._model, "logreg_model.joblib")  # Save the model

# Train Keras Sequential model
pipeline_tf_keras = Pipeline(dataset_train, dataset_val, model=KerasSequentialClassifier())
keras_results = pipeline_tf_keras.execute()
pipeline_tf_keras._model.save("keras_model.h5")  # Save the Keras model

# Print results
print("Logistic Regression Metrics:", logreg_results)
print("Keras Sequential Metrics:", keras_results)

if __name__ == "__main__":
    main()

'''
