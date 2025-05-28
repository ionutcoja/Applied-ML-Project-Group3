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
