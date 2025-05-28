from project_name.pipeline import Pipeline
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
from project_name.models.tf_keras_sequential_model import KerasSequentialClassifier



import pandas as pd

dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')
dataset_val = pd.read_csv('project_name/data/sa_spaeng_validation.csv')

pipeline_log_reg = Pipeline(dataset_train, dataset_val, model = LogisticRegressionClassifier())
pipeline_tf_keras = Pipeline(dataset_train, dataset_val, model = KerasSequentialClassifier())

print(pipeline_tf_keras.execute())