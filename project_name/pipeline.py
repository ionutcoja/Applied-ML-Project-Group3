"""
File containing the ML pipeline.
"""

import pandas as pd
import re
import ast
import string
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
import torch

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# libraries for embeddings
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words

from project_name.models.model import Model


class Pipeline:
    """
    Class to handle the ML pipeline.
    """

    def __init__(self,
                 dataset_train: pd.DataFrame,
                 dataset_val: pd.DataFrame,
                 model: Model) -> None:
        """
        Initialize the pipeline.
        """
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val
        self._model = model


    def _split_data(self) -> None:
        """
        Splits the dataset into training and testing sets based on the
        specified split ratio.
        """
        
        self._dataset_train, self._dataset_test = train_test_split(
            self._dataset_train,
            test_size=0.2,
            random_state=42,
            stratify=self._dataset_train['sa']  # Keeps class distribution
        )
        
    def _preprocess_features(self) -> None:
        parse_words_dataset(self._dataset_train, self._dataset_val, self._dataset_test)
        self._X_train, self._X_val, self._X_test = embedding_words(self._dataset_train, self._dataset_val, self._dataset_test)

        self._y_train = self._dataset_train['sa']
        self._y_val   = self._dataset_val['sa']
        self._y_test  = self._dataset_test['sa']

        label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        self._y_train = self._y_train.map(label_mapping)
        self._y_val   = self._y_val.map(label_mapping)
        self._y_test  = self._y_test.map(label_mapping)

        
        
    def _train(self) -> None:
        """
        Trains the model using the training dataset after compacting the
        input vectors.
        """

        self._model.fit(self._X_train, self._y_train, validation_data=(self._X_val, self._y_val))
        

    def _evaluate(self) -> None:
        """
        Evaluates the model using the test dataset, calculating metrics
        for model performance.
        """
        
        self._metrics_results = self._model.evaluate(self._X_test, self._y_test)

    def execute(self) -> dict:
        """
        Executes the full pipeline process, including feature preprocessing,
        data splitting, model training, and evaluation, returning metrics
        and predictions.

        Returns:
            dict: A dictionary containing training metrics, test metrics,
                  and predictions.
        """
        self._split_data()
        self._preprocess_features()
        self._train()

        self._evaluate()
        test_metrics_results = self._metrics_results
        # test_predictions = self._predictions

        original_X_test_X, original_y_test = self._X_test, self._y_test
        # the part of the dataset that was used for training is now tested
        self._X_test, self._y_test = self._X_train, self._y_train
        self._evaluate()
        train_metrics_results = self._metrics_results
        # set the test data back to the original values
        self._X_test, self._y_test = original_X_test_X, original_y_test

        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results
            #"predictions": test_predictions
        }
