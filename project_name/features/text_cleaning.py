"""Clean and parse the 'words' column"""

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
from keras import layers
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# libraries for embeddings
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


# remove words that contain @ or are RT - reply tweet
def remove_at_words(words):
    return [w for w in words if not (w.startswith("@") or w == "RT")]

# normalize letters so they don t have accents
def remove_accents(text):
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in normalized if not unicodedata.combining(c)])

# remove links
def remove_links(words):
    return [w for w in words if not w.startswith("http")]

def normalize_limited_repeats(word):
    return re.sub(r'(.)\1{2,}', r'\1\1', word)

# normalize multiple characters (like hiiii)
def parse_words(entry):
    if isinstance(entry, str) and entry.strip():
        # remove those chars from the words
        entry = re.sub(r"#^\[|\]$", "", entry)
        entry = entry.replace('"', '')

        entry = remove_accents(entry)

        # Use regex to extract text between single quotes
        words = re.findall(r"'(.*?)'", entry)

        words = remove_at_words(words)
        words = remove_links(words)

        # make all words lowercase
        words = [word.lower() for word in words if word not in string.punctuation]
        return words
    return []


def parse_words_dataset(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    # STEP 4a: Parse words for the NN model
    for df in [df_train, df_val, df_test]:
        df['parsed_words'] = df['words'].apply(parse_words)
        df['joined_text'] = df['parsed_words'].apply(lambda words: ' '.join(words))

    # STEP 4b: Parse and combine words with language for the other models
    '''for df in [df_train, df_val, df_test]:
        df['parsed_words'] = df['words'].apply(parse_words)
        df['words_language'] = list(zip(df['parsed_words'], df['lid']))
        # Convert tuple to single string for embedding: "word1 word2 ... lang"

        # check that the embeddings are for single words

        df['joined_text'] = df['words_language'].apply(lambda t: ' '.join(t[0]) + ' ' + t[1])'''