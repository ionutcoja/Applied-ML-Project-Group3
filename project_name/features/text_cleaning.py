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
from tensorflow.keras import layers
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# libraries for embeddings
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


def remove_accents(text):
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))

def remove_at_words(words, labels):
    filtered = [(w, l) for w, l in zip(words, labels) if not (w.startswith("@") or w == "RT")]
    return zip(*filtered) if filtered else ([], [])

def remove_links(words, labels):
    filtered = [(w, l) for w, l in zip(words, labels) if not w.startswith("http")]
    return zip(*filtered) if filtered else ([], [])

def fix_and_parse_list_like_string(s):
    # Remove leading/trailing brackets if they exist
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # This regex finds quoted words (single or double quoted)
    # e.g. 'After' or "'m" or "hello"
    quoted_words = re.findall(r"""(['"])(.+?)\1""", s)

    # Extract only the inner text
    words = [match[1] for match in quoted_words]

    return words

def fix_label_string(s):
    return re.sub(r"'\s+'", "', '", s.strip())

def parse_words_and_labels(entry, label_str):
    if isinstance(entry, str) and entry.strip():
        try:
            # Fix input string by adding commas between quoted words
            words = fix_and_parse_list_like_string(entry)

            # Clean and extract labels
            #label_str = label_str.replace("'", "")
            #labels = re.findall(r'(other|lang1|lang2|lang3)', label_str)

            labels = fix_and_parse_list_like_string(label_str)

            if len(words) != len(labels):
                print("Mismatch:")
                print("Words:", words)
                print("Labels:", labels)
                raise ValueError(f"Length mismatch: {len(words)} words vs {len(labels)} labels")

            # Remove accents
            words = [remove_accents(w) for w in words]
            labels = [remove_accents(l) for l in labels]

            # Remove @user, RT, links
            words, labels = remove_at_words(words, labels)
            words, labels = remove_links(words, labels)

            return [w.lower() for w in words], labels
        except Exception as e:
            raise ValueError(f"Failed to parse:\n{entry}\n{label_str}\nError: {e}")
    return [], []


def parse_words_dataset(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    # STEP 4a: Parse words for the NN model
    for df in [df_train, df_val, df_test]:
        new_words = []
        new_lids = []
        for entry, lid in zip(df['words'], df['lid']):
            words, labels = parse_words_and_labels(entry, lid)
            new_words.append(words)
            new_lids.append(labels)
        df['words'] = new_words
        df['lid'] = new_lids
        
        df['joined_text'] = df['words'].apply(lambda words: ' '.join(words))
    # STEP 4b: Parse and combine words with language for the other models
    '''for df in [df_train, df_val, df_test]:
        df['parsed_words'] = df['words'].apply(parse_words)
        df['words_language'] = list(zip(df['parsed_words'], df['lid']))
        # Convert tuple to single string for embedding: "word1 word2 ... lang"

        # check that the embeddings are for single words

        df['joined_text'] = df['words_language'].apply(lambda t: ' '.join(t[0]) + ' ' + t[1])'''