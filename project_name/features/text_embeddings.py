from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def embedding_words(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    # STEP 5: Sentence Embedding with multilingual model suitable for code-switching
    model = SentenceTransformer('sentence-transformers/LaBSE')

    X_train = np.array(model.encode(df_train['joined_text'].tolist(), show_progress_bar=True, batch_size=16),dtype=np.float32)
    X_val   = np.array(model.encode(df_val['joined_text'].tolist(), show_progress_bar=True, batch_size=16),dtype=np.float32)
    X_test  = np.array(model.encode(df_test['joined_text'].tolist(), show_progress_bar=True, batch_size=16),dtype=np.float32)

    return X_train, X_val, X_test

"""
# STEP 5: Sentence Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
X_train = model.encode(df_train['joined_text'].tolist(), show_progress_bar=True, batch_size=16)
X_val   = model.encode(df_val['joined_text'].tolist(), show_progress_bar=True, batch_size=16)
X_test  = model.encode(df_test['joined_text'].tolist(), show_progress_bar=True, batch_size=16)


# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.85)
X_train = vectorizer.fit_transform(df_train['joined_text'])
X_val = vectorizer.transform(df_val['joined_text'])
X_test = vectorizer.transform(df_test['joined_text'])
"""