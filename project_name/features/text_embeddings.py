from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


eng_model = SentenceTransformer('sentence-transformers/LaBSE')
spanish_model_name = "dccuchile/bert-base-spanish-wwm-cased"
spanish_tokenizer = AutoTokenizer.from_pretrained(spanish_model_name)
spanish_model = AutoModel.from_pretrained(spanish_model_name)


def get_embeddings(df, spanish_model, spanish_tokenizer, multilingual_model, batch_size=16):
    texts = df['joined_text'].tolist()
    lids_list = df['lid'].tolist()

    # Determine which index uses Spanish model
    use_spanish = [i for i, lids in enumerate(lids_list) if lids.count('lang2') > len(lids) / 2]
    use_multilingual = [i for i in range(len(lids_list)) if i not in use_spanish]

    # Initialize output list
    embeddings = [None] * len(texts)

    # --- Encode Spanish texts (BETO)
    print("Encoding with Spanish model (BETO)...")
    for i in tqdm(range(0, len(use_spanish), batch_size)):
        batch_indices = use_spanish[i:i + batch_size]
        batch_texts = [texts[j] for j in batch_indices]
        inputs = spanish_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = spanish_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        for idx, emb in zip(batch_indices, batch_embeddings):
            embeddings[idx] = emb

    # --- Encode other texts (Multilingual model like LaBSE)
    print("Encoding with Multilingual model...")
    multilingual_texts = [texts[i] for i in use_multilingual]
    multilingual_embeddings = multilingual_model.encode(multilingual_texts, batch_size=batch_size, show_progress_bar=True)
    for idx, emb in zip(use_multilingual, multilingual_embeddings):
        embeddings[idx] = emb

    return np.array(embeddings)


def embedding_words(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    # STEP 5: Sentence Embedding with multilingual model suitable for code-switching
    # model = SentenceTransformer('sentence-transformers/LaBSE')

    X_train = get_embeddings(df_train, spanish_model, spanish_tokenizer, eng_model)
    X_val   = get_embeddings(df_val, spanish_model, spanish_tokenizer, eng_model)
    X_test  = get_embeddings(df_test, spanish_model, spanish_tokenizer, eng_model)
    
    print(X_train[0])  
    print(X_test[0])    

    return np.array(X_train, dtype=np.float32), np.array(X_val, dtype=np.float32), np.array(X_test, dtype=np.float32)
