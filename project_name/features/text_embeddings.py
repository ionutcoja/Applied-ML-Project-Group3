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


def get_embeddings(df, spanish_model, spanish_tokenizer, multilingual_model, batch_size=16)-> np.ndarray:
    texts = df['joined_text'].tolist()
    lids_list = df['lid'].tolist()

    # If the majority of the words are in Spanish, use the Spanish model; otherwise, use the multilingual model
    use_spanish = [i for i, lids in enumerate(lids_list) if lids.count('lang2') > len(lids) / 2]
    use_multilingual = [i for i in range(len(lids_list)) if i not in use_spanish]

    embeddings = [None] * len(texts)

    print("Encoding with Spanish model")
    for i in tqdm(range(0, len(use_spanish), batch_size)):
        batch_indices = use_spanish[i:i + batch_size]
        batch_texts = [texts[j] for j in batch_indices]
        inputs = spanish_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = spanish_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        for idx, emb in zip(batch_indices, batch_embeddings):
            embeddings[idx] = emb

    print("Encoding with Multilingual model")
    multilingual_texts = [texts[i] for i in use_multilingual]
    multilingual_embeddings = multilingual_model.encode(multilingual_texts, batch_size=batch_size, show_progress_bar=True)
    for idx, emb in zip(use_multilingual, multilingual_embeddings):
        embeddings[idx] = emb

    return np.array(embeddings)


def embedding_words(df: pd.DataFrame) -> np.ndarray:
    X = get_embeddings(df, spanish_model, spanish_tokenizer, eng_model)
    return np.array(X, dtype=np.float32)
