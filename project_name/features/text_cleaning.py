import pandas as pd
import re
import unicodedata


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

    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    quoted_words = re.findall(r"""(['"])(.+?)\1""", s)

    words = [match[1] for match in quoted_words]

    return words

def parse_words_and_labels(entry, label_str):
    if isinstance(entry, str) and entry.strip():
        try:

            # adding commas between the words and the language tags so they would be formatted as a proper list
            words = fix_and_parse_list_like_string(entry)
            labels = fix_and_parse_list_like_string(label_str)

            # sanity check
            if len(words) != len(labels):
                print("Mismatch:")
                print("Words:", words)
                print("Labels:", labels)
                raise ValueError(f"Length mismatch: {len(words)} words vs {len(labels)} labels")

            words = [remove_accents(w) for w in words]
            labels = [remove_accents(l) for l in labels]

            words, labels = remove_at_words(words, labels)
            words, labels = remove_links(words, labels)

            return [w.lower() for w in words], labels

        except Exception as e:
            raise ValueError(f"Failed to parse:\n{entry}\n{label_str}\nError: {e}")
    return [], []


def parse_words_dataset(df: pd.DataFrame):
    # parsing words for the model
    new_words = []
    new_lids = []
    for entry, lid in zip(df['words'], df['lid']):
        words, labels = parse_words_and_labels(entry, lid)
        new_words.append(words)
        new_lids.append(labels)
    df['words'] = new_words
    df['lid'] = new_lids

    df['joined_text'] = df['words'].apply(lambda words: ' '.join(words))
