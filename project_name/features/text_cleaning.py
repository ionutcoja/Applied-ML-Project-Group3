import pandas as pd
import re
import unicodedata


def remove_accents(text) -> str:
    """
    Removes accents from a given string using Unicode normalization.

    Args:
        text (str): The input string.
    Returns:
        str: The string with accents removed.
    """
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))


def remove_at_words(words, labels) -> tuple[list[str], list[str]]:
    """
    Removes the words starting with '@' and the word 'RT' from the list (tweet noise).

    Args:
        words (list[str]): List of words.
        labels (list[str]): Corresponding list of labels.

    Returns:
        tuple[list[str], list[str]]: Filtered words and labels.
    """

    filtered = [(w, l) for w, l in zip(words, labels) if not (w.startswith("@") or w == "RT")]
    return zip(*filtered) if filtered else ([], [])


def remove_links(words, labels) -> tuple[list[str], list[str]]:
    """
    Remove words that are links (starting with 'http') from the list.

    Args:
        words (list[str]): List of words.
        labels (list[str]): Corresponding list of labels.
    Returns:
        tuple[list[str], list[str]]: Filtered words and labels without links.
    """

    filtered = [(w, l) for w, l in zip(words, labels) if not w.startswith("http")]
    return zip(*filtered) if filtered else ([], [])


def fix_and_parse_list_like_string(s) -> list[str]:
    """
    Parse a string representation of a list and extract quoted elements.

    Args:
        s (str): String representation of a list.

    Returns:
        list[str]: List of extracted elements.
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # Extract words/labels with a regex knowing that they are between '' or ""
    quoted_words = re.findall(r"""(['"])(.+?)\1""", s)

    # put all the words/labels found into a list
    words = [match[1] for match in quoted_words]

    return words

def parse_words_and_labels(entry, label_str) -> tuple[list[str], list[str]]:
    """
    Parse and clean words and labels from string representations, ensuring they remain aligned.

    Args:
        entry (str): String representation of words.
        label_str (str): String representation of labels.

    Returns:
        tuple[list[str], list[str]]: Cleaned and aligned words and labels.

    Raises:
        ValueError: If the number of words and labels do not match or parsing fails.
    """
    if isinstance(entry, str) and entry.strip():
        try:

            # adding commas between the words and the language tags so they would be formatted as a proper list
            words = fix_and_parse_list_like_string(entry)
            labels = fix_and_parse_list_like_string(label_str)

            # check whether the number of words matches the number of labels (sanity check)
            if len(words) != len(labels):
                print("Mismatch:")
                print("Words:", words)
                print("Labels:", labels)
                raise ValueError(f"Length mismatch: {len(words)} words vs {len(labels)} labels")

            # clean the text, while keeping the words and labels in sync
            words = [remove_accents(w) for w in words]
            labels = [remove_accents(l) for l in labels]

            words, labels = remove_at_words(words, labels)
            words, labels = remove_links(words, labels)

            return [w.lower() for w in words], labels

        except Exception as e:
            raise ValueError(f"Failed to parse:\n{entry}\n{label_str}\nError: {e}")
    return [], []


def parse_words_dataset(df: pd.DataFrame) -> None:
    """
    Parse and clean the 'words' and 'lid' columns in a DataFrame, updating them in place and adding a 'joined_text' column.

    Args:
        df (pd.DataFrame): DataFrame with 'words' and 'lid' columns to process.

    Returns:
        None - as we just modify the dataset there
    """
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
