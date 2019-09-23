import pandas as pd
import numpy as np

def initialize(df):
    new_cols = ['entities', 'lemmas', 'raw_text_length', 'num_words', 'avg_word_len', 'vector_avg', 'polarity', 'subjectivity']
    false_cols = ['starts_conj', 'ends_prep', 'has_colon', 'has_semicolon', 'has_dash']
    for col in new_cols:
        df[col] = np.nan
    for col in false_cols:
        df[col] = np.zeros(shape=(len(df),))
    parts_of_speech = ['ADJ', 'ADV', 'ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'EOL', 'NO_TAG', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
    for pos in parts_of_speech:
        df[pos] = np.zeros(shape=(len(df),)).astype('uint8')
    return df
