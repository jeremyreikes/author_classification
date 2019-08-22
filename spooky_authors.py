import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import gensim
import re
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv('train.csv')
df.drop('id', axis=1, inplace=True)

new_cols = ['entities', 'lemmas', 'raw_text_length', 'num_words', 'avg_word_len', 'vector_avg', 'polarity', 'subjectivity']
false_cols = ['starts_conj', 'ends_prep', 'has_colon', 'has_semicolon', 'has_dash']
for col in new_cols:
    df[col] = np.nan
for col in false_cols:
    df[col] = np.zeros(shape=(len(df),))
parts_of_speech = ['ADJ', 'ADV', 'ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'EOL', 'NO_TAG', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
for pos in parts_of_speech:
    df[pos] = np.zeros(shape=(len(df),)).astype('uint8')
encoder = LabelEncoder()
df['author'] = encoder.fit_transform(df.author.values)
nlp = spacy.load("en_core_web_lg")

def add_features(row):
    text = row.text
    doc = nlp(text)
    lemmas = list()
    entities = list()
    for token in doc:
        if token.text == ':':
            row['has_colon'] = 1
        if token.text == ';':
            row['has_semicolon'] = 1
        if token.text == '-':
            row['has_dash'] = 1
        pos = token.pos_
        row[pos] += 1
        if token.is_stop or not token.is_alpha:
            continue
        lemma = token.lemma_.strip().lower()
        if lemma:
            lemmas.append(lemma)
    for ent in doc.ents:
        entities.append(ent.text)
    lemmas = ' '.join(lemmas)
    blob = TextBlob(text)
    row['subjectivity'] = blob.sentiment.subjectivity
    row['polarity'] = blob.sentiment.polarity
    row['starts_conj'] = int(doc[0].pos_ == 'CONJ')
    row['ends_prep'] = int(doc[0].pos_ == 'PREP')
    row['entities'] = entities
    row['lemmas'] = lemmas
    row['raw_text_length'] = len(text)
    row['num_words'] = len(doc)
    row['avg_word_len'] = row.raw_text_length / row.num_words
    row['vector_avg'] = np.mean(nlp(lemmas).vector)
    return row
df = df.apply(lambda x: add_features(x), axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(df.drop(['author', 'lemmas', 'entities'], axis=1), df.author.values, test_size=0.25, random_state=0)

tfidf = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfidf.fit(np.concatenate([X_train.text, X_valid.text]))
X_train_tfidf = tfidf.transform(X_train.text)
X_valid_tfidf = tfidf.transform(X_valid.text)
X_train_full = X_train.append(pd.DataFrame(X_train_tfidf.toarray()), ignore_index=True)
X_valid_full = X_valid.append(pd.DataFrame(X_valid_tfidf.toarray()), ignore_index=True)

X_train_final = X_train_full.drop('text', axis=1)
X_valid_final = X_valid_full.drop('text', axis=1)

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train_final, y_train)
preds = model.predict(X_valid_final)
acc = accuracy_score(y_valid, preds)
print('XGBOOOST TFIDF Accuracy: ', acc)
