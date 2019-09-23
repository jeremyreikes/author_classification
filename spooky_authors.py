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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
import warnings
from topic_modeling import get_topics
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
df['topic'] = get_topics(df.text.values.tolist(), num_topics = 6)
dummies = pd.get_dummies(df.topic)
df = pd.concat([df, dummies], axis=1)

# df = pd.read_csv('dfwhatup.csv')
remy = pd.read_csv('stacking.csv')
remy.columns = [str(i) + '_x' for i in range(len(remy.columns))]
remy_test = pd.read_csv('stacked_test.csv')
remy_test.shape

remy.shape
remy_test.columns = [str(i) + '_x' for i in range(len(remy_test.columns))]
X_train, X_valid, y_train, y_valid = train_test_split(df.drop(['author', 'lemmas', 'entities'], axis=1), df.author.values, test_size=0.25, random_state=0)

tfidf = CountVectorizer()

tfidf.fit(np.concatenate([X_train.text, X_valid.text]))
X_train_tfidf = tfidf.transform(X_train.text)
X_valid_tfidf = tfidf.transform(X_valid.text)

x_train_array = pd.DataFrame(X_train_tfidf.toarray())
x_valid_array = pd.DataFrame(X_valid_tfidf.toarray())

X_train.reset_index(inplace=True)
X_valid.reset_index(inplace=True)
X_train_full = pd.concat([x_train_array, X_train], axis=1)
X_valid_full = pd.concat([x_valid_array, X_valid], axis=1)
X_train_final = X_train_full.drop(['text', 'index'], axis=1)
X_valid_final = X_valid_full.drop(['text', 'index'], axis=1)

# X_train_final['vector_avg'] = X_train_final['vector_avg'] - X_train_final.vector_avg.min()
# X_valid_final['vector_avg'] = X_valid_final['vector_avg'] - X_valid_final.vector_avg.min()
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from xgboost import XGBClassifier



for col in X_train_final.columns:
    if X_train_final[col].values.min() < 0:
        X_train_final[col] = X_train_final[col] - X_train_final[col].values.min()

for col in X_valid_final.columns:
    if X_valid_final[col].values.min() < 0:
        X_valid_final[col] = X_valid_final[col] - X_valid_final[col].values.min()

model = MultinomialNB(alpha=.1)
model.fit(X_train_final, y_train)
X_train_final.shape
X_valid_final.shape
# model = MultinomialNB(alpha=.1)
# model.fit(X_train_final, y_train)
preds = model.predict(X_valid_final)
acc = accuracy_score(y_valid, preds)
print('TFIDF Accuracy: ', acc)



X_train_final.iloc[1, -35:]

X_train_final.to_csv('X_train.csv')
X_valid_final.to_csv('X_valid.csv')
