import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import re
from collections import Counter
from topic_modeling import get_new_topic_probs, get_topic_probs
import pronouncing
import copy
import textstat
nlp = spacy.load("en_core_web_lg")

def prepare_data():
    '''Initializes DataFrame.'''
    df = pd.read_csv('train.csv')
    df.drop('id', axis=1, inplace=True)
    new_cols = ['raw_text_length', 'num_words', 'avg_word_len', 'vector_avg', 'polarity', 'subjectivity', 'dale_chall', 'rhyme_frequency', 'FleischReadingEase', 'lexicon', 'word_diversity']
    false_cols = ['starts_conj', 'ends_prep', 'has_colon', 'has_semicolon', 'has_dash', 'whom', 'has_had', 'num_ings']
    empty_cols = ['entities', 'lemmas']
    parts_of_speech = ['ADJ', 'ADV', 'ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'EOL', 'NO_TAG', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
    for col in new_cols:
        df[col] = np.zeros(shape=(len(df), ))
    for col in empty_cols:
        df[col] = np.empty(shape=(len(df), ))
    for col in false_cols:
        df[col] = np.zeros(shape=(len(df),))
    for pos in parts_of_speech:
        df[pos] = np.zeros(shape=(len(df),)).astype('uint8')
    encoder = LabelEncoder()
    df['author'] = encoder.fit_transform(df.author.values)
    return df


def rhyme_frequency(text):
    words = list(set([i.lower() for i in text.split(' ')]))
    num_words = len(words)
    num_rhymes = 0
    for i in words:
        other = copy.copy(words)
        other.remove(i)
        rhymes = pronouncing.rhymes(i)
        matches = set(other).intersection(set(rhymes))
        num_rhymes += len(matches)/2
    return num_rhymes/num_words

def add_features(row):
    '''Feature engineering via NLP.'''
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
        if token.text.lower() == 'whom':
            row['whom'] = 1
        if token.text[-3:] == 'ing':
            row['num_ings'] += 1
        if token.text.lower() == 'had':
            row['has_had'] = 1
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
    row['num_ings'] /= row['num_words']
    row['rhyme_frequency'] = rhyme_frequency(row['text'])
    row['dale_chall'] = textstat.dale_chall_readability_score(row['text'])
    row['FleischReadingEase'] = textstat.flesch_reading_ease(row['text'])
    row['lexicon'] = textstat.lexicon_count(row['text'])
    row['word_diversity'] = row.lexicon/row.num_words
    return row

def process_training_data():
    '''Engineers features and performs train test split.'''
    data = prepare_data()
    df = data.apply(lambda x: add_features(x), axis=1)
    df['vector_avg'] = df['vector_avg'] - df['vector_avg'].min()
    topic_probs, lda_model = get_topic_probs(df)
    df = pd.concat([df, topic_probs], axis=1)
    cv = CountVectorizer()
    cv.fit(df.text)
    cv_transformed = cv.transform(df.text).toarray()
    full_df = pd.concat([df, pd.DataFrame(cv_transformed)], axis=1)
    processed_data = train_test_split(full_df.drop(['author', 'lemmas', 'entities'], axis=1), full_df.author.values, test_size=0.1, random_state=0)
    return processed_data # in form X_train, X_val, y_train, y_val
