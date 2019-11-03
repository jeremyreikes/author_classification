import gensim
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess, lemmatize
import re
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import numpy as np

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'use', 'not', 'would', 'say', 'could', '_', 'be',
                   'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank',
                   'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem',
                   'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = simple_preprocess(str(sent), deacc=True)
        yield(sent)

def create_words(df):
    data = df.text.values.tolist()
    data_words = list(sent_to_words(data))
    return data_words

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out

def model_lda(df):
    data_words = create_words(df)
    data_ready = process_words(data_words)
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=4,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=10,
                                               passes=4,
                                               alpha='symmetric',
                                               per_word_topics=True)

    return lda_model, corpus, data_ready

def get_topic_probs(df):
    lda_model, corpus, data_ready = model_lda(df)
    id2word = corpora.Dictionary(data_ready)
    topic_probs = list()
    for doc in corpus:
        row = [0, 0, 0, 0]
        vector = lda_model[doc][0]
        for tup in vector:

            row[tup[0]] = tup[1]
        topic_probs.append(row)
    return np.sqrt(topic_probs), lda_model

def get_new_topic_probs(df, lda_model):
    data_words = create_words(new_df)
    data_ready = process_words(data_words)
    id2word = corpora.Dictionary(data_ready)
    new_corpus = [id2word.doc2bow(text) for text in data_ready]
    topic_probs = list()
    for doc in new_corpus:
        vector = lda_model[doc][0]
        row = [0, 0, 0, 0]
        for tup in vector:
            row[tup[0]] = tup[1]
        topic_probs.append(row)

    return np.sqrt(topic_probs)
'''
to get topic probs for train data, run train_topic_probs.  Then pass in the lda_model for the train topic probs
'''
# df = pd.read_csv('train.csv')
# train_topic_probs, lda_model = get_topic_probs(df)
# new_topic_probs = get_new_topic_probs(val_df, lda_model)
