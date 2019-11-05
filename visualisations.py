import pandas as pd
import spacy
import pyLDAvis.gensim
pyLDAvis.enable_notebook() # visualize
import scattertext as st
from topic_modeling import get_topic_probs
nlp = spacy.load('en')

def create_scatterplot(df, return_corpus=False):
    '''Creates an HTML file to visualize differences in corpora.'''
    corpus = st.CorpusFromPandas(df,
                                 category_col='author',
                                 text_col='text',
                                 nlp=nlp).build()
    if return_corpus:
        return corpus
    html = st.produce_scattertext_explorer(corpus,
                                           category='EAP',
                                           category_name='Edger Allen Poe',
                                           not_category_name='HPL/MWS',
                                           width_in_pixels=1000,
                                           metadata=df['author'])
    open("Author-Visualization.html", 'wb').write(html.encode('utf-8'))

def create_pyLDAvis(df):
    '''.ipynb enabled interactive topic-modeling visualization from pyLDAvis.'''
    lda_model, corpus = get_topic_probs(df, for_vis=True)
    pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # return vis
