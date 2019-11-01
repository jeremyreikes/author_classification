import pandas as pd
import spacy

import scattertext as st
df = pd.read_csv('for_vis.csv')
df = df.dropna()
# df.drop('id', axis=1, inplace=Tru
df.text.isna().sum()
nlp = spacy.load('en')
corpus = st.CorpusFromPandas(df.dropna(),
                             category_col='author',
                             text_col='text',
                             nlp=nlp).build()

html = st.produce_scattertext_explorer(corpus,
                                       category='EAP')
open("Author-Visualization.html", 'wb').write(html.encode('utf-8'))


df.author
df.text
