import pandas as pd
import spacy

import scattertext as st
df = pd.read_csv('train.csv')

df.drop('id', axis=1, inplace=True)
df.head()
nlp = spacy.load('en')
corpus = st.CorpusFromPandas(df,
                             category_col='author',
                             text_col='text',
                             nlp=nlp).build()

html = st.produce_scattertext_explorer(corpus,
                                       category='EAP',
                                       category_name='Edger Allen Poe',
                                       not_category_name='HPL and MWS',
                                       width_in_pixels=1000,
                                       metadata=df['author'])
open("Author-Visualization.html", 'wb').write(html.encode('utf-8'))
