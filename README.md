# Spooky Author Identification
A solution to [Kaggle's Spooky Author Identification Competition](https://www.kaggle.com/c/spooky-author-identification).

## The Problem
Using horror stories from Edgar Allen Poe, Mary Shelley, and HP Lovecraft, build a model to predict the author of unseen sentences.  

The dataset contained ~20,000 sentences from the three authors. The following interactive visualization we made with scattertext shows differences in corpora between Edgar Allen Poe and the other two authors.
![scatterText visualization](scatter_text_vis.png)

## Evaluation
Submissions are evaluated using a multi-class logarithmic loss function as follows as shown in [multi-class log-loss]('https://www.kaggle.com/c/spooky-author-identification/overview/evaluation').

## Our Solution

### Feature Engineering
We leveraged popular NLP packages (Spacy, NLTK, TextStat, TextBlob, and Gensim) to tokenize, vectorize, and parse the dataset.

We created natural language features based around many features such as the following:
- Word choice (count vectorizer)
- Topic modeling (LDA), including bigrams and trigrams
- Grammar/punctuation
- Verb tense
- Rhyming frequency
- Entity recognition
- Sentiment (polarity/subjectivity)
- Readability (using Fleisch and Dale-Chall metrics)
- Etc.

See our [feature engineering file](https://github.com/jeremyreikes/author_classification/blob/master/feature_engineering.py) for more info.

#### TFIDF/Count Vectorizer
Scikit-learn's built in TFIDF and CountVectorizer proved to be our most important feature.  Count Vectorizer performed slightly better so we implemented it instead of TFIDF for our final product.  

After removing stopwords and punctuation, the corpus left us with ~25,000 features for the count vectorizer.

#### Topic Modeling
We utilized Gensim for topic modeling and used Latent Dirichlet Allocation (LDA) to create model.  We added bigram and trigram models to capture patterns in each author's word usage.  The following image shows an interactive visualization for viewing differences between topics.  Our highest coherence score was obtained with 4 topics.
![topicModeling visualization](topic_modeling_vis.png)

#### Other metrics
Our other features were obtained using a combination of Spacy, NLTK, TextBlob, and TextStat.

### Modeling 
