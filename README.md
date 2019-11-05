# Spooky Author Identification
A solution to [Kaggle's Spooky Author Identification Competition](https://www.kaggle.com/c/spooky-author-identification)

## The Problem
Using horror stories from Edgar Allen Poe, Mary Shelley, and HP Lovecraft, build a model to predict the author of unseen sentences.  

The dataset contained ~20,000 sentences from the three authors. The following interactive visualization we made with scattertext shows differences in corpora between Edgar Allen Poe and the other two authors.
![scatterText visualization](scatter_text_vis.png)

## Evaluation
Submissions are evaluated using a multi-class logarithmic loss function as follows.

logloss=−1N∑i=1N∑j=1Myijlog(pij)

## Our Solution
We leveraged popular NLP packages (Spacy, NLTK, TextStat, TextBlob, and Gensim) to tokenize, vectorize, and parse the dataset.

We created natural language features based around word choice, sentence structure, grammar, tense, and more.

## Topic modeling
We utilized Gensim for topic modeling.  The following image shows an interactive visualization for viewing differences between topics.  Our highest coherence score was obtained with 4 topics.
![topicModeling visualization](topic_modeling_vis.png)
