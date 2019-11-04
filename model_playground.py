from feature_engineering import process_training_data
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import train_test_split
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

full_df = process_training_data()
X_train, X_val, y_train, y_val = train_test_split(full_df.drop(['author', 'lemmas', 'entities', 'text'], axis=1), full_df.author.values, test_size=0.2, random_state=0)
for col in X_train.columns:
    X_train[col] = X_train[col] - X_train[col].min()
for col in X_val.columns:
    X_val[col] =  X_val[col] - X_val[col].min()



clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_val)
from sklearn.linear_model import LogisticRegression
print ("logloss: %0.3f " % multiclass_logloss(y_val, predictions))
predictions[0]
predictions[:50]
print(multiclass_logloss(y_val, predictions))
y_val[:5]
