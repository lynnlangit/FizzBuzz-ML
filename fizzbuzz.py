from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import numpy as np
import pickle

#import Algorithmia
#client = Algorithmia.client()

def fizz_buzz(i):
    if i % 15 == 0:
        return "fizzbuzz"
    elif i % 5 == 0:
        return "buzz"
    elif i % 3 == 0:
        return "fizz"
    else:
        return "i"

sequence_length = 15

def build_samples(samples):
    """Create list of features and labels we want to predict"""
    padding = ['PAD'] * sequence_length
    for i in range(len(samples) - sequence_length - 1):
        yield [padding + samples[max(0, i - sequence_length):i], samples[i]]
        padding = padding[1:]


def learn():
    num_samples = 15000

    fizz_buzz_samples = [fizz_buzz(i) for i in range(1, num_samples + 1)]
    print(fizz_buzz_samples)
    samples = list(build_samples(fizz_buzz_samples))

    lb = LabelBinarizer()
    lb.fit(fizz_buzz_samples + ['PAD'])

    X = np.array([np.array(lb.transform(x)).flatten() for x, y in samples])
    y = np.array([y for x, y in samples])

    X_train, X_test, y_train, y_test = X[0:10000], X[10000:15000], y[0:10000], y[10000:15000], 

    # Learn a simple logistic regression classifier
    clf = LogisticRegression(tol=1e-6)
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

    # save binarize and classifier
    with open('binarizer.pkl', 'wb') as binarizer_file:
        pickle.dump(lb, binarizer_file)

    with open('classifier.pkl', 'wb') as classifier_file:
        pickle.dump(clf, classifier_file)

if __name__ == "__main__":
    learn()
