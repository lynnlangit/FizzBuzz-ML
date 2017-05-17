import pickle
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

def fizz_buzz(i):
    if i % 15 == 0:
        return "fizzbuzz"
    elif i % 5 == 0:
        return "buzz"
    elif i % 3 == 0:
        return "fizz"
    else:
        return ""+str(i)

SEQUENCE_LENGTH = 15

def build_labeled_samples(samples):
    padding = ['PAD'] * SEQUENCE_LENGTH
    for i in range(len(samples) - SEQUENCE_LENGTH - 1):
        yield [padding + samples[max(0, i - SEQUENCE_LENGTH):i], samples[i]]
        padding = padding[1:]

def learn():
    num_samples = 15000
    fizz_buzz_samples = [fizz_buzz(i) for i in range(1, num_samples + 1)]
    print "Samples " + str(fizz_buzz_samples)
    samples = list(build_labeled_samples(fizz_buzz_samples))
    labeler = LabelBinarizer()
    labeler.fit(fizz_buzz_samples + ['PAD'])

    X = np.array([np.array(labeler.transform(x)).flatten() for x, y in samples])
    y = np.array([y for x, y in samples])

    X_train, X_test, y_train, y_test = X[0:10000], X[10000:15000], y[0:10000], y[10000:15000],

    classifier = LogisticRegression(tol=1e-6)
    classifier.fit(X_train, y_train)

    print "Score " + str(classifier.score(X_test, y_test))

    with open('binarizer.pkl', 'wb') as binarizer_file:
        pickle.dump(labeler, binarizer_file)
    with open('classifier.pkl', 'wb') as classifier_file:
        pickle.dump(classifier, classifier_file)

if __name__ == "__main__":
    learn()
