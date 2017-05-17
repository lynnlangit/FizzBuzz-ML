import numpy as np
import pickle
import Algorithmia

INPUT = '0'
CLIENT = Algorithmia.client('simhJChv6RPBADABAvO1tXnlrQx1')
ALGO = CLIENT.algo('lynnlangit/fizzbuzzML/0.1.0')
print ALGO.pipe(INPUT)

SEQUENCE_LENGTH = 15

BINARIZER_URL = 'data://lynnlangit/fizzbuzzML/binarizer.pkl'
CLASSIFIER_URL = 'data://lynnlangit/fizzbuzzML/classifier.pkl'
BINARIZER_PATH = CLIENT.file(BINARIZER_URL).getFile().name
CLASSIFIER_PATH = CLIENT.file(CLASSIFIER_URL).getFile().name

with open(BINARIZER_PATH, 'rb') as binarizer_file:
    LABEL = pickle.load(binarizer_file)
with open(CLASSIFIER_PATH, 'rb') as classifier_file:
    CLASSIFIER = pickle.load(classifier_file)

def apply(input):
    pad = ['PAD'] * SEQUENCE_LENGTH
    input = []
    result = []
    for i in range(1, 101):
        val = LABEL.transform(pad + input)
        predicted = CLASSIFIER.predict([np.array(val).flatten()])[0]
        if predicted == "i":
            result.append(i)
        else:
            result.append(predicted)
        input.append(predicted)

        pad = pad[1:]
        if len(input) > SEQUENCE_LENGTH:
            input = input[1:]
    return result
