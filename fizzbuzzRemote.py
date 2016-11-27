import numpy as np
import pickle

import Algorithmia

client = Algorithmia.client()

sequence_length = 15

binarizer_url = 'data://<yourname>/fizzbuzzML/binarizer.pkl'
classifier_url = 'data://<yourname>/fizzbuzzML/classifier.pkl'
binarizer_path = client.file(binarizer_url).getFile().name
classifier_path = client.file(classifier_url).getFile().name
with open(binarizer_path, 'rb') as binarizer_file:
    lb = pickle.load(binarizer_file)
with open(classifier_path, 'rb') as classifier_file:
    clf = pickle.load(classifier_file)

def apply(input):
    pad = ['PAD'] * sequence_length
    input = []
    res = []
    for i in range(1, 101):
        X = lb.transform(pad + input)
        predicted = clf.predict([np.array(X).flatten()])[0]
        if predicted == "i":
            res.append(str(i))
        else:
            res.append(predicted)
        
        input.append(predicted)

        pad = pad[1:]
        if len(input) > sequence_length:
            input = input[1:]
    return res