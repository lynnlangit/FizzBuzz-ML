import numpy as np
import pickle
import Algorithmia

input = 'lynn'
client = Algorithmia.client('simhJChv6RPBADABAvO1tXnlrQx1')
algo = client.algo('lynnlangit/fizzbuzzML/0.1.0')
print algo.pipe(input)

sequence_length = 15

binarizer_url = 'data://lynnlangit/fizzbuzzML/binarizer.pkl'
classifier_url = 'data://lynnlangit/fizzbuzzML/classifier.pkl'
binarizer_path = client.file(binarizer_url).getFile().name
classifier_path = client.file(classifier_url).getFile().name
with open(binarizer_path, 'rb') as binarizer_file:
    lb = pickle.load(binarizer_file)
with open(classifier_path, 'rb') as classifier_file:
    clf = pickle.load(classifier_file)

def apply(input):
    pad = ['PAD'] * sequence_length
    input = []
    result = []
    for i in range(1, 101):
        X = lb.transform(pad + input)
        predicted = clf.predict([np.array(X).flatten()])[0]
        if predicted == "i":
            result.append(str(i))
        else:
            result.append(predicted)
        input.append(predicted)

        pad = pad[1:]
        if len(input) > sequence_length:
            input = input[1:]
    return result
