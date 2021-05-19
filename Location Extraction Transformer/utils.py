import numpy as np

from nltk import download, WordNetLemmatizer

download('wordnet')

MAX_WORDS = 256

def extraction_accuracy(pred, label, verbose=False):
    pred_words = []
    label_words = []
    for i in range(len(pred)):
        if pred[i] == 1:
            pred_words.append(i)
    for i in range(len(label)):
        if label[i] == 1:
            label_words.append(i)
    if verbose:
        print(pred_words)
        print(label_words)
    return all(word in pred_words for word in label_words)


def one_hot_location(words, label_array):
    words_no = words.split(' ')
    Lem = WordNetLemmatizer()
    words = [Lem.lemmatize(word) for word in words_no]
    N = MAX_WORDS
    one_hot_label = np.zeros(N)

    for i in range(len(words)):
        if words[i] in label_array:
            one_hot_label[i] = 1

    return np.array(one_hot_label, dtype=np.float64)
