import torch
import torch.nn as nn
import nltk

from string import punctuation
from tqdm.notebook import tqdm

torch.manual_seed(1)

nltk.download('punkt')

def vocabulary_from_texts(texts, verbose=False):
    vocab = set()

    if verbose:
        texts = tqdm(texts)
    for text in texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        for sentence in sentences:
            lowered_text = (sentence.translate(str.maketrans('', '', punctuation))).lower()
            words_no = lowered_text.split(' ')
            words = [word.strip() for word in words_no]
            vocab.update(words)

    if '' in vocab:
        vocab.remove('')

    return vocab

class VocabularyEmbedding(nn.Embedding):
    def __init__(self, vocabulary, embedding_size):
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary
        self.vocabulary_length = len(vocabulary)
        self.word_to_ix = {word: i for i, word in enumerate(vocabulary)}
        super(VocabularyEmbedding, self).__init__(self.vocabulary_length, self.embedding_size, padding_idx=0)

    def __call__(self, *args, **kwargs):
        return super(VocabularyEmbedding, self).__call__(*args)

