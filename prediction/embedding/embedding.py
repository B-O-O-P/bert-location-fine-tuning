import torch
import torch.nn as nn
import nltk
from string import punctuation

torch.manual_seed(1)

nltk.download('punkt')

def vocabulary_from_texts(texts):
    vocab = set()

    for text in texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        for sentence in sentences:
            lowered_text = (sentence.translate(str.maketrans('', '', punctuation))).lower()
            words_no = lowered_text.split(' ')
            Lem = nltk.WordNetLemmatizer()
            words = [Lem.lemmatize(word) for word in words_no]
            vocab.update(words)

    vocab.remove('')

    return vocab

class VocabularyEmbedding(nn.Embedding):
    def __init__(self, vocabulary, embedding_size):
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary
        self.vocabulary_length = len(vocabulary)
        self.word_to_ix = {word: i for i, word in enumerate(vocabulary)}
        super(VocabularyEmbedding, self).__init__(self.vocabulary_length, self.embedding_size)

    def __call__(self, *args, **kwargs):
        word = args[0]
        lookup_tensor =  torch.tensor([self.word_to_ix[word]], dtype=torch.long)
        return super(VocabularyEmbedding, self).__call__(lookup_tensor)

