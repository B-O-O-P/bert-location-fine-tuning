import logging
import os
from datetime import datetime

import nltk
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, random_split

from string import punctuation
import torch
from embedding.embedding import vocabulary_from_texts, VocabularyEmbedding
from model import BERT
from transformers import BertTokenizer

BACKGROUNDS = ['ancient', 'bay', 'beach', 'city', 'desert', 'field', 'forest', 'highway', 'hill',
               'historical', 'lake', 'luxury', 'modern', 'mountain', 'road', 'rural', 'sea', 'swamp',
               'tropics', 'valley']

# Logging

LOG_DIRECTORY = 'logs'
LOG_FILENAME = 'testing-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO,
                    format='%(message)s')

logging.info('{} BERT PREDICTION TESTING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

nltk.download('punkt')

# %%

data = pd.read_csv('data/dataset_no_location.csv')

un_texts = list(data['text'])

un_labels = list(data['loc'])
un_labels = list(map(lambda l: l.split()[0], un_labels))

texts = []
labels = []

for text, label in list(zip(un_texts, un_labels)):
    sentences = nltk.tokenize.sent_tokenize(text)
    for sentence in sentences:
        lowered_text = (text.translate(str.maketrans('', '', punctuation))).lower()
        texts.append(lowered_text)
        labels.append(label)

print(len(texts))
print(len(labels))

max_words = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

coco_data = pd.read_csv('data/COCO-locations-negative-sampling.csv')

coco_texts = list(coco_data['cap'])
coco_backgrounds = list(coco_data['background'])

vocabulary = vocabulary_from_texts(coco_texts + coco_backgrounds)

embedding_size = 256

embedding = VocabularyEmbedding(vocabulary, embedding_size)

logging.info('Vocabulary: size {}\n'.format(embedding.vocabulary_length))

backgrounds = [embedding.word_to_ix[label] for label in labels]
backgrounds = np.array(backgrounds)
backgrounds = torch.tensor(backgrounds)

input_ids = []
attention_masks = []

max_length = 256

for text in texts:
    encoded_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=256,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

labels = np.array(labels)

batch_size = 8
print(len(input_ids), len(attention_masks), len(backgrounds))
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(backgrounds))

test_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print("cuda selected")
# If not...
else:
    device = torch.device("cpu")
    print("cpu selected")

device = torch.device("cpu")

model = BERT(embedding.vocabulary_length, 256)

torch.cuda.empty_cache()
model.load_state_dict(torch.load('models/bert-location-prediction-transformer-epoch-10.pt'))
model.to(device)

model.eval()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
log_steps = 5

print(5 * "=", "Testing", 5 * "=")

for step, batch in enumerate(test_dataloader):
    b_input_ids = batch[0]
    b_input_mask = batch[1]
    b_labels = batch[2].long()

    with torch.no_grad():
        max_output = None
        for background in BACKGROUNDS:
            embed_background = embedding(embedding.word_to_ix[background])
            logits = model(b_input_ids, background, input_mask=b_input_mask).squeeze()
            logits = logits.detach().cpu().numpy()
            if max_output == None or max_output < logits:
                max_output = logits
            print(logit)
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = 0
        for (logit, label) in zip(logits, label_ids):
            tmp_eval_accuracy += accuracy(logit, label)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += batch_size

    if step % log_steps == 0 and step:
        print('  STEP {} accuracy: {}'.format(step, eval_accuracy / nb_eval_steps * 100))

#print("Test accuracy: {0:.2f}%".format(eval_accuracy / nb_eval_steps * 100))
